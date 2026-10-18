"""Isolated inference worker for notebook 08. Never loads GT."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def state(model, checkpoint):
    import torch
    sd = torch.load(checkpoint, map_location='cpu', weights_only=True)
    for key in ('model_state_dict', 'model', 'state_dict'):
        if key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)


def array(value):
    import torch
    return value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def infer(cfg, model_name, out):
    import torch
    from PIL import Image
    settings = cfg['models'][model_name]
    repo = Path(settings['repo'])
    sys.path.insert(0, str(repo))
    paths = cfg['images']
    device = cfg['device']
    precision = cfg['precision']
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[precision]
    amp = lambda: torch.autocast('cuda', dtype=dtype) if precision != 'fp32' else contextlib.nullcontext()
    checkpoint = settings['checkpoint']
    meta = {'load_strict': True, 'pose_convention': 'world-to-camera, upstream camera axes',
            'depth_semantics': 'native predicted z-depth; direct ERP is an out-of-domain input',
            'protocol': 'direct ERP; no spherical correction', 'wrapper_gpu_validation': 'not run by author'}
    if model_name in ('vggt', 'vggt_omega'):
        if model_name == 'vggt':
            from vggt.models.vggt import VGGT
            from vggt.utils.load_fn import load_and_preprocess_images
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri as decode
            model = VGGT()
            images = load_and_preprocess_images(paths).to(device)
            meta['preprocessing'] = 'upstream load_and_preprocess_images defaults'
        else:
            from vggt_omega.models.vggt_omega import VGGTOmega
            from vggt_omega.utils.load_fn import load_and_preprocess_images
            from vggt_omega.utils.pose_enc import encoding_to_camera as decode
            model = VGGTOmega()
            images = load_and_preprocess_images(paths, image_resolution=settings['resolution']).to(device)
            meta['preprocessing'] = f"upstream image_resolution={settings['resolution']}"
        state(model, checkpoint)
        model.to(device).eval()
        with torch.inference_mode(), amp():
            pred = model(images)
        hw = pred['images'].shape[-2:] if model_name == 'vggt_omega' else images.shape[-2:]
        ext, intr = decode(pred['pose_enc'].float(), hw)
        result = {k: array(v)[0] for k, v in pred.items() if k in ('depth', 'depth_conf', 'world_points', 'world_points_conf')}
        result.update(extrinsics=array(ext)[0], intrinsics=array(intr)[0], input_rgb=array(images).transpose(0, 2, 3, 1))
    elif model_name == 'omnivggt':
        from omnivggt.models.omnivggt import OmniVGGT
        from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri
        from visual_util import load_images_and_cameras
        from safetensors.torch import load_file
        # Staged images contain only the selected input files, in explicit order.
        model = OmniVGGT()
        model.load_state_dict(load_file(checkpoint), strict=True)
        model.to(device).eval()
        images, e, k, d, mask, di, ci = load_images_and_cameras(
            cfg['staged_dir'], camera_folder=None, depth_folder=None, target_size=settings['resolution'])
        if di or ci or len(images) != len(paths):
            raise ValueError('Unexpected auxiliary inputs or image count')
        with torch.inference_mode(), amp():
            pred = model.inference(images=images.to(device), extrinsics=e.to(device), intrinsics=k.to(device),
                                   depth=d.to(device), mask=mask.to(device), depth_gt_index=[], camera_gt_index=[])
        ext, intr = pose_encoding_to_extri_intri(pred['pose_enc'].float(), images.shape[-2:])
        result = {k: array(v)[0] for k, v in pred.items() if k in ('depth', 'depth_conf', 'world_points', 'world_points_conf')}
        result.update(extrinsics=array(ext)[0], intrinsics=array(intr)[0])
        meta['preprocessing'] = f"official load_images_and_cameras target_size={settings['resolution']}; ImageNet normalization"
    elif model_name == 'panovggt':
        from omegaconf import OmegaConf
        spec = importlib.util.spec_from_file_location('benchmark_pano_api', repo / 'inference.py')
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        config = OmegaConf.load(settings['config'])
        OmegaConf.resolve(config)
        mc = config.model
        model = api.PanoVGGTModel(img_size=config.img_size, patch_size=config.patch_size, embed_dim=config.embed_dim,
            enable_camera=mc.enable_camera, enable_depth=mc.enable_depth, enable_point=mc.enable_point,
            aggregator=OmegaConf.to_container(mc.aggregator, resolve=True))
        # Unlike the official demo's strict=False, fail on any incompatible keys.
        state(model, checkpoint)
        model.to(device).eval()
        images = api.load_images_fixed(paths).to(device)
        with torch.inference_mode(), amp():
            pred = model(images.unsqueeze(0))
        result = {k: array(v)[0] for k, v in pred.items() if k in ('depth', 'depth_conf', 'world_points', 'world_points_conf', 'camera_poses') and v is not None}
        result['input_rgb'] = array(images).transpose(0, 2, 3, 1)
        meta.update(protocol='native ERP', preprocessing='official load_images_fixed, 518 x 1036',
                    depth_semantics='native panoramic depth', pose_convention='upstream camera_poses camera-to-world')
    elif model_name == 'vggt360':
        from comparison.upstream_fixes import fix_vggt360
        meta['upstream_fixes'] = [fix_vggt360(repo)]
        from vggt_visfeat.models.vggt import VGGT
        from vggt_visfeat.utils.load_fn2 import load_and_preprocess_images
        from utils.gen_views import view_generation
        from utils.ERP_utils import ERP2Persp, depth_set_to_equirect_attention
        from utils.att_utils import SA_confidence
        model = VGGT()
        state(model, checkpoint)
        model.to(device).eval()
        depths = []
        for i, path in enumerate(paths):
            print(f'Panorama {i + 1}/{len(paths)}', flush=True)
            rgb = np.asarray(Image.open(path).convert('RGB').resize((1024, 512), Image.Resampling.BICUBIC))
            views = view_generation(rgb, settings['fov'])
            faces, masks, valid = [], [], []
            for yaw, pitch, fov in views:
                face = ERP2Persp(rgb, FOV=fov, THETA=yaw, PHI=pitch, height=512, width=512, mode='bilinear')
                m, v = SA_confidence(face)
                faces.append(Image.fromarray(np.clip(face, 0, 255).astype(np.uint8)))
                masks.append(m); valid.append(v)
            images = load_and_preprocess_images(faces).to(device)
            with torch.inference_mode(), amp():
                pred, att = model(images=images, persp_masks=torch.from_numpy(np.asarray(masks)), rgb_masks=torch.from_numpy(np.asarray(valid)))
            distances = np.linalg.norm(array(pred['world_points'])[0], axis=-1)
            depths.append(depth_set_to_equirect_attention(depths=distances, view_params=views, attention_maps=att, H=512, W=1024, interp='linear'))
            np.savez_compressed(out / f'virtual_views_{i:03d}.npz', view_params=np.asarray(views), pose_enc=array(pred['pose_enc']))
            del pred, att, images
        result = {'depth': np.asarray(depths, dtype=np.float32)}
        meta.update(protocol='official adaptive perspective views and attention fusion', preprocessing='ERP resize 512 x 1024; perspective views 512 x 512',
                    depth_semantics='official fused world-point norm, not general camera-centred radial depth',
                    pose_convention='no temporal poses; virtual poses saved separately', validation='experimental local-checkpoint wrapper')
    else:
        raise ValueError('Argus is pending checkpoint access and is not implemented')
    depth = result['depth']
    if depth.ndim == 4 and depth.shape[-1] == 1:
        result['depth'] = depth[..., 0]
    if result['depth'].ndim != 3 or len(result['depth']) != len(paths):
        raise ValueError(f"Unexpected depth shape: {result['depth'].shape}")
    return result, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    import torch
    if cfg['device'] != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This notebook currently requires CUDA; CPU inference is untested')
    if cfg['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is unsupported on this GPU')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    arrays, meta = infer(cfg, args.model, out)
    stats = {}
    for key, value in arrays.items():
        value = np.asarray(value, dtype=np.float32)
        arrays[key] = value
        finite = np.isfinite(value)
        stats[key] = {'shape': list(value.shape), 'finite_fraction': float(finite.mean())}
        if finite.any():
            stats[key].update(min=float(value[finite].min()), max=float(value[finite].max()))
    if not np.any(np.isfinite(arrays['depth']) & (arrays['depth'] > 0)):
        raise ValueError('No positive finite depth predictions')
    np.savez_compressed(out / 'predictions.npz', **arrays)
    torch.cuda.synchronize()
    settings = cfg['models'][args.model]
    repo = settings['repo']
    git = subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True)
    meta.update(model=args.model, request_hash=cfg['request_hash'], input_images=cfg['image_records'],
        checkpoint_sha256=sha256(settings['checkpoint']), settings=settings, precision=cfg['precision'],
        upstream_commit=git.stdout.strip() if git.returncode == 0 else None,
        gt_used=False, metric_scale_verified=False, statistics=stats,
        seconds_including_load=time.perf_counter() - started, peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
        gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
        output_sha256=sha256(out / 'predictions.npz'))
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print('Completed:', out, flush=True)


if __name__ == '__main__':
    main()
