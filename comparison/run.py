"""One model per process. Outputs are qualitative; GT is never loaded here."""
import argparse,hashlib,importlib.util,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from comparison.projection import merge_faces, resize_depth, z_to_radial_intrinsics
ROOT=Path(__file__).resolve().parents[1]

def numpy(t):
    return t.detach().float().cpu().numpy()

def upstream(name):
    spec=json.loads((ROOT/'comparison/sources.json').read_text())[name]
    path=ROOT/'third_party'/spec['folder']
    if not path.exists():raise FileNotFoundError('Run python -m comparison.setup_sources first')
    sha=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if sha!=spec['commit']:raise RuntimeError(f'{name}: upstream commit does not match sources.json')
    sys.path.insert(0,str(path))
    return path,sha

def panovggt(images,checkpoint):
    import torch
    from omegaconf import OmegaConf
    from unittest.mock import patch
    path=ROOT/'third_party/PanoVGGT'
    if not (path/'inference.py').exists():raise FileNotFoundError('Run git submodule update --init --recursive')
    sys.path.insert(0,str(path))
    spec=importlib.util.spec_from_file_location('pano_comparison_upstream',path/'inference.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    from panovggt.models.aggregator import Aggregator
    cfg=OmegaConf.load(path/'training/config/default.yaml');OmegaConf.resolve(cfg);mc=cfg.model
    if not checkpoint or not Path(checkpoint).is_file():raise FileNotFoundError('Provide --checkpoint for PanoVGGT')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('PanoVGGT runner requires BF16-capable CUDA')
    with patch.object(Aggregator,'_try_load_dinov2',lambda *a,**k:None):
        model=mod.PanoVGGTModel(img_size=cfg.img_size,patch_size=cfg.patch_size,embed_dim=cfg.embed_dim,
            enable_camera=mc.enable_camera,enable_depth=mc.enable_depth,enable_point=mc.enable_point,
            aggregator=OmegaConf.to_container(mc.aggregator,resolve=True))
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    for key in ('model_state_dict','model','state_dict'):
        if key in state:state=state[key];break
    state={(k[7:] if k.startswith('module.') else k):v for k,v in state.items()}
    model.load_state_dict(state,strict=True);del state
    model.eval().cuda();pred=mod.run_inference(model,images,'cuda')
    depths=np.asarray(pred['depth'])
    if depths.ndim==4 and depths.shape[-1]==1:depths=depths[...,0]
    if depths.ndim!=3 or len(depths)!=len(images):raise ValueError('Unexpected PanoVGGT depth shape')
    sha=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    return list(depths),pred.get('camera_poses'),{'upstream_commit':sha,'depth_semantics':'radial depth head',
            'scale':'model units; metric accuracy unverified','pose_convention':'upstream camera-to-world',
            'checkpoint':str(Path(checkpoint).resolve())}

def vggt360(images,out,model_id):
    import torch
    _,sha=upstream('vggt360')
    from vggt_visfeat.models.vggt import VGGT
    from vggt_visfeat.utils.load_fn2 import load_and_preprocess_images
    from utils.gen_views import view_generation
    from utils.ERP_utils import ERP2Persp, depth_set_to_equirect_attention
    from utils.att_utils import SA_confidence
    model=VGGT.from_pretrained(model_id).eval().cuda()
    depths=[]
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for idx,path in enumerate(images):
        print(f'VGGT-360 panorama {idx+1}/{len(images)}',flush=True)
        # Match official benchmark input size; keep its sampling and fusion unchanged.
        rgb=np.array(Image.open(path).convert('RGB').resize((1024,512),Image.Resampling.BICUBIC))
        views=view_generation(rgb,110);faces=[];masks=[];valid=[]
        for yaw,pitch,fov in views:
            face=ERP2Persp(rgb,FOV=fov,THETA=yaw,PHI=pitch,height=512,width=512,mode='bilinear')
            m,v=SA_confidence(face);faces.append(Image.fromarray(np.clip(face,0,255).astype(np.uint8)));masks.append(m);valid.append(v)
        inp=load_and_preprocess_images(faces).cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):
            pred,att=model(images=inp,persp_masks=torch.from_numpy(np.array(masks)),rgb_masks=torch.from_numpy(np.array(valid)))
        # Reproduce upstream's world-point norm, which is not a general camera-origin depth conversion.
        radial=np.linalg.norm(numpy(pred['world_points'][0]),axis=-1).astype(np.float32)
        depth=depth_set_to_equirect_attention(depths=radial,view_params=views,attention_maps=att,H=512,W=1024,interp='linear')
        depths.append(np.asarray(depth,dtype=np.float32))
        np.savez_compressed(out/f'virtual_views_{idx:03d}.npz',view_params=np.asarray(views),pose_enc=numpy(pred['pose_enc']))
        del pred,att,inp
    return depths,None,{'upstream_commit':sha,'depth_semantics':'official fused norm of world_points; nominal panorama radial distance',
        'scale':'relative; independent gauge per panorama','model_id':model_id,
        'pose_note':'Virtual-view poses saved separately; no temporal trajectory from this per-panorama method',
        'sampling_note':'Official upstream projection/fusion grids retained; small pixel-grid differences from endpoint ERP are possible'}

def omnivggt(faces,job,checkpoint):
    import torch
    _,sha=upstream('omnivggt')
    from omnivggt.models.omnivggt import OmniVGGT
    from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri
    from visual_util import load_images_and_cameras
    from safetensors.torch import load_file
    if not checkpoint:
        from huggingface_hub import hf_hub_download
        checkpoint=hf_hub_download('Livioni/OmniVGGT','OmniVGGT.safetensors')
    model=OmniVGGT();model.load_state_dict(load_file(str(checkpoint)),strict=True);model.eval().cuda()
    images,e,k,d,m,di,ci=load_images_and_cameras(str(job/'faces'),target_size=518)
    if len(images)!=len(faces) or di or ci:raise ValueError('Unexpected face inputs')
    # Official inference path, no auxiliary camera/depth observations.
    with torch.inference_mode():
        pred=model.inference(images=images.cuda(),extrinsics=e.cuda(),intrinsics=k.cuda(),depth=d.cuda(),mask=m.cuda(),depth_gt_index=[],camera_gt_index=[])
        ext,intr=pose_encoding_to_extri_intri(pred['pose_enc'],images.shape[-2:])
    ext=numpy(ext[0]);poses=np.tile(np.eye(4),(len(ext),1,1));poses[:,:3]=ext;poses=np.linalg.inv(poses)
    z=numpy(pred['depth'][0,...,0]);ks=numpy(intr[0])
    # Radial depth according to the model's own predicted pinhole calibration.
    depths=[]
    for depth,K in zip(z,ks):
        depths.append(z_to_radial_intrinsics(depth,K))
    return depths,poses,{'upstream_commit':sha,'depth_semantics':'z-depth converted to radial using predicted intrinsics',
        'scale':'relative; joint inference gauge','checkpoint':str(checkpoint),
        'pose_convention':'OpenCV camera-to-world','adaptation':'six virtual pinhole views; rays placed on ERP using known crop geometry'}

def mapanything(faces,model_id):
    _,sha=upstream('mapanything')
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images
    model=MapAnything.from_pretrained(model_id).eval().cuda()
    views=load_images(faces,resize_mode='square',size=518)
    pred=model.infer(views,memory_efficient_inference=True,minibatch_size=1,use_amp=True,
                     amp_dtype='bf16',apply_mask=False,mask_edges=False,apply_confidence_mask=False)
    depths=[numpy(p['depth_along_ray'][0,...,0]) for p in pred]
    poses=np.stack([numpy(p['camera_poses'][0]) for p in pred])
    return depths,poses,{'upstream_commit':sha,'depth_semantics':'predicted depth_along_ray',
        'scale':'model predicts metric scale; CARLA accuracy unverified','model_id':model_id,
        'pose_convention':'OpenCV camera-to-world','adaptation':'six virtual pinhole views; image-only, no supplied calibration'}

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--job',required=True);p.add_argument('--model',required=True,choices=['panovggt','vggt360','omnivggt','mapanything'])
    p.add_argument('--checkpoint');p.add_argument('--model-id')
    a=p.parse_args(argv);job=Path(a.job).resolve();manifest_path=job/'manifest.json'
    mf=json.loads(manifest_path.read_text());out=job/'predictions'/a.model
    if out.exists():raise FileExistsError(f'{out} exists; choose a new job or move that model output aside')
    images=[str(job/r['image']) for r in mf['frames']];faces=[str(job/f) for r in mf['frames'] for f in r['faces']]
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA PyTorch environment required; check nvidia-smi and torch.cuda.is_available()')
    out.mkdir(parents=True);started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    if a.model=='panovggt':depths,poses,meta=panovggt(images,a.checkpoint)
    elif a.model=='vggt360':depths,poses,meta=vggt360(images,out,a.model_id or 'facebook/VGGT-1B')
    elif a.model=='omnivggt':depths,poses,meta=omnivggt(faces,job,a.checkpoint)
    else:depths,poses,meta=mapanything(faces,a.model_id or 'facebook/map-anything')
    if a.model in ('omnivggt','mapanything'):
        if len(depths)!=6*len(images):raise ValueError('Incorrect number of virtual-view predictions')
        np.savez_compressed(out/'virtual_view_predictions.npz',radial_depth=np.stack(depths),poses_c2w=poses)
        depths=[merge_faces(depths[i:i+6],mf['height'],mf['width'],mf['face_fov_deg']) for i in range(0,len(depths),6)]
        # Use the front virtual view for each panorama; do not average unrelated rotations.
        poses=poses[::6]
    else:
        depths=[resize_depth(d,mf['height'],mf['width']) for d in depths]
    depth=np.stack(depths).astype(np.float32);depth[~np.isfinite(depth)|(depth<=0)]=np.nan
    if not np.any(np.isfinite(depth)):raise ValueError('No valid depth predictions')
    arrays={'radial_depth':depth,'frame_ids':np.array([r['frame_id'] for r in mf['frames']])}
    if poses is not None:arrays['poses_c2w']=np.asarray(poses)
    np.savez_compressed(out/'predictions.npz',**arrays)
    torch.cuda.synchronize()
    meta.update(model=a.model,manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        seconds_including_load=time.perf_counter()-started,gpu=torch.cuda.get_device_name(),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,torch_version=str(torch.__version__),
        gt_used=False,protocol=mf['protocol'],valid_fraction=float(np.isfinite(depth).mean()))
    # Metadata is the completion marker. Incomplete runs must not be plotted as successful.
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'Completed: {out}',flush=True)
if __name__=='__main__':main()
