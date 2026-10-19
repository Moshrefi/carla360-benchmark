"""Small CPU-only orchestration and plots for notebook 08."""
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('vggt', 'vggt_omega', 'panovggt', 'omnivggt', 'vggt360', 'argus')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def select_images(folder, pattern='*.png', start=0, stop=None, stride=1, limit=None, order='natural'):
    if start < 0 or stride < 1 or (stop is not None and stop < 0) or (limit is not None and limit < 1):
        raise ValueError('Invalid selection: start/stop >= 0, stride/limit >= 1')
    paths = [p.resolve() for p in Path(folder).expanduser().glob(pattern) if p.is_file()]
    if order not in ('natural', 'lexical'):
        raise ValueError('Order must be natural or lexical')
    key = (lambda p: [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', p.name)]) if order == 'natural' else (lambda p: p.name)
    paths = sorted(paths, key=key)[start:stop:stride]
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError('No images selected. Check INPUT_DIR, IMAGE_PATTERN and selection settings.')
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate image paths')
    return paths


def make_plan(paths, run_dir, model_settings, selected, chunk_size, precision='bf16', device='cuda'):
    if chunk_size < 1:
        raise ValueError('CHUNK_SIZE must be >= 1')
    if not selected or len(selected) != len(set(selected)) or any(m not in MODELS for m in selected):
        raise ValueError(f'Select unique models from {MODELS}')
    if precision not in ('bf16', 'fp16', 'fp32') or device != 'cuda':
        raise ValueError('Supported: device=cuda and precision=bf16/fp16/fp32')
    run_dir = Path(run_dir).resolve()
    records = []
    for p in paths:
        if run_dir == p.parent or run_dir in p.parents or run_dir.is_relative_to(p.parent):
            raise ValueError('Keep output runs outside input image folders')
        with Image.open(p) as img:
            width, height = img.size
        records.append({'path': str(p), 'name': p.name, 'sha256': digest(p), 'width': width, 'height': height})
    return {'image_records': records, 'images': [str(p) for p in paths], 'run_dir': str(run_dir),
            'models': model_settings, 'selected': selected, 'chunk_size': chunk_size,
            'precision': precision, 'device': device}


def preflight(plan, python):
    python = str(Path(python).expanduser())
    if not Path(python).is_file():
        raise FileNotFoundError(f'Python not found: {python}')
    check = "import torch, numpy, PIL; print('PyTorch:',torch.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:',torch.cuda.get_device_name())"
    if plan['precision'] == 'bf16':
        check += "; assert torch.cuda.is_bf16_supported(), 'BF16 unsupported'"
    subprocess.run([python, '-c', check], cwd=ROOT, check=True)
    rows = []
    for name in plan['selected']:
        try:
            s = plan['models'][name]
            for key in ('repo', 'checkpoint'):
                if not Path(s[key]).exists():
                    raise FileNotFoundError(f'{name}: missing {key}: {s[key]}')
            if name == 'panovggt' and not Path(s['config']).is_file():
                raise FileNotFoundError(s['config'])
            imports = {
                'vggt': 'from vggt.models.vggt import VGGT; from vggt.utils.load_fn import load_and_preprocess_images',
                'vggt_omega': 'from vggt_omega.models.vggt_omega import VGGTOmega; from vggt_omega.utils.pose_enc import encoding_to_camera',
                'omnivggt': 'from omnivggt.models.omnivggt import OmniVGGT; from visual_util import load_images_and_cameras; import safetensors',
                'panovggt': 'from panovggt.models.panovggt_model import PanoVGGTModel; import omegaconf, cv2',
                'vggt360': 'from comparison.upstream_fixes import fix_vggt360; fix_vggt360(Path(repo)); from vggt_visfeat.models.vggt import VGGT; from utils.gen_views import view_generation',
                'argus': 'from argus.models.argus import Argus; from argus.utils.pose_enc import pose_encoding_to_extri360; from argus.utils.geometry import unproject_depth_to_world_points; import cv2',
            }[name]
            code = f'import sys; from pathlib import Path; repo={s["repo"]!r}; sys.path.insert(0,repo); ' + imports
            subprocess.run([python, '-c', code], cwd=ROOT, check=True)
            rows.append({'model': name, 'status': 'imports passed; weight compatibility checked during strict loading'})
            rows[-1]['ready'] = True
        except Exception as exc:
            rows.append({'model': name, 'status': str(exc), 'ready': False})
    return rows


def prepare(plan):
    run_dir = Path(plan['run_dir'])
    # Selection and chunk boundaries are immutable within a run.
    identity = {k: plan[k] for k in ('image_records', 'chunk_size')}
    manifest = run_dir / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('This RUN_NAME uses different images/chunks. Choose a new RUN_NAME.')
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError('Run directory is not empty and has no manifest')
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(identity, indent=2))
    chunks = []
    for start in range(0, len(plan['images']), plan['chunk_size']):
        end = start + plan['chunk_size']
        chunk = run_dir / f'chunk_{start // plan["chunk_size"]:05d}'
        staged = chunk / 'images'
        staged.mkdir(parents=True, exist_ok=True)
        subset = plan['image_records'][start:end]
        for i, record in enumerate(subset):
            src = Path(record['path'])
            dst = staged / f'{i:06d}{src.suffix.lower()}'
            if digest(src) != record['sha256']:
                raise ValueError(f'Input changed since selection: {src}')
            if not dst.exists():
                shutil.copy2(src, dst)
            if digest(dst) != record['sha256']:
                raise ValueError(f'Staged image changed: {dst}')
        chunks.append((chunk, subset, staged))
    return chunks


def load_result(path, request_hash=None):
    path = Path(path)
    if not (path / 'metadata.json').exists():
        return None
    meta = json.loads((path / 'metadata.json').read_text())
    if request_hash is not None and meta['request_hash'] != request_hash:
        raise ValueError('Cached result has different model settings/weights/code. Use a new RUN_NAME or RERUN=True.')
    if digest(path / 'predictions.npz') != meta['output_sha256']:
        raise ValueError(f'Output integrity check failed: {path}')
    with np.load(path / 'predictions.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    if arrays['depth'].ndim != 3 or len(arrays['depth']) != len(meta['input_images']):
        raise ValueError('Output frame count or depth shape mismatch')
    return arrays, meta


def execute(plan, python, rerun=False, continue_on_error=True):
    chunks = prepare(plan)
    statuses = []
    worker_hash = digest(ROOT / 'comparison/bench_infer.py')
    for name in plan['selected']:
        settings = plan['models'][name]
        weight_hash = digest(settings['checkpoint'])
        repo = Path(settings['repo'])
        # Include local Python/YAML source changes, not only Git HEAD.
        code_hash = hashlib.sha256()
        for p in sorted(repo.rglob('*')):
            if p.is_file() and p.suffix in ('.py', '.yaml', '.yml') and '.git' not in p.parts:
                code_hash.update(str(p.relative_to(repo)).encode()); code_hash.update(digest(p).encode())
        for chunk, records, staged in chunks:
            cfg = {**plan, 'images': [str(staged / f'{i:06d}{Path(r["path"]).suffix.lower()}') for i, r in enumerate(records)],
                   'image_records': records, 'staged_dir': str(staged)}
            identity = {'images': records, 'settings': settings, 'weight_sha256': weight_hash,
                        'source_sha256': code_hash.hexdigest(), 'worker_sha256': worker_hash,
                        'precision': plan['precision'], 'device': plan['device']}
            cfg['request_hash'] = hashlib.sha256(canonical(identity).encode()).hexdigest()
            target = chunk / name
            try:
                if not rerun and load_result(target, cfg['request_hash']) is not None:
                    statuses.append({'model': name, 'chunk': chunk.name, 'status': 'reused'})
                    continue
                if target.exists():
                    backup = chunk / 'archived' / f'{name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}'
                    backup.parent.mkdir(exist_ok=True)
                    target.rename(backup)
                config = chunk / f'{name}_request.json'
                config.write_text(json.dumps(cfg, indent=2))
                log = chunk / f'{name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}.log'
                print(f'Running {name} / {chunk.name} / {len(records)} images', flush=True)
                with log.open('w') as f:
                    p = subprocess.Popen([str(python), '-m', 'comparison.bench_infer', '--config', str(config),
                                          '--model', name, '--out', str(target)], cwd=ROOT, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
                    try:
                        for line in p.stdout:
                            print(line, end='', flush=True); f.write(line); f.flush()
                        rc = p.wait()
                    except BaseException:
                        os.killpg(p.pid, signal.SIGTERM)
                        try:
                            p.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(p.pid, signal.SIGKILL); p.wait()
                        raise
                if rc:
                    raise RuntimeError(f'Exit {rc}; see {log}')
                if load_result(target, cfg['request_hash']) is None:
                    raise RuntimeError('Worker did not create completed output')
                statuses.append({'model': name, 'chunk': chunk.name, 'status': 'completed'})
            except Exception as exc:
                statuses.append({'model': name, 'chunk': chunk.name, 'status': 'failed', 'error': str(exc)})
                print(f'{name}: {exc}', flush=True)
                if not continue_on_error:
                    raise
    return statuses


def depth_plot(plan, chunk_index=0, frame_index=0, normalized=True):
    import matplotlib.pyplot as plt
    chunk = Path(plan['run_dir']) / f'chunk_{chunk_index:05d}'
    manifest = json.loads((Path(plan['run_dir']) / 'manifest.json').read_text())
    records = manifest['image_records'][chunk_index * manifest['chunk_size']:(chunk_index + 1) * manifest['chunk_size']]
    if frame_index not in range(len(records)):
        raise ValueError('PLOT_FRAME is outside this chunk')
    names = plan['selected']
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(4 * (len(names) + 1), 3.5), squeeze=False)
    record = records[frame_index]
    staged = chunk / 'images' / f'{frame_index:06d}{Path(record["path"]).suffix.lower()}'
    axes[0, 0].imshow(Image.open(staged)); axes[0, 0].set_title(record['name'], fontsize=8)
    for ax, name in zip(axes[0, 1:], names):
        result = load_result(chunk / name)
        ax.set_title(name)
        if result is None:
            ax.text(.5, .5, 'No completed output', ha='center', transform=ax.transAxes)
            continue
        arrays, meta = result
        if meta['input_images'] != records:
            raise ValueError(f'{name}: image identities do not match the manifest')
        d = arrays['depth'][frame_index]
        valid = np.isfinite(d) & (d > 0)
        d = np.where(valid, d, np.nan)
        if not valid.any():
            ax.text(.5, .5, 'No positive finite depth', ha='center', transform=ax.transAxes)
            continue
        if normalized:
            low, high = np.percentile(d[valid], [2, 98])
            d = np.clip((d - low) / max(high - low, 1e-8), 0, 1)
        artist = ax.imshow(d, cmap='magma', vmin=0 if normalized else None, vmax=1 if normalized else None)
        fig.colorbar(artist, ax=ax, fraction=.025)
    for ax in axes[0]:
        ax.axis('off')
    fig.suptitle('Per-map percentile normalization (scale hidden)' if normalized else 'Native head values: different depth definitions, NOT comparable metres')
    fig.tight_layout()
    return fig


def geometry_plots(plan, chunk_index=0, max_points=10000):
    import matplotlib.pyplot as plt
    if max_points < 1:
        raise ValueError('MAX_POINTS must be positive')
    chunk = Path(plan['run_dir']) / f'chunk_{chunk_index:05d}'
    figures, rows = [], []
    for name in plan['selected']:
        result = load_result(chunk / name)
        if result is None:
            continue
        arrays, meta = result
        for key, value in arrays.items():
            finite = np.isfinite(value)
            rows.append({'model': name, 'output': key, 'shape': str(value.shape), 'finite_fraction': float(finite.mean())})
        poses = arrays.get('camera_poses')
        if poses is None and 'extrinsics' in arrays:
            ext = arrays['extrinsics']
            if ext.shape[1:] == (3, 4):
                t = np.tile(np.eye(4), (len(ext), 1, 1)); t[:, :3] = ext
                if np.isfinite(t).all():
                    try:
                        poses = np.linalg.inv(t)
                    except np.linalg.LinAlgError:
                        rows.append({'model': name, 'output': 'pose warning', 'shape': 'singular extrinsics'})
        if poses is not None and poses.ndim == 3 and poses.shape[1:] in ((3, 4), (4, 4)) and np.isfinite(poses).all():
            r = poses[:, :3, :3]
            rows.append({'model': name, 'output': 'rotation checks', 'shape': '',
                         'max_RtR_error': float(np.max(np.abs(r.transpose(0, 2, 1) @ r - np.eye(3)))),
                         'max_det_error': float(np.max(np.abs(np.linalg.det(r) - 1)))})
            if len(poses) > 1:
                fig = plt.figure(figsize=(5, 4)); ax = fig.add_subplot(projection='3d')
                centres = poses[:, :3, 3]
                ax.plot(*centres.T, 'o-')
                ax.set_title(f'{name}: chunk {chunk_index} only\nNative axes and scale; no GT comparison')
                for i, point in enumerate(centres): ax.text(*point, str(i))
                figures.append(fig)
        points = arrays.get('world_points')
        if points is not None:
            points = points.reshape(-1, 3)
            points = points[np.isfinite(points).all(axis=1)]
            if len(points):
                points = points[np.linspace(0, len(points) - 1, min(max_points, len(points)), dtype=int)]
                fig = plt.figure(figsize=(5, 4)); ax = fig.add_subplot(projection='3d')
                ax.scatter(*points.T, s=.2)
                ax.set_title(f'{name}: native predicted points\nNo depth unprojection or metric assumption')
                figures.append(fig)
        else:
            rows.append({'model': name, 'output': 'world_points', 'shape': 'not available; not synthesized'})
    return figures, rows
