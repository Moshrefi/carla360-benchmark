"""Plot completed runs with explicit units and independent scale normalization."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
MODELS=['panovggt','vggt360','omnivggt','mapanything']

def load_results(job):
    job=Path(job);mf=json.loads((job/'manifest.json').read_text());results={}
    digest=hashlib.sha256((job/'manifest.json').read_bytes()).hexdigest()
    ids=np.array([r['frame_id'] for r in mf['frames']])
    for model in MODELS:
        out=job/'predictions'/model
        if not (out/'metadata.json').is_file():continue
        meta=json.loads((out/'metadata.json').read_text())
        if meta['manifest_sha256']!=digest:raise ValueError(f'{model}: input manifest mismatch')
        with np.load(out/'predictions.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
        if not np.array_equal(arrays['frame_ids'],ids):raise ValueError(f'{model}: frame ordering mismatch')
        if arrays['radial_depth'].shape!=(len(ids),mf['height'],mf['width']):raise ValueError(f'{model}: wrong depth shape')
        results[model]=(arrays,meta)
    return mf,results

def depth_grid(job,normalized=True,raw_limits=None):
    """Per-image percentile normalization is visual only, never GT alignment."""
    job=Path(job);mf,results=load_results(job)
    fig,axes=plt.subplots(len(mf['frames']),5,figsize=(22,3.3*len(mf['frames'])),squeeze=False)
    for i,record in enumerate(mf['frames']):
        axes[i,0].imshow(Image.open(job/record['image']));axes[i,0].set_title(f'ERP frame {record["frame_id"]}')
        for j,model in enumerate(MODELS,1):
            ax=axes[i,j]
            if model not in results:ax.text(.5,.5,'Not run / incomplete',ha='center',va='center');ax.set_title(model);continue
            arrays,meta=results[model];d=arrays['radial_depth'][i].copy();valid=np.isfinite(d)&(d>0)
            if not valid.any():ax.text(.5,.5,'No valid depth',ha='center');continue
            lo,hi=np.percentile(d[valid],[2,98]);hi=max(float(hi),float(lo)+1e-8)
            if normalized:
                d=(d-lo)/(hi-lo);limits=(0,1);label='Per-image p2–p98 (unitless)'
            else:
                limits=raw_limits if raw_limits is not None else (lo,hi)
                label='Model units (scale unverified)'
            im=ax.imshow(np.ma.masked_invalid(d),cmap='viridis',vmin=limits[0],vmax=limits[1])
            ax.set_title(f'{model}\nvalid {valid.mean():.1%}');fig.colorbar(im,ax=ax,shrink=.65,label=label)
        for ax in axes[i]:ax.axis('off')
    fig.suptitle('Visual shape comparison; independent normalization hides scale differences' if normalized else 'Raw outputs; colorbars may differ. Values are not accuracy scores.')
    fig.tight_layout();return fig

def trajectories(job):
    _,results=load_results(job)
    available=[(k,a,m) for k,(a,m) in results.items() if 'poses_c2w' in a]
    if not available:print('No temporal poses available yet. VGGT-360 does not supply a temporal trajectory.');return None
    fig=plt.figure(figsize=(6*len(available),5))
    for n,(name,arrays,meta) in enumerate(available,1):
        poses=arrays['poses_c2w'];local=np.linalg.inv(poses[0])@poses;t=local[:,:3,3]
        ax=fig.add_subplot(1,len(available),n,projection='3d');ax.plot(*t.T,'o-')
        for pos,frame in zip(t,arrays['frame_ids']):ax.text(*pos,str(frame))
        ax.set(xlabel='X',ylabel='Y',zlabel='Z',title=f'{name}\nFirst-camera frame; own model scale')
        radius=max(np.ptp(t,axis=0).max()/2,1e-6);center=(t.min(axis=0)+t.max(axis=0))/2
        ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]-radius,center[1]+radius);ax.set_zlim(center[2]-radius,center[2]+radius)
        ax.set_box_aspect((1,1,1))
    fig.tight_layout();return fig
