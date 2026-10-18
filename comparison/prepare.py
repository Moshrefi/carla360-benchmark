"""Freeze a shared image selection; source pose values never enter inference."""
import csv,hashlib,json,shutil
from pathlib import Path
from PIL import Image
import numpy as np
from comparison.projection import extract_faces

def prepare_job(sequence,output,frame_ids=(0,20,40),height=256,face_size=518,fov=100):
    sequence=Path(sequence).resolve();out=Path(output).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Use a new job folder: {out}')
    if height<16 or face_size<28 or face_size%14 or not 90<fov<150:raise ValueError('Invalid image size/FOV')
    ids=[int(i) for i in frame_ids]
    if not ids or len(set(ids))!=len(ids) or ids!=sorted(ids):raise ValueError('Frame IDs must be unique and increasing')
    with open(sequence/'poses.csv',newline='') as f:rows=list(csv.DictReader(f))
    if min(ids)<0 or max(ids)>=len(rows):raise ValueError('Selected frame outside sequence')
    out.mkdir(parents=True,exist_ok=True);(out/'images').mkdir();(out/'faces').mkdir()
    records=[]
    for k in ids:
        row=rows[k]
        if int(row['frame_id'])!=k:raise ValueError('Unexpected frame ordering')
        path=(sequence/row['image_name']).resolve()
        if not path.is_relative_to(sequence):raise ValueError('Image path outside sequence')
        rgb=Image.open(path).convert('RGB')
        if rgb.width!=2*rgb.height:raise ValueError(f'Expected 2:1 ERP: {path}')
        dest=out/'images'/f'{k:06d}.png';rgb.save(dest)
        faces=extract_faces(np.array(rgb),face_size,fov)
        names=[]
        for face,arr in enumerate(faces):
            name=f'faces/{k:06d}_{face}.png';Image.fromarray(arr).save(out/name);names.append(name)
        records.append({'frame_id':k,'source_frame_id':row.get('source_frame_id'),
                        'source_name':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'image':f'images/{k:06d}.png','faces':names})
    manifest={'schema_version':1,'source_sequence':str(sequence),'frames':records,'height':height,'width':2*height,
              'face_size':face_size,'face_fov_deg':fov,'gt_used_as_model_input':False,
              'erp_grid':'endpoint-inclusive; x right, y down, z forward',
              'protocol':'PanoVGGT joint ERP; VGGT-360 per ERP; OmniVGGT/MapAnything joint six-face adaptation'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return out
