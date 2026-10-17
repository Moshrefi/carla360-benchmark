"""Image-based relative rotations. Hypotheses are fixed before evaluating new groups.
No camera poses or images are modified; no absolute calibration is inferred.
"""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from datasets.standard_gt import StandardGTDataset
from geometry.poses import carla_rotation, erp_rays, CARLA_CAMERA_FROM_ERP

# Candidate inferred from the four supplied raw samples, NOT fitted per test pair.
D = np.array([[0,0,-1],[-1,0,0],[0,-1,0]])
Q = np.array([[1,0,0],[0,0,-1],[0,1,0]])
S = np.array([[0,1,0],[1,0,0],[0,0,1]])
M = np.array([[0,0,1],[1,0,0],[0,1,0]])
CANDIDATE_BASIS = M @ S @ Q @ D
HYPOTHESES = ['current_gt', 'offset_only', 'axes_only', 'offset_and_axes']


def angular_error(a,b):
    return float(np.rad2deg(Rotation.from_matrix(a.T @ b).magnitude()))


def fit_rotation(a,b):
    """Column-vector R maps bearings in image i into image j."""
    u,_,vt = np.linalg.svd(b.T @ a)
    return u @ np.diag([1.,1.,np.linalg.det(u@vt)]) @ vt


def residual_deg(a,b,r):
    return np.rad2deg(np.arccos(np.clip(np.sum((a@r.T)*b,axis=1),-1,1)))


def estimate_rotation(a,b,seed=17,iterations=2000,threshold_deg=.6):
    """RANSAC on 70% of matches; remaining 30% independently checks consistency."""
    if len(a)<40:
        raise ValueError('Fewer than 40 mutual feature matches')
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(a)); split=int(.7*len(a)); train=order[:split]; heldout=order[split:]
    aa,bb=a[train],b[train];best=np.zeros(len(train),bool)
    for _ in range(iterations):
        ix=rng.choice(len(train),3,replace=False)
        if np.linalg.svd(aa[ix],compute_uv=False)[1]<1e-3:continue
        r=fit_rotation(aa[ix],bb[ix]);mask=residual_deg(aa,bb,r)<threshold_deg
        if mask.sum()>best.sum():best=mask
    if best.sum()<3:raise ValueError('No stable rotation consensus')
    r=fit_rotation(aa[best],bb[best]);best=residual_deg(aa,bb,r)<threshold_deg
    if best.sum()<3:raise ValueError('Consensus lost during refinement')
    r=fit_rotation(aa[best],bb[best])
    best=residual_deg(aa,bb,r)<threshold_deg
    train_error=residual_deg(aa,bb,r);test_error=residual_deg(a[heldout],b[heldout],r)
    test_good=test_error<threshold_deg
    trusted=(best.sum()>=25 and best.mean()>=.35 and test_good.sum()>=10 and test_good.mean()>=.35)
    quality={'matches':len(a),'train_inliers':int(best.sum()),'train_inlier_fraction':float(best.mean()),
             'heldout_matches':len(heldout),'heldout_inliers':int(test_good.sum()),
             'heldout_inlier_fraction':float(test_good.mean()),
             'heldout_inlier_median_deg':float(np.median(test_error[test_good])) if test_good.any() else None,
             'estimated_turn_deg':float(np.rad2deg(Rotation.from_matrix(r).magnitude())),
             'quality_pass':bool(trusted)}
    return r,quality,residual_deg(a,b,r)<threshold_deg


def load_inputs(sequence,source_csv):
    ds=StandardGTDataset(sequence)
    if ds.metadata.get('geometry_profile')!='source_matlab_v1':
        raise ValueError('This diagnostic is specific to source_matlab_v1')
    src=np.loadtxt(source_csv,delimiter=',',ndmin=2,encoding='utf-8-sig')
    if src.shape[1]!=6 or not np.isfinite(src).all():raise ValueError('Expected original headerless six-column CSV')
    rows=[]
    for row in ds.rows:
        k=int(row.get('source_csv_row') or row['source_frame_id'])
        expected=np.array([float(row['source_'+x]) for x in ['x','y','z','pitch_deg','yaw_deg','roll_deg']])
        if k<0 or k>=len(src) or not np.allclose(src[k],expected,atol=1e-8,rtol=0):
            raise ValueError(f'Original CSV does not agree with exported source pose at row {k}')
        rows.append(k)
    return ds,src,np.asarray(rows)


def choose_groups(ds,src,rows,groups_per_region=2,centre_tolerance_m=.001):
    """Three contiguous source rows sharing a centre under BOTH offset hypotheses."""
    if groups_per_region<1:raise ValueError('groups_per_region must be positive')
    n=len(rows);candidates={name:[] for name in ['beginning','middle','end']}
    ids=np.array([int(r['source_frame_id']) for r in ds.rows])
    for i in range(n-2):
        ix=np.arange(i,i+3);rr=rows[ix]
        if np.any(ids[ix]<=5):continue  # Exclude calibration samples from validation.
        if rr[0]<1 or not np.all(np.diff(rr)==1) or not np.all(np.diff(ids[ix])==1):continue
        if any(np.max(np.linalg.norm(src[rr+offset,:3]-src[rr[0]+offset,:3],axis=1))>centre_tolerance_m for offset in [0,-1]):continue
        pos=i/max(1,n-1)
        region='beginning' if pos<=.2 else 'middle' if .4<=pos<=.6 else 'end' if pos>=.8 else None
        if region:candidates[region].append(ix.tolist())
    selected=[]
    for region, options in candidates.items():
        # Disjoint triples, then evenly spaced groups within each region.
        disjoint=[];last=-1
        for group in options:
            if group[0]>last:disjoint.append(group);last=group[-1]
        if not disjoint:continue
        picks=np.unique(np.linspace(0,len(disjoint)-1,min(groups_per_region,len(disjoint))).round().astype(int))
        selected.extend({'region':region,'frames':disjoint[k]} for k in picks)
    return selected


def predicted_relative(ds,src,rows,i,j,hypothesis):
    if hypothesis=='current_gt':return ds.poses[j,:3,:3].T @ ds.poses[i,:3,:3]
    if hypothesis not in HYPOTHESES:raise ValueError(hypothesis)
    offset=-1 if hypothesis in ('offset_only','offset_and_axes') else 0
    basis=CANDIDATE_BASIS if hypothesis in ('axes_only','offset_and_axes') else CARLA_CAMERA_FROM_ERP
    ri=carla_rotation(*src[rows[i]+offset,3:]);rj=carla_rotation(*src[rows[j]+offset,3:])
    return basis.T @ rj.T @ ri @ basis


def image_features(path):
    image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if image is None:raise ValueError(f'Cannot read {path}')
    h,w=image.shape
    if w!=2*h:raise ValueError('Expected 2:1 ERP')
    mask=np.zeros((h,w),np.uint8);py=max(1,round(h*.1));px=max(1,round(w*.028))
    mask[py:-py,px:-px]=255
    keypoints,descriptors=cv2.SIFT_create(nfeatures=7000).detectAndCompute(image,mask)
    uv=np.array([k.pt for k in keypoints],dtype=float).reshape(-1,2)
    return uv,descriptors,(h,w)


def matched_bearings(f1,f2):
    uv1,d1,hw1=f1;uv2,d2,hw2=f2
    if d1 is None or d2 is None or min(len(d1),len(d2))<2:raise ValueError('Insufficient image features')
    matcher=cv2.BFMatcher()
    reverse={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(d2,d1,k=2) if m.distance<.7*n.distance}
    ms=[m for m,n in matcher.knnMatch(d1,d2,k=2) if m.distance<.7*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
    a=uv1[[m.queryIdx for m in ms]];b=uv2[[m.trainIdx for m in ms]]
    return (erp_rays(a[:,0],a[:,1],hw1[1],hw1[0]),erp_rays(b[:,0],b[:,1],hw2[1],hw2[0]),
            a,b,hw1,hw2)


def run_diagnostic(sequence,source_csv,groups_per_region=2,seed=17):
    cv2.setNumThreads(2)
    ds,src,rows=load_inputs(sequence,source_csv)
    groups=choose_groups(ds,src,rows,groups_per_region)
    if not groups:raise ValueError('No eligible same-centre triples; do not substitute moving pairs without reviewing parallax')
    cache={};results=[]
    for gidx,group in enumerate(groups):
        for i,j in itertools.combinations(group['frames'],2):
            item={'region':group['region'],'group':gidx,'frame_i':i,'frame_j':j,
                  'source_id_i':int(ds.rows[i]['source_frame_id']),'source_id_j':int(ds.rows[j]['source_frame_id']),
                  'csv_row_i':int(rows[i]),'csv_row_j':int(rows[j])}
            print(f"Checking {group['region']}: frames {i} -> {j}",flush=True)
            try:
                for k in [i,j]:
                    if k not in cache:cache[k]=image_features(ds.paths[k])
                a,b,uv1,uv2,hw1,hw2=matched_bearings(cache[i],cache[j])
                r,quality,inliers=estimate_rotation(a,b,seed=seed+len(results))
                item.update(quality);item['estimated_R_j_from_i']=r.tolist()
                # Coverage guards against a tiny consistent patch dominating the fit.
                sectors=np.floor(uv1[inliers,0]/hw1[1]*8).astype(int)
                item['occupied_longitude_sectors']=int(len(np.unique(sectors)))
                item['quality_pass'] &= item['occupied_longitude_sectors']>=3
                halves=[]
                for is_left in [True,False]:
                    region=(uv1[:,0]<hw1[1]/2)==is_left
                    region &= ((uv2[:,0]<hw2[1]/2)==is_left)
                    if region.sum()<40:halves.append(None);continue
                    try:
                        hr,hq,_=estimate_rotation(a[region],b[region],seed=seed+gidx,iterations=1000)
                        halves.append(hr if hq['quality_pass'] else None)
                    except ValueError:halves.append(None)
                item['hemisphere_disagreement_deg']=angular_error(*halves) if all(x is not None for x in halves) else None
                if item['hemisphere_disagreement_deg'] is not None and item['hemisphere_disagreement_deg']>1:
                    item['quality_pass']=False
                for hypothesis in HYPOTHESES:
                    item[hypothesis+'_error_deg']=angular_error(r,predicted_relative(ds,src,rows,i,j,hypothesis))
                item['status']='usable' if item['quality_pass'] else 'inconclusive_image_fit'
            except ValueError as e:
                item.update(status='inconclusive_image_fit',quality_pass=False,reason=str(e))
            results.append(item)
    digest=hashlib.sha256(Path(source_csv).read_bytes()).hexdigest()
    return {'sequence_id':ds.metadata['sequence_id'],'source_csv_sha256':digest,'seed':seed,
            'candidate_basis':CANDIDATE_BASIS.tolist(),'source_row_offset_candidate':-1,
            'groups':groups,'results':results,'regions_missing':sorted(set(['beginning','middle','end'])-{g['region'] for g in groups}),
            'notes':['Relative rotation only; not absolute calibration or translation validation.',
                     'Same-centre filtering uses metadata under both hypotheses; actual capture centres remain unverified.',
                     'Candidates fixed using earlier samples; no per-pair axis or offset search.',
                     'Low image-fit quality means inconclusive, not GT failure.',
                     'Repeated pose schedules may make a one-row lag non-unique; acquisition timing is not proven.',
                     'No GT files or source images were changed.']}


def save_report(report,output_dir):
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose an empty/new diagnostic output directory')
    out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    fields=sorted(set().union(*(r.keys() for r in report['results']))-{'estimated_R_j_from_i'})
    with open(out/'pairs.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(report['results'])
    return out
