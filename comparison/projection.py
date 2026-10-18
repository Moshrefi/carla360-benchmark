"""Known virtual pinhole views of an ERP. Independent of world poses/GT.
ERP rays: x right, y down, z forward; endpoint-inclusive longitude/latitude.
"""
import numpy as np
from scipy.ndimage import map_coordinates

# Face 0 shares the ERP camera axes. Columns are right, down, forward.
FACE_ROTATIONS = np.array([
    [[1,0,0],[0,1,0],[0,0,1]],
    [[0,0,1],[0,1,0],[-1,0,0]],
    [[-1,0,0],[0,1,0],[0,0,-1]],
    [[0,0,-1],[0,1,0],[1,0,0]],
    [[1,0,0],[0,0,-1],[0,1,0]],
    [[1,0,0],[0,0,1],[0,-1,0]],
],dtype=np.float64)

def erp_grid(height, width):
    lon,lat=np.meshgrid(np.linspace(-np.pi,np.pi,width),np.linspace(np.pi/2,-np.pi/2,height))
    return np.stack([np.cos(lat)*np.sin(lon),-np.sin(lat),np.cos(lat)*np.cos(lon)],axis=-1)

def face_rays(size, fov=100):
    t=np.tan(np.deg2rad(fov)/2)
    x,y=np.meshgrid(np.linspace(-t,t,size),np.linspace(-t,t,size))
    return np.stack([x,y,np.ones_like(x)],axis=-1)

def sample(array,u,v,order=1):
    if array.ndim==2:
        return map_coordinates(array,[v,u],order=order,mode='nearest')
    return np.stack([sample(array[...,c],u,v,order) for c in range(array.shape[-1])],axis=-1)

def extract_faces(rgb,size=518,fov=100):
    h,w=rgb.shape[:2]; q=face_rays(size,fov);q/=np.linalg.norm(q,axis=-1,keepdims=True)
    faces=[]
    for rot in FACE_ROTATIONS:
        d=q@rot.T
        u=(np.arctan2(d[...,0],d[...,2])+np.pi)/(2*np.pi)*(w-1)
        v=(np.pi/2-np.arcsin(np.clip(-d[...,1],-1,1)))/np.pi*(h-1)
        faces.append(np.clip(sample(rgb.astype(float),u,v),0,255).astype(np.uint8))
    return faces

def z_to_radial(depth,fov=100):
    if depth.ndim!=2 or depth.shape[0]!=depth.shape[1]:raise ValueError('Expected square z-depth')
    return depth*np.linalg.norm(face_rays(depth.shape[0],fov),axis=-1)

def merge_faces(depths,height,width,fov=100):
    """Choose the most central valid face at each ERP pixel; no scale fitting."""
    if len(depths)!=6:raise ValueError('Six radial-depth faces required')
    rays=erp_grid(height,width); out=np.full((height,width),np.nan,np.float32)
    score=np.full((height,width),-np.inf);t=np.tan(np.deg2rad(fov)/2)
    for depth,rot in zip(depths,FACE_ROTATIONS):
        depth=np.asarray(depth,dtype=np.float32)
        if depth.ndim!=2 or depth.shape[0]!=depth.shape[1]:raise ValueError('Square faces required')
        d=rays@rot;z=d[...,2];safe=np.where(z>0,z,1)
        x=d[...,0]/safe;y=d[...,1]/safe
        inside=(z>0)&(np.abs(x)<=t)&(np.abs(y)<=t)
        u=(x/t+1)/2*(depth.shape[1]-1);v=(y/t+1)/2*(depth.shape[0]-1)
        # Nearest sampling preserves invalid pixels and avoids blending across depth edges.
        values=sample(depth,u,v,order=0)
        valid=inside&np.isfinite(values)&(values>0)&(z>score)
        out[valid]=values[valid];score[valid]=z[valid]
    return out

def resize_depth(depth,height,width):
    u,v=np.meshgrid(np.linspace(0,depth.shape[1]-1,width),np.linspace(0,depth.shape[0]-1,height))
    return sample(np.asarray(depth,dtype=np.float32),u,v,order=0)


def z_to_radial_intrinsics(depth, intrinsic):
    """Convert z-depth using predicted camera calibration at this resolution."""
    depth=np.asarray(depth);K=np.asarray(intrinsic)
    if depth.ndim!=2 or K.shape!=(3,3) or not np.isfinite(K).all():
        raise ValueError('Expected 2D depth and finite 3x3 intrinsics')
    u,v=np.meshgrid(np.arange(depth.shape[1]),np.arange(depth.shape[0]))
    rays=np.stack([u,v,np.ones_like(u)],axis=-1)@np.linalg.inv(K).T
    return depth*np.linalg.norm(rays,axis=-1)/rays[...,2]
