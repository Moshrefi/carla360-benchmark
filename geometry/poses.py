"""Float64, column-vector transforms. See docs/gt_convention.md."""
import numpy as np
from scipy.spatial.transform import Rotation

WORLD_FROM_CARLA = np.diag([1.0, -1.0, 1.0])
# For the supplied FisheyeToSphere.m, including Rx(+90), row/column swap,
# optical-to-CARLA axes and the documented ERP longitude convention.
CARLA_CAMERA_FROM_ERP = np.diag([-1.0, 1.0, 1.0])


def carla_rotation(pitch, yaw, roll):
    """CARLA GetMatrix rotation (degrees); not the usual Rz Ry Rx signs."""
    p, y, r = np.deg2rad([pitch, yaw, roll])
    cp, sp, cy, sy, cr, sr = np.cos(p), np.sin(p), np.cos(y), np.sin(y), np.cos(r), np.sin(r)
    return np.array([[cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr],
                     [cp*sy, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr],
                     [sp, -cp*sr, cp*cr]], dtype=np.float64)


def validate_poses(poses):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses):
        raise ValueError("Expected nonempty (N,4,4) camera-to-world matrices")
    if not np.isfinite(poses).all():
        raise ValueError("Nonfinite pose")
    rotations = poses[:, :3, :3]
    orth = float(np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3))))
    det = float(np.max(np.abs(np.linalg.det(rotations) - 1)))
    bottom = float(np.max(np.abs(poses[:, 3] - [0, 0, 0, 1])))
    if max(orth, det, bottom) > 1e-10:
        raise ValueError("Invalid rigid camera-to-world transformation")
    return {"max_orthogonality_error": orth, "max_determinant_error": det,
            "max_homogeneous_row_error": bottom}


def source_to_c2w(source_pose):
    """Input [x,y,z,pitch,yaw,roll], using the source_matlab_v1 profile."""
    source_pose = np.asarray(source_pose, dtype=np.float64)
    if source_pose.shape != (6,) or not np.isfinite(source_pose).all():
        raise ValueError("Expected six finite source pose values")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = WORLD_FROM_CARLA @ carla_rotation(*source_pose[3:]) @ CARLA_CAMERA_FROM_ERP
    pose[:3, 3] = WORLD_FROM_CARLA @ source_pose[:3]
    return pose


def source_transform_from_c2w(pose):
    """Recover native CARLA matrix, avoiding Euler ambiguity at gimbal lock."""
    out = np.eye(4)
    out[:3, :3] = WORLD_FROM_CARLA.T @ pose[:3, :3] @ CARLA_CAMERA_FROM_ERP.T
    out[:3, 3] = WORLD_FROM_CARLA.T @ pose[:3, 3]
    return out


def quaternion_xyzw(rotations):
    """Unit quaternion, scalar last; deterministic sign (w >= 0)."""
    return Rotation.from_matrix(rotations).as_quat(canonical=True)


def erp_rays(u, v, width, height):
    """Exact source endpoint grid; u,v are zero-based integer or float indices."""
    if width < 2 or height < 2:
        raise ValueError("ERP dimensions must be >= 2")
    lon = 2*np.pi*np.asarray(u)/(width-1)-np.pi
    lat = np.pi/2-np.pi*np.asarray(v)/(height-1)
    lon, lat = np.broadcast_arrays(lon, lat)
    return np.stack([np.cos(lat)*np.sin(lon), -np.sin(lat),
                     np.cos(lat)*np.cos(lon)], axis=-1)


def relative_c2w(pose_i, pose_j):
    """T_i_from_j: transforms a point in camera j into camera i."""
    return np.linalg.solve(pose_i, pose_j)
