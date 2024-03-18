import torch
import numpy as np

@torch.jit.script
def quat_rotate_inverse(q, v):
        shape = q.shape
        q_w = q[:, 0]
        q_vec = q[:, 1:]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
        return (a - b + c)

# q is (4,1) and v is (3,1)
def quat_rotate_inverse_numpy(q,v):
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v, axis=0) * q_w * 2.0
    c = q_vec * np.dot(q_vec.T, v) * 2.0
    return (a - b + c)