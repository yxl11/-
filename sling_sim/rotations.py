"""Quaternion convention: scalar-first, body to world, body angular velocity."""
import numpy as np


def quaternion_matrix(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    out = np.empty(q.shape[:-1] + (3, 3))
    out[..., 0, 0] = 1 - 2*(y*y + z*z)
    out[..., 0, 1] = 2*(x*y - w*z)
    out[..., 0, 2] = 2*(x*z + w*y)
    out[..., 1, 0] = 2*(x*y + w*z)
    out[..., 1, 1] = 1 - 2*(x*x + z*z)
    out[..., 1, 2] = 2*(y*z - w*x)
    out[..., 2, 0] = 2*(x*z - w*y)
    out[..., 2, 1] = 2*(y*z + w*x)
    out[..., 2, 2] = 1 - 2*(x*x + y*y)
    return out


def matrix_quaternion(r):
    # Symmetric eigenproblem is robust near 180 degrees; used at initialization.
    k = np.array([
        [r[0, 0]-r[1, 1]-r[2, 2], r[0, 1]+r[1, 0], r[0, 2]+r[2, 0], r[2, 1]-r[1, 2]],
        [r[0, 1]+r[1, 0], r[1, 1]-r[0, 0]-r[2, 2], r[1, 2]+r[2, 1], r[0, 2]-r[2, 0]],
        [r[0, 2]+r[2, 0], r[1, 2]+r[2, 1], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
        [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)],
    ]) / 3
    _, v = np.linalg.eigh(k)
    q = v[:, -1][[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q


def quaternion_derivative(q, omega):
    return 0.5*np.concatenate((
        -np.sum(q[..., 1:]*omega, axis=-1, keepdims=True),
        q[..., :1]*omega + np.cross(q[..., 1:], omega)), axis=-1)


def desired_rotation(force, yaw=0.0):
    b3 = force / max(np.linalg.norm(force), 1e-12)
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    b2 = np.cross(b3, heading)
    if np.linalg.norm(b2) < 1e-8:
        b2 = np.cross(b3, np.array([0.0, 0.0, 1.0]))
    b2 /= np.linalg.norm(b2)
    return np.column_stack((np.cross(b2, b3), b2, b3))


def vee(a):
    return np.array([a[2, 1], a[0, 2], a[1, 0]])

