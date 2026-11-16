"""Smooth area-preserving horizontal deformations and shared tangent orientation."""
import numpy as np
BASE_SIGMA=np.array([[13.,9.,3.5],[5.5,4.5,3.],[4.8,4.5,2.8]])
SWEEP_LONG=np.array([7.5,4.0,4.3])
ORBIT_LONG=np.array([14.5,6.,5.5])
def ramp(t,a,b):
    u=np.clip((t-a)/(b-a),0,1)
    return u**3*(10+u*(-15+6*u)),30*u**2*(1-u)**2/(b-a)
def shape_axes(t):
    """Scalar t. Columns of R are the material axes in the world frame."""
    t=float(t)
    a,da=ramp(t,8.,24.); b,db=ramp(t,68.,86.)
    L=np.log(SWEEP_LONG/BASE_SIGMA[:,0]); M=np.log(ORBIT_LONG/SWEEP_LONG)
    factor=np.exp(a*L+b*M)
    sigma=BASE_SIGMA.copy(); sigma[:,0]*=factor;sigma[:,1]/=factor
    rate=da*L+db*M
    ds=np.column_stack((sigma[:,0]*rate,-sigma[:,1]*rate,np.zeros(3)))
    u=np.clip((t-86)/16,0,1)
    omega=2*np.pi/72
    theta=omega*(16*u**4*(2.5-3*u+u**2)+max(0,t-102))
    w=omega*u**3*(10+u*(-15+6*u))
    c,s=np.cos(theta),np.sin(theta)
    rot=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    dr=w*np.array([[-s,-c,0.],[c,-s,0.],[0.,0.,0.]])
    return rot,dr,sigma,ds
def shape_state(t):
    a=np.asarray(t)
    if a.ndim:
        pairs=[shape_state(x) for x in a.flat]
        return tuple(np.stack(items).reshape(a.shape+(3,3,3)) for items in zip(*pairs))
    R,dR,s,ds=shape_axes(float(a))
    return R[None]*s[:,None,:],dR[None]*s[:,None,:]+R[None]*ds[:,None,:]
