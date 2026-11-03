"""Shared rear-axle Stanley-style control; steering is in radians."""
import numpy as np

def steering_radians(x,y,yaw,v,waypoints,k=0.3,softening=20.0):
    w=np.asarray(waypoints,dtype=float)[:,:2]
    if len(w)<2: return 0.0
    a=w[:-1]; d=np.diff(w,axis=0); lengths=np.sum(d*d,axis=1)
    valid=lengths>1e-12
    if not np.any(valid): return 0.0
    a=a[valid]; d=d[valid]; lengths=lengths[valid]
    t=np.clip(np.sum((np.array([x,y])-a)*d,axis=1)/lengths,0,1)
    q=a+t[:,None]*d; j=int(np.argmin(np.sum((q-[x,y])**2,axis=1)))
    tangent=d[j]/np.sqrt(lengths[j]); offset=np.array([x,y])-q[j]
    signed=tangent[0]*offset[1]-tangent[1]*offset[0]
    heading=np.arctan2(tangent[1],tangent[0])-yaw
    heading=np.arctan2(np.sin(heading),np.cos(heading))
    return float(np.clip(heading-np.arctan2(k*signed,softening+max(v,0)), -np.pi/6,np.pi/6))
