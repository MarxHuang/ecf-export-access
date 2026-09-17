"""Participant-local Exp3 response, with completed own-delivery feedback only.

This is a behavioral sensitivity, not an incentive-compatible or foresighted
controller. The coordinator's ECF implementation is not modified.
"""
import hashlib
import numpy as np

ACTIONS=np.array([0.50,0.75,1.00,1.25,1.50],dtype=float)
EXPLORATION=0.10

def random_draws(seed,date,interval,count):
    key=f'ECF_ADAPTATION_189|{seed}|{date}|{interval}'.encode()
    integer=int.from_bytes(hashlib.sha256(key).digest()[:16],'big')
    return np.random.Generator(np.random.PCG64(integer)).random(count)

class LocalResponse:
    def __init__(self,capacity,mode='ADAPTIVE'):
        self.capacity=np.asarray(capacity,dtype=float).copy()
        if self.capacity.ndim!=1 or not np.all(np.isfinite(self.capacity)) or np.any(self.capacity<=0):
            raise ValueError('Every participant requires positive registered capacity')
        if mode not in ['ADAPTIVE','STATIC_RANDOM','FIXED_REFERENCE']:raise ValueError(mode)
        self.mode=mode;self.log_weights=np.zeros((len(self.capacity),len(ACTIONS)))
        self.pending=None;self.completed=0

    def probabilities(self):
        w=np.exp(self.log_weights-self.log_weights.max(axis=1,keepdims=True))
        return (1-EXPLORATION)*w/w.sum(axis=1,keepdims=True)+EXPLORATION/len(ACTIONS)

    def choose(self,forecast_request,uniform):
        if self.pending is not None:raise RuntimeError('Complete the previous interval before choosing again')
        base=np.asarray(forecast_request,dtype=float);u=np.asarray(uniform,dtype=float)
        if base.shape!=self.capacity.shape or u.shape!=base.shape:raise ValueError('Shape mismatch')
        if not np.all(np.isfinite(base)) or np.any(base<0) or np.any(base>self.capacity+1e-12):raise ValueError('Invalid forecast request')
        if not np.all(np.isfinite(u)) or np.any(u<0) or np.any(u>=1):raise ValueError('Draw must lie in [0,1)')
        p=self.probabilities();cdf=np.cumsum(p,axis=1);cdf[:,-1]=1
        action=(u[:,None]>=cdf).sum(axis=1)
        if self.mode=='FIXED_REFERENCE':action[:]=2
        raw=np.minimum(base*ACTIONS[action],self.capacity)
        # This condition uses the known request, never contemporaneous PV truth.
        eligible=base>0
        self.pending=(action.copy(),p.copy(),eligible.copy(),raw.copy())
        return raw.copy(),action.copy(),p.copy()

    def observe(self,completed_delivery):
        if self.pending is None:raise RuntimeError('No chosen action to credit')
        y=np.asarray(completed_delivery,dtype=float)
        action,p,eligible,raw=self.pending
        if y.shape!=self.capacity.shape or not np.all(np.isfinite(y)) or np.any(y<0) or np.any(y>raw+1e-10):
            raise ValueError('Delivery is not a valid completed outcome of this request')
        reward=y/self.capacity
        if self.mode=='ADAPTIVE':
            idx=np.flatnonzero(eligible)
            self.log_weights[idx,action[idx]]+=EXPLORATION*reward[idx]/(len(ACTIONS)*p[idx,action[idx]])
            # A row-wise constant does not alter choice probabilities.
            self.log_weights-=self.log_weights.max(axis=1,keepdims=True)
        self.pending=None;self.completed+=1
        return reward

    def state(self):
        if self.pending is not None:raise RuntimeError('Cannot checkpoint an unfinished interval')
        return self.log_weights.copy()

    def restore(self,weights,completed):
        w=np.asarray(weights,dtype=float)
        if self.pending is not None or w.shape!=self.log_weights.shape or not np.all(np.isfinite(w)):
            raise ValueError('Invalid policy checkpoint')
        if self.mode!='ADAPTIVE' and np.any(w!=0):raise ValueError('Static policy cannot contain learned preferences')
        if int(completed)!=completed or completed<0:raise ValueError('Invalid completed count')
        self.log_weights=w.copy();self.completed=int(completed)
