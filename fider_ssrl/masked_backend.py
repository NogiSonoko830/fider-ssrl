"""Final real-data RL / SSRL* with harmonic latent initialization."""
import numpy as np
from scipy import fft,ndimage,sparse
from scipy.sparse.linalg import spsolve

def conv(x,h):return fft.irfft2(fft.rfft2(x,workers=1)*h,s=x.shape[-2:],workers=1)


class Fill:
    """Deterministic interpolation for filter input/latent initialization only.
    The likelihood still uses exactly the original observation mask.
    Periodic graph is consistent with the reconstruction canvas.
    """
    def __init__(self,valid):
        self.valid=valid;self.shape=valid.shape;self.holes=np.flatnonzero(~valid)
        self.nearest=ndimage.distance_transform_edt(~valid,return_distances=False,return_indices=True)
        ind=np.full(valid.size,-1);ind[self.holes]=np.arange(len(self.holes));n,m=valid.shape
        rr,cc=np.unravel_index(self.holes,valid.shape)
        self.neighbors=np.stack([((rr-1)%n)*m+cc,((rr+1)%n)*m+cc,rr*m+(cc-1)%m,rr*m+(cc+1)%m],axis=1)
        rows=np.repeat(np.arange(len(self.holes)),4);cols=ind[self.neighbors].ravel();sel=cols>=0
        self.matrix=sparse.eye(len(self.holes),format='csc')*4-sparse.coo_matrix((np.ones(sel.sum()),(rows[sel],cols[sel])),shape=(len(self.holes),len(self.holes))).tocsc()
    def apply(self,x,kind):
        x=np.asarray(x,float);out=np.where(self.valid,x,0.)
        if not len(self.holes):return out
        if kind=='zero':return out
        if kind=='nearest':return np.where(self.valid,x,out[tuple(self.nearest)])
        assert kind=='harmonic'
        flat=out.ravel();rhs=flat[self.neighbors].sum(axis=1);flat[self.holes]=spsolve(self.matrix,rhs)
        assert np.isfinite(out).all();return out


def solve(net,b,valid,h,algo,ks,fill,kind):
    """Masked total-RL or explicitly named signed-source SSRL extension.
    Returns full-canvas checkpoints plus diagnostics; no output normalization.
    """
    total=net+b
    if not np.isfinite(total[valid]).all() or total[valid].min()<0:
        return None,dict(status='INVALID_OBSERVED_TOTAL',minimum=float(np.nanmin(total[valid])))
    sensitivity=conv(valid.astype(float),h.conj());constrained=sensitivity>1e-12*sensitivity.max()
    data=np.where(valid,total,0.)
    if algo=='RL':x=np.where(valid,total,b)
    else:x=fill.apply(net,kind)
    sign0=np.sign(x);zero0=x==0
    frozen=np.zeros_like(valid);snaps=[];records=[];freeze_count=0
    for k in range(1,max(ks)+1):
        pred=conv(x,h)
        ratio=np.zeros_like(x)
        if algo=='RL':np.divide(data,np.maximum(pred,1e-12),out=ratio,where=valid)
        else:
            pred+=b;new=valid&(~frozen)&(pred<=0);frozen|=new;freeze_count+=int(new.sum())
            ratio[frozen]=1.;np.divide(data,pred,out=ratio,where=valid&(~frozen))
        adj=conv(ratio,h.conj());factor=np.ones_like(x)
        if adj.min() < -1e-6*max(sensitivity.max(),1):
            return None,dict(status='NEGATIVE_CORRECTION',iteration=k,minimum=float(adj.min()))
        np.divide(np.maximum(adj,0),sensitivity,out=factor,where=constrained);x*=factor
        if not np.isfinite(x).all():return None,dict(status='NONFINITE',iteration=k)
        if k in ks:
            output=x-b if algo=='RL' else x
            snaps.append(output.copy())
            records.append(dict(iteration=k,sum=float(output.sum()),weighted_sum=float((output*sensitivity).sum()),
                observed_net_sum=float(net[valid].sum()),minimum_prediction=float(pred[valid].min()),frozen=freeze_count,
                negative_source_pixels=int((output<0).sum()),zero_source_pixels=int((output==0).sum())))
    sign_ok=bool(np.all((x[~zero0]==0)|(np.sign(x[~zero0])==sign0[~zero0])) and np.all(x[zero0]==0)) if algo!='RL' else None
    return np.array(snaps),dict(status='OK',history=records,sign_zero_preserved=sign_ok,
        minimum_input_total=float(total[valid].min()),unconstrained=int((~constrained).sum()))

