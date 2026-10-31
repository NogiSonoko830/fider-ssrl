"""Retained 225-canvas real-data front end, including subpixel calibration.
The final outputs here use the native mask; paper evaluation uses the common
mask through masked_backend.solve, as shown in examples/replay_real.py.
"""
import numpy as np
from scipy.fft import rfft2,irfft2
from .fider_core import FFTModel
from .masked_rl import masked_rl
from .psf_sampling import PSFSampler

def reconstruct(cp,source,valid,psf):
    C,b,beta=cp['C'],cp['b'],cp['beta'];model=FFTModel(psf,225)
    t=PSFSampler(psf,'cubic').at(cp['xc'],cp['yc']);t=np.maximum(t,0);t/=t.sum();target=np.zeros((225,225));target[57:168,57:168]=t
    response=masked_rl(model,target+beta,valid,1000,initial_background=beta)[0]-beta
    k=rfft2(np.fft.ifftshift(response));tf=rfft2(np.fft.ifftshift(target));matching=k.conj()*tf/(abs(k)**2+3e-14);matching[0,0]=1.
    total=C*source+b;raw=masked_rl(model,total,valid,40,initial_background=b)[0]
    deep=masked_rl(model,total,valid,1000,initial_background=b)[0];pre=irfft2(rfft2(deep)*matching,s=(225,225));fi=masked_rl(model,pre,valid,40,initial_background=b)[0]
    out=dict(raw=(raw-b)/C,fider=(fi-b)/C,prefilter=(pre-b)/C,response=response,matching=matching,deep=(deep-b)/C)
    diag=dict(minimum_valid_total=float(total[valid].min()),minimum_valid_prefilter=float(pre[valid].min()))
    assert all(np.isfinite(v).all() for v in out.values())
    return out,diag

