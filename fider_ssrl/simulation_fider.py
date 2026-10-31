"""Original PASP front end: total RL, beta calibration and unit DC matching."""
import numpy as np
from scipy import fft
from .simulation_rl import embed,conv,otf,update,EPS
ETA=3e-14

def kernels(p,b,flux,n=225):
 target=embed(p,n);beta=b/flux;y=target+beta;h=otf(p,n);x=np.maximum(y,EPS);ops={};rows=[]
 for k in range(1,1001):
  x=update(x,y,h)
  if k in [200,1000]:
   ke_image=x-beta;ke=fft.rfft2(np.fft.ifftshift(ke_image));m=np.conj(ke)*h/(abs(ke)**2+ETA);m[0,0]=1;ops[k]=m
   matched=conv(ke_image,m);rows.append(dict(k=k,beta=beta,response_max_abs=float(abs(matched-target).max()),response_relative_l2=float(np.linalg.norm(matched-target)/np.linalg.norm(target)),ke_min=float(ke_image.min()),ke_sum=float(ke_image.sum()),matching_max_otf=float(abs(m).max()),matching_dc_real=float(m[0,0].real)))
 return ops,rows


def fider(source,p,b,ops,n=225):
 single=source.ndim==2
 if single:source=source[None]
 s=source.shape[-1];pad=(n-s)//2;y=embed(source,n)+b;assert y.min()>0
 x=np.maximum(y,EPS);h=otf(p,n);out={};diag=[]
 for k in range(1,1001):
  x=update(x,y,h)
  if k in [200,1000]:
   matched=conv(x,ops[k])-b;a=matched[...,pad:pad+s,pad:pad+s].copy();out['FiDeR'+str(k)]=a[0] if single else a
   diag.append(dict(k=k,latent_total_conservation_max_counts=float(abs(x.sum((-2,-1))-y.sum((-2,-1))).max()),source_matching_total_conservation_max_counts=float(abs(matched.sum((-2,-1))-source.sum((-2,-1))).max()),minimum_output_source=float(a.min()),source_output_clipped=False))
 return out,diag

