"""Original same-grid simulation RL; EPS floors are retained."""
import numpy as np
from scipy import fft
EPS=1e-12
KS=[10,20,30,40]

def embed(a,n):
 a=np.asarray(a,float);s=a.shape[-1];assert a.shape[-2]==s and (n-s)%2==0
 pad=(n-s)//2;z=np.zeros((*a.shape[:-2],n,n));z[...,pad:pad+s,pad:pad+s]=a;return z


def conv(a,h):return fft.irfft2(fft.rfft2(a,axes=(-2,-1),workers=1)*h,s=a.shape[-2:],axes=(-2,-1),workers=1)


def otf(p,n):return fft.rfft2(np.fft.ifftshift(embed(p,n)),workers=1)


def update(x,y,h,b=0):
 pred=np.maximum(conv(x,h)+b,EPS);cor=conv(y/pred,np.conj(h));assert cor.min()>-1e-8
 return np.maximum(x*np.maximum(cor,0),EPS)


def deconv(source,p,b,algo,n=225,ks=KS):
 s=source.shape[-1];pad=(n-s)//2;im=embed(source,n)+b;assert im.min()>0
 h=otf(p,n);x=np.maximum(im,EPS).copy();out=[];tot=[]
 for k in range(1,max(ks)+1):
  x=update(x,im,h,b if algo=='Snyder-RL' else 0)
  if k in ks:
   est=x if algo=='Snyder-RL' else x-b;out.append(est[...,pad:pad+s,pad:pad+s].copy());tot.append(est.sum(axis=(-2,-1)))
 out=np.array(out);tot=np.array(tot)
 if algo=='RL':assert np.max(abs(tot-source.sum(axis=(-2,-1))))<1e-3
 return out,tot

