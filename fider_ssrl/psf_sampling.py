"""Retained cubic/Fourier sampling of a supplied detector PSF."""
import numpy as np
from scipy.fft import fft2,ifft2,fftfreq
from scipy.ndimage import map_coordinates,spline_filter

class PSFSampler:
    def __init__(self, psf, kind, canvas=225):
        self.p=psf/psf.sum(); self.kind=kind; self.n=psf.shape[0]
        self.yy,self.xx=np.indices(psf.shape); self.pc=(self.n-1)/2
        if kind=='cubic': self.coeff=spline_filter(self.p)
        elif kind=='fourier':
            start=(canvas-self.n)//2;self.crop=np.s_[start:start+self.n,start:start+self.n]
            a=np.zeros((canvas,canvas));a[self.crop]=self.p
            self.ft=fft2(a);self.f=fftfreq(canvas)
        else:raise ValueError(kind)

    def at(self,xc,yc):
        if self.kind=='cubic':
            return map_coordinates(self.coeff,[self.yy-yc+self.pc,self.xx-xc+self.pc],
                                   order=3,prefilter=False,mode='constant',cval=0.)
        phase=np.exp(-2j*np.pi*(self.f[:,None]*(yc-self.pc)+self.f[None,:]*(xc-self.pc)))
        return ifft2(self.ft*phase).real[self.crop]

