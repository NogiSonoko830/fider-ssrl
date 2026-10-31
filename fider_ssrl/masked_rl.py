"""Classical RL with missing measurements on the FFTModel canvas.

The observation mask is in measurement space. It is NOT a zero-valued
observation and is NOT a constraint fixing the corresponding latent pixel.
This extends only classical RL. It does not silently redefine the paper's
signed SSRL or assert that FiDeR retains one PSF with a nonuniform mask.
"""
import numpy as np


def masked_rl(model,total,valid,iterations=40,initial_background=1.,floor=1e-12,
              record_objective=False):
    total=np.asarray(total,dtype=float);valid=np.asarray(valid,dtype=bool)
    if total.shape!=model.shape or valid.shape!=model.shape:
        raise ValueError('Data and binary observation mask must match the canvas.')
    if not valid.any():raise ValueError('No valid observations.')
    if not np.all(np.isfinite(total[valid])) or np.any(total[valid]<0):
        raise ValueError('Observed data must be finite and nonnegative.')
    if int(iterations)!=iterations or iterations<0 or floor<=0:
        raise ValueError('Invalid iteration count or floor.')
    if initial_background<=0 or not np.isfinite(initial_background):
        raise ValueError('Missing-pixel initial intensity must be positive.')
    sensitivity=model.adjoint(valid.astype(float))
    constrained=sensitivity>1e-12*float(sensitivity.max())
    x=np.where(valid,total,initial_background)
    # np.where sanitizes masked values for arithmetic; the mask determines
    # all likelihood contributions. This is not zero-filling missing data.
    data=np.where(valid,total,0.)
    history=[]
    def objective():
        mu=np.maximum(model.forward(x),floor)
        return float(np.sum(mu[valid]-data[valid]*np.log(mu[valid])))
    if record_objective:history.append(objective())
    for _ in range(iterations):
        mu=np.maximum(model.forward(x),floor)
        ratio=np.zeros(model.shape)
        np.divide(data,mu,out=ratio,where=valid)
        correction=np.ones(model.shape)
        np.divide(model.adjoint(ratio),sensitivity,out=correction,where=constrained)
        x*=np.maximum(correction,0.)
        if record_objective:history.append(objective())
    diagnostics=dict(observed_pixels=int(valid.sum()),missing_pixels=int((~valid).sum()),
                     unconstrained_latent_pixels=int((~constrained).sum()),
                     minimum_constrained_sensitivity=float(sensitivity[constrained].min()),
                     poisson_negative_log_likelihood=history)
    return x,diagnostics
