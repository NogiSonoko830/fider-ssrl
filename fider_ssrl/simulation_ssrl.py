"""Original signed same-grid reduction, not the UPDC geometry engine."""
import numpy as np
from .simulation_rl import embed,conv,otf
KS=[10,20,30,40]

def signed_step(x, y, h, b, frozen, denominator):
    raw_mu = conv(x, h) + b
    assert np.isfinite(raw_mu).all()
    new = (~frozen) & (raw_mu <= 0)
    frozen |= new
    ratio = np.ones_like(raw_mu)
    np.divide(y, raw_mu, out=ratio, where=~frozen)
    assert np.isfinite(ratio).all() and ratio.min() >= 0
    correction = conv(ratio, np.conj(h))
    assert correction.min() >= -1e-6 * denominator.max()
    # UPDC1.2.0 clips only a tiny negative numerical correction; never x.
    factor = np.maximum(correction, 0) / denominator
    estimate = x * factor
    assert np.isfinite(estimate).all()
    return estimate, new, raw_mu


def signed_deconv(source, p, b, n=225, ks=KS):
    s = source.shape[-1]; pad = (n-s)//2
    x = embed(source, n).copy(); y = x + b
    floor = max(1e-10, np.sqrt(b)/1e8)
    assert y.min() > floor, 'Current positive-SCI mock does not require native flooring'
    h = otf(p, n); denominator = conv(np.ones((n,n)), np.conj(h))
    assert denominator.min() > 1e-8
    frozen = np.zeros_like(x, dtype=bool); snapshots=[]; totals=[]; records=[]
    single = x.ndim == 2
    for k in range(1, max(ks)+1):
        x, new, raw_mu = signed_step(x, y, h, b, frozen, denominator)
        for j in range(1 if single else len(x)):
            nm = new if single else new[j]; fm = frozen if single else frozen[j]
            mm = raw_mu if single else raw_mu[j]; xx = x if single else x[j]
            records.append(dict(iteration=k, realization=j, n_new_frozen=int(nm.sum()),
                n_total_frozen=int(fm.sum()), min_raw_mu=float(mm.min()),
                mu_marker=1e-12, frozen_ratio=1, negative_source_pixels=int((xx<0).sum())))
        if k in ks:
            snapshots.append(x[...,pad:pad+s,pad:pad+s].copy())
            totals.append(x.sum(axis=(-2,-1)))
    return np.array(snapshots), np.array(totals), records

