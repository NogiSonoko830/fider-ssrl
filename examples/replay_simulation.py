"""Replay the original two-band fixed inputs. Run from an installed checkout."""
from pathlib import Path
import json
import numpy as np
from fider_ssrl.simulation_fider import kernels,fider
from fider_ssrl.simulation_rl import deconv
from fider_ssrl.simulation_ssrl import signed_deconv

def main():
    data=Path(__file__).resolve().parents[1]/'data';rows=[]
    settings=json.loads((data/'settings.json').read_text())
    with np.load(data/'miri_inputs.npz') as z,np.load(data/'miri_references.npz') as ref,np.load(data/'simulation_expected.npz') as saved:
        for c in settings:
            ds=c['dataset'];p=z[ds+'__psf'];b=c['background'];n=c['canvas']
            ops,diagnostics=kernels(p,b,c['calibration_flux'],n)
            fi,_=fider(z[ds+'__raw0'],p,b,ops,n)
            want=saved[ds+'__mother_prefilter'];error=float(np.max(abs(fi['FiDeR1000']-want))/abs(want).max());assert error<1e-10
            rows.append(dict(dataset=ds,stage='prefilter1000',archived_relative_max=error))
            for backend in ['RL','SSRL']:
                solve=lambda x: deconv(x,p,b,'RL',n,ks=[40])[0][0] if backend=='RL' else signed_deconv(x,p,b,n,ks=[40])[0][0]
                truth=solve(z[ds+'__truth']);expected=ref[ds+'__'+('RL' if backend=='RL' else 'SignedX')]
                err=float(np.max(abs(truth-expected))/max(abs(expected).max(),1e-30));assert err<1e-10
                raw=solve(z[ds+'__raw0']);filtered=solve(fi['FiDeR1000'])
                for label,result in [('Raw',raw),('FiDeR',filtered)]:
                    want=saved[ds+'__mother_'+backend+'_'+label];error=float(np.max(abs(result-want))/abs(want).max());assert error<1e-10
                    rows.append(dict(dataset=ds,stage=backend+'/'+label,archived_relative_max=error))
                rows.append(dict(dataset=ds,stage=backend+'/noiseless_reference',archived_relative_max=err))
    print(json.dumps(rows,indent=2));return rows
if __name__=='__main__':main()
