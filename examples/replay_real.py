"""Replay native-mask FiDeR then common-mask RL / SSRL* for both paper examples."""
from pathlib import Path
import json
import numpy as np
from fider_ssrl.real_fider import reconstruct
from fider_ssrl.fider_core import FFTModel
from fider_ssrl.masked_backend import Fill,solve

def main():
    data=Path(__file__).resolve().parents[1]/'data';rows=[]
    for band in ['f1000','f1800']:
        c=json.loads((data/f'real_{band}.json').read_text())['case']
        with np.load(data/f'real_{band}.npz') as z:
            a={k:z[k] for k in z.files}
        out,_=reconstruct(c,a['source'],a['valid'],a['psf'])
        def check(label,x,expected):
            err=float(np.max(abs(x-expected))/max(abs(expected).max(),1e-30))
            assert err<1e-10,(label,err)
            rows.append(dict(band=band,stage=label,relative_max=err))
        check('native_prefilter1000',out['prefilter'],a['expected_prefilter'])
        h=FFTModel(a['psf'],225).otf;mask=a['common_mask'];fill=Fill(mask)
        for i,backend in enumerate(['RL','SSRL_masked']):
            inputs=[a['source'],out['prefilter'],a['coadd']]
            expected=[a['expected_outputs'][i,0],a['expected_outputs'][i,1],a['expected_references'][i]]
            for label,x,want in zip(['Raw','FiDeR','coadd_reference'],inputs,expected):
                result,diag=solve(c['C']*x,c['b'],mask,h,backend,[40],fill,'harmonic')
                assert diag['status']=='OK'
                check(backend+'/'+label,result[0]/c['C'],want)
    print(json.dumps(rows,indent=2));return rows
if __name__=='__main__':main()
