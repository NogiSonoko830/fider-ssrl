import unittest
import numpy as np
from fider_ssrl.fider_core import FFTModel
from fider_ssrl.masked_backend import Fill,solve
from fider_ssrl.simulation_ssrl import signed_step

class Conventions(unittest.TestCase):
    def test_missing_measurements_are_not_observed_zeros(self):
        y,x=np.indices((7,7));p=np.exp(-((x-3)**2+(y-3)**2)/3);h=FFTModel(p,15).otf
        net=np.random.default_rng(71).normal(size=(15,15));mask=np.ones_like(net,dtype=bool);mask[4:6,5:8]=False
        changed=net.copy();changed[~mask]=np.nan;fill=Fill(mask)
        for backend in ['RL','SSRL_masked']:
            a,da=solve(net,20.,mask,h,backend,[10],fill,'harmonic')
            b,db=solve(changed,20.,mask,h,backend,[10],fill,'harmonic')
            self.assertEqual(da['status'],'OK');self.assertEqual(db['status'],'OK');np.testing.assert_array_equal(a,b)
    def test_signed_freeze_is_persistent_and_zero_absorbing(self):
        p=np.zeros((5,5));p[2,2]=1;h=FFTModel(p,5).otf
        x=np.full((5,5),2.);x[0,0]=-2.;x[1,1]=0.;y=np.full_like(x,3.)
        frozen=np.zeros_like(x,dtype=bool);den=np.ones_like(x)
        result,new,_=signed_step(x,y,h,1.,frozen,den)
        self.assertTrue(new[0,0]);self.assertLess(result[0,0],0);self.assertEqual(result[1,1],0)
        result[0,0]=10.;again,_,_=signed_step(result,y,h,1.,frozen,den)
        self.assertTrue(frozen[0,0]);self.assertAlmostEqual(again[0,0],10.)
    def test_complete_domain_rl_count_identity(self):
        y,x=np.indices((7,7));p=np.exp(-((x-3)**2+(y-3)**2)/3);model=FFTModel(p,15)
        total=np.random.default_rng(92).poisson(50.,(15,15)).astype(float)
        result,_=model.iterate(total,10,'RL');self.assertAlmostEqual(result.sum(),total.sum(),places=8)

if __name__=='__main__':unittest.main()
