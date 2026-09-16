"""Check surveyed coordinate transforms with independent physical examples."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('frames',Path(__file__).resolve().parents[1]/'gazebo_sling/export_local_frames.py')
frames=importlib.util.module_from_spec(spec);spec.loader.exec_module(frames)

class HardwareFrameChecks(unittest.TestCase):
    def test_rotated_origin_preserves_physical_position_and_derivatives(self):
        # Local x points map +y; its origin is map (10,-2,3). A map +y
        # displacement/velocity/acceleration must therefore become local +x.
        p=np.array([[10.,0.,4.]]);v=np.array([[0.,3.,0.]]);a=np.array([[0.,4.,0.]])
        lp,lv,la,yaw=frames.transform(p,v,a,np.array([np.pi/2]),[10,-2,3],np.pi/2)
        np.testing.assert_allclose(lp,[[2,0,1]],atol=1e-12)
        np.testing.assert_allclose(lv,[[3,0,0]],atol=1e-12)
        np.testing.assert_allclose(la,[[4,0,0]],atol=1e-12)
        np.testing.assert_allclose(yaw,[0],atol=1e-12)
    def test_origin_translation_does_not_change_velocity_or_acceleration(self):
        p=np.array([[3.,2.,1.]]);v=np.array([[1.,-2.,.5]]);a=np.array([[.1,.2,.3]])
        _,lv,la,yaw=frames.transform(p,v,a,np.array([.2]),[100,-50,20],0)
        np.testing.assert_array_equal(lv,v);np.testing.assert_array_equal(la,a)
        np.testing.assert_allclose(yaw,[.2],atol=1e-12)

if __name__=='__main__':unittest.main()
