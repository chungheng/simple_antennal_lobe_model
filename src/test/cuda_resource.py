"""
  FileName    [ cuda_resource.py ]
  PackageName [ ]
  Synopsis    [ Print out the GPU attributes of Huxley ]
  Author      [ Chung-Heng Yeh <chyeh@ee.columbia.edu> ]
  Copyright   [ Copyleft(c) 2012-2014 Bionet Group at Columbia University ]
  Note        []
"""
# Use device 4 on huxley.ee.columbia.edu
import pycuda.driver as drv
import atexit 
drv.init()
mydev = drv.Device(4)
myctx = mydev.make_context()
atexit.register(myctx.pop)
import numpy as np
import pycuda.gpuarray as gpuarray

print mydev.get_attributes()
