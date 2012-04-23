"""
  FileName    [ gpu_data_size.py ]
  PackageName [ gpu ]
  Synopsis    [ report size of built-in cuda datatype ]
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

# Set matplotlib backend so that plots can be generated without a
# display:
from pycuda.compiler import SourceModule
import progressbar as pb


cuda_source = """
__global__ void size_of_type(int *data)
{
    data[0] = sizeof(bool);
    data[1] = sizeof(void*);
    data[2] = sizeof(int);
}
"""

cuda_func = SourceModule(cuda_source, options = ["--ptxas-options=-v"])
cuda_data_size = cuda_func.get_function("size_of_type")

if __name__ == "__main__":
    data = np.zeros(3).astype(np.int32)
    cuda_data_size(drv.Out(data),block=(1,1,1))
 
    print "CUDA Datatype size:\n" + \
          " bool  : " + repr(data[0]) + "\n" + \
          " void* : " + repr(data[1]) + "\n" + \
          " int   : " + repr(data[2])

