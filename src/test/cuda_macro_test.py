"""
  FileName    [ gpu_macro_test.py ]
  PackageName []
  Synopsis    [ Define class CmdParser ]
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

import matplotlib as mp
mp.rc('savefig', dpi=150)
mp.use('AGG')
import pylab as p
import time
from pycuda.compiler import SourceModule


cuda_source = """

struct LeakyIAF
{
    double V;
    double Vr;
    double Vt;
    double tau;
    double R;
    double I;
};
# define multiply2(idx,data) \
{                            \
    data[idx] *= data[idx];          \
}

__global__ void macro_test(int num, int *a)
{
    const int tid = threadIdx.x+threadIdx.y*blockDim.x;
    if( tid < num ) multiply2(tid,a)
}
"""

cuda_func = SourceModule(cuda_source, options = ["--ptxas-options=-v"])
macro_test = cuda_func.get_function("macro_test")

if __name__ == "__main__":
    n = 10
    a = np.arange(n,0,-1).astype(np.int32)
    g_a = gpuarray.to_gpu(a)
    macro_test( np.int32(n), g_a,
                block=(int(1024),int(1),int(1)), grid=(int(1),int(1)))
    r_a = g_a.get()
    print r_a

