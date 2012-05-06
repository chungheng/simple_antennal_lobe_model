"""
  FileName    [ cuda_accArray.py ]
  PackageName [ gpu 
  Synopsis    [ Test Accumulate Array Structure on GPU  ]
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
mp.rc('savefig', dpi=300)
mp.use('AGG')
import pylab as p
from pycuda.compiler import SourceModule
from time import gmtime, strftime
import sys
import progressbar as pb


cuda_source = """
#define MAX_THREAD 1024
struct AlphaSyn
{
    int    num;    // number of innerved neuron 
    int    offset; // Offset in the array
};
struct AlphaSyn_Neu
{
    long   idx;
    double value;
};
#define alpha_sum( out, neu_list, num ) \
{                                       \
    out = 0;                            \
    for( int i=0; i<num; ++i )          \
        out += neu_list[i].value;       \
}
__global__ void test( int num, double *ans, AlphaSyn *arr, AlphaSyn_Neu *dat )
{
    const int tid = threadIdx.x;
    if( tid >= num ) return;
    AlphaSyn_Neu* tmp = dat + arr[tid].offset;
    //ans[tid] = tmp[].idx;
    alpha_sum( ans[tid], tmp, arr[tid].num  );
    //ans[tid] = arr[tid].offset;
}
"""

cuda_func = SourceModule(cuda_source, options = ["--ptxas-options=-v"])
cuda_gpu_run = cuda_func.get_function("test")



if __name__ == "__main__":
    n = 5
    idx = []
    dat = []
    for i in xrange(n):
        idx.append( np.arange(i+1) )
        dat.append( np.ones(i+1)*(i+1) )
    stk_idx = []
    stk_dat = []
    g_len   = np.zeros(n).astype(np.int32);
    for i in xrange(n):
        g_len[i] = len( idx[i] )
        stk_idx.extend( idx[i] )
        stk_dat.extend( dat[i] )
    g_off = np.insert(g_len[:-1].cumsum(),0,0).astype(np.int32)
    g_dat = np.array( zip( stk_idx, stk_dat ), dtype=('i8,f8') )
    g_idx = np.array(zip( g_len, g_off ),dtype=np.int32)
    g_ans = np.zeros(n).astype(np.float64)
    print g_idx
    print g_dat
    cuda_gpu_run( np.int32(n), drv.Out(g_ans),
                  drv.In(g_idx), drv.In(g_dat),
                  block=(10,1,1) )

    print g_ans
