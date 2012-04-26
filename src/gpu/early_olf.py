"""
  FileName    [ early_olf.py ]
  PackageName [ gpu ]
  Synopsis    [ CPU/GPU simulation for the Early Olfaction of Drosphila ]
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
struct LeakyIAF
{
    double V;
    double Vr;
    double Vt;
    double tau;
    double R;
    int    num;
    int    offset;
};
struct AlphaSynNL
{
    int    neu_idx;
    double neu_coe;
};
struct AlphaSyn
{
    double g;
    double gmax;
    double tau;
    double sign;
    int    num;    // number of innerved neuron 
    int    offset; // Offset in the array
};
#define update_V( neuron, bh, spk )  \
{                                    \
    neuron.V = neuron.V*bh;          \
    spk = 0;                         \
    if( neuron.V >= neuron.Vt )      \
    {                                \
        neuron.V = neuron.Vr;        \
        spk = 1;                     \
    }                                \
}
#define compute_ode_coef( neuron, bh, ic )  \
{                                           \
    bh = exp( -dt/neuron.tau );             \
    ic = neuron.R*( 1.0-bh );               \
}
#define neu_thread_copy( neuron, bh, ic, n_s_list,               \
                         all_neu_syn_list )                      \
{                                                                \
    bh = exp( -dt/neuron.tau );                                  \
    ic = neuron.R*( 1.0-bh );                                    \
    n_s_list = all_neu_syn_list + neuron.offset;                 \
}
#define syn_thread_copy( synapse, tr, gmax, s_n_list,            \
                         all_syn_neu_list )                      \
{                                                                \
    gmax = synapse.gmax;                                         \
    tr = 1/synapse.tau;                                          \
    s_n_list = all_syn_neu_list + synapse.offset;                \
}

#define update_G( synapse, g_old, g_new, gmax, tr,               \
                  s_n_list,  spk_list  )                         \
{                                                                \
    /* Update g(t) */                                            \
    g_new[0] = g_old[0];                                         \
    if( g_new[0] < 0.0 ) g_new[0] = 0.0;                         \
                                                                 \
    /* Update g'(t) */                                           \
    g_new[1] = g_old[1] + dt*g_old[2];                           \
    for( int j=0; j<synapse.num; ++j)                            \
        if( spk_list[ s_n_list[j].neu_idx ] )                    \
            g_new[1] += s_n_list[j].neu_coe;                     \
                                                                 \
    /* Update g"(t) */                                           \
    g_new[2] = (-2.0*g_old[1] - tr*g_old[2])*tr;                 \
    /* Copy g_old to g_new */                                    \
    for( int i=0; i<3; ++i ) g_old[i] = g_new[i];                \
    synapse.g = gmax*g_new[0];                                   \
    __syncthreads();                                             \
}

__global__ void gpu_run( int N, double dt, 
                         int neu_num, LeakyIAF *neuron, 
                         int *neu_syn_list,
                         int syn_num, AlphaSyn *synapse, 
                         AlphaSynNL *syn_neu_list,
                         int *spike_list
                       )
{
    // Constant for neuron update
    //__shared__ double BH[MAX_THREAD];
    //__shared__ double IC[MAX_THREAD]; 

    const int tid = threadIdx.x+threadIdx.y*blockDim.x;
    const int uid = tid;  // unit id; unit is either neuron or synapse
    int sid = uid;        // spike id, updated per dt
    
    

    // local copy of neuron parameters
    int* n_s_list;
    double bh, ic;
    int *dt_spk_list = spike_list;
    if( uid < neu_num )
        neu_thread_copy( neuron[uid], bh, ic, n_s_list, neu_syn_list );


    // local copy of synapse parameters
    double g_new[3],tau_r, gmax;
    double g_old[3] = {0, 0, 0};
    AlphaSynNL *s_n_list;
    if( uid < syn_num )
        syn_thread_copy( synapse[uid], tau_r, gmax, s_n_list, syn_neu_list );


    // Compute coefficients for Exponential Euler Method
    //if( uid < neu_num )
    //    compute_ode_coef( neuron[uid], bh, ic )
    
    // Simulation Loop
    for( int i = 0; i<N; ++i )
    {
        // Update Neuron Membrane Voltage
        if( uid < neu_num )
            update_V( neuron[uid], bh, spike_list[sid] );

        // Update Synapse Status
        if( uid < syn_num )
            update_G( synapse[uid], g_old, g_new, 
                      gmax, tau_r, s_n_list, dt_spk_list );
        
        // Update Spike ID and dt-spike array address
        sid += neu_num;
        dt_spk_list += neu_num;
    }
}
"""

cuda_func = SourceModule(cuda_source, options = ["--ptxas-options=-v"])
cuda_gpu_run = cuda_func.get_function("gpu_run")

class AlphaSyn:
    def __init__(self, neu_list, neu_coef, gmax, tau, sign=1):
        self.neu_list = neu_list
        self.neu_coef = neu_coef
        self.tau = tau
        self.gmax = gmax
        self.gvec = np.array([0, 1/tau**2, 0]) #[g(t) g'(t) g"(t)] 
        self.sign = sign # -1:inhibitory; 1:excitatory

    def update(self,dt,spk_list):
        g_new = np.zeros(3);
        # update g(t)
        g_new[0] = max([0,self.gvec[0] + dt * self.gvec[1]]);
        # update g'(t)
        g_new[1] = self.gvec[1] + dt * self.gvec[2]
        #g_new[1] = self.gvec[1] + np.dot(self.neu_coef,spk_list[self.neu_list])
                
        for n,w in zip(self.neu_list,self.neu_coef):
            if spk_list[n]:
                g_new[1] += w
        # upate g"(t)
        g_new[2] = (-2*self.gvec[1] - 1/self.tau*self.gvec[0])/self.tau;
        self.gvec = g_new
        
    def _get_g(self):
        return self.gvec[0]*self.gmax
        
    g = property(_get_g)

class IAFNeu:
    def __init__(self,V0,Vr,Vt,tau,R,syn_list):
        self.V  = V0
        self.Vr = Vr
        self.Vt = Vt

        self.tau = tau
        self.R   = R
        
        self.syn_list = syn_list
        self.I = 0
        self.spk = False
        
        # NOt clear when to set -dt/tau
        #self.bh = np.exp(-dt/self.tau)
    def update_BH(self,dt):
        self.bh = np.exp(-dt/self.tau)
    
    def update_V(self):
        # Euler Exponetial Method
        self.V = self.V*self.bh + self.R*self.I*(1-self.bh) 
        self.spk = False
        # not sure where to store spiking information
        if self.V >= self.Vt:
            self.V = self.Vr
            self.spk = True
        
    def update_I(self,syn_list,I_ext=0):
        g = 0
        for i in self.syn_list:
            s = syn_list[i]
            g += s.g*s.sign
        self.I = I_ext + g*(self.V-self.Vr)

class Early_olfaction_Network:
    def readNeuron(self,f,neu_num):
        for i in xrange(neu_num):
            lineInFile = f.readline()
            name, V0, Vr, Vt, tau, R = lineInFile.split(' ')
            if self.neu_name.has_key( name ): 
                sys.exit('Deplicate declaration of Neuron: ' + name)
            self.neu_name[ name ] = len( self.neu_name )
            self.neu_list.append( IAFNeu(float(V0),float(Vr),float(Vt),
                                  float(tau), float(R), []) )
                                  
    def neuAppendSyn(self, post_neu, syn_idx = -1 ):
        if syn_idx == -1: syn_idx = len(self.syn_list)
        self.neu_list[ self.neu_name[post_neu] ].syn_list.append( syn_idx )
    
    def readSynapse(self,f,syn_num):
        for i in xrange(syn_num):
            lineInFile = f.readline()
            pre_neu, post_neu, gmax, tau, coef, sign = lineInFile.split(' ')
            name = pre_neu + '-' + post_neu
            if self.syn_name.has_key( name ): 
                sys.exit('Deplicate declaration of Synapse: ' + name)
            if self.neu_name.has_key( pre_neu ) == False:
                sys.exit('No such Presynaptic Neuron: ' + pre_neu)
            if self.neu_name.has_key( post_neu ) == False:
                sys.exit('No such Postsynaptic Neuron: ' + post_neu)
            self.syn_name[ name ] = len(self.syn_name)
            self.neuAppendSyn( post_neu )
            self.syn_list.append( AlphaSyn( [self.neu_name[pre_neu]], 
                                  [float(coef)], float(gmax), float(tau), 
                                   float(sign)))
            
        
    def readPreSyn(self,f,presyn_num):
        for i in xrange(presyn_num):
            lineInFile = f.readline()
            ln_neu, pre_neu, post_neu, coef = lineInFile.split(' ')
            syn_name = pre_neu + '-' + post_neu
            if self.neu_name.has_key( ln_neu ) == False: 
                sys.exit('No such Local Neuron: ' + ln_neu)
            if self.syn_name.has_key( syn_name ) == False: 
                sys.exit('No such Synapse: ' + syn_name)
            syn_idx = self.syn_name[ syn_name ]
            self.syn_list[syn_idx].neu_list.append( self.neu_name[ ln_neu ] )
            self.syn_list[syn_idx].neu_coef.append( float(coef) )
            
    def readIgnore(self,f,ignore_num):
        for i in xrange(ignore_num):
            f.readline()
            
    def __init__(self,filename):
        self.neu_list = []
        self.neu_name = {}
        self.syn_list = []
        self.syn_name = {}
        
        f = open(filename,'r')
        while True:
            s = f.readline()
            if s == '': break
            try:
                dtype, dnum = s.split(' ')
            except:
                sys.exit("Usage: <Neuron/Synapse/PreSyn> DataNum\n" + s)
            if dtype == 'Neuron': self.readNeuron(f, int(dnum))
            if dtype == 'Synapse': self.readSynapse(f, int(dnum))
            if dtype == 'PreSyn': self.readPreSyn(f, int(dnum))
            if dtype == 'Ignore': self.readIgnore(f, int(dnum))
                
        self.spk_list = []
        self.neu_num = len(self.neu_list)
        self.syn_num = len(self.syn_list)

    def basic_prepare(self,dt,dur):
        self.Nt = int(dur/dt)
        self.dt = dt
        self.dur = dur
        self.spk_list = np.zeros((self.Nt,self.neu_num), np.int32)
 
    def cpu_prepare(self,dt,dur):
        self.basic_prepare(dt,dur)
        for neu in self.neu_list:
            neu.update_BH(dt)
 
    def cpu_run(self,dt,dur,I_ext=np.empty((0,0))):
        self.cpu_prepare(dt,dur)
        pbar = pb.ProgressBar(maxval=self.Nt).start()
        dt_spk_list = np.empty(self.neu_num).astype(np.bool)
        for i in xrange(self.Nt):
            pbar.update(i)
            for j in xrange(self.neu_num):
                self.neu_list[j].update_V()
                dt_spk_list[j] = self.neu_list[j].spk
            for syn in self.syn_list:
                syn.update(dt,dt_spk_list)
            for j in xrange(self.neu_num):
                if j < I_ext.shape[0] and i < I_ext.shape[1] :
                    self.neu_list[j].update_I(self.syn_list,I_ext[j,i])
                else:
                    self.neu_list[j].update_I(self.syn_list)
            self.spk_list[i,:] = dt_spk_list
        print ""
            
    def gpu_prepare(self,dt,dur):
        self.basic_prepare(dt,dur)
        
        # Merge Neuron data
        gpu_neu_list = np.zeros( self.neu_num, dtype=('f8,f8,f8,f8,f8,i4,i4') )
        offset, agg_syn = 0,[]
        for i in xrange( self.neu_num ):
            n = self.neu_list[i]
            gpu_neu_list[i] = ( n.V, n.Vr, n.Vt, n.tau,
                                n.R, len(n.syn_list), offset)
            offset += len( n.syn_list )
            agg_syn.extend( n.syn_list  )
        gpu_neu_syn_list = np.array( agg_syn, dtype=np.int32 )
        
        # Merge Synapse data
        gpu_syn_list = np.zeros( (self.syn_num,6),dtype=('f8,f8,f8,f8,i4,i4') )
        offset, agg_neu, agg_coe = 0, [], []
        for i in xrange( self.syn_num ):
            s = self.syn_list[i]
            gpu_syn_list[i,:] = (s.g, s.gmax, s.tau, s.sign, 
                                 len(s.neu_list), offset)
            offset += len(s.neu_list)
            agg_neu.extend( s.neu_list )
            agg_coe.extend( s.neu_coef )
        gpu_syn_neu_list = np.array( zip(agg_neu,agg_coe), dtype=('i4,f8') )

        # Determine Bloack and Grid size
        # TODO
        return gpu_neu_list, gpu_neu_syn_list, gpu_syn_list, gpu_syn_neu_list

    def gpu_run(self,dt,dur):
        neu_list,neu_syn_list,syn_list,syn_neu_list = self.gpu_prepare(dt,dur)
        cuda_gpu_run( np.int32(self.Nt), np.double( dt ), 
                      np.int32(self.neu_num), 
                      drv.In(neu_list), drv.In(neu_syn_list),
                      np.int32(self.syn_num),
                      drv.In(syn_list), drv.In(syn_neu_list),
                      drv.Out( self.spk_list ),block=(1024,1,1) )

    def compare_cpu_gpu(self,dt,dur):
        self.gpu_run(dt,dur)
        gpu_spk = self.spk_list
        self.cpu_run(dt,dur)
        compare = self.spk_list == gpu_spk
        if gpu_spk.size == compare.sum():
            print "Cool, cpu and gpu gives the same result!!"
        else:
            print "Bomb!! cpu and gpu gives different reults!!"


    def plot_raster(self, show_stems=True, show_axes=True, show_y_ticks=True,
                    marker='.', markersize=5, fig_title='', file_name=''):
        """
        This function is based on the homework 2 solution given by 
        Lev Givon.
        """
        p.clf()
        p.gcf().canvas.set_window_title(fig_title)
        ax = p.gca()
        ax.axis([0, self.dur, -0.5, self.neu_num-0.5 ])
        
        if show_y_ticks:
            p.yticks(xrange(self.neu_num))
        else:
            for tick_label in ax.get_yticklabels():
                tick_label.set_visible(False)
        for y in xrange(self.neu_num):
            spk_time = self.spk_list[:,y].nonzero()[0]*self.dt
            if show_axes:
                p.axhline(y, 0, 1, color='b', hold=True)
            p.plot(spk_time, y*np.ones(len( spk_time )), 
                   marker, hold=True,
                   color=(1-1./self.neu_num*y,0.5,1./self.neu_num*y),
                   markersize=markersize,
                   scalex=False, scaley=False)
            if show_stems:
                for t in spk_time:
                    ax.add_line(mp.lines.Line2D([t,t], [-0.5, y], 
                                color='r', linestyle=':'))
        ax.xaxis.set_major_locator(mp.ticker.MultipleLocator(
                                   10.0**np.ceil(np.log10(self.dur/10))))
        p.xlabel('time, sec')
        p.ylabel('Neuron')
        if fig_title:
            p.title(fig_title)

        p.draw_if_interactive()
        if file_name:
            p.savefig(file_name)
        

datapath = '../../data/'
picpath  = '../../../pic/'

if __name__=='__main__':
    if len(sys.argv) == 1:
        sys.exit()
        
if sys.argv[1] == 'Read_Olf':
    dt = 1e-5
    dur = 1.
    curtime = strftime("[%a_%d_%b_%Y_%H_%M_%S]", gmtime())
    filename = sys.argv[2]
    olfnet = Early_olfaction_Network( datapath + filename)
    olfnet.cpu_run(dt,dur)
    olfnet.plot_raster(show_stems=False, show_axes=False, 
                            show_y_ticks=False, markersize=5,
                            file_name=picpath+filename+curtime+'cpu.png',
                            fig_title='cpu')
    olfnet = Early_olfaction_Network( datapath + filename)
    olfnet.gpu_run(dt,dur)
    olfnet.plot_raster(show_stems=False, show_axes=False, 
                            show_y_ticks=False, markersize=5,
                            file_name=picpath+filename+curtime+'gpu.png',
                            fig_title='gpu')
 
if sys.argv[1] == 'compare_cpu_gpu':
    dt = 1e-5
    dur = 1.
    filename = sys.argv[2]
    olfnet = Early_olfaction_Network( datapath + filename)
    olfnet.compare_cpu_gpu(dt,dur)
        
if sys.argv[1] == 'synapse':
    dt = 1e-5
    t = np.arange(0,10,1e-5)
    syn = AlphaSyn([0,1],[4,-1],1,0.25)
    g = np.zeros_like(t);
    spk_list = (np.random.rand(2,t.shape[0]) < 1e-5)
    for i in xrange(t.shape[0]):
        syn.update(dt,spk_list[:,i])
        g[i] = syn.g

    p.figure()
    
    p.subplot(2,1,1);p.plot(t,g)
    p.subplot(2,1,2);p.plot(t,spk_list[0,:],t,spk_list[1,:])
    p.legend(['Excitatory','Inhibitory'])
    p.savefig('./pic/test_syn.png')

