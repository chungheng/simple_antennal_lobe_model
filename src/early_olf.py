import numpy as np
import pickle
from time import gmtime, strftime
import sys
import matplotlib as mp
mp.rc('savefig', dpi=300)
mp.use('AGG')
import pylab as p
import progressbar as pb


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
        return self.gvec[0]
        
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
        if self.V > self.Vt:
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
        self.Num = len(self.neu_list)

    def prepare(self,dt,dur):
        self.Nt = int(dur/dt)
        self.dt = dt
        self.dur = dur
        self.spk_list = np.zeros((self.Num, self.Nt), np.int32)
        for neu in self.neu_list:
            neu.update_BH(dt)

    def run(self,dt,dur,I_ext=np.empty((0,0))):
        self.prepare(dt,dur)
        pbar = pb.ProgressBar(maxval=self.Nt).start()
        dt_spk_list = np.empty(self.Num).astype(np.bool)
        for i in xrange(self.Nt):
            pbar.update(i)
            for j in xrange(self.Num):
                self.neu_list[j].update_V()
                dt_spk_list[j] = self.neu_list[j].spk
            for syn in self.syn_list:
                syn.update(dt,dt_spk_list)
            for j in xrange(self.Num):
                if j < I_ext.shape[0] and i < I_ext.shape[1] :
                    self.neu_list[j].update_I(self.syn_list,I_ext[j,i])
                else:
                    self.neu_list[j].update_I(self.syn_list)
                    
            self.spk_list[:,i] = dt_spk_list
 

    def plot_raster(self, show_stems=True, show_axes=True, show_y_ticks=True,
                    marker='.', markersize=5, fig_title='', file_name=''):
        """
        This function is based on the homework 2 solution given by 
        Lev Givon.
        """
        p.clf()
        p.gcf().canvas.set_window_title(fig_title)
        ax = p.gca()
        ax.axis([0, self.dur, -0.5, self.Num-0.5 ])
        
        if show_y_ticks:
            p.yticks(xrange(self.Num))
        else:
            for tick_label in ax.get_yticklabels():
                tick_label.set_visible(False)
        for y in xrange(self.Num):
            spk_time = self.spk_list[y].nonzero()[0]*self.dt
            if show_axes:
                p.axhline(y, 0, 1, color='b', hold=True)
            p.plot(spk_time, y*np.ones(len( spk_time )), 
                   marker, hold=True,
                   color=(1-1./self.Num*y,0.5,1./self.Num*y),
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
        
if __name__=='__main__':
    if len(sys.argv) == 1:
        sys.exit()
        
if sys.argv[1] == 'Read_Olf':
    dt = 1e-5
    dur = 1.
    curtime = strftime("[%a_%d_%b_%Y_%H_%M_%S]", gmtime())
    filename = sys.argv[2]
    #I_pkl = 'odor1.pkl'    
    #I_ext = np.zeros((3,int(dur/dt)))
    #I_ext[0,:] = 2
    #I_ext[1,:] = 0.8
    #I_ext[2,:] = 0.6
    #I_ext = pickle.load(open(I_pkl, 'rb'))
    #I_ext *= 0.5
    olfnet = Early_olfaction_Network('./data/' + filename)
    olfnet.run(dt,dur)
    olfnet.plot_raster(show_stems=False, show_axes=False, 
                            show_y_ticks=False, markersize=5,
                            file_name='./pic/test/'+filename+curtime+'.png')
if sys.argv[1] == 'gen_odor':
    output = open('odor1.pkl', 'wb')
    
    I_ext = np.zeros((3,30000))
    I_ext[0,:] = 2
    I_ext[1,:] = 0.8
    I_ext[2,:] = 0.6
    pickle.dump(I_ext,output)
    output = open('odor2.pkl', 'wb')

    I_ext = np.zeros((3,30000))
    I_ext[0,:] = 0.4
    I_ext[1,:] = 2.2
    I_ext[2,:] = 0.3
    pickle.dump(I_ext,output)
    output = open('odor3.pkl', 'wb')
    I_ext = np.zeros((3,30000))
    I_ext[0,:] = 0.4
    I_ext[1,:] = 0.2
    I_ext[2,:] = 1.9
    pickle.dump(I_ext,output)
        
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

