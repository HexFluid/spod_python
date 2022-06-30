"""
Application of SPOD to compressor blade surface blade data by DDES.
Details of the data can be found in the following:
    
  He, X., Zhao, F., & Vahdati, M. (2022). Detached Eddy Simulation: Recent
  Development and Application to Compressor Tip Leakage Flow. ASME Journal
  of Turbomachinery, 144(1), 011009.

Xiao He (xiao.he2014@imperial.ac.uk)
Last update: 24-Sep-2021
"""

# -------------------------------------------------------------------------
# 0. Import libraries
# standard python libraries
import sys
import time
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
import psutil
import os
import numpy as np
import h5py
import pylab
import io

# SPOD path
current_path = os.getcwd()
parrent_path = os.path.dirname(current_path)
SPOD_path    = parrent_path

# SPOD library
sys.path.insert(0, SPOD_path)
import spod

# -------------------------------------------------------------------------
# 1. Load input data for SPOD
# data shape: nrow = nt (total number of snapshots)
#             ncol = ngrid*nvar (number of grid point * number of variable)
# grid shape: nrow = ngrid (number of grid)
#             ncol = 3 (e.g., x, r, control volume size)
# In this case, nt = 1701, ngrid = 7086, and nvar = 1 (e.g., p)
# -------------------------------------------------------------------------

# Sec. 1 start time
start_sec1 = time.time()

# data path
data_path = os.path.join(current_path, 'comp_data')

# option to save SPOD results
save_fig  = True # postprocess figs
save_path = data_path

# load data from h5 format
h5f  = h5py.File(os.path.join(data_path,'compDDES.h5'),'r')
data = h5f['data'][:]        # flow fields
grid = h5f['grid'][:]        # grid points
dt   = h5f['dt'][0]          # unit in seconds
ng   = int(grid.shape[0]) # number of grid point
nt   = data.shape[0]         # number of snap shot
nx   = data.shape[1]         # number of grid point * number of variable
nvar = int(nx/ng)         # number of variables
BPF  = 1100/60*17            # blade passing frequency = shaft speed * number of blade
SS_idxs = h5f['SS_idxs'][:]  # min/max index of blade suction surface
PS_idxs = h5f['PS_idxs'][:]  # min/max index of blade pressure surface
h5f.close()

# calculate weight
weight_grid = grid[:,2]                         # control volume weighted
weight_grid = weight_grid/np.mean(weight_grid)  # normalized by mean values for better scaling
weight_phy  = np.ones(ng)                       # sigle variable does not require weights from physics
weight      = weight_grid*weight_phy # element-wise multiplation

# Sec. 1 end time
end_sec1 = time.time()

print('--------------------------------------'    )
print('SPOD input data summary:'                  )
print('--------------------------------------'    )
print('number of snapshot   :', nt                )
print('number of grid point :', ng                )
print('number of variable   :', nvar              )
print('--------------------------------------'    )
print('SPOD input data loaded!'                   )
print('Time lapsed: %.2f s'%(end_sec1-start_sec1) )
print('--------------------------------------'    )
      

# -------------------------------------------------------------------------
# 2. Run SPOD
# function spod.spod(data_matrix,timestep)
# -------------------------------------------------------------------------

# Sec. 2 start time
start_sec2 = time.time()

# main function
spod.spod(data, dt, save_path, weight, method='fast')

# Sec. 2 end time
end_sec2 = time.time()

print('--------------------------------------'    )
print('SPOD main calculation finished!'           )
print('Time lapsed: %.2f s'%(end_sec2-start_sec2) )
print('--------------------------------------'    )

    
# -------------------------------------------------------------------------
# 3. Read SPOD result
# -------------------------------------------------------------------------

# Sec. 3 start time
start_sec3 = time.time()

# load data from h5 format
SPOD_LPf  = h5py.File(os.path.join(save_path,'SPOD_LPf.h5'),'r')
L = SPOD_LPf['L'][:,:]    # modal energy E(f, M)
P = SPOD_LPf['P'][:,:,:]  # mode shape
f = SPOD_LPf['f'][:]      # frequency
SPOD_LPf.close()

# Sec. 3 end time
end_sec3 = time.time()

print('--------------------------------------'    )
print('SPOD results read in!'                     )
print('Time lapsed: %.2f s'%(end_sec3-start_sec3) )
print('--------------------------------------'    )


# -------------------------------------------------------------------------
# 4. Plot SPOD result
# Figs: 1. f-mode energy; 
#       2. mode shape at given mode number and frequency
#       3. animation of original flow field
#       4. animation of reconstructed flow field
# -------------------------------------------------------------------------

# Sec. 4 start time
start_sec4 = time.time()

# -------------------------------------------------------------------------
### 4.0 pre-defined function
params={
'axes.labelsize': '20',       
'xtick.labelsize': '16',
'ytick.labelsize': '16',
'lines.linewidth': 1.5,
'legend.fontsize': '14',
'figure.figsize': '8, 6'    # set figure size
}
pylab.rcParams.update(params)
    
def figure_format(xtitle, ytitle, zoom, legend):
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.axis(zoom)
    if legend != 'None':
        plt.legend(loc=legend)

def comp_contour(q, qlevels, qname, x, y, SS_idxs, PS_idxs, colormap=cm.coolwarm):
    '''
    Purpose: template for comp 2D contour plot
    '''
    
    fig, axs = plt.subplots(1,2,figsize=(8, 6))
    
    # suction surface
    cntr1 = axs[0].tricontourf(x[SS_idxs[0]:SS_idxs[1]],y[SS_idxs[0]:SS_idxs[1]],
               q[SS_idxs[0]:SS_idxs[1]], qlevels,cmap=colormap,extend='both')

    # suction surface
    cntr2 = axs[1].tricontourf(x[PS_idxs[0]:PS_idxs[1]],y[PS_idxs[0]:PS_idxs[1]],
               q[PS_idxs[0]:PS_idxs[1]], qlevels,cmap=colormap,extend='both')

    # colorbar
    fig.colorbar(cntr1,ax=axs[0],ticks=np.linspace(qlevels[0],qlevels[-1],3),shrink=0.8,extendfrac='auto',\
                 orientation='horizontal', pad=0.15, label=qname)
    fig.colorbar(cntr2,ax=axs[1],ticks=np.linspace(qlevels[0],qlevels[-1],3),shrink=0.8,extendfrac='auto',\
                 orientation='horizontal', pad=0.15, label=qname)
    
    # figure format
    fig.subplots_adjust(wspace=0.4)
    axs[0].text(0.305,0.305,'SS',fontsize=16)
    axs[1].text(0.305,0.305,'PS',fontsize=16)
    for jcol in range(2):
        # hub wall
        axs[jcol].plot([0.3,0.5],[0.3,0.3],color='black',linewidth=1)
        axs[jcol].fill_between([0.3,0.5], 0.3, 0, facecolor='whitesmoke')
        
        # casing wall
        axs[jcol].plot([0.3,0.5],[0.5,0.5],color='black',linewidth=1)        
        axs[jcol].fill_between([0.3,0.5], 0.5, 1, facecolor='whitesmoke')
        
        # blade
        axs[jcol].plot([0.34623,0.352893,0.468616,0.475381],
           [0.3,0.4965,0.4965,0.3],color='black',linewidth=1)
        
        # axis
        axs[jcol].set_xlabel('x (m)')
        axs[jcol].set_ylabel('r (m)')        
        axs[jcol].axis([0.3,0.5,0.29,0.51])
        axs[jcol].set_xticks(np.linspace(0.3,0.5,3))
        axs[jcol].set_yticks(np.linspace(0.3,0.5,3))
        axs[jcol].tick_params(direction='in')
    
    return fig

def comp_contour_anim(t_start, t_end, t_delta, dt, ani_save_name, q, qlevels, 
                       qname, x, y, SS_idxs, PS_idxs, colormap=cm.coolwarm):
    '''
    Purpose: plot and save animation of comp 2D contour plot
    '''
    with imageio.get_writer(os.path.join(save_path,ani_save_name), mode='I') as writer:
        # loop over snapshots
        for ti in range(t_start,t_end,t_delta):
            fig = comp_contour(q[ti,:], qlevels, qname, x, y, SS_idxs, PS_idxs)
            fig.suptitle(r't $\cdot$ BPF = %.2f'%(ti*dt*BPF),fontsize=16,y=0.93)  
            
            # convert Matplotib figure to png file
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=50)
            buf.seek(0)
            
            # read png in and plot gif
            image = imageio.imread(buf)
            writer.append_data(image)
            
            # release RAM
            plt.close(fig)
        
    return

# -------------------------------------------------------------------------
### 4.1 Energy spectrum
fig = spod.plot_spectrum(f/BPF,L,hl_idx=5)

# figure format
figure_format(xtitle='f / BPF', ytitle='SPOD mode energy', 
              zoom=[10**-1, 3*10**1, 10**0, 10**8], legend='best')

if save_fig:
    plt.savefig(os.path.join(save_path,'Spectrum.png'), dpi=300, bbox_inches='tight')
    plt.close()

print('Plot spectrum finished')

# -------------------------------------------------------------------------
### 4.2 Mode shape
plot_modes = [[0,5],
              [0,25]] # [[M1,f1],[M2,f2],...] to be plotted

# plot mode shape contour
for i in range(len(plot_modes)):
    Mi = plot_modes[i][0]
    fi = plot_modes[i][1]

    fig = plt.figure(figsize=(4,6))
    fig = comp_contour(np.real(P[fi,:,Mi]), np.arange(-0.1,0.11,0.01), r'$\phi_p$',
                        grid[:,0], grid[:,1], SS_idxs, PS_idxs)
    fig.suptitle('Mode '+str(Mi+1)+r', f / BPF = %.2f'%(f[fi]/BPF),fontsize=16,y=0.93)

    if save_fig:
        fig.savefig(os.path.join(save_path,'M'+str(Mi)+'f'+str(fi)+
                                 '_p_mode_shape.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

print('Plot mode shape finished')

# -------------------------------------------------------------------------
### 4.3 Original flow field
data_mean     = np.mean(data,axis=0) # time-averaged data

# plot snapshot flow field
plot_snapshot = [0,10] # [t1,t2,...] to be plotted

for i in range(len(plot_snapshot)):
    ti = plot_snapshot[i]

    fig = plt.figure(figsize=(6,4))
    fig = comp_contour(data[ti,:]-data_mean, np.arange(-500,550,50),
                        r'$p-\bar{p} $ (Pa)', grid[:,0], grid[:,1], SS_idxs, PS_idxs)
    fig.suptitle(r't $\cdot$ BPF = %.2f'%(ti*dt*BPF),fontsize=16,y=0.93)
    
    if save_fig:
        fig.savefig(os.path.join(save_path,'t'+str(ti)+'_p.png'), dpi=300, bbox_inches='tight')
        plt.close()

# plot animation of flow field
t_start = 0
t_end   = 240
t_delta = 2

if save_fig:
    comp_contour_anim(t_start, t_end, t_delta, dt, 'ori_p_anim.gif',
                       data-data_mean, np.arange(-500,550,50),
                       r'$p-\bar{p} $ (Pa)', grid[:,0], grid[:,1],
                       SS_idxs, PS_idxs, colormap=cm.coolwarm)

print('Plot original flow field finished')

# -------------------------------------------------------------------------
### 4.4 Reconstructed flow field
# time series to be reconstructed
t_start = 0
t_end   = 240
t_delta = 2

# modes and frequencies used for reconstruction
Ms = np.arange(0,L.shape[1])
fs = np.arange(0,f.shape[0])

#Ms = np.array([0]) # for nozzle pressure wave spike
#fs = np.array([5])

# plot animation of reconstructed flow field
if save_fig:
    # reconstruction function
    data_rec = spod.reconstruct_time_method(data-data_mean,dt,f,P,Ms,fs,weight=weight)
    comp_contour_anim(t_start, t_end, t_delta, dt, ani_save_name='rec_p_anim.gif',
                      q=data_rec, qlevels=np.arange(-500,550,50),
                      qname=r'$p-\bar{p} $ (Pa)', x=grid[:,0], y=grid[:,1],
                      SS_idxs=SS_idxs, PS_idxs=PS_idxs, colormap=cm.coolwarm)

print('Plot reconstructed flow field finished')

# Sec. 4 end time
end_sec4 = time.time()

print('--------------------------------------'    )
print('SPOD results postprocessed!'               )
print('Figs saved to the directory:'              )
print( save_path                                  )
print('Time lapsed: %.2f s'%(end_sec4-start_sec4) )
print('--------------------------------------'    )


# -------------------------------------------------------------------------
# -1. print memory usage
# -------------------------------------------------------------------------
process   = psutil.Process(os.getpid())
RAM_usage = np.around(process.memory_info().rss/1024**3, decimals=2) # unit in GBs
print('Total memory usage is: %.2f GB'%RAM_usage)

# End