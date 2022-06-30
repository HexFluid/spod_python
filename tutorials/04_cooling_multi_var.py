"""
Application of SPOD to film cooling 2D slice data by DDES.
Details of the data can be found in the following:
    
  Wang, R., & Yan, X. (2021). Delayed-Detached Eddy Simulations of 
  Film Cooling Effect On Trailing Edge Cutback with Land Extensions. 
  ASME Journal of Engineering for Gas Turbines and Power, 143(11), 111004.

Xiao He (xiao.he2014@imperial.ac.uk), Ruiqin Wang
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
#             ncol = 3 (e.g., x/H, y/H, control volume size/H**3; H is cooling path width)
# In this case, nt = 600, ngrid = 3647, and nvar = 4 (e.g., rho,u,v,T)
# -------------------------------------------------------------------------

# Sec. 1 start time
start_sec1 = time.time()

# data path
data_path = os.path.join(current_path, 'cooling_data')

# option to save SPOD results
save_fig  = True  # postprocess figs
save_path = data_path

# load data from h5 format
h5f  = h5py.File(os.path.join(data_path,'coolingDDES.h5'),'r')
data = h5f['data'][:]        # flow fields
grid = h5f['grid'][:]        # grid points
dt   = h5f['dt'][0]          # unit in seconds
ng   = int(grid.shape[0]) # number of grid point
nt   = data.shape[0]         # number of snap shot
nx   = data.shape[1]         # number of grid point * number of variable
nvar = int(nx/ng)         # number of variables
h5f.close()

# calculate grid weight
weight_grid = grid[:,2]                         # control volume weighted
weight_grid = weight_grid/np.mean(weight_grid)  # normalized by mean values for better scaling

# multivar weights based on total energy of disturbance (Chu, B., 1965, Acta Mechanica)
# air constants
R     = 287
gamma = 1.4

# density mean
rho_data = data[:,0:ng]
rho_mean = np.mean(rho_data, axis=0)
del rho_data # release RAM

# temperature mean
T_data = data[:,(3*ng):(4*ng)]
T_mean = np.mean(T_data, axis=0)
del T_data # release RAM

# weights for vars
weight_rho = R*T_mean/rho_mean
weight_u   = rho_mean
weight_T   = R*rho_mean/T_mean/(gamma-1)

# concatenate weights
weight = np.concatenate((weight_rho*weight_grid,
                         weight_u*weight_grid,
                         weight_u*weight_grid,
                         weight_T*weight_grid),axis=0)

# Sec. 1 end time
end_sec1 = time.time()

print('--------------------------------------'    )
print('SPOD input data summary:'                  )
print('--------------------------------------'    )
print('number of snapshot   :', nt                )
print('number of grid point :', ng                )
print('number of variable   :', nvar              )
print('dt   :', dt                                )
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
spod.spod(data, dt, save_path, weight, nOvlp=64,nDFT=128, method='lowRAM')

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

def cooling_contour(q, qlevels, qname, x, y, colormap=cm.coolwarm):
    '''
    Purpose: template for film cooling 2D contour plot !!!
    '''
    
    cntr = plt.tricontourf(x,y,q, qlevels,cmap=colormap,extend='both')

    # colorbar
    plt.colorbar(cntr,ticks=np.linspace(qlevels[0],qlevels[-1],3),shrink=0.8,extendfrac='auto',\
                 orientation='horizontal', pad=0.25, label=qname)
        
    # wall boundary
    plt.fill_between([-2,15,20], [-0.5,2.47,2.47], -1, facecolor='whitesmoke')
    plt.plot([-2,15,20], [-0.5,2.47,2.47],color='black',linewidth=1)
    plt.fill_between([-2,0], [1.9,1.9], [0.7,1.2], facecolor='lightgrey')
    plt.plot([-2,0,0,-2], [1.9,1.9,1.2,0.7],color='black',linewidth=1)
    
    # figure format
    figure_format('x/H','y/H', [-1,17,-1,4],'None')    
    
    return fig

def cooling_contour_anim(t_start, t_end, t_delta, dt, ani_save_name, q, qlevels, 
                       qname, x, y, colormap=cm.coolwarm):
    '''
    Purpose: plot and save animation of backward-facing step 2D contour plot !!!
    '''
    with imageio.get_writer(os.path.join(save_path,ani_save_name), mode='I') as writer:
        # loop over snapshots
        for ti in range(t_start,t_end,t_delta):
            plt.figure(figsize=(6,4))
            cooling_contour(q[ti,:], qlevels, qname, x, y)
            plt.text(12,-0.5,'t = %.4f s'%(ti*dt), fontsize=14)   
            
            # convert Matplotib figure to png file
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # read png in and plot gif
            image = imageio.imread(buf)
            writer.append_data(image)
            
            # release RAM
            plt.close()
        
    return

# -------------------------------------------------------------------------   
### 4.1 Energy spectrum
fig = spod.plot_spectrum(f,L,hl_idx=2)
        
# figure format
figure_format(xtitle='Frequency (Hz)', ytitle='SPOD mode energy', 
              zoom=[10**1, 10**4, 10**3, 4*10**6], legend='best')

if save_fig:
    plt.savefig(os.path.join(save_path,'Spectrum.png'), dpi=300, bbox_inches='tight')
    plt.close()

print('Plot spectrum finished')

# -------------------------------------------------------------------------
### 4.2 Mode shape
var_names  = ['rho','u','v','T']
var_levels = [10**(-4)*np.arange(-1,1.1,0.1),
              10**(-3)*np.arange(-5,5.5,0.5),
              10**(-3)*np.arange(-5,5.5,0.5),
              10**(-2)*np.arange(-3,3.3,0.3)]

plot_vars  = [0,1,2,3]   # vars to be plotted; 0-3: rho, u, v, T
plot_modes = [[0,22]] # [[M1,f1],[M2,f2],...] to be plotted

# loop over each var
for vari in plot_vars:
    var_name  = var_names[vari]
    var_level = var_levels[vari]
    # plot mode shape contour
    for i in range(len(plot_modes)):
        Mi = plot_modes[i][0]
        fi = plot_modes[i][1]
        
        fig = plt.figure(figsize=(6,4))
        fig = cooling_contour(np.real(P[fi, int(vari*ng):int((vari+1)*ng), Mi]), 
                            var_level, r'$\phi_{'+var_name+'}$', grid[:,0], grid[:,1])
        plt.text(7.5,-0.5,'Mode '+str(Mi+1)+', f = %.2f Hz'%(f[fi]), fontsize=14)
        
        if save_fig:
            plt.savefig(os.path.join(save_path,'M'+str(Mi)+'f'+str(fi)+'_'+
                                     var_name+'_mode_shape.png'), dpi=300,
                        bbox_inches='tight')
            plt.close()

print('Plot mode shape finished')

# -------------------------------------------------------------------------
### 4.3 Original flow field
var_names  = ['rho','u','v','T']
var_units  = ['kg/m^3','m/s','m/s','K']
var_levels = [np.arange(-0.3,0.33,0.03),
              np.arange(-15,16,1),
              np.arange(-15,16,1),
              np.arange(-100,110,10)]
plot_vars  = [0,1,2,3]   # vars to be plotted; 0-3: rho, u, v, T
data_mean  = np.mean(data,axis=0) # time-averaged data

# loop over each var
for vari in plot_vars:
    var_name  = var_names[vari]
    var_level = var_levels[vari]
    var_unit  = var_units[vari]
    
    # plot snapshot flow field
    plot_snapshot = [0,10] # [t1,t2,...] to be plotted
    
    for i in range(len(plot_snapshot)):
        ti = plot_snapshot[i]
        
        fig = plt.figure(figsize=(6,4))
        fig = cooling_contour(data[ti,int(vari*ng):int((vari+1)*ng)]-
                                 data_mean[int(vari*ng):int((vari+1)*ng)], var_level, 
                            '$'+str(var_name)+'$'+r'$-\overline{'+str(var_name)+'} $ ('+str(var_unit)+')', 
                            grid[:,0], grid[:,1])
        plt.text(12,-0.5,'t = %.4f s'%(ti*dt), fontsize=14)
        
        if save_fig:
            plt.savefig(os.path.join(save_path,'t'+str(ti)+'_'+str(var_name)+'.png'), dpi=300,
                        bbox_inches='tight')
            plt.close()

# plot animation of flow field
t_start = 0
t_end   = int(nt/10)
t_delta = 1

if save_fig:
    # loop over each var    
    for vari in plot_vars:
        var_name  = var_names[vari]
        var_level = var_levels[vari]
        var_unit  = var_units[vari]
    
        cooling_contour_anim(t_start, t_end, t_delta, dt, ani_save_name='ori_'+str(var_name)+'_anim.gif', 
                           q=(data-data_mean)[:,int(vari*ng):int((vari+1)*ng)],
                           qlevels=var_level, 
                           qname='$'+str(var_name)+'$'+r'$-\bar{'+str(var_name)+'} $ ('+str(var_unit)+')', 
                           x=grid[:,0], y=grid[:,1], colormap=cm.coolwarm)   

print('Plot original flow field finished')

# -------------------------------------------------------------------------
### 4.4 Reconstructed flow field
# time series to be reconstructed
t_start = 0
t_end   = int(nt/10)
t_delta = 1

# modes and frequencies used for reconstruction
Ms = np.arange(0,L.shape[1])
fs = np.arange(0,f.shape[0])

#Ms = np.array([0]) # for vortex shedding spike
#fs = np.array([22])

# reconstruction function
data_rec = spod.reconstruct_frequency_method(data-data_mean,f,P,Ms,fs,weight=weight,
                                             nOvlp=64,nDFT=128)
#data_rec = spod.reconstruct_direct(f,L,P,Ms,fs,dt*np.arange(t_start,t_end,t_delta)) # for vortex shedding spike

# plot animation of reconstructed flow field
if save_fig:
    # loop over each var    
    for vari in plot_vars:
        var_name  = var_names[vari]
        var_level = var_levels[vari]
        var_unit  = var_units[vari]
        
        cooling_contour_anim(t_start, t_end, t_delta, dt,
                           ani_save_name='rec_'+str(var_name)+'_anim.gif', 
                           q=data_rec[:,int(vari*ng):int((vari+1)*ng)], 
                           qlevels=var_level, 
                           qname='$'+str(var_name)+'$'+r'$-\bar{'+str(var_name)+'} $ ('+str(var_unit)+')', 
                           x=grid[:,0], y=grid[:,1], colormap=cm.coolwarm)       

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