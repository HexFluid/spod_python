"""
Application of SPOD to jet 2D slice data by LES.
Details of the data can be found in the following:

  G. A. Brès, P. Jordan, M. Le Rallic, V. Jaunet, A. V. G. Cavalieri, A. Towne,
  S. K. Lele, T. Colonius, O. T. Schmidt, Importance of the nozzle-exit boundary
  layer state in subsonic turbulent jets, J. of Fluid Mech. 851, 83-124, 2018

Zhou Fang, Xiao He (xiao.he2014@imperial.ac.uk)
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
#             ncol = 3 (e.g., x/D, y/D, control volume size/D**3)
# In this case, nt = 2500, ngrid = 6825, and nvar = 1 (e.g., p)
# -------------------------------------------------------------------------

# Sec. 1 start time
start_sec1 = time.time()

# data path
data_path = os.path.join(current_path, 'jet_data')

# option to save SPOD results
save_fig  = True  # postprocess figs
save_path = data_path

# load data from h5 format
h5f  = h5py.File(os.path.join(data_path,'jetLES.h5'),'r')
data = h5f['data'][:]        # flow fields
grid = h5f['grid'][:]        # grid points
dt   = h5f['dt'][0]          # unit in seconds
ng   = int(grid.shape[0]) # number of grid point
nt   = data.shape[0]         # number of snap shot
nx   = data.shape[1]         # number of grid point * number of variable
nvar = int(nx/ng)         # number of variables
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

def jet_contour(q, qlevels, qname, x, y, colormap=cm.coolwarm):
    '''
    Purpose: template for jet 2D contour plot
    '''

    cntr = plt.tricontourf(x,y,q, qlevels,cmap=colormap,extend='both')

    # colorbar
    plt.colorbar(cntr,ticks=np.linspace(qlevels[0],qlevels[-1],3),shrink=0.8,extendfrac='auto',\
                 orientation='horizontal', pad=0.25, label=qname)

    # figure format
    figure_format('x/D','r/D', [0,10,0,2],'None')

    return fig

def jet_contour_anim(t_start, t_end, t_delta, dt, ani_save_name, q, qlevels,
                       qname, x, y, colormap=cm.coolwarm):
    '''
    Purpose: plot and save animation of jet 2D contour plot
    '''
    with imageio.get_writer(os.path.join(save_path,ani_save_name), mode='I') as writer:
        # loop over snapshots
        for ti in range(t_start,t_end,t_delta):
            plt.figure(figsize=(6,4))
            jet_contour(q[ti,:], qlevels, qname, x, y)
            plt.text(7.5,1.7,'t = %.2f s'%(ti*dt), fontsize=14)

            # convert Matplotib figure to png file
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=50)
            buf.seek(0)

            # read png in and plot gif
            image = imageio.imread(buf)
            writer.append_data(image)

            # release RAM
            plt.close()

    return

# -------------------------------------------------------------------------
### 4.1 Energy spectrum
fig = spod.plot_spectrum(f,L,hl_idx=5)

# figure format
figure_format(xtitle='Frequency (Hz)', ytitle='SPOD mode energy',
              zoom=[10**-2, 3*10**0, 10**-7, 10**-1], legend='best')

if save_fig:
    plt.savefig(os.path.join(save_path,'Spectrum.png'), dpi=300, bbox_inches='tight')
    plt.close()

print('Plot spectrum finished')

# -------------------------------------------------------------------------
### 4.2 Mode shape
plot_modes = [[0,5],
              [3,5]] # [[M1,f1],[M2,f2],...] to be plotted

# plot mode shape contour
for i in range(len(plot_modes)):
    Mi = plot_modes[i][0]
    fi = plot_modes[i][1]

    fig = plt.figure(figsize=(6,4))
    fig = jet_contour(np.real(P[fi,:,Mi]), np.arange(-0.06,0.066,0.006), r'$\phi_p$',
                        grid[:,0], grid[:,1])
    plt.text(5.5,1.7,'Mode '+str(Mi+1)+', f = %.2f Hz'%(f[fi]), fontsize=14)

    if save_fig:
        plt.savefig(os.path.join(save_path,'M'+str(Mi)+'f'+str(fi)+
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
    fig = jet_contour(data[ti,:]-data_mean, np.arange(-0.04,0.044,0.004),
                        r'$p-\bar{p} $ (Pa)', grid[:,0], grid[:,1])
    plt.text(7.5,1.7,'t = %.2f s'%(ti*dt), fontsize=14)

    if save_fig:
        plt.savefig(os.path.join(save_path,'t'+str(ti)+'_p.png'), dpi=300, bbox_inches='tight')
        plt.close()

# plot animation of flow field
t_start = 0
t_end   = int(nt/10)
t_delta = 1

if save_fig:
    jet_contour_anim(t_start, t_end, t_delta, dt, ani_save_name='ori_p_anim.gif',
                       q=data-data_mean, qlevels=np.arange(-0.04,0.044,0.004),
                       qname=r'$p-\bar{p} $ (Pa)', x=grid[:,0], y=grid[:,1],
                       colormap=cm.coolwarm)

print('Plot original flow field finished')
#%%
# -------------------------------------------------------------------------
### 4.4 Reconstructed flow field
# time series to be reconstructed
t_start = 0
t_end   = int(nt/10)
t_delta = 1

# modes and frequencies used for reconstruction
Ms = np.arange(0,L.shape[1])
fs = np.arange(0,f.shape[0])

Ms = np.arange(0,3)
fs = np.arange(0,2)

# plot animation of reconstructed flow field
if save_fig:
    # reconstruction function
    data_rec = spod.reconstruct_time_method(data-data_mean,dt,f,P,Ms,fs,weight=weight,
                                            method='lowRAM',save_path=save_path)
    # jet_contour_anim(t_start, t_end, t_delta, dt,
    #                  ani_save_name='rec_p_anim.gif',
    #                  q=data_rec, qlevels=np.arange(-0.04,0.044,0.004),
    #                  qname=r'$p-\bar{p} $ (Pa)', x=grid[:,0], y=grid[:,1],
    #                  colormap=cm.coolwarm)

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