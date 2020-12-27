"""
SPOD Python toolkit
The script was originally written for analyzing compressor tip leakage flows:
    
  He, X., Fang, Z., Vahdati, M. & Rigas, G., (2021). Spectral Proper Orthogonal 
  Decomposition of Compressor Tip Leakage Flow. ETC Paper No. ETC2021-491.
  
An explic reference to the work above is highly appreciated if this script is
useful for your research.  

Zhou Fang, Xiao He (xiao.he2014@imperial.ac.uk)
Last update: 27-Dec-2020
"""

# -------------------------------------------------------------------------
# import libraries
import numpy as np
from scipy.fftpack import fft
import time
import os
import psutil
import h5py
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# main function
def spod(x,dt,weight='default',nOvlp='default',nDFT='default',window='default'):
    '''
    Purpose: main function of spectral proper orthogonal decomposition
        
    Parameters
    ----------
    x      : 2D numpy array, float; space-time flow field data.
             nrow = number of time snapshots;
             ncol = number of grid point * number of variable
    dt     : float; time step between adjacent snapshots
    weight : 1D numpy array; weight function (default unity)
             n = number of grid point * number of variable
    nOvlp  : int; number of overlap (default 50% nDFT)
    nDFT   : int; number of DFT points (default about 10% of number of time snapshots)    
    window : 1D numpy array, float; window function values (default Henning)
             n = nDFT (default nDFT calculated from number of time snapshots)
        
    Returns
    -------
    L : 2D numpy array, float; modal energy E(f, M)
        nrow = number of frequencies
        ncol = number of modes (ranked in descending order by modal energy)
    P : 3D numpy array, complex; mode shape
        P.shape[0] = number of frequencies;
        P.shape[1] = number of grid point * number of variable
        P.shape[2] = number of modes (ranked in descending order by modal energy)
    f : 1D numpy array, float; frequency
    '''
    
    time_start=time.time()

    print('--------------------------------------')  
    print('SPOD starts...'                        )
    print('--------------------------------------')    
    
    # ---------------------------------------------------------
    # 1. calculate SPOD parameters
    # ---------------------------------------------------------
    nt = np.shape(x)[0]
    nx = np.shape(x)[1]
     
    # SPOD parser
    [weight, window, nOvlp, nDFT, nBlks] = spod_parser(nt, nx, window, weight, nOvlp, nDFT)


    #---------------------------------------------------------
    # 2. loop over number of blocks and generate Fourier 
    #    realizations (DFT)
    #---------------------------------------------------------
    print('Calculating temporal DFT'              )
    print('--------------------------------------')
 
    # calculate time-averaged result
    x_mean  = np.mean(x,0)

    # obtain frequency axis
    f     = np.arange(0,int(np.ceil(nDFT/2)+1))
    f     = f/dt/nDFT
    nFreq = f.shape[0]   
    
    # initialize whole domain result 
    Q_hat = np.zeros((nx,nFreq,nBlks),dtype = complex) #!!! RAM demanding here
    Q_blk = np.zeros((nDFT,nx))
    
    # loop over each block
    for iBlk in range(nBlks):
        
        # get time index for present block
        it_end   = min(iBlk*(nDFT-nOvlp)+nDFT,nt)
        it_start = it_end-nDFT
                    
        print('block', iBlk+1, '/', nBlks, '(', it_start+1, ':', it_end, ')')
        
        # subtract time-averaged results from the block
        Q_blk = x[it_start:it_end,:] - x_mean # column-wise broadcasting
        
        # add window function to block
        Q_blk = Q_blk.T * window # row-wise broadcasting
        
        # Fourier transform on block
        Q_blk_hat = 1/np.mean(window)/nDFT*fft(Q_blk)
        Q_blk_hat = Q_blk_hat[:,0:nFreq] 
        
        # correct Fourier coefficients for one-sided spectrum
        Q_blk_hat[:,1:(nFreq-1)] *= 2
        
        # save block result to the whole domain result 
        Q_hat[:,:,iBlk] = Q_blk_hat
    
    # remove vars to release RAM
    del x, Q_blk
        
    #---------------------------------------------------------
    # 3. loop over all frequencies and calculate SPOD
    #---------------------------------------------------------
    print('--------------------------------------')
    print('Calculating SPOD'                      )
    print('--------------------------------------')
    
    # initialize output vars
    L = np.zeros((nFreq,nBlks))
    P = np.zeros((nFreq,nx,nBlks),dtype = complex) #!!! RAM demanding here
    
    # loop over each frequency
    for iFreq in range(nFreq):
        print('Frequency',iFreq+1,'/',nFreq,'(f = %.3f'%f[iFreq],')')
        
        Q_hat_f = Q_hat[:,iFreq,:]
        
        M = np.dot(np.conjugate(Q_hat_f).T * weight, Q_hat_f)/nBlks
        
        # solve eigenvalue problem
        [Lamda, Theta] = np.linalg.eig(M)
        
        # sort modes by eigenvalues
        sort_idx = np.argsort(-Lamda)
        Lamda = Lamda[sort_idx]
        Theta = Theta[:,sort_idx]

        # save result to the output
        P[iFreq,:,:] = np.dot(np.dot(Q_hat_f,Theta),np.diag(1./np.sqrt(Lamda)/np.sqrt(nBlks)))
        L[iFreq,:] = abs(np.array(Lamda))

    # memory usage
    process   = psutil.Process(os.getpid())
    RAM_usage = process.memory_info().rss/1024**3 # unit in GBs
        
    time_end=time.time()
    print('--------------------------------------'     )
    print('SPOD finished'                              )
    print('Memory usage: %.2f GB'%RAM_usage            )
    print('Run time    : %.2f s'%(time_end-time_start) )
    print('--------------------------------------'     )
            
    return L,P,f


# -------------------------------------------------------------------------
# sub-functions
def spod_parser(nt, nx, window, weight, nOvlp, nDFT):
    '''
    Purpose: determine data structure/shape for SPOD
        
    Parameters
    ----------
    nt     : int; number of time snapshots
    nx     : int; number of grid point * number of variable
    window : expect 1D numpy array, float; specified window function values
    weight : expect 1D numpy array; specified weight function
    nOvlp  : expect int; specified number of overlap
    nDFT   : expect int; specified number of DFT points (expect to be same as weight.shape[0])

    
    Returns
    -------
    weight : 1D numpy array; calculated/specified weight function
    window : 1D numpy array, float; window function values
    nOvlp  : int; calculated/specified number of overlap
    nDFT   : int; calculated/specified number of DFT points
    nBlks  : int; calculated/specified number of blocks
    '''
    
    # check specified weight function value
    try:
        # user specified weight
        nweight = weight.shape[0]
        if nweight != nx:
            print('WARNING: weight does not match with x')
            raise ValueError
        else:
            wgt_name = 'user specified'
            print('Using user specified weight...')
            
    except:        
        # default weight
        weight   = np.ones(nx)
        wgt_name = 'unity'
        print('Using default weight...')
                
    
    # calculate or specify window function value
    try:
        # user sepcified window
        nWinLen  = window.shape[0]
        win_name = 'user specified'
        nDFT     = nWinLen # use window shape to over-write nDFT (if specified)
        print('Using user specified nDFT from window length...')         
        print('Using user specified window function...')  
        
    except:
        # default window with specified/default nDFT
        try:
            # user specified nDFT
            nDFT  = np.int(nDFT)
            nDFT  = np.int(2**(np.floor(np.log2(nDFT)))) # round-up to 2**n type int
            print('Using user specified nDFT ...')             
                
        except:
            # default nDFT
            nDFT  = 2**(np.floor(np.log2(nt/10))) #!!! why /10 is recommended?
            nDFT  = np.int(nDFT)
            print('Using default nDFT...')
            
        window   = hammwin(nDFT)
        win_name = 'Hamming'
        print('Using default Hamming window...') 

    # calculate or specify nOvlp
    try:
        # user specified nOvlp
        nOvlp = np.int(nOvlp)
        
        # test feasibility
        if nOvlp > nDFT-1:
            print('WARNING: nOvlp too large')
            raise ValueError
        else:
            print('Using user specified nOvlp...')
            
    except:            
        # default nOvlp
        nOvlp = np.int(np.floor(nDFT/2))
        print('Using default nOvlp...')


    # calculate nBlks from nOvlp and nDFT    
    nBlks = np.int(np.floor((nt-nOvlp)/(nDFT-nOvlp)))
 
    # test feasibility
    if (nDFT < 4) or (nBlks < 2):
        raise ValueError('User sepcified window and nOvlp leads to wrong nDFT and nBlk.')
        

    print('--------------------------------------')
    print('SPOD parameters summary:'              )
    print('--------------------------------------')
    print('number of DFT points :', nDFT          )
    print('number of blocks is  :', nBlks         )
    print('number of overlap is :', nOvlp         )
    print('Window function      :', win_name      )
    print('Weight function      :', wgt_name      )
    
    return weight, window, nOvlp, nDFT, nBlks

def hammwin(N):
    '''
    Purpose: standard Hamming window
    
    Parameters
    ----------
    N : int; window lengh

    Returns
    -------
    window : 1D numpy array; containing window function values
             n = nDFT
    '''
    
    window = np.arange(0, N)
    window = 0.54 - 0.46*np.cos(2*np.pi*window/(N-1))
    window = np.array(window)

    return window

def save_results(save_path,L,P,f,filetype='HDF5'):
    '''
    Purpose: save SPOD results

    Parameters
    ----------
    save_path : str; path to save the data
    L,P,f     : numpy arrays; outputs from SPOD   
    '''
    
    print('--------------------------------------')
    print('Saving SPOD results...'                )
    
    # start saving
    if filetype == 'npy':
        np.save(os.path.join(save_path,'L.npy'), L)
        np.save(os.path.join(save_path,'P.npy'), P)
        np.save(os.path.join(save_path,'f.npy'), f)
    elif filetype == 'HDF5':
        h5f = h5py.File(os.path.join(save_path,'SPOD_LPf.h5'), 'w')
        h5f.create_dataset('L', data=L, compression="gzip")
        h5f.create_dataset('P', data=P, compression="gzip")
        h5f.create_dataset('f', data=f, compression="gzip")
        h5f.close()
    
    print('SPOD results saved to '+filetype+' file' )
    print('--------------------------------------'  )
    return

def reconstruct(f,L,P,Ms,fs,ts):
    '''
    Purpose: reconstruct flow field using SPOD mode shapes
    
    Parameters
    ----------
    f  : 1D numpy array, float; frequency; output of SPOD main function
    L  : 2D numpy array, float; modal energy E(f, M); output of SPOD main function
         nrow = number of frequencies
         ncol = number of modes (ranked in descending order by modal energy)
    P  : 3D numpy array, complex; mode shape; output of SPOD main function
         shape[0] = number of frequencies;
         shape[1] = number of grid point * number of variable
         shape[2] = number of modes(ranked in descending order by modal energy)              
    Ms : 1D numpy array, int; index of modes used for reconstruction
    fs : 1D numpy array, int; index of frequencies used for reconstruction
    ts : 1D numpy array, float; time series at which flow field is reconstructed

    Returns
    -------
    data_recons : 1D numpy array; reconstructed flow field data
    '''

    # initialize reconstructed flow field data
    data_rec = np.zeros((ts.shape[0],np.shape(P)[1]))
        
    # calculate reconstruction    
    for i in range(ts.shape[0]):
        ti = ts[i]
        for fi in fs:
            for Mi in Ms:
                data_rec[i,:] += np.real(np.sqrt(L[fi,Mi])*P[fi,:,Mi]*np.exp(2j*np.pi*f[fi]*ti))
                
    return data_rec

def plot_spectrum(f,L,hl_idx=5):
    '''
    Purpose: plot SPOD energy spectrum
    
    Parameters
    ----------
    f  : 1D numpy array, float; frequency; output of SPOD main function
    L  : 2D numpy array, float; modal energy E(f, M); output of SPOD main function
         nrow = number of frequencies
         ncol = number of modes (ranked in descending order by modal energy)
    hl_idx : int; max index of mode to be plotted in color
         
    Returns
    -------
    fig : matplotlib figure object
    '''

    fig = plt.figure()
    
    # loop over each mode
    for imode in range(L.shape[1]):
        if imode < hl_idx:  # highlight modes with colors
            plt.loglog(f[0:-1],L[0:-1,imode],label='Mode '+str(imode+1)) # truncate last frequency
        elif imode == L.shape[1]-1:
            plt.loglog(f[0:-1],L[0:-1,imode],color='lightgrey',label='Others')
        else:
            plt.loglog(f[0:-1],L[0:-1,imode],color='lightgrey',label='')
    
    # figure format
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SPOD mode energy')
    plt.legend(loc='best')
    
    return fig

# End