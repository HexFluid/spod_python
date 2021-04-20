"""
SPOD Python toolkit Ver 1.1

The script was originally written for analyzing compressor tip leakage flows:
    
  He, X., Fang, Z., Vahdati, M. & Rigas, G., (2021). Spectral Proper Orthogonal 
  Decomposition of Compressor Tip Leakage Flow. ETC Paper No. ETC2021-491.
  
An explict reference to the work above is highly appreciated if this script is
useful for your research.  

Xiao He (xiao.he2014@imperial.ac.uk), Zhou Fang
Last update: 15-April-2021
"""

# -------------------------------------------------------------------------
# import libraries
import numpy as np
from scipy.fft import fft
import time
import os
import psutil
import h5py
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# main function
def spod(x,dt,save_path,weight='default',nOvlp='default',nDFT='default',window='default',
         method='lowRAM'):
    '''
    Purpose: main function of spectral proper orthogonal decomposition
             (Towne, A., Schmidt, O., and Colonius, T., 2018, arXiv:1708.04393v2)
        
    Parameters
    ----------
    x         : 2D numpy array, float; space-time flow field data.
                nrow = number of time snapshots;
                ncol = number of grid point * number of variable
    dt        : float; time step between adjacent snapshots
    save_path : str; path to save the output data
        
    weight    : 1D numpy array; weight function (default unity)
                n = number of grid point * number of variable
    nOvlp     : int; number of overlap (default 50% nDFT)
    nDFT      : int; number of DFT points (default about 10% of number of time snapshots)    
    window    : 1D numpy array, float; window function values (default Henning)
                n = nDFT (default nDFT calculated from number of time snapshots)
    method    : string; 'fast' for fastest speed, 'lowRAM' for lowest RAM usage


    Return
    -------
    The SPOD results are written in the file '<save_path>/SPOD_LPf.h5'
    SPOD_LPf['L'] : 2D numpy array, float; modal energy E(f, M)
                    nrow = number of frequencies
                    ncol = number of modes (ranked in descending order by modal energy)
    SPOD_LPf['P'] : 3D numpy array, complex; mode shape
                    P.shape[0] = number of frequencies;
                    P.shape[1] = number of grid point * number of variable
                    P.shape[2] = number of modes (ranked in descending order by modal energy)
    SPOD_LPf['f'] : 1D numpy array, float; frequency
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
    [weight, window, nOvlp, nDFT, nBlks] = spod_parser(nt, nx, window, weight, nOvlp, nDFT, method)


    #---------------------------------------------------------
    # 2. loop over number of blocks and generate Fourier 
    #    realizations (DFT)
    #---------------------------------------------------------
    print('Calculating temporal DFT'              )
    print('--------------------------------------')
 
    # calculate time-averaged result
    x_mean  = np.mean(x,axis=0)

    # obtain frequency axis
    f     = np.arange(0,int(np.ceil(nDFT/2)+1))
    f     = f/dt/nDFT
    nFreq = f.shape[0]   
    
    # initialize all DFT result in frequency domain
    if method == 'fast':
        Q_hat = np.zeros((nx,nFreq,nBlks),dtype = complex) # RAM demanding here       
        
    elif method == 'lowRAM':
        Q_hat = h5py.File('Q_hat.h5', 'w')
        Q_hat.create_dataset('Q_hat', shape=(nx,nFreq,nBlks), chunks=True, dtype = complex, compression="gzip")
        
    # initialize block data in time domain
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
        if method == 'fast':
            Q_hat[:,:,iBlk] = Q_blk_hat
            
        elif method == 'lowRAM':
            Q_hat['Q_hat'][:,:,iBlk] = Q_blk_hat

    # remove vars to release RAM
    del x, Q_blk
        
    #---------------------------------------------------------
    # 3. loop over all frequencies and calculate SPOD
    #---------------------------------------------------------
    print('--------------------------------------')
    print('Calculating SPOD'                      )
    print('--------------------------------------')
    
    # initialize output vars
    if method == 'fast':
        L = np.zeros((nFreq,nBlks))
        P = np.zeros((nFreq,nx,nBlks),dtype = complex) # RAM demanding here
        
    elif method == 'lowRAM':
        h5f = h5py.File(os.path.join(save_path,'SPOD_LPf.h5'), 'w')
        h5f.create_dataset('L', shape=(nFreq,nBlks), compression="gzip")
        h5f.create_dataset('P', shape=(nFreq,nx,nBlks), chunks=True, dtype = complex, compression="gzip")
        h5f.create_dataset('f', data=f, compression="gzip")     
    
    # loop over each frequency
    for iFreq in range(nFreq):
        print('Frequency',iFreq+1,'/',nFreq,'(f = %.3f'%f[iFreq],')')
        
        if method == 'fast':
            Q_hat_f = Q_hat[:,iFreq,:]
        
        elif method == 'lowRAM':
            Q_hat_f = Q_hat['Q_hat'][:,iFreq,:]
        
        M = np.dot(np.conjugate(Q_hat_f).T * weight, Q_hat_f)/nBlks
        
        # solve eigenvalue problem
        [Lambda, Theta] = np.linalg.eig(M)
        
        # sort modes by eigenvalues
        sort_idx = np.argsort(-Lambda)
        Lambda = Lambda[sort_idx]
        Theta = Theta[:,sort_idx]

        # save result to the output
        if method == 'fast':
            P[iFreq,:,:] = np.dot(np.dot(Q_hat_f,Theta),np.diag(1./np.sqrt(Lambda)/np.sqrt(nBlks)))
            L[iFreq,:] = abs(np.array(Lambda))
        
        elif method == 'lowRAM':
            h5f['P'][iFreq,:,:] = np.dot(np.dot(Q_hat_f,Theta),np.diag(1./np.sqrt(Lambda)/np.sqrt(nBlks)))
            h5f['L'][iFreq,:] = abs(np.array(Lambda))
    
    # save results to file
    print('--------------------------------------')
    print('Saving SPOD results...'                )
    
    if method == 'fast':
        h5f = h5py.File(os.path.join(save_path,'SPOD_LPf.h5'), 'w')
        h5f.create_dataset('L', data=L, compression="gzip")
        h5f.create_dataset('P', data=P, compression="gzip")
        h5f.create_dataset('f', data=f, compression="gzip")
        h5f.close()
    
    elif method == 'lowRAM':
        h5f.close()
        Q_hat.close()

    print('SPOD results saved to HDF5 file' )
    print('--------------------------------------'  )
                
    # memory usage
    process   = psutil.Process(os.getpid())
    RAM_usage = process.memory_info().rss/1024**3 # unit in GBs
        
    time_end=time.time()
    print('--------------------------------------'     )
    print('SPOD finished'                              )
    print('Memory usage: %.2f GB'%RAM_usage            )
    print('Run time    : %.2f s'%(time_end-time_start) )
    print('--------------------------------------'     )
    return


# -------------------------------------------------------------------------
# sub-functions
def spod_parser(nt, nx, window, weight, nOvlp, nDFT, method):
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
    method : expect string; specified running mode of SPOD
    
    Returns
    -------
    weight : 1D numpy array; calculated/specified weight function
    window : 1D numpy array, float; window function values
    nOvlp  : int; calculated/specified number of overlap
    nDFT   : int; calculated/specified number of DFT points
    nBlks  : int; calculated/specified number of blocks
    '''

    # check SPOD running method
    try:
        # user specified method
        if method not in ['fast', 'lowRAM']:
            print('WARNING: user specified method not supported')
            raise ValueError
        else:
            print('Using user specified method...')
            
    except:        
        # default method
        method = 'lowRAM'
        print('Using default low RAM method...')
    
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
    print('Running method       :', method        )
    
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

def reconstruct(f,L,P,Ms,fs,ts):
    '''
    Purpose: reconstruct flow field using SPOD mode shapes
             !!!(this form needs debug - initial phase is not recovered by the 
                 imaginary part of P)
    
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
    data_rec : 1D numpy array; reconstructed flow field data
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

def reconstruct_time_method(x,dt,f,L,P,Ms,fs,weight='default'):
    '''
    Purpose: reconstruct flow field using the time domain method 
             (Nekkanti, A. and Schmidt, O., 2020, arXiv:2011.03644v1)
    
    Parameters
    ----------
    x  : 2D numpy array, float; space-time flow field data (mean=0).
         nrow = number of time snapshots;
         ncol = number of grid point * number of variable
    dt : float; time step between adjacent snapshots
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
    weight : 1D numpy array; weight function (default unity)
             n = number of grid point * number of variable
             
    Returns
    -------
    data_rec : 1D numpy array; reconstructed flow field data
    '''

    # initialize reconstructed flow field data
    data_rec = np.zeros((x.shape[0],x.shape[1]))

    # get weight
    nx = np.shape(x)[1]
    try:
        # user specified weight
        nweight = weight.shape[0]
        if nweight != nx:
            print('WARNING: weight does not match with x')
            raise ValueError
        else:
            print('Using user specified weight for reconstruction...')
    except:        
        # default weight
        weight   = np.ones(nx)
        print('Using default weight for reconstruction...')

    # calculate phi_tilda
    phi_tilda = np.zeros((P.shape[1],Ms.shape[0]*fs.shape[0]), dtype=complex)
    for i in range(P.shape[1]):
        phi_tilda[i,:] = np.ravel(P[:,i,:][fs,:][:,Ms], order='C')
    
    # calculate approximated phi_inv
    [D, U] = np.linalg.eig(np.dot(np.conjugate(phi_tilda).T * weight, phi_tilda)) # eigen-decomposition
    D_inv = np.identity(D.shape[0], dtype=complex)
    eps=1e-2 # limiter: truncated eigenvalue / max eigenvalue
    for i in range(D.shape[0]):
        if np.real(D[i]) < eps*np.max(np.real(D)):
            D_inv[i,i] = 0
        else:
            D_inv[i,i] = 1/D[i]
    phi_inv = np.dot(np.dot(U, D_inv), np.conjugate(U).T)

    # calculate A_tilda
    A_tilda = np.dot(np.dot(phi_inv, np.conjugate(phi_tilda).T)*weight, x.T)
   
    # calculate data_rec
    data_rec = np.real(np.dot(phi_tilda, A_tilda).T)
                
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