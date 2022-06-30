"""
SPOD Python toolkit Ver 1.2

The script was originally written for analyzing compressor tip leakage flows:
    
  He, X., Fang, Z., Vahdati, M. & Rigas, G., (2021). Spectral Proper Orthogonal 
  Decomposition of Compressor Tip Leakage Flow. Physics of Fluids, 33(10).
  
An explict reference to the work above is highly appreciated if this script is
useful for your research.  

Xiao He (xiao.he2014@imperial.ac.uk), Zhou Fang
Last update: 24-Sep-2021
"""

# -------------------------------------------------------------------------
# import libraries
import numpy as np
from scipy.fft import fft, ifft
from scipy.special import gammaincinv
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
    print('--------------------------------------')
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
        Q_hat = h5py.File(os.path.join(save_path,'Q_hat.h5'), 'w')
        Q_hat.create_dataset('Q_hat', shape=(nx,nFreq,nBlks), chunks=True, dtype = complex, compression="gzip")
        
    # initialize block data in time domain
    Q_blk = np.zeros((nDFT,nx))
    Q_blk_hat = np.zeros((nx,nFreq),dtype = complex)
    
    # loop over each block
    for iBlk in range(nBlks):
        
        # get time index for present block
        it_end   = min(iBlk*(nDFT-nOvlp)+nDFT, nt)
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
    del x, Q_blk, Q_blk_hat
        
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
            L[iFreq,:] = np.abs(np.array(Lambda))
        
        elif method == 'lowRAM':
            h5f['P'][iFreq,:,:] = np.dot(np.dot(Q_hat_f,Theta),np.diag(1./np.sqrt(Lambda)/np.sqrt(nBlks)))
            h5f['L'][iFreq,:] = np.abs(np.array(Lambda))
    
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
            nDFT  = int(nDFT)
            nDFT  = int(2**(np.floor(np.log2(nDFT)))) # round-up to 2**n type int
            print('Using user specified nDFT ...')             
                
        except:
            # default nDFT
            nDFT  = 2**(np.floor(np.log2(nt/10))) #!!! why /10 is recommended?
            nDFT  = int(nDFT)
            print('Using default nDFT...')
            
        window   = hammwin(nDFT)
        win_name = 'Hamming'
        print('Using default Hamming window...') 

    # calculate or specify nOvlp
    try:
        # user specified nOvlp
        nOvlp = int(nOvlp)
        
        # test feasibility
        if nOvlp > nDFT-1:
            print('WARNING: nOvlp too large')
            raise ValueError
        else:
            print('Using user specified nOvlp...')
            
    except:            
        # default nOvlp
        nOvlp = int(np.floor(nDFT/2))
        print('Using default nOvlp...')


    # calculate nBlks from nOvlp and nDFT    
    nBlks = int(np.floor((nt-nOvlp)/(nDFT-nOvlp)))
 
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

def reconstruct_direct(f,L,P,Ms,fs,ts):
    '''
    Purpose: reconstruct flow field using SPOD mode shapes
             (WARNING: this form of reconstruction only works for one mode at a time, because the phase
             difference is not recovered by the imaginary part of P)
    
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

# def reconstruct_time_method(x,dt,f,P,Ms,fs,weight='default'): # backup
#     '''
#     Purpose: reconstruct flow field using the time domain method 
#              (Nekkanti, A. and Schmidt, O., 2021, JFM, 926)
    
#     Parameters
#     ----------
#     x  : 2D numpy array, float; space-time flow field data (mean=0).
#          nrow = number of time snapshots;
#          ncol = number of grid point * number of variable
#     dt : float; time step between adjacent snapshots
#     f  : 1D numpy array, float; frequency; output of SPOD main function
#     P  : 3D numpy array, complex; mode shape; output of SPOD main function
#          shape[0] = number of frequencies;
#          shape[1] = number of grid point * number of variable
#          shape[2] = number of modes(ranked in descending order by modal energy)              
#     Ms : 1D numpy array, int; index of modes used for reconstruction
#     fs : 1D numpy array, int; index of frequencies used for reconstruction
#     weight : 1D numpy array; weight function (default unity)
#              n = number of grid point * number of variable
             
#     Returns
#     -------
#     data_rec : 1D numpy array; reconstructed flow field data
#     '''

#     # initialize reconstructed flow field data
#     nt = x.shape[0]
#     nx = x.shape[1]
#     data_rec = np.zeros((nt,nx))

#     # SPOD parser
#     [weight, _, _, _, _] = spod_parser(nt, nx, window='default', \
#              weight=weight, nOvlp='default',nDFT='default', method='lowRAM')

#     # calculate phi_tilda
#     phi_tilda = np.zeros((nx,Ms.shape[0]*fs.shape[0]), dtype=complex)   
#     for i in range(nx):
#         phi_tilda[i,:] = np.ravel(P[:,i,:][fs,:][:,Ms], order='C')
    
#     # calculate approximated phi_inv
#     [D, U] = np.linalg.eig(np.dot(np.conjugate(phi_tilda).T * weight, phi_tilda)) # eigen-decomposition
#     D_inv = np.identity(D.shape[0], dtype=complex)
#     eps=1e-2 # limiter: truncated eigenvalue / max eigenvalue
#     for i in range(D.shape[0]):
#         if np.abs(D[i]) < eps*np.max(np.abs(D)):
#             D_inv[i,i] = 0
#         else:
#             D_inv[i,i] = 1/D[i]
#     phi_inv = np.dot(np.dot(U, D_inv), np.conjugate(U).T)

#     # calculate A_tilda
#     A_tilda = np.dot(np.dot(phi_inv, np.conjugate(phi_tilda).T)*weight, x.T)
   
#     # calculate data_rec
#     data_rec = np.real(np.dot(phi_tilda, A_tilda).T)
                
#     return data_rec

def reconstruct_time_method(x,dt,f,P,Ms,fs,weight='default',save_path=os.getcwd(),method='lowRAM'):
    '''
    Purpose: reconstruct flow field using the time domain method 
             (Nekkanti, A. and Schmidt, O., 2021, JFM, 926)
    
    Parameters
    ----------
    x  : 2D numpy array, float; space-time flow field data (mean=0).
         nrow = number of time snapshots;
         ncol = number of grid point * number of variable
    dt : float; time step between adjacent snapshots
    f  : 1D numpy array, float; frequency; output of SPOD main function
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

    time_start=time.time()

    print('--------------------------------------')
    print('Time-method reconstruction ...'        )
    print('--------------------------------------')
    
    # initialize reconstructed flow field data
    nt = x.shape[0]
    nx = x.shape[1]
    data_rec = np.zeros((nt,nx))

    # SPOD parser
    [weight, _, _, _, _] = spod_parser(nt, nx, window='default', \
             weight=weight, nOvlp='default',nDFT='default', method=method)

    # calculate phi_tilda
    if method == 'fast':
        phi_tilda = np.zeros((nx,Ms.shape[0]*fs.shape[0]), dtype=complex)
        for i in range(nx):
            phi_tilda[i,:] = np.ravel(P[:,i,:][fs,:][:,Ms], order='C')
        
    elif method == 'lowRAM':
        h5f = h5py.File(os.path.join(save_path,'reconstruct.h5'), 'w')
        h5f.create_dataset('phi_tilda', shape=(nx,Ms.shape[0]*fs.shape[0]), 
                            chunks=True, dtype = complex, compression="gzip")
        for i in range(nx):
            h5f['phi_tilda'][i,:] = np.ravel(P[:,i,:][fs,:][:,Ms], order='C')
    
    print('Calculate phi_tilda finished ...')
    
    process   = psutil.Process(os.getpid())
    RAM_usage = process.memory_info().rss/1024**3 # unit in GBs
    print('Memory usage: %.2f GB'%RAM_usage            )       
    
    # calculate approximated phi_inv
    if method == 'fast':
        [D, U] = np.linalg.eig(np.dot(np.conjugate(phi_tilda).T * weight, phi_tilda)) # eigen-decomposition
    elif method == 'lowRAM':
        [D, U] = np.linalg.eig(np.dot(np.conjugate(h5f['phi_tilda']).T * weight, h5f['phi_tilda'])) # eigen-decomposition
    D_inv = np.identity(D.shape[0], dtype=complex)
    eps=1e-2 # limiter: truncated eigenvalue / max eigenvalue
    for i in range(D.shape[0]):
        if np.abs(D[i]) < eps*np.max(np.abs(D)):
            D_inv[i,i] = 0
        else:
            D_inv[i,i] = 1/D[i]
    phi_inv = np.dot(np.dot(U, D_inv), np.conjugate(U).T)
    print(phi_inv.shape)
    del D_inv, D, U

    print('Calculate phi_inv finished ...')
    
    # process   = psutil.Process(os.getpid())
    # RAM_usage = process.memory_info().rss/1024**3 # unit in GBs
    # print('Memory usage: %.2f GB'%RAM_usage            )   


    # calculate A_tilda
    if method == 'fast':    
        A_tilda = np.dot(np.dot(phi_inv, np.conjugate(phi_tilda).T)*weight, x.T)
    elif method == 'lowRAM':
        A_tilda = np.dot(np.dot(phi_inv, np.conjugate(h5f['phi_tilda']).T)*weight, x.T)
    print(A_tilda.shape)
    del phi_inv
    print('Calculate A_tilda finished ...')
       
    # process   = psutil.Process(os.getpid())
    # RAM_usage = process.memory_info().rss/1024**3 # unit in GBs
    # print('Memory usage: %.2f GB'%RAM_usage            )   
        
    # calculate data_rec
    if method == 'fast':
        data_rec = np.real(np.dot(phi_tilda, A_tilda).T)
    elif method == 'lowRAM':
        data_rec = np.real(np.dot(h5f['phi_tilda'], A_tilda).T)
        
    # process   = psutil.Process(os.getpid())
    # RAM_usage = process.memory_info().rss/1024**3 # unit in GBs
    # print('Memory usage: %.2f GB'%RAM_usage            )   
        
    # memory usage
    process   = psutil.Process(os.getpid())
    RAM_usage = process.memory_info().rss/1024**3 # unit in GBs
    
    time_end=time.time()
    print('--------------------------------------'     )
    print('Time-method reconstruction finished'        )
    print('Memory usage: %.2f GB'%RAM_usage            )
    print('Run time    : %.2f s'%(time_end-time_start) )
    print('--------------------------------------'     )
                
    return data_rec

def reconstruct_frequency_method(x,f,P,Ms,fs,window='default',weight='default',
                                 nOvlp='default',nDFT='default'):
    '''
    Purpose: reconstruct flow field using the frequency domain method 
             (Nekkanti, A. and Schmidt, O., 2021, JFM, 926)
    
    Parameters
    ----------
    x  : 2D numpy array, float; space-time flow field data (mean=0).
         nrow = number of time snapshots;
         ncol = number of grid point * number of variable
    f  : 1D numpy array, float; frequency; output of SPOD main function
    P  : 3D numpy array, complex; mode shape; output of SPOD main function
         shape[0] = number of frequencies;
         shape[1] = number of grid point * number of variable
         shape[2] = number of modes(ranked in descending order by modal energy)              
    Ms : 1D numpy array, int; index of modes used for reconstruction
    fs : 1D numpy array, int; index of frequencies used for reconstruction
    window : 1D numpy array, float; window function values (default Henning)
                n = nDFT (default nDFT calculated from number of time snapshots)      
    weight : 1D numpy array; weight function (default unity)
             n = number of grid point * number of variable
    nOvlp  : int; number of overlap (default 50% nDFT)
    nDFT   : int; number of DFT points (default about 10% of number of time snapshots)    
           
    Returns
    -------
    data_rec : 1D numpy array; reconstructed flow field data
    '''
    
    # SPOD parser
    nt = x.shape[0]
    nx = x.shape[1]    
    [weight, window, nOvlp, nDFT, nBlks] = spod_parser(nt, nx, window=window, \
             weight=weight, nOvlp=nOvlp, nDFT=nDFT, method='lowRAM')

    # initialize reconstructed flow field data
    nt_rec = nBlks*nDFT-nOvlp*(nBlks-1)
    data_rec = np.zeros((nt_rec,nx))
    
    ## calculate Q_hat
    nFreq = P.shape[0]    
    Q_hat = np.zeros((nx,nFreq,nBlks),dtype = complex) # RAM demanding here  
    Q_blk = np.zeros((nDFT,nx))
    Q_blk_hat = np.zeros((nx,nFreq),dtype = complex)
    
    # loop over each block
    for iBlk in range(nBlks):
        
        # get time index for present block
        it_end   = min(iBlk*(nDFT-nOvlp)+nDFT, nt)
        it_start = it_end-nDFT
                    
        print('block', iBlk+1, '/', nBlks, '(', it_start+1, ':', it_end, ')')
        
        # subtract time-averaged results from the block
        Q_blk = x[it_start:it_end,:] # column-wise broadcasting
 
        # add window function to block
        Q_blk = Q_blk.T * window # row-wise broadcasting
        
        # Fourier transform on block
        Q_blk_hat = 1/np.mean(window)/nDFT*fft(Q_blk)
        Q_blk_hat = Q_blk_hat[:,0:nFreq]
            
        # correct Fourier coefficients for one-sided spectrum
        Q_blk_hat[:,1:(nFreq-1)] *= 2
                
        # save block result to the whole domain result 
        Q_hat[:,:,iBlk] = Q_blk_hat

        
    ## calculate the matrix of expansion coefficient A 
    A = np.zeros((nFreq,nBlks,nBlks),dtype = complex)

    # select frequency
    for iFreq in range(nFreq):
        if iFreq in fs:
            A[iFreq,:,:] = np.dot(np.conjugate(P[iFreq,:,:]).T*weight, Q_hat[:,iFreq,:])

    # select mode
    Ms_mask = np.ones(nBlks, dtype=bool)
    Ms_mask[Ms] = False
    A[:,Ms_mask,:] = 0        

    ## calculate Q_hat_rec
    Q_hat_rec = np.zeros((nx,nFreq,nBlks),dtype = complex) # RAM demanding here  
    for iFreq in range(nFreq):
        Q_hat_rec[:,iFreq,:] = np.dot(P[iFreq,:,:], A[iFreq,:,:])

    ## calculate data_rec
    for iBlk in range(nBlks):
        Q_blk_hat_rec_half = Q_hat_rec[:,:,iBlk]
        Q_blk_hat_rec_half[:,1:(nFreq-1)] /= 2
        Q_blk_hat_rec_half_rev = np.flip(Q_blk_hat_rec_half[:,1:(nFreq-1)], axis=1)
        Q_blk_hat_rec_half_rev = np.real(Q_blk_hat_rec_half_rev) - np.imag(Q_blk_hat_rec_half_rev)*1j
        Q_blk_hat_rec = np.concatenate((Q_blk_hat_rec_half, Q_blk_hat_rec_half_rev), axis=1)
        Q_blk_hat_rec*= np.mean(window)*nDFT       
        Q_blk_rec = (np.real(ifft(Q_blk_hat_rec)/window)).T
       
        # get time index for present block
        it_end   = min(iBlk*(nDFT-nOvlp)+nDFT, nt)
        it_start = it_end-nDFT

        # fill in data_rec
        for itime in range(0, nDFT, 1):
            # overlaping with previous block
            if (itime>=0) and (itime<=nOvlp-1):
                if iBlk==0: # first block； no previous block for overlapping
                    data_rec[(it_start+itime), :] += Q_blk_rec[itime,:]
                else:
                    win_itime_crt = window[itime]
                    win_itime_pre = window[nDFT-nOvlp+itime]
                    win_weight = win_itime_crt/(win_itime_crt+win_itime_pre) # use win_weighted avearge
#                    win_weight = 1 # always use next block
#                    win_weight = 0 # always use previous block
                    data_rec[(it_start+itime), :] += Q_blk_rec[itime,:]*win_weight
            
            # non-overlaping region
            elif (itime>nOvlp-1) and (itime<nDFT-nOvlp):
                data_rec[(it_start+itime), :] += Q_blk_rec[itime,:]
            
            # overlaping with next block
            elif (itime>=nDFT-nOvlp) and (itime<=nDFT-1):
                if iBlk==nBlks-1: # last block; no next block for overlapping
                    data_rec[(it_start+itime), :] += Q_blk_rec[itime,:]
                else:
                    win_itime_crt = window[itime]
                    win_itime_aft = window[nOvlp-nDFT+itime]
                    win_weight = win_itime_crt/(win_itime_crt+win_itime_aft) # use win_weighted avearge
#                    win_weight = 0 # always use next block
#                    win_weight = 1 # always use previous block
                    data_rec[(it_start+itime), :] += Q_blk_rec[itime,:]*win_weight
        
    return data_rec

#def reconstruct_direct_method(x,f,L,P,Ms,fs,ts,window='default',weight='default',
#                                 nOvlp='default',nDFT='default'):
#    '''
#    Purpose: reconstruct flow field directly !!! under development
#             (Towne, A., Schmidt, O., and Colonius, T., 2018, arXiv:1708.04393v2)
#    
#    Parameters
#    ----------
#    x  : 2D numpy array, float; space-time flow field data (mean=0).
#         nrow = number of time snapshots;
#         ncol = number of grid point * number of variable
#    f  : 1D numpy array, float; frequency; output of SPOD main function
#    L  : 2D numpy array, float; modal energy E(f, M); output of SPOD main function
#         nrow = number of frequencies
#         ncol = number of modes (ranked in descending order by modal energy)
#    P  : 3D numpy array, complex; mode shape; output of SPOD main function
#         shape[0] = number of frequencies;
#         shape[1] = number of grid point * number of variable
#         shape[2] = number of modes(ranked in descending order by modal energy)              
#    Ms : 1D numpy array, int; index of modes used for reconstruction
#    fs : 1D numpy array, int; index of frequencies used for reconstruction
#    ts : 1D numpy array, float; time series at which flow field is reconstructed   
#    window : 1D numpy array, float; window function values (default Henning)
#                n = nDFT (default nDFT calculated from number of time snapshots)      
#    weight : 1D numpy array; weight function (default unity)
#             n = number of grid point * number of variable
#    nOvlp  : int; number of overlap (default 50% nDFT)
#    nDFT   : int; number of DFT points (default about 10% of number of time snapshots)    
#           
#    Returns
#    -------
#    data_rec : 1D numpy array; reconstructed flow field data
#    '''
#
#    # initialize reconstructed flow field data
#    data_rec = np.zeros((ts.shape[0],np.shape(P)[1]))
#
#    # SPOD parser
#    nt = x.shape[0]
#    nx = x.shape[1]    
#    [weight, window, nOvlp, nDFT, nBlks] = spod_parser(nt, nx, window=window, \
#             weight=weight, nOvlp=nOvlp,nDFT=nDFT, method='lowRAM')
#
#    # calculate Q_hat
#    nFreq = P.shape[0]    
#    Q_hat = np.zeros((nx,nFreq,nBlks),dtype = complex) # RAM demanding here  
#    Q_blk = np.zeros((nDFT,nx))
#    Q_blk_hat = np.zeros((nx,nFreq),dtype = complex)
#    
#    # loop over each block
#    for iBlk in range(nBlks):
#        
#        # get time index for present block
#        it_end   = min(iBlk*(nDFT-nOvlp)+nDFT, nt)
#        it_start = it_end-nDFT
#                    
#        print('block', iBlk+1, '/', nBlks, '(', it_start+1, ':', it_end, ')')
#        
#        # subtract time-averaged results from the block
#        Q_blk = x[it_start:it_end,:] # column-wise broadcasting
# 
#        # add window function to block
#        Q_blk = Q_blk.T * window # row-wise broadcasting
#        
#        # Fourier transform on block
#        Q_blk_hat = 1/np.mean(window)/nDFT*fft(Q_blk)
#        Q_blk_hat = Q_blk_hat[:,0:nFreq]
#            
#        # correct Fourier coefficients for one-sided spectrum
#        Q_blk_hat[:,1:(nFreq-1)] *= 2
#                
#        # save block result to the whole domain result 
#        Q_hat[:,:,iBlk] = Q_blk_hat
#
#    # calculate reconstruction    
#    for i in range(ts.shape[0]):
#        ti = ts[i]
#        for fi in fs:
#            for Mi in Ms:
##                temp_zeta = np.conjugate(Q_hat[:,fi,Mi])*P[fi,:,Mi]*np.sqrt(nBlks*L[fi,Mi])
##                temp_a = np.dot(Q_hat[:,fi,Mi]*weight, temp_zeta)
##                temp_data_rec += temp_a*temp_zeta
#
#                a_fi_Mi = np.dot(Q_hat[:,fi,0]*weight, P[fi,:,Mi])
#                data_rec[i,:] += np.real(a_fi_Mi*P[fi,:,Mi]*np.exp(2j*np.pi*f[fi]*ti))
#                
#    return data_rec

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

def plot_confidence_bounds(nBlks, conf_level = 0.95):
    '''
    Purpose: plot confidence bounds of SPOD energy
             i.e., lambda_lower/lambda and lambda_upper/lambda
             (Schmidt, O., and Colonius, T., 2020, AIAA J, 58(3), 1023-1033)
    
    Parameters
    ----------
    nBlks  : int; calculated/specified number of blocks; also number of modes, i.e., L.shape[1]
         
    Returns
    -------
    lambda_lower: float; lower bound of energy (relative)
    lambda_upper: float; upper bound of energy (relative)
    fig1 : matplotlib figure object; plot upper and lower bounds
    fig2 : matplotlib figure object; plot difference between upper and lower bounds
    '''

    # calculate bounds for all nBlks
    Nbs = np.linspace(5,100,20)
    xi2_uppers    = 2 * gammaincinv(Nbs, 1 - conf_level)
    xi2_lowers    = 2 * gammaincinv(Nbs, conf_level)
    lambda_uppers = 2 * Nbs/xi2_uppers
    lambda_lowers = 2 * Nbs/xi2_lowers
    lambda_intervals = lambda_uppers-lambda_lowers
    
    # calculate bounds for the input nBlk
    lambda_upper = nBlks/gammaincinv(nBlks, 1 - conf_level)
    lambda_lower = nBlks/gammaincinv(nBlks, conf_level)  
    lambda_interval = lambda_upper-lambda_lower
    
    # plot upper and lower bounds
    fig1 = plt.figure()
    plt.loglog(Nbs, lambda_lowers, linestyle='-', color='lightgrey', label='Lower bound')
    plt.loglog(Nbs, lambda_uppers, linestyle='--', color='lightgrey',label='Upper bound')
    plt.scatter([nBlks,nBlks],[lambda_upper,lambda_lower], 
                facecolor='white', edgecolor='steelblue', label='Input $n_{Blks}$',zorder=3)
    plt.text(nBlks*1.1, lambda_upper*1.1, '(%.i, %.4f)'%(nBlks,lambda_upper))
    plt.text(nBlks*1.1, lambda_lower*0.7, '(%.i, %.4f)'%(nBlks,lambda_lower))    

    plt.xlabel('Number of modes')
    plt.ylabel('Normalized confidence bounds')
    plt.legend(loc='best')
    plt.axis([5,100,0.1,10])
    
    # plot difference between upper and lower bounds
    fig2 = plt.figure()
    plt.loglog(Nbs, lambda_intervals, color='lightgrey',label='confidence interval')
    plt.scatter(nBlks,lambda_interval, facecolor='white', edgecolor='steelblue',
                label='Input $n_{Blks}$',zorder=3)
    plt.text(nBlks*1.1, lambda_interval*1.1, '(%.i, %.4f)'%(nBlks,lambda_interval))

    plt.xlabel('Number of modes')
    plt.ylabel('Normalized confidence interval')
    plt.legend(loc='best')
    plt.axis([5,100,0.1,2])
    
    return lambda_lower, lambda_upper, fig1, fig2

# End