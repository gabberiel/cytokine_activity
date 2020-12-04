import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from vae_dense_wf import get_vae 
from cvae_dense_wf import get_cvae
from scipy.io import loadmat
from os import path
import warnings

def load_waveforms(path_to_wf,matlab_key, verbose=1):
    """
    Load waveform-matlab file specified by "path_to_wf" and returns it as numpy array.
    
    Parameters
    ----------
    path_to_wf : 'path/to/file.mat' string_type

    matlab_key : string_type
            key as specified when saving file in MATLAB.
    
    standardize : boolean_type
            If True, then data is standardized by subtracting column mean and devide by 
            column standard diviation.

    Returns
    -------
    waveforms : (number_of_waveforms, size_of_waveform) array_like
            Numpy version of loaded matlab matrix. 

    """
    if verbose>0:
        print()
        print('Loading matlab waveforms files...')
        print()
    # Load matlab
    waveforms = loadmat(path_to_wf)[matlab_key]
    if verbose>0:
        print('waveforms loaded succesfully...')
        print()
        print(f'Shape of waveforms: {waveforms.shape}.')
    
    return waveforms

def load_timestamps(path_to_ts,matlab_key,verbose=1):
    """Load timestamp-matlab file specified by "path_to_ts" and returns it as numpy array.
    
    See Also
    --------
        doc_string in "load_waveforms()" for more extensive description.
    """

    if verbose>0:
        print()
        print('Loading matlab timestamps file...')
        print()
    # Load matlab
    timestamps = loadmat(path_to_ts)[matlab_key]
    if verbose>0:
        print('timestamps loaded succesfully...')
        print()
        print(f'Shape of timestamps: {timestamps.shape}.')
    return timestamps


def train_model(data_train,latent_dim=2, nr_epochs=50, batch_size=128, path_to_weights=None, 
                continue_train=False, verbose=1,ev_label=None):
    """
    Initiates or continues training of vae-model depending on existance of path_to_weights-file.

    If ev_label is None the it assumes a VAE-model.
    Otherwise Conditional-VAE
    
    Parameters
    ----------
    data_train : (num_train_data_pts, size_of_waveform) array_like
    
    latent_dim : integer_like
            dimension of latent space.
    
    ... 

    Returns
    -------
    encoder, decoder, vae : keras model classes.

    """
    assert np.isnan(np.sum(data_train))==False, 'Nans in "data_train"'
    waveform_shape = data_train.shape[-1]
    if ev_label is None:
        encoder,decoder,vae = get_vae(waveform_shape,latent_dim)
    else:
        encoder,decoder,cvae = get_cvae(waveform_shape,latent_dim,label_dim=3)
    
    if path.isfile(path_to_weights+'.index'):
        if verbose>0:
            print()
            print(f'Loading {path_to_weights}...')
            print()
        if ev_label is None:
            vae.load_weights(path_to_weights)
        else:
            cvae.load_weights(path_to_weights)
        if continue_train == True:
            if verbose>0:
                print()
                print(f'Continue training for {nr_epochs} epochs...')
                print()
            if ev_label is None:
                history = vae.fit(data_train , data_train, epochs=nr_epochs, batch_size=batch_size)
                vae.save_weights(path_to_weights)
            else:
                history = cvae.fit([data_train, ev_label] , data_train, epochs=nr_epochs, batch_size=batch_size)
                cvae.save_weights(path_to_weights)
            if verbose>1:
                plt.plot(history.history['loss'])
                plt.show() 
    else:
        if verbose>0:
            print()
            print(f'Start training from scratch for {nr_epochs} epochs...')
            print(f'Weights will be saved as {path_to_weights}')
            print()
        if ev_label is None:
            history = vae.fit(data_train , data_train, epochs=nr_epochs, batch_size=batch_size)
            vae.save_weights(path_to_weights)
        else:
            history = cvae.fit([data_train,ev_label] , data_train, epochs=nr_epochs, batch_size=batch_size)
            cvae.save_weights(path_to_weights)
        if verbose>1:
            plt.plot(history.history['loss'])
            plt.show() 
    if ev_label is None:
         return encoder,decoder,vae 
    else:
         return encoder,decoder,cvae 

def __cluster__(vae,x,eta,gamma,m):
    ''' The Gradient decent loop used in "pdf_GD". '''
    count = 0
    assert np.isnan(np.sum(x))==False, 'Nans in input data..'
    for i in range(m):
        # Estimate time of loop, (ETA).
        if i==0:
            t0 = time.time()
        elif i%100==0:
            count += 1
            ti = time.time()
            ETA_t = m/100 * (ti-t0)/(count) - (ti-t0) 
            print(f'Running pdf-GD, iteration={i}')
            print(f'ETA: {round(ETA_t)} seconds..')
            print()

        #x_hat = x + eta*tf.random.normal(shape=x.shape)
        x_hat = x + eta * np.random.normal(size=x.shape)
        x_rec = vae.predict(x_hat)
        x = x - gamma*(x_hat-x_rec)

    return x

def __cluster_CVAE__(cvae,x,label,eta,gamma,m):
    ''' The Gradient decent loop used in "pdf_GD". '''
    count = 0
    assert np.isnan(np.sum(x))==False, 'Nans in input data..'
    for i in range(m):
        # Estimate time of loop, (ETA).
        if i==0:
            t0 = time.time()
        elif i%100==0:
            count += 1
            ti = time.time()
            ETA_t = m/100 * (ti-t0)/(count) - (ti-t0) 
            print(f'Running pdf-GD, iteration={i}')
            print(f'ETA: {round(ETA_t)} seconds..')
            print()

        #x_hat = x + eta*tf.random.normal(shape=x.shape)
        x_hat = x + eta * np.random.normal(size=x.shape)
        x_rec = cvae.predict([x_hat,label])
        x = x - gamma*(x_hat-x_rec)

    return x

def pdf_GD(vae, data_points,ev_label=None, m=1000, gamma=0.01, eta=0.01, path_to_hpdp=None,verbose=1):
    '''
    Gradient decent of approximate input probability space using VAEs.
    I.e Normal approximation of input distribution.

    If path_to_hpdp exists then GD is continued using the saved data.

    OBS : hpdp = "High probability data-points.

    Parameters
    ----------
    vae : kera.Model class_instance
        Full trained VAE model. 
    data_points : (number_of_wf, dim_of_wf) array_like
        Only used to initiate GD if "path_to_hpdp" does not exist.
    ev_labels : (number_of_wf, 3) array_like or None
        If None, then VAE is assumed. Otherwise CVAE
    m,gamma,eta : integer/floats 
        Parameters for GD of pdf
    path_to_hpdp : 'path/to/hpdp.npy'
        If None then "data_points" is used to start GD.

    Returns
    -------
    if m>0:
        hpdp_x : (number_of_wf, dim_of_wf) array_like
            The resulting waveforms after running GD on all.
    if m=0:
        data_points : (number_of_wf, dim_of_wf) array_like
            Saved hpdp if "path_to_hpdp" exist. Otherwise raises warning and returns the input.
    '''
    if m>0:
        if path.isfile(path_to_hpdp+'.npy'):
            if verbose>0:
                print()
                print(f'Loading {path_to_hpdp} to continue pdf-GD...')
                print()
            data_points = np.load(path_to_hpdp+'.npy')
            assert np.isnan(np.sum(data_points))==False, 'NaNs in input data..'

            if verbose>0:
                print(f'Saved clusters: "{path_to_hpdp}" loaded Succesfully...')
                print()
                print(f'Continues GD on file: {path_to_hpdp} for {m} iterations...')
            
            if ev_label is None:
                hpdp_x = __cluster__(vae,data_points,eta,gamma,m)
            else:
                hpdp_x = __cluster_CVAE__(vae,data_points,ev_label,eta,gamma,m)
            assert np.isnan(np.sum(hpdp_x))==False, 'NaNs in hpdp_x efter GD..'
            np.save(path_to_hpdp,hpdp_x)

            if verbose>0:
                print()
                print(f'High prob. data-points (hpdp): "{path_to_hpdp}" saved Succesfully...')
                print()
        else:
            if verbose>0:
                print()
                print(f'Starting fresh for {m} iterations....')
                print()
            if ev_label is None:
                hpdp_x = __cluster__(vae,data_points,eta,gamma,m)
            else:
                hpdp_x = __cluster_CVAE__(vae,data_points,ev_label,eta,gamma,m)
            assert np.isnan(np.sum(hpdp_x))==False, 'NaNs in hpdp_x after GD..'
            np.save(path_to_hpdp,hpdp_x)
            if verbose>0:
                print()
                print(f'High prob. data-points (hpdp): "{path_to_hpdp}" saved Succesfully...')
                print()
        return hpdp_x 
    else:
        if path.isfile(path_to_hpdp+'.npy'):
            if verbose>0:
                print()
                print(f'Loading {path_to_hpdp} as hpdp without performing GD...')
                print()
            data_points = np.load(path_to_hpdp+'.npy')
            assert np.isnan(np.sum(data_points))==False, 'NaNs loaded hpdp...'
            if verbose>0:
                print()
                print(f'High prob. data-points (hpdp): "{path_to_hpdp}" loaded Succesfully...')
                print()
        else:
            warnings.warn(f'{path_to_hpdp} not found and number of iterations set to 0. Returning input datapoints.')
        
        return data_points


def plot_waveforms(waveforms,labels=None):
    ''' 
    OBSOBS: old version, use version in plot_functions_wf.py
    If labels are given, then the meadian of each waveform - cluster is ploted. 
    x_axis assumes each waveform is 3.5ms long.
    '''
    warnings.warn('Old version of function plt_waveforms() is being used. Use version in plot_functions_wf.py instead.',DeprecationWarning)
    num_wf = waveforms.shape[0]
    wf_size = waveforms.shape[-1]
    cols = ['r','b',]
    if labels is not None:
        t_axis = np.arange(0,3.5,3.5/wf_size)

        for cluster in labels:
            wf_clust = waveforms[labels==cluster]
            plt.plot(t_axis.T,np.median(wf_clust,axis=0).T)# ,color = (0.7,0.7,0.7),lw=0.2)
    #plt.plot(time,np.median(waveforms[ind,:,0],axis=0),color = (0.2,0.2,0.2),lw=1)
    else: 
        t_axis = np.arange(0,3.5,3.5/wf_size)*np.ones((num_wf,1))
        plt.plot(t_axis.T,waveforms.T)
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(\mu V)$')
    plt.show()




if __name__ == "__main__":
    '''
    ######## TESTING: #############
    '''

    path_to_wf = 'matlab_files/gg_waveforms-R10_IL1B_TNF_03.mat'
    path_to_ts = 'matlab_files/gg_timestamps.mat'
    path_to_weights = 'models/main_funs'

    waveforms, mean, std = load_waveforms(path_to_wf,'waveforms',standardize=True, verbose=1)
    timestamps = load_timestamps(path_to_ts,'gg_timestamps',verbose=1)
    
    
    # ********************** PLOTS *******************************************
    if False:
        for i in range(10):
            plt.plot(waveforms[i])
        plt.show()
        plt.plot(timestamps[1:-1:100])
        plt.show()
    # ************************************************************************

    encoder,decoder,vae = train_model(waveforms[:1000], nr_epochs=5, batch_size=128, path_to_weights=path_to_weights, 
                                        continue_train=False, verbose=1)
    
    
    path_to_hpdp = "numpy_hpdp/second_run"
    
    hpdp = pdf_GD(vae, waveforms[:1000], m=100, gamma=0.01, eta=0.01, path_to_hpdp=path_to_hpdp,verbose=1)
