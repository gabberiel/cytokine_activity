import time
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os import path
from scipy.io import loadmat
from vae_dense_wf import get_vae 
from cvae_dense_wf import get_cvae
from plot_functions_wf import plot_encoded


def load_mat_file(path_to_file, matlab_key, verbose=1):
    """
    Load waveform- or timestamps-matlab file specified by "path_to_file" and returns it as numpy array.
    Files saved locally on Asus-computer have matlab_key='waveforms'

    Parameters
    ----------
    path_to_file : 'path/to/file.mat' string_type
            Path to saved .mat file
    matlab_key : string_type
            key as specified when saving file in MATLAB.
            Either 'waveforms' or 'timestamps'
    verbose : integer
            > 0 => prints about progress.

    Returns
    -------
    mat_file : (number_of_waveforms, size_of_waveform) array_like or
               (number_of_waveforms, 1) array_like
            Numpy version of loaded matlab matrix. 

    """
    if verbose > 0:
        print('\n Loading matlab'+ matlab_key +' files... \n' )
    # Load matlab
    mat_file = loadmat(path_to_file)[matlab_key]
    if verbose > 0:
        print(f'{matlab_key} loaded succesfully... \n')
        print(f'Shape of {matlab_key}: {mat_file.shape}.')
    return mat_file


def get_pdf_model(data_train, hypes, ev_label=None,path_to_weights=None, 
                continue_train=False, verbose=1):
    """
    Initiates or continues training of cvae/vae-model depending on existance of path_to_weights-file.

    If ev_label is ``None`` the it assumes a VAE-model.
    Otherwise Conditional-VAE
    
    Parameters
    ----------
    data_train : (num_train_data_pts, dim_of_waveform) array_like
            CAP-waveforms to train model on
    hypes : .json file
            Containing the hyperparameters:
                latent_dim : integer
                        dimension of latent space.
                nr_epochs : integer
                        number of epochs in training.
                batch_size : integer
                        number of samples in each SGD step.
    ev_label : (num_train_data_pts, 3) array_like or ``None``
            one-hot representation of labels for each CAP-waveform in CVAE.
            If no labels are given, then function assumes that a VAE is to be used.

    path_to_weights : string or ``None``
            specifies path to saved model-weights if any.
    continue_train : boolean
            determines if the training should continue or not if pretrained model exists
    verbose : integer
            level of verbosity
    ... 

    Returns
    -------
    encoder, decoder, cvae/vae : keras model classes.

    """
    latent_dim = hypes["cvae"]["latent_dim"]
    nr_epochs = hypes["cvae"]["nr_epochs"]
    batch_size = hypes["cvae"]["batch_size"]

    assert np.isnan(np.sum(data_train))==False, 'Nans in "data_train"'
    
    #ops.reset_default_graph()
    tf.keras.backend.clear_session()    # When building new models in for-loop, the stored graph in keras.backend
    # creats an error since it then is missmatches between the different models..

    waveform_shape = data_train.shape[-1]
    if ev_label is None:
        encoder,decoder,vae = get_vae(waveform_shape,latent_dim)
    else:
        encoder,decoder,cvae = get_cvae(hypes)
    
    if path.isfile(path_to_weights+'.index'):
        if verbose > 0:
            print(f'\n Loading {path_to_weights}... \n')
        if ev_label is None:
            vae.load_weights(path_to_weights).expect_partial()
        else:
            cvae.load_weights(path_to_weights).expect_partial()
        if continue_train == True:
            if verbose > 0:
                print(f'\nContinue training for {nr_epochs} epochs... \n')
            if ev_label is None:
                history = vae.fit(data_train , data_train, epochs=nr_epochs, batch_size=batch_size)
                vae.save_weights(path_to_weights)
            else:
                history = cvae.fit([data_train, ev_label] , data_train, epochs=nr_epochs, batch_size=batch_size)
                cvae.save_weights(path_to_weights)
            if verbose > 1:
                plt.plot(history.history['loss'])
                plt.show() 
    else:
        if verbose > 0:
            print(f'\n Start training from scratch for {nr_epochs} epochs...')
            print(f'Weights will be saved as {path_to_weights} \n')
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



def run_pdf_GD(waveforms, cvae, ev_labels, hypes, 
            matlab_file='',
            unique_string_for_figs='',
            path_to_hpdp='',
            verbose=False, 
            view_GD_result=False, 
            encoder=None):
    '''
    Function to be called for running grandient decent of pdf using cvae.

    Encoder needed if view_GD_results is True.

    Parameters
    ----------
    waveforms : (number_of_wf, dim_of_wf) array_like
        Only used to initiate GD if "path_to_hpdp" does not exist.
    cvae : kera.Model class_instance
        Fully trained CVAE/VAE model. 
    ev_labels : (number_of_wf, 3) array_like or ``None``
        If ``None``, then VAE is assumed. Otherwise CVAE
    hypes : .json file
            Containing the hyperparameters:
                m,gamma,eta : integer/floats
                    Parameters for GD of pdf
                labels_to_evaluate : python list
                    Specifies which labels that is to be used
                downsample_threshold : integer
                    If the number of waveforms with a given labele is > downsample_threshold,
                    then every 4th wf is used. This to speed up computations.
    path_to_hpdp : 'path/to/hpdp.npy'
        If ``None`` then "data_points" is used to start GD.
    verbose : boolean
        verbosity of code..
    view_GD_result : boolean
        If True, then the encoded latent space is plotted before/after GD is performed for each label
    encoder : ``None`` or keras.Model instance
        Used for plotting.
        Must be passed if view_GD_result is True.

    Returns
    -------
    hpdp_list : numpy list
        Each element contains:
            (number_of_wf, dim_of_wf) array_like
                The resulting waveforms after running GD on all.
    '''
    labels_to_evaluate = hypes["pdf_GD"]["labels_to_evaluate"]
    downsample_threshold = hypes["pdf_GD"]["downsample_threshold"]
    dim_of_wf = hypes["preprocess"]["dim_of_wf"]
    hpdp_list = []
    ev_label_corr_shape_list = []

    for label_on in labels_to_evaluate: # Either 0, 1 (or 2)
        waveforms_increase = waveforms[ev_labels[:,label_on]==1]

        print(f'waveforms_increase injection {label_on+1} : {waveforms_increase.shape}\n')

        if waveforms_increase.shape[0] == 0:
            waveforms_increase = np.append(np.zeros((1, dim_of_wf)), waveforms_increase).reshape((1, dim_of_wf))
            ev_label_corr_shape = np.zeros((waveforms_increase.shape[0], 3))
            ev_label_corr_shape[:, label_on] = 1
            print('*************** OBS ***************')
            print(f'No waveforms with increased event rate at injection {label_on + 1} was found.')
            print(f'This considering the recording {matlab_file}')

        elif waveforms_increase.shape[0] > downsample_threshold: # Speed up process during param search..
            waveforms_increase = waveforms_increase[::4, :]
            ev_label_corr_shape = np.zeros((waveforms_increase.shape[0], 3))
            ev_label_corr_shape[:, label_on] = 1

        else:
            ev_label_corr_shape = np.zeros((waveforms_increase.shape[0], 3))
            ev_label_corr_shape[:, label_on] = 1

        hpdp = __pdf_GD__(cvae, waveforms_increase, hypes, 
                          ev_label=ev_label_corr_shape,
                          path_to_hpdp=path_to_hpdp+str(label_on), 
                          verbose=verbose)

        hpdp_list.append(hpdp)
        ev_label_corr_shape_list.append(ev_label_corr_shape)
        if view_GD_result:
            encoded_hpdp_title = 'Visualisation of the Latent Variable Mean.'
            save_figure = 'figures/encoded_decoded/' + unique_string_for_figs
            print(f'Visualising decoded latent space of hpdp... \n')
            plt.figure(1)
            plot_encoded(encoder, waveforms, saveas=save_figure+'_encoded_ho_wf'+str(label_on),
                         verbose=1, ev_label=ev_labels, title=encoded_hpdp_title) 
            plt.figure(2)
            plot_encoded(encoder, hpdp, saveas=save_figure+'_encoded_hpdp'+str(label_on), 
                        verbose=1, ev_label=ev_label_corr_shape, title=encoded_hpdp_title)
    if view_GD_result:
        plt.figure(3)
        for i in range(2):
            plot_encoded(encoder, hpdp_list[i], saveas=None, 
                        verbose=0, ev_label=ev_label_corr_shape_list[i], title=encoded_hpdp_title)
        plt.show()
    # view_GD_result = False  # TODO: REMOVE!!!
    if view_GD_result:     
        continue_to_Clustering = input('Continue to Clustering? (yes/no) :')
        all_fine = False
        while all_fine is False:
            if continue_to_Clustering == 'no':
                exit()
            elif continue_to_Clustering == 'yes':
                print('Continues to "run_GD"')
                all_fine = True
            else:
                continue_to_Clustering = input('Invalid input, continue to Clustering? (yes/no) :')
    return hpdp_list

def __pdf_GD__(vae, data_points,hypes, 
               ev_label=None, path_to_hpdp=None, 
               verbose=1):
    '''
    Gradient decent of approximate probability ditribution using VAEs.

    If path_to_hpdp exists then GD is continued using the saved data.

    Note : hpdp = "High probability data-points".

    Parameters
    ----------
    vae : kera.Model class_instance
        Fully trained CVAE/VAE model. 
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
    if m > 0:
        hpdp_x : (number_of_wf, dim_of_wf) array_like
            The resulting waveforms after running GD on all CAPs.
    if m=0:
        data_points : (number_of_wf, dim_of_wf) array_like
            Saved hpdp if "path_to_hpdp" exist. Otherwise raises warning and returns the input.
    
    '''
    m = hypes["pdf_GD"]["m"] 
    gamma = hypes["pdf_GD"]["gamma"]
    eta = hypes["pdf_GD"]["eta"]
    if m > 0:
        if path.isfile(path_to_hpdp + '.npy'):
            if verbose > 0:
                print(f'n Loading {path_to_hpdp} to continue pdf-GD... \n')
            data_points = np.load(path_to_hpdp + '.npy')
            assert np.isnan(np.sum(data_points))==False, 'NaNs in input data..'

            if verbose > 0:
                print(f'Saved clusters: "{path_to_hpdp}" loaded Succesfully... \n')
                print(f'Continues GD on file: {path_to_hpdp} for {m} iterations...')
            
            if ev_label is None:
                hpdp_x = __cluster__(vae, data_points, eta, gamma, m)
            else:
                hpdp_x = __cluster_CVAE__(vae, data_points, ev_label, eta, gamma, m)
            hpdp_x = hpdp_x[~np.isnan(hpdp_x).any(axis=1)] # Remove CAPs containing nans.
            assert np.isnan(np.sum(hpdp_x))==False, 'NaNs in hpdp_x efter GD..'
            np.save(path_to_hpdp, hpdp_x)

            if verbose > 0:
                print(f'\n High prob. data-points (hpdp): "{path_to_hpdp}" saved Succesfully... \n')
        else:
            if verbose > 0:
                print(f'\n Starting fresh for {m} iterations.... \n')
            if ev_label is None:
                hpdp_x = __cluster__(vae, data_points, eta, gamma, m)
            else:
                hpdp_x = __cluster_CVAE__(vae, data_points, ev_label, eta, gamma, m)
            hpdp_x = hpdp_x[~np.isnan(hpdp_x).any(axis=1)] # Remove CAPs containing nans.
            assert np.isnan(np.sum(hpdp_x))==False, 'NaNs in hpdp_x after GD..'
            np.save(path_to_hpdp, hpdp_x)
            if verbose > 0:
                print(f'\n High prob. data-points (hpdp): "{path_to_hpdp}" saved Succesfully... \n')
        return hpdp_x 
    else:
        if path.isfile(path_to_hpdp+'.npy'):
            if verbose > 0:
                print(f'\n Loading {path_to_hpdp} as hpdp without performing GD... \n')
            data_points = np.load(path_to_hpdp+'.npy')
            assert np.isnan(np.sum(data_points))==False, 'NaNs loaded hpdp...'
            if verbose > 0:
                print(f'\n High prob. data-points (hpdp): "{path_to_hpdp}" loaded Succesfully... \n')
        else:
            warnings.warn(f'{path_to_hpdp} not found and number of iterations set to 0. Returning input datapoints.')
        
        return data_points

def __cluster_CVAE__(cvae, x, label, eta, gamma, m):
    ''' 
    CVAE version..
    Gradient decent iterations used in "__pdf_GD__()". 
    
    '''
    count = 0
    assert np.isnan(np.sum(x))==False, 'Nans in input data..'
    for i in range(m):
        # Estimate time of loop, (ETA).
        if i==0:
            t0 = time.time()
            if np.max(x) > 20: # Fix too hopefully keep code running even if some GD diverges...
                print('Too large value encountered for x in GD. Stops the iterations..')
                break
        elif i % 100==0:
            count += 1
            ti = time.time()
            ETA_t = m/100 * (ti - t0) / (count) - (ti - t0) 
            print(f'Running pdf-GD, iteration={i}')
            print(f'ETA: {round(ETA_t)} seconds.. \n')
            #assert np.isnan(np.sum(x))==False, 'NaNs in hpdp_x after GD..'
            if np.max(x) > 20: # Fix too hopefully keep code running even if some GD diverges...
                print('Too large value encountered for x in GD. Stops the iterations..')
                break

        #x_hat = x + eta*tf.random.normal(shape=x.shape)
        x_hat = x + eta * np.random.normal(size=x.shape)
        x_rec = cvae([x_hat, label]).numpy()
        x = x - gamma * (x_hat - x_rec)

    return x

def __cluster__(vae, x, eta, gamma, m):
    '''
    VAE-version... TODO: DELETE? 
    The Gradient decent loop used in "__pdf_GD__()". 
    '''
    count = 0
    assert np.isnan(np.sum(x))==False, 'Nans in input data..'
    for i in range(m):
        # Estimate time of loop, (ETA).
        if i==0:
            t0 = time.time()
        elif i%100==0:
            count += 1
            ti = time.time()
            ETA_t = m/100 * (ti - t0) / (count) - (ti - t0) 
            print(f'Running pdf-GD, iteration={i}')
            print(f'ETA: {round(ETA_t)} seconds.. \n')

        #x_hat = x + eta*tf.random.normal(shape=x.shape)
        x_hat = x + eta * np.random.normal(size=x.shape)
        x_rec = vae(x_hat)
        x = x - gamma*(x_hat - x_rec)

    return x


if __name__ == "__main__":
    '''
    ######## TESTING: #############
    '''

    path_to_wf = 'matlab_files/gg_waveforms-R10_IL1B_TNF_03.mat'
    path_to_ts = 'matlab_files/gg_timestamps.mat'
    path_to_weights = 'models/main_funs'

    waveforms, mean, std = load_waveforms(path_to_wf,'waveforms', verbose=1)
    timestamps = load_timestamps(path_to_ts,'gg_timestamps',verbose=1)
    
    
    # ********************** PLOTS *******************************************
    if False:
        for i in range(10):
            plt.plot(waveforms[i])
        plt.show()
        plt.plot(timestamps[1:-1:100])
        plt.show()
    # ************************************************************************

    encoder,decoder,vae = get_pdf_model(waveforms[:1000], hypes, path_to_weights=path_to_weights, 
                                        continue_train=False, verbose=1)
    
    
    path_to_hpdp = "numpy_hpdp/second_run"
    
    hpdp = __pdf_GD__(vae, waveforms[:1000],hypes, path_to_hpdp=path_to_hpdp,verbose=1)
