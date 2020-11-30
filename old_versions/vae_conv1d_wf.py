# BUILD MODEL
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.mixture import GaussianMixture



def sample_z(mu,sigma):
    '''
    TODO : 
    '''
    batch     = keras.backend.shape(mu)[0]
    dim       = keras.backend.int_shape(mu)[1]
    eps       = keras.backend.random_normal(shape=(batch, dim))
    return mu + keras.backend.exp(sigma / 2) * eps

class Add_kl_loss(layers.Layer):
    '''
    Layer instance which returns the the Kullbeck-Lieberg Divergence between latent distribution q(z|x) and prior p(z)= N(0,1).
    '''
    def call(self,inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        #self.add_loss(kl_loss + 27)
        #self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        return kl_loss


def create_encoder(encoder_input,latent_dim):
    '''
    TODO:
    Description....

    Parameters
    ----------
    inp_layer : 

    Returns
    -------
    out : 

    See Also
    --------
        Ref to similar funtions...
    
    Notes
    -----
    More mathematicall description

    References
    ----------
    .. [1] Wikipedia, "Convolution", http://en.wikipedia.org/wiki/Convolution.
    Examples
    --------
    blabla
    >>> np.convolve([1, 2, 3], [0, 1, 0.5])
    array([ 0. ,  1. ,  2.5,  4. ,  1.5])
    Only return the middle values of the convolution.
    '''
    #e_i    = layers.Input(shape=(waveform_dim, ), name='encoder_input')
    cx     = keras.layers.Conv1D(filters=8, kernel_size=2, strides=1, padding='same', activation='relu',input_shape=(141,1))(encoder_input)
    cx     = keras.layers.BatchNormalization()(cx)
    cx     = keras.layers.Conv1D(filters=16, kernel_size=2, strides=1, padding='same', activation='relu')(cx)
    cx     = keras.layers.BatchNormalization()(cx)
    x      = keras.layers.Flatten()(cx)
    x      = keras.layers.Dense(100, activation='relu')(x)
    x      = keras.layers.BatchNormalization()(x)

    #conv_shape = keras.backend.int_shape(cx)

    z_mean      = layers.Dense(latent_dim, name='latent_mu')(x)
    z_log_var   = layers.Dense(latent_dim, name='latent_sigma')(x)

    z = sample_z(z_mean,z_log_var)

    # CALCULATE LOSSES:
    kl_loss = Add_kl_loss()([z_mean,z_log_var])    

    # Instantiate encoder
    encoder = keras.Model(inputs=[encoder_input], outputs=[z_mean,z_log_var, z], name='encoder')
    encoder.add_loss(kl_loss)
    encoder.add_metric(kl_loss, name='kl_loss', aggregation='mean')

    # TODO: add gmm_KL_loss
    #encoder.add_loss(GMM_kl_loss)
    #encoder.add_metric(GMM_kl_loss, name='GMM_kl_loss', aggregation='mean')

    #encoder.summary()
    
    return encoder#, conv_shape
    

def create_decoder(waveform_dim,latent_dim):    
    '''
    TODO:
    Description....

    Parameters
    ----------
    a : 

    Returns
    -------
    out : 

    See Also
    --------
        Ref to similar funtions...
    
    Notes
    -----
    More mathematicall description

    References
    ----------
    .. [1] Wikipedia, "Convolution", http://en.wikipedia.org/wiki/Convolution.
    Examples
    --------
    blabla
    >>> np.convolve([1, 2, 3], [0, 1, 0.5])
    array([ 0. ,  1. ,  2.5,  4. ,  1.5])
    Only return the middle values of the convolution.
    '''

    d_i = layers.Input(shape=(latent_dim, ), name='decoder_input')
    x   = keras.layers.Dense(20 , activation='relu')(d_i)
    x   = keras.layers.Dense(100, activation='relu')(x)
    x   = keras.layers.BatchNormalization()(x)
    x   = keras.layers.Dense(100, activation='relu')(x)
    x   = keras.layers.BatchNormalization()(x)
    #x   = keras.layers.Dense(100, activation='relu')(x)
    #x   = keras.layers.BatchNormalization()(x)
    o   = keras.layers.Dense(waveform_dim, activation=None)(x)
    #x   = keras.layers.Reshape((waveform_dim, 1))(x)
    #cx  = keras.layers.Conv1DTranspose(filters=16, kernel_size=2, strides=1, padding='same', activation='relu')(x)
    #cx  = keras.layers.BatchNormalization()(cx)
    #o  = keras.layers.Conv1DTranspose(filters=8, kernel_size=2, strides=2, padding='same',  activation='relu')(cx)
    #cx  = keras.layers.BatchNormalization()(cx)
    #x = keras.layers.Conv1DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)
    #o = keras.layers.Dense(141,activation='relu')(x)
    decoder = keras.Model(inputs=[d_i],outputs=o, name='decoder')
    return decoder
    
def get_vae(waveform_dim,latent_dim):
    '''
    TODO:
    Description....

    Parameters
    ----------
    a : 

    Returns
    -------
    out : 

    '''
    encoder_input = layers.Input(shape=(waveform_dim, 1), name='encoder_input')
    encoder = create_encoder(encoder_input, latent_dim)
    decoder = create_decoder(waveform_dim, latent_dim)
    
    reconstruction = decoder(encoder(encoder_input)[2])
    vae = keras.Model(inputs=encoder_input, outputs=reconstruction, name='vae')
    vae.add_loss(encoder.losses)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    #reconstruction_loss = tf.keras.losses.MeanSquaredError(reconstruction, inp_layer)
    #vae.add_loss(reconstruction_loss)
    
    vae.compile(optimizer,loss=reconstruction_loss)
    #vae.summary()
    return encoder,decoder,vae

def reconstruction_loss(data,target):
    '''
    TODO:
    Description....

    Parameters
    ----------
    a : 

    Returns
    -------
    out : 

    '''
    reconstruction_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data, target)
            )
    reconstruction_loss *= 141
    return reconstruction_loss

    

if __name__ == "__main__":
    '''
    ######## TESTING: #############
    Shape of waveforms: (136259, 141).
    Shape of timestamps: (136259, 1).
    OBS takes about 6.4 milliseconds to call "get_event_rates()" (mean of 100 runs)
    '''

    import numpy as np
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    import time
    from plot_functions_wf import *
    print()
    print('Loading matlab files...')
    print()
    wf_name = '../matlab_files/gg_waveforms-R10_IL1B_TNF_03.mat'
    ts_name = '../matlab_files/gg_timestamps.mat'

    waveforms = loadmat(wf_name)
    waveforms = waveforms['waveforms']
    #timestamps = loadmat(ts_name)['gg_timestamps']
    print('MATLAB files loaded succesfully...')
    print()
    print(f'Shape of waveforms: {waveforms.shape}.')
    '''
    print()
    print(f'Shape of timestamps: {timestamps.shape}.')
    print()
    #assert waveforms.shape[0] == timestamps.shape[0], 'Missmatch of waveforms and timestamps shape.'
    # CREATE LABELS FOR TESTS
    labels = np.zeros((waveforms.shape[0]))
    first_injection_time = 30*60
    second_injection_time = 60*60

    labels[timestamps[:,0] < first_injection_time] = 1;
    labels[(first_injection_time < timestamps[:,0]) & (timestamps[:,0] < second_injection_time)] = 2;
    labels[timestamps[:,0] > second_injection_time] = 3;
    '''
    # ------------------------------------------------------------------------------------
    # --------------------- TEST FUNCTIONS: ----------------------------
    # ------------------------------------------------------------------------------------
    Train = False
    continue_train = False
    nr_epochs = 10
    saved_weights = 'models_tests/first_test'
    save_figure = 'figures_tests/wf_latent_decoded'
    waveform_dim = waveforms.shape[-1]
    xx = waveforms[:136000,:]
    # Standardize input data.
    mean = np.mean(xx, axis=-1)
    std  = np.std(xx, axis=-1)  
    xx_mean_shifted = xx - mean[:,None]
    xx_standardized = xx/std[:,None]

    x_train = xx_standardized.reshape((xx.shape[0],xx.shape[1],1))    
    
    encoder,decoder,vae = get_vae(waveform_dim,2)
    #x_train = np.
    #x_train = tf.linalg.normalize(x_train, axis=-1)
    #print(x_train[0:300,:])
    if Train == True:
        history = vae.fit(x_train.reshape((xx.shape[0],xx.shape[1])) , x_train.reshape((xx.shape[0],xx.shape[1])), epochs=nr_epochs, batch_size=128,verbose=1)
        vae.save_weights(saved_weights)
        plt.plot(history.history['loss'])
        plt.show() 
    else:
        print()
        print(f'Loading {saved_weights}...')
        print()
        vae.load_weights(saved_weights)
        if continue_train == True:
            print()
            print(f'Continue training for {nr_epochs} epochs...')
            print()
            history = vae.fit(x_train , x_train, epochs=nr_epochs, batch_size=128,verbose=1)
            vae.save_weights(saved_weights)
            plt.plot(history.history['loss'])
            plt.show() 
    print()
    print(f'Visualising decoded latent space...')
    print()
    plot_decoded_latent(decoder,saveas=save_figure+'_decoded',verbose=1)
    plot_encoded(encoder, x_train, saveas=None, verbose=1)
    
