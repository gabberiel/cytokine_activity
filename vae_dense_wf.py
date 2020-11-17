# BUILD MODEL
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers



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
    Layer instance which returns the the Kullbeck-Lieberg Divergence.
    
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
    #e_i    = layers.Input(shape=(waveform_shape, ), name='encoder_input')
    x      = layers.Dense(120, activation='relu')(encoder_input)
    x      = layers.BatchNormalization()(x)
    x      = layers.Dense(80, activation='relu')(x)
    x      = layers.BatchNormalization()(x)
    x      = layers.Dense(40, activation='relu')(x)
    x      = layers.BatchNormalization()(x)
    z_mean      = layers.Dense(latent_dim, name='latent_mu')(x)
    z_log_var   = layers.Dense(latent_dim, name='latent_sigma')(x)

    z = sample_z(z_mean,z_log_var)
    kl_loss = Add_kl_loss()([z_mean,z_log_var])

    # Instantiate encoder
    encoder = keras.Model(inputs=[encoder_input], outputs=[z_mean,z_log_var, z], name='encoder')
    encoder.add_loss(kl_loss)
    encoder.add_metric(kl_loss, name='kl_loss', aggregation='mean')
    #encoder.summary()
    
    return encoder
    

def create_decoder(waveform_shape,latent_dim):    
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

    d_i    = layers.Input(shape=(latent_dim, ), name='decoder_input')
    x      = layers.Dense(40, activation='relu')(d_i)
    x      = layers.BatchNormalization()(x)
    x      = layers.Dense(80, activation='relu')(x)
    x      = layers.BatchNormalization()(x)
    x      = layers.Dense(120, activation='relu')(x)
    x      = layers.BatchNormalization()(x)
    o      = layers.Dense(waveform_shape, activation=None)(x)

    decoder = keras.Model(inputs=[d_i],outputs=o, name='decoder')
    return decoder
    
def get_vae(waveform_shape,latent_dim):
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
    encoder_input = layers.Input(shape=(waveform_shape, ), name='encoder_input')
    encoder = create_encoder(encoder_input, latent_dim)
    decoder = create_decoder(waveform_shape, latent_dim)
    
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
    >>> ...
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
    wf_name = 'matlab_files/gg_waveforms-R10_IL1B_TNF_03.mat'
    ts_name = 'matlab_files/gg_timestamps.mat'

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
    nr_epochs = 50
    saved_weights = 'models/first_test'
    save_figure = 'figures/wf_latent_decoded'
    waveform_shape = waveforms.shape[-1]
    encoder,decoder,vae = get_vae(waveform_shape,2)
    xx = waveforms[1:30000,:]
    # Standardize input data.
    mean = np.mean(xx, axis=-1)
    std  = np.std(xx, axis=-1)  
    xx_mean_shifted = xx - mean[:,None]
    xx_standardized = xx/std[:,None]

    x_train = xx_standardized    
    #x_train = np.
    #x_train = tf.linalg.normalize(x_train, axis=-1)
    #print(x_train[0:300,:])
    if Train == True:
        history = vae.fit(x_train , x_train, epochs=nr_epochs, batch_size=128,verbose=1)
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
    plot_decoded_latent_2(decoder,saveas=save_figure+'_decoded',verbose=1)
    plot_label_clusters(encoder, x_train, saveas=None, verbose=1)


