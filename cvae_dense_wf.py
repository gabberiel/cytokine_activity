# Conditional Variational Autoencoder using dense architechture
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


def create_encoder(encoder_input,latent_dim,inp_label):
    '''
    TODO:
    Description....

    Parameters
    ----------
    inp_layer : 

    Returns
    -------
    out : 

    '''
    #e_i    = layers.Input(shape=(waveform_shape, ), name='encoder_input')
    xi     = layers.Concatenate(axis=1)([encoder_input, inp_label])
    x      = layers.Dense(120, activation='relu')(xi)
    x      = layers.BatchNormalization()(x)
    x      = layers.Dense(100, activation='relu')(x)
    x      = layers.BatchNormalization()(x)
    x      = layers.Dense(80, activation='relu')(x)
    x      = layers.BatchNormalization()(x)
    z_mean      = layers.Dense(latent_dim, name='latent_mu')(x)
    z_log_var   = layers.Dense(latent_dim, name='latent_sigma')(x)

    z = sample_z(z_mean,z_log_var)

    # CALCULATE LOSSES:
    kl_loss = Add_kl_loss()([z_mean,z_log_var])


    # Instantiate encoder
    encoder = keras.Model(inputs=[encoder_input,inp_label], outputs=[z_mean,z_log_var, z], name='encoder')
    encoder.add_loss(kl_loss)
    encoder.add_metric(kl_loss, name='kl_loss', aggregation='mean')

    # TODO: add gmm_KL_loss
    #encoder.add_loss(GMM_kl_loss)
    #encoder.add_metric(GMM_kl_loss, name='GMM_kl_loss', aggregation='mean')

    #encoder.summary()
    
    return encoder


def create_decoder(waveform_shape,latent_dim,inp_label):    
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
    xi = layers.Concatenate(axis=1)([d_i, inp_label])
    x      = layers.Dense(80, activation='relu')(xi)
    x      = layers.BatchNormalization()(x)
    x      = layers.Dense(100, activation='relu')(x)
    x      = layers.BatchNormalization()(x)
    x      = layers.Dense(120, activation='relu')(x)
    x      = layers.BatchNormalization()(x)
    o      = layers.Dense(waveform_shape, activation=None)(x)

    decoder = keras.Model(inputs=[d_i,inp_label],outputs=o, name='decoder')
    return decoder
    
def get_cvae(waveform_shape,latent_dim,label_dim=3):
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
    encoder_input = layers.Input(shape=(waveform_shape, ), name='encoder_input')
    inp_label = layers.Input(shape=(label_dim, ), name='ev_label_input')
    encoder = create_encoder(encoder_input, latent_dim, inp_label)
    decoder = create_decoder(waveform_shape, latent_dim, inp_label)
    
    reconstruction = decoder([encoder([encoder_input,inp_label])[2],inp_label])
    cvae = keras.Model(inputs=[encoder_input,inp_label], outputs=reconstruction, name='cvae')
    cvae.add_loss(encoder.losses)
    optimizer = tf.keras.optimizers.Adam()
    #reconstruction_loss = tf.keras.losses.MeanSquaredError(reconstruction, inp_layer)
    #cvae.add_loss(reconstruction_loss)
    
    cvae.compile(optimizer,loss=reconstruction_loss)
    #cvae.summary()
    return encoder,decoder,cvae

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
    variance = 0.5

    reconstruction_loss = ( 0.5 / variance ) * tf.reduce_sum(
                keras.losses.mean_squared_error(data, target)
            )
    #reconstruction_loss *= 141 # Corresponding to assuming a variance of 0.5... Looks reasonable when simulating new waveforms with noise..
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
    continue_train = True
    nr_epochs = 30
    saved_weights = 'models_tests/Cvae_first_test_var=0.5'
    path_to_EVlabels = "../numpy_files/EV_labels/deleteme_25nov"
    save_figure = 'figures_tests/encoded_decoded/cvae_wf_latent_decoded_var05'


    ev_labels_wf = np.load(path_to_EVlabels+'.npy')
    waveform_shape = waveforms.shape[-1]
    encoder,decoder,cvae = get_cvae(waveform_shape,2)
    xx = waveforms[:130000,:]
    ev_labels = ev_labels_wf[:,:130000].T

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
        history = cvae.fit([x_train,ev_labels] , x_train, epochs=nr_epochs, batch_size=128,verbose=1)
        cvae.save_weights(saved_weights)
        plt.plot(history.history['loss'])
        plt.show() 
    else:
        print()
        print(f'Loading {saved_weights}...')
        print()
        cvae.load_weights(saved_weights)
        if continue_train == True:
            print()
            print(f'Continue training for {nr_epochs} epochs...')
            print()
            history = cvae.fit([x_train,ev_labels] , x_train, epochs=nr_epochs, batch_size=128,verbose=1)
            cvae.save_weights(saved_weights)
            plt.plot(history.history['loss'])
            plt.show() 
    
    
    consider_cluster = 0
    use_label = np.zeros((3,))

    for cluster in [0,1,2]:
        use_label = np.zeros((3,))
        use_label[cluster] = 1
        print()
        print(f'Visualising decoded latent space for label {use_label}...')
        print()
        ev_label = use_label.reshape((1,3))
        plot_decoded_latent(decoder,saveas=save_figure+'decoded_clsuter_'+str(cluster),verbose=0,ev_label=ev_label)
        x_same_label = x_train[ev_labels[:,cluster]==1,:]
        ev_label = np.ones((x_same_label.shape[0],1))*use_label
        plot_encoded(encoder, x_same_label, saveas=save_figure+'_encoded_clsuter_'+str(cluster), verbose=0,ev_label=ev_label)
    
    '''
    for jj in [10,212,3120,100000]:
        saveas = 'figures_tests/model_assessment/vae_wf_'+str(jj)
        x = x_train[jj,:]
        print(jj)
        x_rec = cvae.predict(x.reshape((1,141))).reshape((141,))
        time = np.arange(0,3.5,3.5/x.shape[0])

        plt.figure()
        plt.plot(time,x,color = (0,0,0),lw=1,label='$x$')
        plt.plot(time,x_rec,color = (1,0,0),lw=1,label='$\mu_x$')

        for i in range(4):
            x_sample = np.random.multivariate_normal(x_rec,np.eye(x.shape[0])*0.5)
            if i==0:
                plt.plot(time,x_sample,color = (0.1,0.1,0.1),lw=0.3, label='$x_{sim}$')
            else:
                plt.plot(time,x_sample,color = (0.1,0.1,0.1),lw=0.3)

            #plt.plot(time,waveforms[original_idx,:],color = (1,0,0),lw=1, label='Original')
        plt.xlabel('Time $(ms)$')
        plt.ylabel('Voltage $(\mu V)$')
        plt.title('Model Assessment: Simulating $x \sim \mathcal{N}(\mu_x,0.5 \mathcal{I} )$')
        plt.legend(loc='upper right')
        if saveas is not None:
            plt.savefig(saveas,dpi=150)
        plt.close()
        #plt.show()
    print('Done.')
    '''