'''
Conditional Variational Autoencoder using dense architechture.

main function: 
 * get_cvae()
    (hypes) --> (keras.Model)

'''
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
# from sklearn.mixture import GaussianMixture
from tensorflow.keras import backend as K

# Supress tensorflow warnings..:
#tf.logging.set_verbosity(tf.logging.ERROR) najn

def sample_z(mu, sigma):
    '''
    Reparametrisation step of encoder to consider the stochastic sampling of latent variable
    as an input to decoder network -- enabeling backprop.

    '''
    batch     = K.shape(mu)[0]
    dim       = K.int_shape(mu)[1]
    eps       = K.random_normal(shape=(batch, dim))
    return mu + K.exp(sigma / 2) * eps

class Add_kl_loss(layers.Layer):
    '''
    Layer instance which returns the the Kullbeck-Lieberg Divergence
    between latent distribution q(z|x) and prior p(z)= N(0,I).
    '''
    def call(self, inputs):
        z_mean, z_log_var = inputs
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        return kl_loss


def __create_encoder__(encoder_input, inp_label, hypes):
    '''
    Initiates encoder of conditional variational autoencoder.

    Called by "get_cvae()"

    Parameters
    ----------
    encoder_input : keras.layers.Input - type
        Input layer of CVAE

    inp_label : keras.layers.Input - type
        Input label to condition on in the conditional VAE.
    hypes : dict.
        hyperparams.

    Returns
    -------
    out : keras.Model
        Dense encoder model of CVAE

    '''
    latent_dim = hypes["cvae"]["latent_dim"]
    dense_net_nodes = hypes["cvae"]["dense_net_nodes"]
    activation = hypes["cvae"]["activation"]

    x = layers.Concatenate(axis=1)([encoder_input, inp_label])
    for num_nodes in dense_net_nodes:
        x = layers.Dense(num_nodes, activation=activation)(x)
        x = layers.BatchNormalization()(x)
    z_mean = layers.Dense(latent_dim, name='latent_mu')(x)
    z_log_var = layers.Dense(latent_dim, name='latent_sigma')(x)

    z = sample_z(z_mean, z_log_var)

    # CALCULATE LOSSES:
    kl_loss = Add_kl_loss()([z_mean, z_log_var])

    # Initiate encoder
    encoder = keras.Model(inputs=[encoder_input,inp_label], outputs=[z_mean, z_log_var, z], name='encoder')
    encoder.add_loss(kl_loss)
    encoder.add_metric(kl_loss, name='kl_loss', aggregation='mean')

    # encoder.summary()
    
    return encoder


def __create_decoder__(waveform_dim, inp_label, hypes):    
    '''
    Initiates decoder of conditional variational autoencoder.

    Called by "get_cvae()"

    Parameters
    ----------
    wavefrom_dim : Integer - type
        Number of features in each waveform. To be used in reconstructed output
    inp_label : keras.layers.Input - type
        Input label to condition on in the conditional VAE.
    hypes : 
    laten_dim : Integer 
        dimension of latent space

    Returns
    -------
    out : keras.Model
        Dense decoder model of CVAE

    '''
    latent_dim = hypes["cvae"]["latent_dim"]
    dense_net_nodes = hypes["cvae"]["dense_net_nodes"]
    activation = hypes["cvae"]["activation"]

    d_i = layers.Input(shape=(latent_dim, ), name='decoder_input')
    x = layers.Concatenate(axis=1)([d_i, inp_label])

    for num_nodes in dense_net_nodes[::-1]:
        x = layers.Dense(num_nodes, activation=activation)(x)
        x = layers.BatchNormalization()(x)

    o = layers.Dense(waveform_dim, activation=None)(x)

    decoder = keras.Model(inputs=[d_i, inp_label], outputs=o, name='decoder')
    return decoder
    
def get_cvae(hypes):
    '''
    Builds a Conditional Autoencoder with "dense" input layer with dimension specified by waveform_dim.\\
    Conditiones input with label of dimension label_dim.

    Parameters
    ----------
    wavefrom_dim : Integer - type
        Number of features in each waveform. To be used in reconstructed output
    laten_dim : Integer 
        dimension of latent space
    label_dim : Integer
        Dimension of label to condition on in the conditional VAE.

    Returns
    -------
    encoder : keras.Model
        Dense encoder model of CVAE
    decoder : keras.Model
        Dense decoder model of CVAE
    cvae : keras.Model
        Full cvae model to be trained.
    '''
    global model_variance
    model_variance = hypes["cvae"]["model_variance"]
    waveform_dim = hypes["preprocess"]["dim_of_wf"]
    label_dim = hypes["cvae"]["label_dim"]

    encoder_input = layers.Input(shape=(waveform_dim, ), name='encoder_input')
    inp_label = layers.Input(shape=(label_dim, ), name='ev_label_input')
    encoder = __create_encoder__(encoder_input, inp_label, hypes)
    decoder = __create_decoder__(waveform_dim, inp_label, hypes)
    
    reconstruction = decoder([encoder([encoder_input, inp_label])[2], inp_label])
    cvae = keras.Model(inputs=[encoder_input, inp_label], outputs=reconstruction, name='cvae')
    cvae.add_loss(encoder.losses) # Add KL-loss from encoder
    optimizer = tf.keras.optimizers.Adam()
    
    cvae.compile(optimizer, loss=__reconstruction_loss__)

    return encoder, decoder, cvae

def __reconstruction_loss__(data, target):
    '''
    Specifies reconstruction loss under multivariate normal assumption of the input data.
    
    Called when compiling cvae-keras.Model.
    
    Tensorflow requiers two inputs only and gives the inputs: data and target.

    Parameters
    ----------
    data : tensorflow object 
        output from tensorflow prediction
    target : tensorflow object
        Label for the corresponing input data.

    Returns
    -------
    out : scalar
        Loss considered in backpropagation. 

    '''
    variance = model_variance
    reconstruction_loss = (0.5 / variance) * tf.reduce_sum(
                keras.losses.mean_squared_error(data, target))
    return reconstruction_loss
