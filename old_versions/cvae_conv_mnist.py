# BUILD MODEL
from tensorflow.keras import backend as K

def sample_z(mu,sigma):
    batch     = K.shape(mu)[0]
    dim       = K.int_shape(mu)[1]
    eps       = K.random_normal(shape=(batch, dim))
    return mu + K.exp(sigma / 2) * eps

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

def create_encoder(inp_layer,latent_dim,inp_label):
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

    cx      = layers.Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(inp_layer)
    cx      = layers.BatchNormalization()(cx)
    cx      = layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(cx)
    cx      = layers.BatchNormalization()(cx)
    x       = layers.Flatten()(cx)
    #print('tjo')
    xi      = layers.Concatenate(axis=1)([x, inp_label])
    #print('tjo')


    x       = layers.Dense(20, activation='relu')(xi)
    x       = layers.BatchNormalization()(x)
    z_mean      = layers.Dense(latent_dim, name='latent_mu')(x)
    z_log_var   = layers.Dense(latent_dim, name='latent_sigma')(x)
    # Used in Decoder:
    z = sample_z(z_mean,z_log_var)
    kl_loss = Add_kl_loss()([z_mean,z_log_var])
    # Get Conv2D shape for Conv2DTranspose operation in decoder
    conv_shape = K.int_shape(cx)
    
    # Instantiate encoder
    encoder = keras.Model(inputs=[inp_layer,inp_label], outputs=[z_mean,z_log_var, z], name='encoder')
    encoder.add_loss(kl_loss)
    encoder.add_metric(kl_loss, name='kl_loss', aggregation='mean')
    #encoder.summary()
    
    return encoder, conv_shape
    

def create_decoder(latent_dim,conv_shape,inp_label):    
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

    d_i   = layers.Input(shape=(latent_dim, ), name='decoder_input')
    inputs = layers.Concatenate(axis=1)([d_i, inp_label])

    x     = layers.Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(inputs)
    x     = layers.BatchNormalization()(x)
    x     = layers.Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
    cx    = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    cx    = layers.BatchNormalization()(cx)
    cx    = layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same',  activation='relu')(cx)
    cx    = layers.BatchNormalization()(cx)
    o     = layers.Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)
    
    decoder = keras.Model(inputs=[d_i,inp_label],outputs=o, name='decoder')
    return decoder
    
def get_vae(inp_layer,inp_label,latent_dim):
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

    encoder, conv_shape = create_encoder(inp_layer, latent_dim,inp_label)
    decoder = create_decoder(latent_dim, conv_shape,inp_label)
    
    reconstruction = decoder([encoder([inp_layer,inp_label])[2],inp_label])
    vae = keras.Model(inputs=[inp_layer,inp_label], outputs=reconstruction, name='vae')
    vae.add_loss(encoder.losses)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    #reconstruction_loss = tf.keras.losses.MeanSquaredError(reconstruction, inp_layer)
    #vae.add_loss(reconstruction_loss)
    
    vae.compile(optimizer,loss=reconstruction_loss)
    vae.summary()
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
    reconstruction_loss *= 28 * 28
    return reconstruction_loss

