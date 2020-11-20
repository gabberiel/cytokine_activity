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

class Add_GMM_kl_loss(layers.Layer):
    '''
    Layer instance which returns the the Kullbeck-Lieberg Divergence between latent distribution q(z|x) and GMM distribution p(z\gmm_params).

    '''
    def call(self,inputs):
        z_mean_vae, z_mean_gmm, z_log_var_vae, z_log_var_gmm = inputs

        gmm_kl_loss = - (tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1))
        #self.add_loss(kl_loss + 27)
        #self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        return gmm_kl_loss

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

    # CALCULATE LOSSES:
    kl_loss = Add_kl_loss()([z_mean,z_log_var])

    # TODO: test to include GMM KL-loss:
    # 1. Update gmm means and variances using fit:
    # GMM.fit(data) --- will have to change the class-train function..?
    # z_mean_gmm = GMM.get_gaussian_means()
    # z_var_gmm = GMM.get_gaussian_var()
    # z_log_var_gmm = ... 
    
    # 2. GMM_kl_loss = Add_GMM_kl_loss()([z_mean, z_mean_gmm, z_log_var, z_log_var_gmm])
    

    # Instantiate encoder
    encoder = keras.Model(inputs=[encoder_input], outputs=[z_mean,z_log_var, z], name='encoder')
    encoder.add_loss(kl_loss)
    encoder.add_metric(kl_loss, name='kl_loss', aggregation='mean')

    # TODO: add gmm_KL_loss
    #encoder.add_loss(GMM_kl_loss)
    #encoder.add_metric(GMM_kl_loss, name='GMM_kl_loss', aggregation='mean')

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

class Gmm_Vae_Model(keras.Model):
    '''keras.Model subclass including GMM-training and loss in train.'''
    def __init__(self,encoder,decoder,gmm):
        super(Gmm_Vae_Model, self).__init__(name='Gmm_Vae') #name='Gmm_Vae', **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.gmm = gmm

    '''
    def call(self,input):
        z_mean,z_log_var,zc = self.encoder(input)
        X_rec = self.decoder(zc)
        return X_rec
    '''

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # KL-loss to prior of z:
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var) )

            # GMM_update and corresponding kl_loss
            gmm.fit(z) # Potentially time consuming...
            gmm_labels = gmm.classify(z)
            gmm_var = gmm.get_gaussian_var()
            gmm_means = gmm.get_gaussian_means()
            acces_all_rows = np.arange(len(labels))
            z_mean_gmm = gmm_means[acces_all_rows,labels] # Correct syntax..?
            z_var_gmm = gmm_var[acces_all_rows,labels]

            # add float to denominatior to avoid inf..
            gmm_kl_loss = -(0.5 + z_log_var - tf.log(z_var_gmm) + (z_log_var_gmm + tf.square(z_mean_gmm - z_mean))/(2*tf.exp(z_log_var) + 1e-3))
            gmm_kl_loss = - (tf.reduce_mean(gmm_kl_loss))

            reconstruction_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data, reconstruction)
            )
            #reconstruction_loss *= 141

            total_loss = reconstruction_loss + kl_loss + gmm_kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "gmm_kl_loss": gmm_kl_loss
        }



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

    '''
    encoder_input = layers.Input(shape=(waveform_shape, ), name='encoder_input')
    encoder = create_encoder(encoder_input, latent_dim)
    decoder = create_decoder(waveform_shape, latent_dim)
    
    reconstruction = decoder(encoder(encoder_input)[2])
    gmm = Latent_GMM(num_components=3, covariance_type='diag')
    vae = keras.Model(encoder,decoder,gmm,inputs=encoder_input, outputs=reconstruction, name='vae')
    vae.add_loss(encoder.losses)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    #reconstruction_loss = tf.keras.losses.MeanSquaredError(reconstruction, inp_layer)
    #vae.add_loss(reconstruction_loss)
    
    vae.compile(optimizer) #,loss=reconstruction_loss)
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

class Latent_GMM():
    ''' 
    Gaussian mixture model of latent space encoded by VAE.
    Makes use of sklearn.mixture.GaussianMixture or sklearn.mixture.BayesianGaussianMixture
    
    Used to classify/label waveforms. The probability model is furthermore used to include 
    a loss-term in the VAE-optimization as a KL-divergence: KL( p(z|gmm_params) || q(z|x) )

    Attributes
    ----------

    Methods
    -------
        fit(data) : 
        get_gaussian_means() : 
        get_gaussian_var() :
        soft_assignment() : 
        classify() : 
    References
    ----------
        Sckit documentation:
            https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
            https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture


     '''
    def __init__(self,num_components=5,covariance_type='diag'):
        self.n_components = num_components
        self.covariance_type = covariance_type
        self.model = GaussianMixture(n_components=self.n_components, 
                                                     covariance_type=self.covariance_type,
                                                     warm_start=True)

    def fit(self, data):
        self.model.fit(data)
    
    def get_gaussian_means(self,):
        '''
        Get mean-vector of each gaussian component in model.

        Returns
        -------
            means : (n_components, n_features) array-like
                The mean of each mixture component.

        '''
        
        return self.model.means_

    def get_gaussian_var(self,):
        '''
        OBS: The shape depends on covariance_type
        '''
        return self.model.covariances_

    def soft_assignment(self,data_points):
        '''
        Predict posterior probability of each component given the data.

        Parameters
        ----------
            data_points : (n_samples, n_features) array_like

        Returns
        -------
            X : array, shape:(n_samples, n_features)
                Randomly generated sample

            y : array, shape (nsamples,)
                Component labels
        '''
        cluster_probs = self.model.predict_proba(data_points)
        return cluster_probs
        

    def classify(self, data_points):
        '''
        Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
            data_points : (n_samples, n_features) array_like

        Returns
        -------
            labels : (n_points,) array_like
        '''
        labels = self.model.predict(data_points)
        return labels
    

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
    nr_epochs = 50
    saved_weights = 'models_tests/first_test'
    save_figure = 'figures_tests/wf_latent_decoded'
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
    #print()
    #print(f'Visualising decoded latent space...')
    #print()
    #plot_decoded_latent(decoder,saveas=save_figure+'_decoded',verbose=1)
    #plot_encoded(encoder, x_train, saveas=None, verbose=1)
    z_mean,_,zz = encoder.predict(x_train)

    gmm_components = 5
    Gmm = Latent_GMM(gmm_components,covariance_type='full')
    Gmm.fit(zz)
    g_means = Gmm.get_gaussian_means()
    g_var = Gmm.get_gaussian_var()
    probs = Gmm.soft_assignment(zz)
    labels = Gmm.classify(zz)

    def plot_gmm(data,labels):
        print()
        print(f'Visualising gaussian mixture model...')
        print()
        for ii in range(gmm_components):
            plt.scatter(data[labels==ii][::2,0],data[labels==ii][::2,1])
        
        plt.show()

    plot_gmm(zz,labels)

    print('Done.')
