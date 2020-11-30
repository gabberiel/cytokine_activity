import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Input, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras.models import Model

## PREPARE DATA
'''
(x_train, y_train),(x_test, y_test) = mnist.load_data()
zerInd = np.where(np.any([y_train == 0, y_train == 1,y_train == 2], axis = 0))
zerIndTest = np.where(np.any([y_test == 0, y_test == 1, y_test == 2],axis = 0))

x_train = x_train.reshape(x_train.shape + (1,)).astype('float32') / 255.
x_test = x_test.reshape(x_test.shape + (1,)).astype('float32') / 255.
x_train4 = x_train[zerInd]
x_test4 = x_test[zerIndTest]

batch_size = 100
train_dataset = tf.data.Dataset.from_tensor_slices(x_train4).shuffle(x_train4.shape[0]).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(x_test4).shuffle(x_test4.shape[0]).batch(batch_size)
'''
## MODEL

class VAE(tf.keras.Model):
    
    def reparameterize(mean,log_var):
        eps = tf.random.normal(shape=mean.shape)
        return eps*tf.exp(log_var * 0.5) + mean
    
    @tf.function
    def calc_loss(self,x):
        mean, logvar = self.transform(x)
        z = VAE.reparameterize(mean,logvar)
        x_rec = self.generate(z)
        
        if self.waveforms:
            rec_loss = 0.5*tf.reduce_sum(tf.math.squared_difference(x_rec, x), axis = [1,2])
        else:
            rec_loss = 0.5*tf.reduce_sum(tf.math.squared_difference(x_rec, x), axis = [1,2,3])
        #rec_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = x_rec, labels = x), axis=[1,2,3])
        kl_loss = 0.5*tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - logvar -1,axis = 1)
        return tf.reduce_mean(rec_loss + self.beta*kl_loss)
    
    @tf.function
    def apply_gradient(self,x):
        with tf.GradientTape() as tape:
            loss = self.calc_loss(x)
        grad = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad,self.trainable_variables))
        
    def __init__(self, latent_dim= 2, beta = 1, batch_size = 32, arch_type = 2, signal_shape = (28,28,1), waveforms = False):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.batch_size = batch_size
        self.signal_shape = signal_shape
        self.optimizer = tf.keras.optimizers.Adam()
        self.arch_type = arch_type
        self.waveforms = waveforms
        
        if arch_type == 1:
            self.encoder = self.create_encoder_CNN()
            self.decoder = self.create_decoder_CNN()
        elif arch_type == 2:
            self.encoder = self.create_encoder_dense()
            self.decoder = self.create_decoder_dense()
        elif arch_type == 3:
            self.encoder = self.create_encoder_CNN_pool()
            self.decoder = self.create_decoder_CNN_pool()
        else:
            self.encoder = self.create_encoder_CNN()
            self.decoder = self.create_decoder_CNN()
        
    def create_encoder_CNN(self):
        inp = Input(shape=self.signal_shape)
        conv_1 = Conv2D(filters = 32, kernel_size=3, strides = (2,2), activation='relu')(inp)
        conv_2 = Conv2D(filters = 64, kernel_size=3, strides = (2,2), activation='relu')(conv_1)
        fl = Flatten()(conv_2)
        mean_out = Dense(self.latent_dim)(fl)
        log_var_out = Dense(self.latent_dim)(fl)
        
        enc_model = Model(inp, [mean_out,log_var_out])
        return enc_model
        
    def create_decoder_CNN(self):
        inp_c = Input(shape=(self.latent_dim,))
        den = Dense(units=7*7*32, activation='relu')(inp_c)
        rs = Reshape(target_shape=(7,7,32))(den)
        convt_1 = Conv2DTranspose(filters = 32, kernel_size=3, strides = (2,2), padding = 'same', activation='relu')(rs)
        convt_2 = Conv2DTranspose(filters = 64, kernel_size=3, strides = (2,2),padding = 'same', activation='relu')(convt_1)
        image_out = Conv2DTranspose(filters=1,kernel_size=3,strides=(1,1),padding = 'same')(convt_2)
        dec_model = Model(inp_c, image_out)
        return dec_model
    
    
    def create_encoder_dense(self):
        inp_d = Input(shape=self.signal_shape)
        fl = Flatten()(inp_d)
        den_1 = Dense(100,activation='relu')(fl)
        den_2 = Dense(100, activation='relu')(den_1)
        den_3 = Dense(100, activation='relu')(den_2)
        mean_out = Dense(self.latent_dim)(den_3)
        log_var_out = Dense(self.latent_dim)(den_3)
        
        enc_model = Model(inp_d, [mean_out,log_var_out])
        return enc_model
        
    def create_decoder_dense(self):
        inp_d = Input(shape=(self.latent_dim,))
        den_1 = Dense(units=100, activation='relu')(inp_d)
        den_2 = Dense(100, activation='relu')(den_1)
        den_3 = Dense(100, activation='relu')(den_2)
        if self.waveforms:
            den_4 = Dense(141)(den_3)
        else:
            den_4 = Dense(28*28)(den_3)
        image_out = Reshape(target_shape=self.signal_shape)(den_4)
        dec_model = Model(inp_d, image_out)
        return dec_model
    
    def create_encoder_CNN_pool(self):
        inp = Input(shape=self.signal_shape)
        conv_1 = Conv2D(filters = 32, kernel_size=3, strides = (1,1),padding = 'same', activation='relu')(inp)
        conv_2 = Conv2D(filters = 64, kernel_size=3, strides = (1,1),padding = 'same', activation='relu')(conv_1)
        mp1 = MaxPool2D(pool_size=(2,2), padding='valid')(conv_2)
        conv_3 = Conv2D(filters = 64, kernel_size=3, strides = (1,1),padding = 'same', activation='relu')(mp1)
        conv_4 = Conv2D(filters = 64, kernel_size=3, strides = (1,1),padding = 'same', activation='relu')(conv_3)
        mp2 = MaxPool2D(pool_size=(2,2), padding='valid')(conv_4)
        fl = Flatten()(mp2)
        mean_out = Dense(self.latent_dim)(fl)
        log_var_out = Dense(self.latent_dim)(fl)
        
        enc_model = Model(inp, [mean_out,log_var_out])
        return enc_model
        
    def create_decoder_CNN_pool(self):
        inp_c = Input(shape=(self.latent_dim,))
        den = Dense(units=7*7*32, activation='relu')(inp_c)
        rs = Reshape(target_shape=(7,7,32))(den)
        convt_1 = Conv2DTranspose(filters = 32, kernel_size=3, strides = (2,2), padding = 'same', activation='relu')(rs)
        convt_2 = Conv2DTranspose(filters = 64, kernel_size=3, strides = (1,1), padding = 'same', activation='relu')(convt_1)
        convt_3 = Conv2DTranspose(filters = 64, kernel_size=3, strides = (2,2),padding = 'same', activation='relu')(convt_2)
        convt_4 = Conv2DTranspose(filters = 64, kernel_size=3, strides = (1,1),padding = 'same', activation='relu')(convt_3)
        image_out = Conv2DTranspose(filters=1,kernel_size=3,strides=(1,1),padding = 'same')(convt_4)
        dec_model = Model(inp_c, image_out)
        return dec_model
    
    def generate(self,z,sigmoid = False):
        return self.decoder(z)
    
    def transform(self,images):
        return self.encoder(images)
    
    def sample_image(self,n):
        z_sample = tf.random.normal(shape=(n,self.latent_dim))
        return self.generate(z_sample)
    
    def train(self,epochs,x_dataset,x_test):
        
        for ep in range(1,epochs+1):
            for x_batch in x_dataset:
                self.apply_gradient(x_batch)
            loss = tf.keras.metrics.Mean()
            for test_x in x_test:
                loss(self.calc_loss(test_x))
            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, '.format(ep,elbo))
                
    def kl_loss(self,x):
        mean, logvar = self.transform(x)
        z = VAE.reparameterize(mean,logvar)
        x_rec = self.generate(z)
        
        rec_loss = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = x_rec, labels = x), axis=[1,2,3])
        kl_loss = -0.5*tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - logvar -1,axis = 1)
        return kl_loss, mean, logvar
    
    def save_model(self,name):
        string_enc = name + '_{}d_encoder.h5'.format(self.latent_dim)
        string_dec = name + '_{}d_decoder.h5'.format(self.latent_dim)
        self.encoder.save(string_enc)
        self.decoder.save(string_dec)
    
    def load_model(self,name):
        string_enc = name + '_{}d_encoder.h5'.format(self.latent_dim)
        string_dec = name + '_{}d_decoder.h5'.format(self.latent_dim)
        self.encoder = tf.keras.models.load_model(string_enc, compile=False)
        self.decoder = tf.keras.models.load_model(string_dec, compile=False)

## RUN PROGRAM
'''
vae = VAE(latent_dim=2,arch_type = 2)
vae.train(50,train_dataset,test_dataset)
plt.plot(vae.sample_image(1).numpy().reshape((28,28)))
plt.show()
'''

