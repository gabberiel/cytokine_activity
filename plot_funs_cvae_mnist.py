
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
def plot_latent(encoder, decoder, saveas=None,verbose=0, conditional=None):
    # display a n*n 2D manifold of digits
    n = 30
    digit_size = 28
    scale = 2.0
    figsize = 10
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            #rows = np.ones((conditional.shape[0],1))
            #z_sample = rows*z_sample
            if conditional is not None:
                x_decoded = decoder([z_sample,conditional])
                x_decoded = x_decoded.numpy()
            else:
                x_decoded = decoder(z_sample)
                x_decoded = x_decoded.numpy()
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range 
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose>0:
        plt.show()

def plot_label_clusters(encoder, decoder, data, inp_labels,labels, saveas=None,verbose=0):
    '''
    Inputs:
        data:
        inp_labels:
        labels: 
    '''
    # display a 2D plot of the digit classes in the latent space
    print('hej')
    z_mean, _, _ = encoder([data,inp_labels])
    z_mean = z_mean.numpy()

    plt.figure(figsize=(10, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose>0:
            plt.show()

def plot_1d_latent_dist(encoder,data,labels, saveas=None,verbose=0):
        # OBS completely inefficient... would be sufficient with one forward pass
        # But did not work...
        NN=200
        z_samples = np.empty(shape=(NN,10))
        #print(z_samples.shape)
        for i in range(10):
            i_idx = np.where(labels==i)[0][0] 
            X = data[i_idx,:,:,:]
            X = np.reshape(X,(1,28,28,1))
            NN=200
            for jj in range(NN):
                _,_,z = encoder(X)
                z_samples[jj,i] = z[0]
        
        mu = 0
        variance = 1
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma),label='Prior')
        for i in range(10):
            plt.hist(z_samples[:,i], density=True, bins=10,label=f'x={i}')
        plt.title('Latent Space Distributions of Test Data')
        plt.legend()
        if saveas is not None:
            plt.savefig(saveas, dpi=150)
        if verbose>0:
            plt.show()