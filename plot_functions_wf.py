import numpy as np
import matplotlib.pyplot as plt

# VERISON USED FOR ARAMS CODE ON WAVEFORM DATA
def plot_decoded_latent(decoder,resolution=6,saveas=None, verbose=1):
    '''
    Plots (resolution x resolution) sized grid in of waveform from the
    corresponding latent variable in the range 



    Notes
    -----
    Sample (z1,z2) in grid [-2,2] x [-2,2] . 

    z1,z2 in R^2 --- decoder --> waveform in R^141

    '''
    print()
    print('Constructing plot of decoded latent space...')
    print()
    fig = plt.figure(constrained_layout=True)
    resolution = 6
    spec = fig.add_gridspec(ncols=resolution+1, nrows=resolution+1)
    for i,z1 in enumerate(np.arange(-2,2 + 4/resolution,4/resolution)):
        for j, z2 in enumerate(np.arange(-2,2 + 4/resolution,4/resolution)):
            ax = fig.add_subplot(spec[j, i])
            latent_z = np.array((z1,-z2))
            #print(latent_z)
            latent_z = np.reshape(latent_z, (1,2))
            reconstructed = decoder.predict(latent_z)
            ax.plot(reconstructed[0,0:-1:5])
            ax.set(xticks=[], yticks=[])
            if j == resolution:
                plt.xlabel(np.round(z1, decimals=1))
            if i == 0:
                plt.ylabel(np.round(-z2, decimals=1))
    for ax in fig.get_axes():
        ax.spines['top'].set_visible(ax.is_first_row())
        ax.spines['bottom'].set_visible(ax.is_last_row())
        ax.spines['left'].set_visible(ax.is_first_col())
        ax.spines['right'].set_visible(ax.is_last_col())
    plt.savefig(saveas, dpi=150)
    #plt.title('Decoded Latent Space')
    plt.show()


def plot_encoded(encoder, data, saveas=None,verbose=1):
    '''
    Display a 2D plot of the latent space mean. 
    Will show plot if verbose>0.
    Will save figure if saveas is a valid path.

    Parameters
    ----------
    encoder : keras.Model class_instance
        Encoder part of VAE
    data : (num_of_wf, size_of_wf) array_like
        Data to be visualised in laten space.
    saveas : 'path/to/save_fig' string_like
        if None then the figure is not saved
    verbose : integer_like
        verbose>0 => plt.show()

    '''
    z_mean, log_var_out, z  = encoder.predict(data)
    
    assert z_mean.shape[-1] == 2, 'PLOT ONLY WORK FOR 2D LATENT SPACE'
    plt.figure(figsize=(10, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1])#, c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose>0:
            plt.show()
