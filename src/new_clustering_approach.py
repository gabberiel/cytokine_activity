'''

USED FOR TESTS.. 

Compare results with a dimentionallity reduction step folled by clustering such as DBSCAN.

'''

import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt


def runTSNE(waveforms, reduced_dim=2, verbose=False):
    '''
    Runs t-SNE on waveforms and returns the low-dimensional representation.

    args:
    -----
    waveforms : (n_wf, dim_wf) array_like
    
    returns:
    --------
    tsne_wf : (n_wf, reduced_dim) numpy_array

    '''
    print('Running T-sne...')
    wf_embedded = TSNE(n_components=reduced_dim, perplexity=60, verbose=1).fit_transform(waveforms)

    if verbose:
        plt.scatter(wf_embedded[::10,0], wf_embedded[::10,1])
        plt.xlabel('t-sne dim1')
        plt.ylabel('t-sne dim2')
        plt.show()

    return wf_embedded

def runDBSCAN(wf_embedded, db_eps=2, db_min_sample=2):
    '''
    Runs DBSCAN on t-SNE embedded waveforms.


    '''
    # db_eps = hypes['DBSCAN']['db_eps']
    # db_min_sample = hypes['DBSCAN']['db_min_sample']
    dbscan = DBSCAN(eps=db_eps, min_samples=db_min_sample, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
    dbscan.fit(wf_embedded)
    labels = dbscan.labels_
    

    return labels


'''
# #####################%%%%%%%%%%%%¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤###################
# TODO: TESTS OF T-SNE + DBSCAN Approach TODO TODO TODO TODO TODO TODO
# Runs t-SNE dimensionallity reduction on waveforms and clusters the 
# 2-dim manyfold using DBSCAN. 

tsne_wf = runTSNE(wf0[0:55000, :], reduced_dim=2, verbose=True)
# tsne_wf = np.load('aa.npy')
persistent_hom = []
eps_range =  np.arange(1.5, 10, 0.2)
for epsilon in eps_range:
    labels = runDBSCAN(tsne_wf,  db_eps=epsilon, db_min_sample=100)
    n_clusters = np.sum(np.unique(labels)) - 1
    persistent_hom.append( n_clusters )
plt.plot(eps_range, persistent_hom)
plt.show()
event_rate_results = get_event_rates(timestamps[0:55000], hypes, labels=labels, consider_only=None)
plot_event_rates(event_rate_results, timestamps[0:55000], conv_width=10)    

# #####################%%%%%%%%%%%%¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤###################
'''