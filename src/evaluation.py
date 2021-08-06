'''
Functions used to evaluate results after training is complete. 
The main file for "auto" - evaluation is "run_evaluation()".

The evaluation assumes that the the resulting .npy files from main_train.py exists.

'''
import numpy as np
import matplotlib.pyplot as plt
import preprocess_wf 
import json
from load_and_GD_funs import load_mat_file
from os import scandir
from wf_similarity_measures import wf_correlation, similarity_SSQ
from event_rate_funs import __get_EV_stats, get_event_rates
from plot_functions_wf import plot_similar_wf, plot_event_rates, plot_encoded, plot_waveforms
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

def run_visual_evaluation(waveforms, timestamps, 
                        hpdp_list, encoder, hypes, 
                        unique_title_for_figs='no_unique_string_given',
                        path_to_save_candidate='no_path_given'):
    '''
    Perform the clustering of the gradient-descent results manually. \\
    If using k-means, the number of clusters are to be given by user 
    after visual inspection of encoded hpdp in latent space. \\
    If a CAP-shape occurance is found to increase significantly after cytokine-injection, 
    it can be saved by user.

    Parameters
    ----------
    waveforms : (n_wf,d_wf), array_like
        The waveforms which are used for similarity measure to evaluate candidates.
    
    timestamps : (n_wf,) array_like
        Corresponding timestamps
    
    hpdp_list : (2,) python list with elements : (n_hpdp, d_wf) array_like
        The high probability datapointes under consideration to find cytokine-candidate for each injection.
    
    encoder : keras.Model
        Encoder part of tensorflow CVAE model.
    
    hypes : .json file
        Containing the hyperparameters:
            labels_to_evaluate : python_list
                list of which labels to consider: 0 <=> "increase after first injection", 1 <=> "increase after second injection"  
                i.e. labels_to_evaluate = [0,1] => will consider increase after both injections.
            similarity_measure : string, 'corr' or 'ssq'
                Specifies which similarity measure to be used.
            similarity_thresh : float
                Threshold compatible with chosen similarity measure. 
            assumed_model_varaince : float
                Assumed variance used in 'ssq'
    
    unique_title_for_figs : 'path/to/figure_results' string_like _or_ None
        If None then the figures are not saved
    
    path_to_save_candidate : 'path/to/evaluation_results' string_like _or_ None
        If None then the results are not saved

    Returns
    -------
        ``None``

    Saves:
    ------
    cytokine_candidates : (n_user_inputs, dim_wf)
        CAPs as specified by user-input.

    '''
    labels_to_evaluate = hypes["pdf_GD"]["labels_to_evaluate"]
    clustering_method = hypes["evaluation"]["clustering_method"]

    if clustering_method == 'dbscan':
        db_eps = hypes["evaluation"]["db_eps"]
        db_min_sample = hypes["evaluation"]["db_min_sample"]

    cytokine_candidates = np.zeros((2,waveforms.shape[-1]))
    for label_on in labels_to_evaluate:
        hpdp = hpdp_list[label_on]
        bool_labels = np.ones((hpdp.shape[0])) == 1 # Label all as True (same cluster) to plot the average form of increased EV-hpdp
        saveas = 'figures/hpdp/' + unique_title_for_figs
        title = 'CAPs With Identical Labels'
        #plot_similar_wf(0, hpdp, bool_labels, None, saveas=saveas+'_wf'+str(label_on),verbose=True)
        plot_waveforms(hpdp, saveas=saveas+'_wf'+str(label_on), verbose=True,title=title)

        #PLOT ENCODED wf_increase... :
        save_figure = 'figures/encoded_decoded/'+unique_title_for_figs
        #plot_encoded(encoder,  waveforms_increase, ev_label =ev_label_corr_shape, saveas=save_figure+'_encoded'+str(label_on), verbose=True)
        ev_label_corr_shape = np.zeros((hpdp.shape[0],3))
        ev_label_corr_shape[:,label_on] = 1
        plot_encoded(encoder, hpdp, saveas=save_figure+'_encoded_hpdp'+str(label_on), 
                     verbose=1,ev_label=ev_label_corr_shape,title='Encoded hpdp') 

        if clustering_method=='k-means':
            K_string  = input('Number of clusters? (integer) :')
            try:
                K_int = int(K_string)
            except Exception as e:
                print(f'Invalid input: "{K_string}". Must be interpretable as integer. \n')
                print(f'Full Error: \n {e}')
                print('Number of clusters set to 10!')
                K_int = 10
            encoded_hpdp,_,_ = encoder([hpdp,ev_label_corr_shape])
            kmeans = KMeans(n_clusters=K_int, random_state=0).fit(encoded_hpdp)
            k_labels = kmeans.labels_

        elif clustering_method=='dbscan':
            dbscan = DBSCAN(eps=db_eps, min_samples=db_min_sample, metric='euclidean')
            encoded_hpdp,_,_ = encoder([hpdp,ev_label_corr_shape])
            dbscan.fit(encoded_hpdp)
            k_labels = dbscan.labels_

        saveas = 'figures/event_rate_labels/'+unique_title_for_figs + str(label_on)

        possible_wf_candidates = __evaluate_hpdp_candidates__(waveforms, timestamps, 
                                                            hpdp, k_labels, hypes, 
                                                            saveas=saveas, verbose=True, 
                                                            return_candidates=True)
        k_candidate  = input('which CAP-cluster seems most likely to encode the cytokine? (integer or None) :')
        if k_candidate != 'None':

            cytokine_candidates[label_on, :] = possible_wf_candidates[int(k_candidate),:]
    if k_candidate != 'None':
        np.save(path_to_save_candidate, cytokine_candidates)

def __evaluate_hpdp_candidates__(waveforms, timestamps, hpdp, k_labels, hypes, 
                                saveas='saveas_not_specified',
                                verbose=False, return_candidates=False):
    '''
    Called by "run_visual_evaluation()".
    Allows to visually evaluate the clustered hpdp to find main-candidate waveforms.

    Uses the specified similarity measure to find the event rate using the median of each hpdp cluster as "main candidate". 
    The clusters as specified by k_labels.
    
    Parameters
    ----------
    waveforms : (n_wf,d_wf), array_like
        The waveforms which are used for similarity measure to evaluate candidates.
    
    timestamps : (n_wf,) array_like
        Corresponding timestamps
    
    hpdp : (n_hpdp, d_wf) array_lika
        The high probability datapointes under consideration to find cytokine-candidate.
    
    k_labels : (n_hpdp,) array_like
        labels for hpdp -- should correpond to the different maximas of conditional pdf.
    
    hypes : .json file
        Containing the hyperparameters:
            similarity_measure : string, 'corr' _or_ 'ssq'
                Specifies which similarity measure to be used.
            similarity_thresh : float
                Threshold compatible with chosen similarity measure. 
            assumed_model_varaince : float
                Assumed variance used in 'ssq'
    
    saveas : 'path/to/save_fig' string_like _or_ None
            If None then the figure is not saved
    
    verbose : Boolean
        True => plt.show()
    
    return_candidates : boolean
        If true, a candidate CAP for each cluster is returned

    Returns
    -------
    if return_candidates is True:
        candidate_wf : (n_clusters, d_wf) array_like
            The median of each hpdp-cluster.
    
    '''
    similarity_measure = hypes["evaluation"]["similarity_measure"] 
    similarity_thresh = hypes["evaluation"]["similarity_threshold"]
    assumed_model_varaince = hypes["evaluation"]["assumed_model_varaince"]

    k_clusters = np.unique(k_labels)  
    candidate_wf = np.empty((k_clusters.shape[0],waveforms.shape[-1]))
    for cluster in k_clusters:
        hpdp_cluster = hpdp[k_labels==cluster]
        MAIN_CANDIDATE = np.median(hpdp_cluster,axis=0) # Median more robust to outlier..

        added_main_candidate_wf = np.concatenate((MAIN_CANDIDATE.reshape((1,MAIN_CANDIDATE.shape[0])), waveforms),axis=0)
        assert np.sum(MAIN_CANDIDATE) == np.sum(added_main_candidate_wf[0,:]), 'Something wrong in concatenate..'

        print(f'Shape of test-dataset (now considers all observations): {added_main_candidate_wf.shape}')
        # Get correlation cluster for Delta EV - increased_second hpdp
        
        if similarity_measure=='corr':
            print('Using "corr" to evaluate final result')
            correlations = wf_correlation(0,added_main_candidate_wf)
            bool_labels = correlations > similarity_thresh
        if similarity_measure=='ssq':
            print('Using "ssq" to evaluate final result')
            if assumed_model_varaince is False:
                #added_main_candidate_wf = added_main_candidate_wf/assumed_model_varaince  # (0.7) Assumed var in ssq
                bool_labels,_ = similarity_SSQ(0,added_main_candidate_wf,epsilon=similarity_thresh,standardised_input=False)
            else:
                added_main_candidate_wf = added_main_candidate_wf/assumed_model_varaince  # (0.7) Assumed var in ssq
                bool_labels,_ = similarity_SSQ(0,added_main_candidate_wf,epsilon=similarity_thresh)
        event_rates = get_event_rates(timestamps, hypes, labels=bool_labels[1:], consider_only=1)
        wf_title = 'CAP-Cluster Mean'
        plt.figure(1)
        median_wf = plot_similar_wf(0, added_main_candidate_wf, bool_labels, saveas=saveas+'Main_cand_wf',
                            verbose=False, show_clustered=False,return_cand=True, title=wf_title)
        candidate_wf[cluster,:] = median_wf
        plt.figure(2)
        bool_labels[bool_labels==True] = cluster
        plot_event_rates(event_rates, timestamps, hypes, saveas=saveas+'Main_cand_ev', 
                         verbose=False, linewidth=1) 
    plt.figure(3)
    event_rates = get_event_rates(timestamps, hypes )
    plot_event_rates(event_rates, timestamps, hypes, saveas=saveas+'overall_EV', tot_EV=True,
                     verbose=False)     
    
    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose:
        plt.show()
    if return_candidates:
        return candidate_wf

def run_evaluation(waveforms, timestamps, 
                    hpdp_list, encoder, 
                    hypes, saveas=None):
    '''
    Main function to evaluate the hpdp-results. i.e results obtained by performing CVAE-gradient decent.

    - Runs clustering method on hpdp.
    - Each cluster is represented by its mean or median. ( => CAP-candidate).
    - Each CAP-candidates event-rate is estimated.
    - CAP-candidate is defined to be a "responder" if there is a "significant" increase in the event-rate
    after one of the injection events.

    See a further discription in the "__evaluate_cytokine_candidates__()" docstring.
    
    Clustering-method:
    ----
    If ``k_clusters=None (null)`` in hypes, then DBSCAN is used with a persistent homology approach, 
    where the user gives epsilon as input after looking at # of clusters vs. epsilon. 
    Observe that the ``min_sample`` parameter is fixed in this approach and may not be optimal 
    depending on the total number of data-points with a specific EV-label..\\
    Elif ``k_clusters = integer``, then k-means is applied with the number of clusters specified 
    by the ``k_clusters`` hypes-param.

    The saved results are to be interpreted by the function "find_reponders()" (in ``main_evaluation.py``)

    Parameters
    ----------
    waveforms : (n_wf,d_wf), array_like
        The waveforms which are used for similarity measure to evaluate hpdp-candidates. (Usually all observations.)
    
    timestamps : (n_wf,) array_like
        Corresponding timestamps.
    
    hpdp_list : (2,) python list with elements : (n_hpdp, d_wf) array_like
        The high probability datapointes under consideration to find cytokine-candidate for each injection.
    
    encoder : keras.Model
        Encoder part of tensorflow CVAE model.
    
    hypes : .json file
        Containing the hyperparameters:
            k_SD_eval : float
                multiplier-param in threshold for finding "responder". See docstring in "__evaluate_cytokine_candidates__()"
            SD_min_eval : float
                Minimum standard deviation in threshold for finding "responder".
            labels_to_evaluate : python_list
                list of which labels to consider: 0 <=> "increase after first injection", 1 <=> "increase after second injection" . \\
                i.e. labels_to_evaluate = [0,1] => will consider increase after both injections. 
            k_clusters : integer or ``None``
                Determines method to cluster hpdp. \\
                If ``None`` then DBSCAN is used, elif integer, then k-means is used with specified number of clusters. 
            db_eps, db_min_sample : float, Integer
                params for DBSCAN if that is chosen to be used.
            similarity_measure : string, 'corr' or 'ssq'
                Specifies which similarity measure to be used.
            similarity_thresh : float
                Threshold compatible with chosen similarity measure. 
            assumed_model_varaince : float
                Assumed variance used in 'ssq'
    
    saveas : 'path/to/evaluation_results' string_like _or_ ``None``
        If ``None`` then the results is not saved

    Returns
    -------
    responder_results : (2,) Nested numpy_array, each element containing:
        Empty numpy array if no responders where found. 
        Otherwise:
        (responder_CAP, tuple_object) numpy_array
            responder_CAP : (dim_of_wf,) numpy_array \\
            tuple_object : (MU, SD, responder, time_above_thresh) 
                of type: (float, float, Boolean, int)

    Saves:
    ------
    responder_results : numpt-array
        This is sadly not the most pleasant data-structure but the code in "find_reponders()"
        is used to load and read the results.

    '''
    labels_to_evaluate = hypes["pdf_GD"]["labels_to_evaluate"]
    k_clusters = hypes["evaluation"]["k_clusters"]
    dim_of_wf =  hypes["preprocess"]["dim_of_wf"]

    if k_clusters is None:
        # db_eps = hypes["evaluation"]["db_eps"]    # Current version takes this param as input from user.
        db_min_sample = hypes["evaluation"]["db_min_sample"]

    responder_results = []
    for label_on in labels_to_evaluate:
        hpdp = hpdp_list[label_on]   # Extract hpdp for one of the labels
        ev_label_corr_shape = np.zeros((hpdp.shape[0],3))   # Create corresponding labels with the correct shape. 
        ev_label_corr_shape[:,label_on] = 1   # Create corresponding labels with the correct shape. 
        encoded_hpdp,_,_ = encoder([hpdp, ev_label_corr_shape])
        encoded_hpdp = encoded_hpdp.numpy()
        if k_clusters is not None:
            # Run k-means clustering on latent-space-representation of hpdp.

            if (hpdp.shape[0]<8) and (hpdp.shape[0] != dim_of_wf):
                kmeans = KMeans(n_clusters=1, random_state=0).fit(encoded_hpdp)
            else:
                kmeans = KMeans(n_clusters=k_clusters, random_state=0).fit(encoded_hpdp)
            k_labels = kmeans.labels_

        else:
            # "Persistent Homology" approach using DBSCAN.
            # Run DBSCAN for different scales of epsilon, fixing the number of min_samples.
            # Give most persistent epsilon manually as input.

            sample_size = 5000
            if encoded_hpdp.shape[0] > sample_size:
                subsample_idx = np.random.choice(np.arange(encoded_hpdp.shape[0]), sample_size, replace=False)
                encoded_hpdp_sample = encoded_hpdp[subsample_idx, :] 
                hpdp = hpdp[subsample_idx, :]

            else:
                print(f'[run_evaluation()] OBS! All hpdp datapoints is given to DBCSAN..')
                encoded_hpdp_sample = encoded_hpdp

            eps_range =  np.arange(0.02, 2, 0.02)
            persistent_hom = []
            for epsilon in eps_range:
                dbscan = DBSCAN(eps=epsilon, min_samples=db_min_sample, metric='euclidean')
                
                #dbscan.fit(encoded_hpdp)
                dbscan.fit(encoded_hpdp_sample)
                k_labels = dbscan.labels_
                n_clusters = np.sum(np.unique(k_labels) != -1)

                persistent_hom.append( n_clusters )
            plt.figure(figsize=(10,8))
            plt.plot(eps_range, persistent_hom)
            plt.title('Persistent homology (H0 using DBSCAN) on latent space.')
            plt.xlabel('Epsilon')
            plt.ylabel('# of clusters')
            plt.show()

            input_epsilon = float(input('Most persistent epsilon : '))
            dbscan = DBSCAN(eps=input_epsilon, min_samples=db_min_sample, metric='euclidean')
            dbscan.fit(encoded_hpdp_sample)
            k_labels = dbscan.labels_

            for k_label in np.unique(k_labels):
                plt.title(f'DBSCAN clusters using e = {np.round(input_epsilon, decimals=1)}.')
                plt.xlabel('z1 (latent dim)')
                plt.ylabel('z2 (latent dim)')
                if k_label == -1:
                    plt.scatter(encoded_hpdp_sample[k_labels==k_label, 0], encoded_hpdp_sample[k_labels==k_label, 1], marker='+', label=k_label)
                else:
                    plt.scatter(encoded_hpdp_sample[k_labels==k_label, 0], encoded_hpdp_sample[k_labels==k_label, 1] , label=k_label)
            plt.show()
            # Plot resulting kluster in latent space. 
        
        responders = __evaluate_cytokine_candidates__(waveforms, timestamps, 
                                                      hpdp, k_labels, hypes, 
                                                      injection=label_on+1)
        responder_results.append(np.array(responders))
    responder_results = np.array(responder_results)
    if saveas is not None:
        np.save(saveas, np.squeeze(responder_results))
        print(f'Results for evaluation saved sucessfully as {saveas}.')
    return responder_results

def __evaluate_cytokine_candidates__(waveforms, timestamps, 
                                     hpdp, k_labels, hypes, 
                                     injection=1):
    '''
    Called by "run_evaluation()"

    Evaluates the results of clustered hpdp using the median _or_ mean (depending in param in hypes) of each hpdp cluster 
    as a "cytokine-candidate". \\
    Makes use of the similarity measure to find the event-rate of each candidate and evaluates 
    if there is a sufficient increase in the firing rate at time of injection. If so, the mice under consideration is considered
    to be a "responder" of the corresponding injection. 
    
    The evaluation of the event-rate increase is defined as follows,
    - Measure standart deviation, SD, of the baseline activity. (10-30 min from initial recording.)
    - Measure mean firing rate, MU, 4 min before the considered injection.
    - Measure past-injection firing rate, EV_past. (10-30 min after injection.)
    - Set threshold to ``k * max(SD_min, SD)`` and consider mice as responer if ``EV_past > k * max(SD_min, SD)``
      for at least 1/3 of the considered post-injection time interval. (e.g. 7 out of 20 min.)
    
    The SD_min paramater is manually set to prevent the method to be sensitive to insignificant 
    changes in event-rate. 
    
    Parameters
    ----------
    waveforms : (n_wf,d_wf), array_like
        The waveforms which are used for similarity measure to evaluate candidates.
    
    timestamps : (n_wf,) array_like
        Corresponding timestamps
    
    hpdp : (n_hpdp, d_wf) array_lika
        The high probability datapointes under consideration to find cytokine-candidate.
    
    k_labels : (n_hpdp,) array_like
        labels for hpdp -- should correpond to the different maximas of conditional pdf.
    
    hypes : .json file
        Containing the hyperparameters.
            
    injection : integer
        Specifies which injection to consider (1 _or_ 2)

    Returns
    -------
    prel_results: nested python list
        responder_CAP, (MU, SD, responder, time_above_thresh)
    
    '''
    # Extraxt hyperparams
    similarity_measure = hypes["evaluation"]["similarity_measure"] 
    similarity_thresh = hypes["evaluation"]["similarity_threshold"]
    assumed_model_varaince = hypes["evaluation"]["assumed_model_varaince"]
    k = hypes["evaluation"]["k_SD_eval"]
    SD_min = hypes["evaluation"]["SD_min_eval"]
    cluster_representation = hypes["evaluation"]["cluster_representation"]
    injection_t_period = hypes["experiment_setup"]["injection_t_period"]

    t_delay_post_injection = hypes["experiment_setup"]["t_delay_post_injection"]
    
    t0_baseline_SD = hypes["labeling"]["t0_baseline_SD"]
    time_baseline_MU = hypes["labeling"]["time_baseline_MU"]
    # ---------------------------------------------------------------------

    # Times of interest (in seconds)
    t0_baseline_SD = t0_baseline_SD * 60 # Initial time for measure baseline SD
    time_baseline_MU = time_baseline_MU * 60 # length og time period measureing  baseline MU

    t_injection = injection_t_period * 60 * injection # Time of injection
    
    assert timestamps[0] < 60*injection_t_period, f'Invalid time range. Start time {timestamps[0]}, need to be before first injection.'
    assert timestamps[-1] > 60*injection_t_period, f'Invalid time range. End time {timestamps[-1]}, need to be After second injection.'

    k_clusters = np.unique(k_labels)  
    candidate_wf = np.empty((k_clusters.shape[0],waveforms.shape[-1]))
    prel_results = [] # Store result about responders. If no responders, then this remains empty. 
    print(f'Shape of test-dataset (now considers all observations): {waveforms.shape}')
    for cluster in k_clusters:
        hpdp_cluster = hpdp[k_labels==cluster]

        # Choice between using median and mean as main candidate.. 
        if cluster_representation == 'mean':
            MAIN_CANDIDATE = np.mean(hpdp_cluster,axis=0) # Mean more smooth.. 
        elif cluster_representation == 'median':
            MAIN_CANDIDATE = np.median(hpdp_cluster,axis=0) # Median more robust to outlier.
        
        added_main_candidate_wf = np.concatenate((MAIN_CANDIDATE.reshape((1,MAIN_CANDIDATE.shape[0])),waveforms),axis=0)
        assert np.sum(MAIN_CANDIDATE) == np.sum(added_main_candidate_wf[0,:]), 'Something wrong in concatenate..'

        if similarity_measure=='corr':
            print('Using "corr" to evaluate final result')
            correlations = wf_correlation(0, added_main_candidate_wf)
            bool_labels = correlations > similarity_thresh
        if similarity_measure=='ssq':
            print('Using "ssq" to evaluate final result')
            if assumed_model_varaince is None:
                #added_main_candidate_wf = added_main_candidate_wf/assumed_model_varaince  # (0.7) Assumed var in ssq
                bool_labels,_ = similarity_SSQ(0, added_main_candidate_wf, epsilon=similarity_thresh, standardised_input=False)
            else:
                added_main_candidate_wf = added_main_candidate_wf/assumed_model_varaince  # (0.7) Assumed var in ssq
                bool_labels, _ = similarity_SSQ(0, added_main_candidate_wf, epsilon=similarity_thresh) 
        event_rate = get_event_rates(timestamps, hypes, labels=bool_labels[1:], consider_only=1)
        
        _, baseline_SD = __get_EV_stats(event_rate, start_time=t0_baseline_SD, end_time=injection_t_period*60)
        baseline_MU, _ =  __get_EV_stats(event_rate, start_time=t_injection-time_baseline_MU, end_time=t_injection)
        
        SD_thesh = k * np.max((SD_min,  baseline_SD))
        cytokine_stats = __get_EV_stats(event_rate, start_time=t_injection+t_delay_post_injection*60, end_time=t_injection+injection_t_period*60, 
                                            compare_to_theshold=baseline_MU + SD_thesh, conv_width=5)
        #print(f'Cytokine candidate responder result for injection 1 is : {cytokine_stats[2]}')
        #print(f'Cytokine candidate responder result for injection 2 is : {second_cytokine_stats[2]}')
        if cytokine_stats[2] is True:
            prel_results.append(np.array([MAIN_CANDIDATE, cytokine_stats]))
            print(f'CAP nr. {cluster} found to have a sufficient increase in firing rate for injection {injection}.')
    return prel_results
        


def find_reponders(candidate_directory, hypes, 
                   start_string='', end_string='',
                   specify_recordings='R10', 
                   verbose=False, saveas=None,
                   return_main_candidates=False):
    '''
    Main function to "inspect" the saved results given by "run_evaluation()".

    Looks through the stored "cytokine_candidates" numpy files, which are saved in ``main_train.py`` in 
    the "candidate_directory". (Output of "run_evaluation()"). 

    All calculations are done prior to this function, this is just a file to process (visually inspect) the results.

    The "start_string" defines which training-run we are considering and should be the hypes-file-name. \\
    The input "specify_recordings", e.g. 'R10', is used to disciminating between the files that was considered in this 
    training-run. This to e.g. seperate the cytokine-cases (R10), from saline-control (R12) (If Zanos..).


    The responders file-names are saved to hypes.json file under "Results" and figures of the CAP-Candidate, Event-rate etc.
    are saved / plotted (if any).

    Parameters
    ----------
    candidate_directory : "path/to/CAP_candidates_directory" string
        Specify directory containing the saved CAP-candidates .npy files.
    
    start/end_string : string
        Specify start/end string of files to consider. 
        - The start_string should be the hypes-file-name.
        - The end_string is "auto_assesment.npy" if the "auto_assessment" is True in ``main_train.py``.
    
    specify_recordings : string
        Specify further which recordings to consider. \\
        i.e, the saved candidates which is to be considered are saved as "start_string" + "specify_recordings" in "candidate_directory".
    
    matlab_directory : "path/to/data.mat"
        Specify directory containing waveforms and timestamps .mat files 
    
    verbose : boolean
        If True, then show plots of "total event rate", 
        "candidate-CAP event rate" and 
        "candidate-CAP" of the main-candidate/responder
    
    saveas : 'path/to/save_fig' string_like _or_ None
        If None then the figures is not saved
    
    return_main_candidates : boolean
        Wether to return the responder CAPs or not.

    Returns
    -------
    if return_main_candidates is True:
        responders : (number_of_considered_recordings, ) python list
            Elements are 1 if responder is found, otherwise 0.
        main_candidates : (number_of_considered_recordings, ) python list
            Elements : (dim_of_wf, ) array_like
                the responders CAP-waveform.
    else: 
        responders : (number_of_considered_recordings, ) python list
            Elements are 1 if responder is found, otherwise 0.

    Saves:
    ------
    Writes the recording-file-names where a responder was found under "Results" in .json hypes-file.
    ''' 
    responders = [] # We will ad a 1 if a recording in specified candidate_directory corresponds to a "responder", otherwise 0. 
                    # This is used to find how many, out of all considered recordings, that are showing promising results. 
    main_candidates = []   # Best responder-results.. Can be saved and returned.
    responder_files = []   # To be saved as results in .json file
    responder_labels = []   # To be saved as results in .json file

    for entry in scandir(candidate_directory):
        if entry.path.startswith(candidate_directory+start_string+specify_recordings) & entry.path.endswith(end_string): # & ~entry.path.startswith(candidate_directory+start_string+'R10_Exp3'): # Specify any uniquness in the name of the files to be considered. 
            result = np.load(entry.path, allow_pickle=True)
            responder_bool = False
            injection = ["first", "second"]   # Used for prints.
            for jj, injection_res in enumerate(result): 
                # "result" is a nested list. The two elements in injection_res corresponds to the results of the first / second injections (if responder).
                if len(injection_res)==0: 
                    # No responders where found for this injection
                    pass
                else: 
                    # A responder was found
                    responder_bool=True
                    recording = entry.path[len(candidate_directory+start_string):-len(end_string)] # Get the matlab_file-string of recording
                    print(f'{"*"*40} \n The recording which was found to be a responder : {recording}')
                    print(f' This regarding the {injection[jj]} injection')
                    responder_files.append(recording)
                    responder_label = [0,0,0]   # To be saved in .json file
                    responder_label[jj] = 1   
                    responder_labels.append(responder_label)
                    if injection_res.shape==(2,): 
                        # => Only one cluster fullfilled the requerement to be defined as a responder
                        waveform = injection_res[0]
                        main_candidates.append(waveform)  
                        if verbose:
                            __evaluate_responder__(waveform, 
                                                   recording, hypes,
                                                   saveas=saveas + '_' + str(injection[jj]))
                            
                    else: 
                        '''
                        >1 clusters where found with qualities as responders (most often corresponding to very similar CAPs) 
                        Loop through the different cluster-candidate CAPs to find the one with the "clearest" result as responder.
                        The "clearest result" is in terms of "Total time above threshold":
                        '''

                        times_above_thresh = np.zeros((len(injection_res,))) 
                        for ii, responder in enumerate(injection_res):
                            stats = responder[1]
                            times_above_thresh[ii] = stats[3] # time above threshold
                            # waveform = responder[0]
                        main_candidate = np.argmax(times_above_thresh) # The main_candidate is defined to be the cluster-candidate with most time
                                                                       # above threshold after injection
                        waveform = injection_res[main_candidate][0]
                        main_candidates.append(waveform)
                        if verbose: # Show plots of "total event rate", "candidate-CAP event rate" and "candidate-CAP" of the main-candidate/responder.
                            __evaluate_responder__(waveform, 
                                                   recording, hypes,
                                                   saveas=saveas + '_' + str(injection[jj]))
            if responder_bool:
                responders.append(1)
            else: 
                responders.append(0)
    print(f'Number of responders: {np.sum(responders)} out of {len(responders)}')

    # Add responder mat-file titels as "results" in .json file
    with open('hypes/' + start_string + '.json') as json_file: 
        data = json.load(json_file)
        temp = data['Results']
        temp.update({"Responders" : responder_files}) 
        temp.update({"labels" : responder_labels})
    with open('hypes/' + start_string + '.json', 'w') as f: 
        json.dump(data, f, indent=3) 

    if return_main_candidates:
        return responders, main_candidates
    else:
        return responders

def __evaluate_responder__(cytokine_candidate, 
                           file_name, hypes,
                           saveas=None, verbose=True):
    '''
    Visual evaluation of "cytokine_candidate".
    Plots similarity-clusters and event-rate for "cytokine_candidate"-waveform. This needs the matlab-recording to be specified.

    Called by "find_responers()" and "eval_candidateCAP_on_multiple_recordings()"

    Parameters
    ----------
    cytokine_candidate :  (dim_wf, ) array_like
        The CAP-candidate under evaluation.

    file_name : string.
        Specify which file/files to go through and plot similar waveforms to candidate.
        The matlabfiles that are named "'ts' + 'file_name'" will be considered.
    
    hypes : dict.
        Hyperparams
    
    saveas : 'path/to/save_fig' string_like _or_ None
        If None then the figures is not saved.
    
    verbose : Boolean
        ``True`` => plt.show() \\
        ``False`` => plt.close('all')

    Returns:
    --------
    ``None``.
    '''
    matlab_directory = hypes["dir_and_files"]["matlab_dir"]
    similarity_measure = hypes["evaluation"]["similarity_measure"] 
    similarity_thresh = hypes["evaluation"]["similarity_threshold"] 
    assumed_model_varaince = hypes["evaluation"]["assumed_model_varaince"] 
    for entry in scandir(matlab_directory):
        if entry.path.startswith(matlab_directory + '\\ts' +  file_name): # Find unique recording string

            matlab_file = entry.path[len(matlab_directory) + 3:-4] # Find unique recording string'

            print('************* PLOTTING RESPONDER RESULTS ***********************')
            print('*' * 40)
            print(f'Considering MATLAB-file : {matlab_file}')
            print()
            print(f'Using {similarity_measure} as similarity measure to evaluate final result')
            path_to_wf = matlab_directory + '/wf'+matlab_file +'.mat'
            path_to_ts = matlab_directory + '/ts'+matlab_file +'.mat'

            savefig = saveas + matlab_file.replace('.','_') #file_names['figure_strings'][run_i]

            wf0 = load_mat_file(path_to_wf, 'waveforms', verbose=0) # Load the candidate's corresponding matlab file 
            ts0 = load_mat_file(path_to_ts, 'timestamps', verbose=0)

            wf0, ts0 = preprocess_wf.get_desired_shape(wf0, ts0, hypes, training=False)
            # wf0,ts0 = preprocess_wf.get_desired_shape(wf0,ts0,start_time=10,end_time=90,dim_of_wf=141,desired_num_of_samples=None)
            wf0 = preprocess_wf.standardise_wf(wf0, hypes)
            
            added_main_candidate_wf = np.concatenate((cytokine_candidate.reshape((1,cytokine_candidate.shape[0])),wf0),axis=0)
            assert np.sum(cytokine_candidate) == np.sum(added_main_candidate_wf[0,:]), 'Something wrong in concatenate..'

            if similarity_measure == 'corr':
                correlations = wf_correlation(0,added_main_candidate_wf)
                bool_labels = correlations > similarity_thresh

            elif similarity_measure == 'ssq':
                added_main_candidate_wf = added_main_candidate_wf / assumed_model_varaince  # (0.7) Assumed var in ssq
                bool_labels,_ = similarity_SSQ(0, added_main_candidate_wf, epsilon=similarity_thresh)
            event_rates = get_event_rates(ts0, hypes, labels=bool_labels[1:], consider_only=1)
            # Plot titles etc.
            wf_title = 'Candidate-CAP'
            overall_ev_title = 'Event-Rate for all Observed CAPs'
            cluster_ev_title = 'Event-Rate for Candidate-CAP'
            plt.figure(1)
            plot_similar_wf(0, added_main_candidate_wf, bool_labels, 
                            saveas=savefig+'Main_cand'+'_wf',
                            verbose=False, show_clustered=False,title=wf_title)
            plt.figure(4)
            plot_similar_wf(0, added_main_candidate_wf, bool_labels, 
                            saveas=savefig+'Main_cand'+'_wf_cluster',
                            verbose=False, show_clustered=True, title=wf_title)
            plt.figure(2)
            plot_event_rates(event_rates, ts0, hypes, saveas=savefig + 'Main_cand' + '_ev', 
                             verbose=False,title=cluster_ev_title) 
            plt.figure(3)
            event_rates = get_event_rates(ts0, hypes )
            plot_event_rates(event_rates, ts0, hypes, saveas=savefig + 'overall_EV', tot_EV=True,
                             verbose=False, title=overall_ev_title ) 
            if verbose is True:
                plt.show()
            else:
                plt.close('all')

def eval_candidateCAP_on_multiple_recordings(candidate_CAP, hypes, 
                                             file_name='', 
                                             saveas='Not_specified', 
                                             verbose=True):
    '''
    Use CAP-Candidate to get event-rate in multiple different recordings. 
    
    Parameters:
    -----------
    candidate_CAP : (dim_of_waveform, ) array_like
        The CAP-candidate under evaluation. This will e.g. be the mean-value if 
        the "ssq" method is used.

    hypes : dict.
        Hyperparams

    file_name : String.
        The shared file-name for the recordings of interest. \\
        Specify which file/files to go through and get event-rates / Figures. \\
        The matlab-files that are named "'ts' + 'file_name'" will be considered.

    saveas : 'path/to/save_fig' string_like _or_ None
        If None then the figures is not saved.

    verbose : Boolean
        ``True`` => plt.show() \\
        ``False`` => plt.close('all')

    Returns:
    --------
    None.
    '''
    # similarity_measure=similarity_measure, 
    # similarity_thresh = similarity_thresh, 
    # assumed_model_varaince=assumed_model_varaince
    __evaluate_responder__(candidate_CAP, file_name, hypes, saveas=saveas, verbose=verbose)


def marginal_log_likelihood(x, label, encoder, decoder, hypes):
    '''
    Returns the estimated marginal likelihood p(x|label) using importance sampling,
    from the CVAE model.
    
    Only used for some tests..

    Parameters
    ----------
    x : (dim_of_waveform, ) array_like

    label : (3, ) array_like

    encoder/decoder : keras.Model 

    hypes : .json file
        Containing hyperparameters
    
    Returns:
    --------
    log_prob_x
    '''
    L = hypes["marginal_likelihood"]["MC_sample_size"]
    model_var = hypes["cvae"]["model_variance"]
    # Create L samples of the needed variables.
    #x_samples = np.ones((L, 1)) * x.reshape((1, x.shape[0]))
    label_samples = np.ones((L, 1)) * label.reshape((1, label.shape[0]))
    z_mean, z_log_var, _ = encoder([x.reshape((1, x.shape[0])), label.reshape((1, label.shape[0]))])
    z_cov = np.array( [[np.exp(z_log_var[0,0]), 0], [0, np.exp(z_log_var[0,1])]] )
    z_samples = np.random.multivariate_normal(z_mean[0,:], z_cov, size=(L,))
    x_means = decoder([z_samples, label_samples]).numpy()

    # Evaluate the nessassary probabilities.
    posterior_log_probs = multivariate_normal.logpdf(z_samples, mean=z_mean[0,:], cov=np.exp(z_log_var[0,:]))
    prior_log_probs = multivariate_normal.logpdf(z_samples, mean=[0, 0], cov=[1, 1])
    likelihood_log_probs = np.zeros((L,))
    for ii in range(L):
        likelihood_log_probs[ii] = multivariate_normal.logpdf(x, mean=x_means[ii,:], cov=np.ones(x.shape[0])*model_var)

    # Calculate the MC estimate of marginal log-likelihood of x. 
    argsum = likelihood_log_probs + prior_log_probs - posterior_log_probs
    log_prob_x = logsumexp(argsum) - np.log(L)
    return log_prob_x

def run_DBSCAN_evaluation(wf_ho, wf0, ts0, 
                        ev_label_ho, hypes, 
                        saveas=None, np_saveas=None, 
                        matlab_file=None):
    '''
    Skips everything after labeling and perform clustering using DBSCAN on labeled high-occurance waveforms. 
    
    Parameters
    ----------

    Returns
    -------
            No returns.
    '''
    labels_to_evaluate = hypes["pdf_GD"]["labels_to_evaluate"] 
    db_eps = hypes["DBSCAN"]["db_eps"] 
    db_min_sample = hypes["DBSCAN"]["db_min_sample"] 
    dim_of_wf = hypes["preprocess"]["dim_of_wf"]


    cytokine_candidates = np.empty((2,wf_ho.shape[-1])) # To save the main candidates
    for label_on in labels_to_evaluate:
        waveforms_increase = wf_ho[ev_label_ho[:,label_on]==1]
        if waveforms_increase.shape[0] == 0:
            waveforms_increase = np.append(np.zeros((1,dim_of_wf)),waveforms_increase).reshape((1,dim_of_wf))
            print('*************** OBS ***************')
            print(f'No waveforms with increased event rate at injection {label_on+1} was found.')
            print(f'This considering the recording {matlab_file}')

        elif waveforms_increase.shape[0] > 3000: # Downsample to speed up process during param search..
            waveforms_increase = waveforms_increase[::4,:]
        #bool_labels = np.ones((waveforms_increase.shape[0])) == 1 # Label all as True (same cluster) to plot the average form of increased EV-hpdp
        #plot_similar_wf(0,waveforms_increase,bool_labels,None,saveas=saveas+'_wf'+str(label_on),verbose=True)
        #dist_vec = cdist(waveforms_increase, waveforms_increase, 'euclid')
        #plt.hist(dist_vec)
        #plt.show()

        print()
        print(f'Running DBSCAN on high occurance waveforms after injection {label_on+1}...')
        print()
        dbscan = DBSCAN(eps=db_eps, min_samples=db_min_sample, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
        dbscan.fit(waveforms_increase)
        labels = dbscan.labels_ #  Noisy samples are given the label -1

        possible_wf_candidates = __evaluate_hpdp_candidates__(wf0, ts0, waveforms_increase, 
                                                            labels, hypes, saveas=saveas,
                                                            verbose=True, return_candidates=True)
        
        k_candidate  = input('which CAP-cluster seems most likely to encode the cytokine? (integer or None) :')
        if k_candidate != 'None':
            cytokine_candidates[label_on,:] = possible_wf_candidates[int(k_candidate),:]
    if k_candidate is not None:
        np.save(np_saveas+'DBSCAN', cytokine_candidates)

