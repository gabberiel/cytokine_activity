# Functions used to evaluate results after training is complete. 
import numpy as np
import matplotlib.pyplot as plt
import preprocess_wf 
from load_and_GD_funs import load_timestamps,load_waveforms
from os import path, scandir
from wf_similarity_measures import wf_correlation, similarity_SSQ, label_from_corr
from event_rate_funs import get_ev_labels, get_event_rates
from plot_functions_wf import plot_similar_wf, plot_event_rates, plot_encoded,plot_waveforms
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN

def run_visual_evaluation(waveforms,timestamps,hpdp_list,encoder,labels_to_evaluate=[0,1],clustering_method='k-means', db_eps=0.15, db_min_sample=5,
                     similarity_measure='ssq', similarity_thresh=0.4, assumed_model_varaince=0.5, unique_string_for_figs='no_unique_string_given',
                     path_to_cytokine_candidate='no_path_given'):
    ''' 
    '''
    cytokine_candidates = np.zeros((2,waveforms.shape[-1]))
    for label_on in labels_to_evaluate:
        hpdp = hpdp_list[label_on]
        bool_labels = np.ones((hpdp.shape[0])) == 1 # Label all as True (same cluster) to plot the average form of increased EV-hpdp
        saveas = 'figures/hpdp/'+unique_string_for_figs
        title = 'CAPs With Identical Labels'
        #plot_similar_wf(0,hpdp,bool_labels,None,saveas=saveas+'_wf'+str(label_on),verbose=True)
        plot_waveforms(hpdp,saveas=saveas+'_wf'+str(label_on),verbose=True,title=title)
        #waveforms_increase = wf_ho[ev_label_ho[:,label_on]==1]
        ##print(f'waveforms_increase injection {label_on+1} : {waveforms_increase.shape}')
        #if waveforms_increase.shape[0] == 0:
        #    waveforms_increase = np.append(np.zeros((1,141)),waveforms_increase).reshape((1,141))
        #    ev_label_corr_shape = np.zeros((waveforms_increase.shape[0],3))
        #    ev_label_corr_shape[:,label_on] = 1
        #    print('*************** OBS ***************')
        #    print(f'No waveforms with increased event rate at injection {label_on+1} was found.')
        #    print(f'This considering the recording {matlab_file}')
        #elif waveforms_increase.shape[0] > 3000: # Speed up process during param search..
        #    waveforms_increase = waveforms_increase[::4,:]
        #    ev_label_corr_shape = np.zeros((waveforms_increase.shape[0],3))
        #    ev_label_corr_shape[:,label_on] = 1
        #else:
        #    ev_label_corr_shape = np.zeros((waveforms_increase.shape[0],3))
        #    ev_label_corr_shape[:,label_on] = 1   
        
        #PLOT ENCODED wf_increase... :
        save_figure = 'figures/encoded_decoded/'+unique_string_for_figs
        #plot_encoded(encoder,  waveforms_increase, ev_label =ev_label_corr_shape, saveas=save_figure+'_encoded'+str(label_on), verbose=True)
        ev_label_corr_shape = np.zeros((hpdp.shape[0],3))
        ev_label_corr_shape[:,label_on] = 1
        plot_encoded(encoder, hpdp, saveas=save_figure+'_encoded_hpdp'+str(label_on), verbose=1,ev_label=ev_label_corr_shape,title='Encoded hpdp') 

        if clustering_method=='k-means':
            K_string  = input('Number of clusters? (integer) :')
            encoded_hpdp,_,_ = encoder([hpdp,ev_label_corr_shape])
            kmeans = KMeans(n_clusters=int(K_string), random_state=0).fit(encoded_hpdp)
            k_labels = kmeans.labels_

        elif clustering_method=='DBSCAN':
            dbscan = DBSCAN(eps=db_eps, min_samples=db_min_sample, metric='euclidean')
            encoded_hpdp,_,_ = encoder([hpdp,ev_label_corr_shape])
            dbscan.fit(encoded_hpdp)
            k_labels = dbscan.labels_

        saveas = 'figures/event_rate_labels/'+unique_string_for_figs + str(label_on)

        possible_wf_candidates = evaluate_hpdp_candidates(waveforms,timestamps,hpdp,k_labels,saveas=saveas, similarity_measure=similarity_measure,
                                similarity_thresh=similarity_thresh, assumed_model_varaince=assumed_model_varaince, verbose=True, return_candidates=True)
        k_candidate  = input('which CAP-cluster seems most likely to encode the cytokine? (integer or None) :')
        if k_candidate != 'None':
            cytokine_candidates[label_on,:] = possible_wf_candidates[int(k_candidate),:]
    if k_candidate != 'None':
        np.save(path_to_cytokine_candidate, cytokine_candidates)

def evaluate_hpdp_candidates(wf0,ts0,hpdp,k_labels,similarity_measure='corr', similarity_thresh=0.4, 
                            assumed_model_varaince=0.5,saveas='saveas_not_specified',verbose=False, return_candidates=False):
    '''
    Main function to visually evaluate the clustered hpdp to find main-candidate waveforms.

    Uses the specified similarity measure to find the event rate using the median of each hpdp cluster as "main candidate". 
    The clusters as specified by k_labels.
    
    Parameters
    ----------
    wf0 : (n_wf,d_wf), array_like
        The waveforms which are used for similarity measure to evaluate candidates.
    ts0 : (n_wf,) array_like
        Corresponding timestamps
    hpdp : (n_hpdp, d_wf) array_lika
        The high probability datapointes under consideration to find cytokine-candidate.
    k_labels : (n_hpdp,) array_like
        labels for hpdp -- should correpond to the different maximas of conditional pdf.
    similarity_measure : sring, 'corr' or 'ssq'
        Specifies which similarity measure to be used.
    similarity_thresh : float
        Threshold compatible with chosen similarity measure. 
    assumed_model_varaince : float
        Assumed variance used in 'ssq'
    saveas : 'path/to/save_fig' string_like _or_ None
            If None then the figure is not saved
    verbose : Booleon
        True => plt.show()
    return_candidates : booleon
        If true, a candidate CAP for each cluster is returned

    Returns
    -------
    if return_candidates is True:
        candidate_wf : (n_clusters, d_wf) array_like
            The median of each hpdp-cluster.
    
    '''
    k_clusters = np.unique(k_labels)  
    candidate_wf = np.empty((k_clusters.shape[0],wf0.shape[-1]))
    for cluster in k_clusters:
        hpdp_cluster = hpdp[k_labels==cluster]
        MAIN_CANDIDATE = np.median(hpdp_cluster,axis=0) # Median more robust to outlier..

        added_main_candidate_wf = np.concatenate((MAIN_CANDIDATE.reshape((1,MAIN_CANDIDATE.shape[0])),wf0),axis=0)
        assert np.sum(MAIN_CANDIDATE) == np.sum(added_main_candidate_wf[0,:]), 'Something wrong in concatenate..'
        

        # QUICK FIX FOR WAVEFORMS AMPLITUDE INCREASING AFTER GD-- standardise it.
        # Should not be needed if GD works properly...
        
        #added_main_candidate_wf = preprocess_wf.standardise_wf(added_main_candidate_wf)

        print(f'Shape of test-dataset (now considers all observations): {added_main_candidate_wf.shape}')
        # Get correlation cluster for Delta EV - increased_second hpdp
        
        if similarity_measure=='corr':
            print('Using "corr" to evaluate final result')
            correlations = wf_correlation(0,added_main_candidate_wf)
            bool_labels = label_from_corr(correlations,threshold=similarity_thresh,return_boolean=True)
        if similarity_measure=='ssq':
            print('Using "ssq" to evaluate final result')
            if assumed_model_varaince is False:
                #added_main_candidate_wf = added_main_candidate_wf/assumed_model_varaince  # (0.7) Assumed var in ssq
                bool_labels,_ = similarity_SSQ(0,added_main_candidate_wf,epsilon=similarity_thresh,standardised_input=False)
            else:
                added_main_candidate_wf = added_main_candidate_wf/assumed_model_varaince  # (0.7) Assumed var in ssq
                bool_labels,_ = similarity_SSQ(0,added_main_candidate_wf,epsilon=similarity_thresh)
        event_rates, _ = get_event_rates(ts0,bool_labels[1:],bin_width=1,consider_only=1)
        wf_title = 'CAP-Cluster Mean'
        plt.figure(1)
        median_wf = plot_similar_wf(0,added_main_candidate_wf,bool_labels,similarity_thresh,saveas=saveas+'Main_cand_wf',
                            verbose=False, show_clustered=False,cluster=cluster,return_cand=True, title=wf_title)
        candidate_wf[cluster,:] = median_wf
        plt.figure(2)
        bool_labels[bool_labels==True] = cluster
        plot_event_rates(event_rates,ts0,noise=None,conv_width=100,saveas=saveas+'Main_cand_ev', verbose=False,cluster=cluster) 
    plt.figure(3)
    #plt.hist(ts0,bins=200)
    event_rates, _ = get_event_rates(ts0,np.ones((ts0.shape[0],)),bin_width=1,consider_only=1)
    plot_event_rates(event_rates,ts0,noise=None,conv_width=100,saveas=saveas+'overall_EV', verbose=False)     
    
    if saveas is not None:
        plt.savefig(saveas, dpi=150)
    if verbose:
        plt.show()
    if return_candidates:
        return candidate_wf



def run_evaluation(waveforms,timestamps,hpdp_list,encoder,k_SD_eval=1,SD_min_eval=0.15,labels_to_evaluate=[0,1],k_clusters=None, db_eps=0.15, db_min_sample=5,
                     similarity_measure='ssq', similarity_thresh=0.4, assumed_model_varaince=0.5, saveas=None):
    '''
    Main function to evaluate the hpdp-results. i.e results obtained by performing gradient decent on the conditional probability function approximated by CVAE.

    Runs quantitative evaluation of the hpdp for the different conditionals, i.e increase after first/second injections. 
    Evaluation is done according to description in "__evaluate_cytokine_candidates__()"
    If k_clusters=None, then DBSCAN is used with specified params.
    Else, k-means with the number of clusters specified by k_clusters.

    The saved results are to be interpreted by the function "find_reponders()" 
    Parameters
    ----------
    waveforms : (n_wf,d_wf), array_like
        The waveforms which are used for similarity measure to evaluate hpdp-candidates.
    timestamps : (n_wf,) array_like
        Corresponding timestamps
    hpdp_list : (2,) python list with elements : (n_hpdp, d_wf) array_like
        The high probability datapointes under consideration to find cytokine-candidate for each injection.
    encoder : keras.Model
        Encoder part of tensorflow CVAE model.
    k_SD_eval : float
        multiplier-param in threshold for finding "responder". See docstring in "__evaluate_cytokine_candidates__()"
    k_SD_eval : float
        Minimum standard deviation in threshold for finding "responder".
    labels_to_evaluate : python_list
        list of which labels to consider: 0 <=> "increase after first injection", 1 <=> "increase after second injection"  
        e.g labels_to_evaluate = [0,1] => will consider increase after both injections. 
    k_clusters : integer or None
        Determines method to cluster hpdp.
        If None then DBSCAN is used, elif integer then k-means is used with specified number of clusters. 
    db_eps, db_min_sample : float, Integer
        params for DBSCAN if that is chosen to be used.
    similarity_measure : sring, 'corr' or 'ssq'
        Specifies which similarity measure to be used.
    similarity_thresh : float
        Threshold compatible with chosen similarity measure. 
    assumed_model_varaince : float
        Assumed variance used in 'ssq'
    saveas : 'path/to/evaluation_results' string_like _or_ None
            If None then the results is not saved
    Returns
    -------
    responder_results : (2,) Nested numpy_array, each element containing:
        Empty numpy array if no responders where found. 
        Otherwise:
        (responder_CAP, tuple_object) numpy_array
            responder_CAP : (dim_of_wf,) numpy_array
            tuple_object : (MU, SD, responder, time_above_thresh), of type: (float,float,Booleon, int)
    '''
    responder_results = []
    for label_on in labels_to_evaluate:
        hpdp = hpdp_list[label_on]
        ev_label_corr_shape = np.zeros((hpdp.shape[0],3))
        ev_label_corr_shape[:,label_on] = 1
        encoded_hpdp,_,_ = encoder([hpdp,ev_label_corr_shape])
        if k_clusters is not None:
            if (hpdp.shape[0]<8) and (hpdp.shape[0] != 141):
                kmeans = KMeans(n_clusters=1, random_state=0).fit(encoded_hpdp)
            else:
                kmeans = KMeans(n_clusters=k_clusters, random_state=0).fit(encoded_hpdp)
            k_labels = kmeans.labels_
        else:
            dbscan = DBSCAN(eps=db_eps, min_samples=db_min_sample, metric='euclidean')
            dbscan.fit(encoded_hpdp)
            k_labels = dbscan.labels_

        responders = __evaluate_cytokine_candidates__(waveforms, timestamps, hpdp, k_labels, injection=label_on+1, similarity_measure=similarity_measure,
                                    similarity_thresh=similarity_thresh, assumed_model_varaince=assumed_model_varaince, k=k_SD_eval, SD_min=SD_min_eval,
                                    saveas=None)
        responder_results.append(np.array(responders))
    responder_results = np.array(responder_results)
    if saveas is not None:
        np.save(saveas,np.squeeze(responder_results))
        print(f'Results for evaluation saved sucessfully as {saveas}.')
    return responder_results




def __evaluate_cytokine_candidates__(waveforms, timestamps, hpdp, k_labels, injection=1, similarity_measure='ssq', similarity_thresh=0.4, 
                            assumed_model_varaince=0.5, k=1, SD_min=1, saveas='saveas_not_specified'):
    '''
    Called by "run_evaluation()"

    Evaluates the results of clustered hpdp using the median of each hpdp cluster as a "cytokine-candidate". 
    We make use of the specified similarity measure to find the corresponding event-rate of each candidate and evaluates 
    if there is a sufficient increase in the firing rate at time of injection. If so, the mice under consideration is considered
    to be a "responder" of the corresponding cytokine. 
    
    The evaluation of the event-rate increase is defined as follows,
    - Measure standart deviation, SD, of the baseline activity. (10-30 min from initial recording.)
    - Measure mean firing rate, MU, 4 min before the considered injection.
    - Measure past-injection firing rate, EV_past. (10-30 min after injection.)
    - Set threshold to k*max(SD_min,SD) and consider mice as responer if EV_past > k*max(SD_min,SD) for at least 1/3 of the considered 
      past-injection-time-interval. (7 out of 20 min.)
    
    The SD_min paramater is manually set to prevent the method to be sensitive to insignificant changes in event-rate. 
    
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
    
    saveas : 'path/to/save_fig' string_like _or_ None
            If None then the figure is not saved

    
    Returns
    -------
    prel_results: nested python list
        responder_CAP, (MU, SD, responder, time_above_thresh)
    
    '''

    # Times of interest (in seconds)
    t0_baseline_SD = 10*60 # Initial time for measure baseline SD
    time_baseline_MU = 4*60 # length og time period measureing  baseline MU

    if injection==1:
        t_injection = 30*60 # Time of first injection
    elif injection==2:
        t_injection = 60*60 # Time of second injection
    assert timestamps[0] < 60*30, f'Invalid time range. Start time {timestamps[0]}, need to be before first injection.'
    assert timestamps[-1] > 60*60, f'Invalid time range. End time {timestamps[-1]}, need to be After second injection.'

    k_clusters = np.unique(k_labels)  
    candidate_wf = np.empty((k_clusters.shape[0],waveforms.shape[-1]))
    prel_results = [] # Store result about responders. If no responders, then this remains empty. 
    print(f'Shape of test-dataset (now considers all observations): {waveforms.shape}')
    for cluster in k_clusters:
        hpdp_cluster = hpdp[k_labels==cluster]
        MAIN_CANDIDATE = np.median(hpdp_cluster,axis=0) # Median more robust to outlier. should however not be a problem using DBSCAN..

        added_main_candidate_wf = np.concatenate((MAIN_CANDIDATE.reshape((1,MAIN_CANDIDATE.shape[0])),waveforms),axis=0)
        assert np.sum(MAIN_CANDIDATE) == np.sum(added_main_candidate_wf[0,:]), 'Something wrong in concatenate..'

        if similarity_measure=='corr':
            print('Using "corr" to evaluate final result')
            correlations = wf_correlation(0,added_main_candidate_wf)
            bool_labels = label_from_corr(correlations,threshold=similarity_thresh,return_boolean=True)
        if similarity_measure=='ssq':
            print('Using "ssq" to evaluate final result')
            if assumed_model_varaince is None:
                #added_main_candidate_wf = added_main_candidate_wf/assumed_model_varaince  # (0.7) Assumed var in ssq
                bool_labels,_ = similarity_SSQ(0,added_main_candidate_wf,epsilon=similarity_thresh,standardised_input=False)
            else:
                added_main_candidate_wf = added_main_candidate_wf/assumed_model_varaince  # (0.7) Assumed var in ssq
                bool_labels,_ = similarity_SSQ(0,added_main_candidate_wf,epsilon=similarity_thresh)
        event_rate, _ = get_event_rates(timestamps,bool_labels[1:],bin_width=1,consider_only=1)
        
        _, baseline_SD = __get_ev_stats__(event_rate,start_time=t0_baseline_SD, end_time=30*60)
        baseline_MU,_ =  __get_ev_stats__(event_rate,start_time=t_injection-time_baseline_MU, end_time=t_injection)
        #baseline_MU_2,_ =  __get_ev_stats__(event_rate,timestamps,start_time=t_injection_2-time_baseline_MU, end_time=t_injection_2)
        
        SD_thesh = k * np.max((SD_min, baseline_SD))
        cytokine_stats = __get_ev_stats__(event_rate,start_time=t_injection+10*60, end_time=t_injection+30*60, 
                                            compare_to_theshold=baseline_MU+SD_thesh, conv_width=5)
        #print(f'Cytokine candidate responder result for injection 1 is : {cytokine_stats[2]}')
        #print(f'Cytokine candidate responder result for injection 2 is : {second_cytokine_stats[2]}')
        if cytokine_stats[2] is True:
            #plt.figure(1)
            #bool_labels[bool_labels==True] = cluster
            #plt.plot(event_rate)
            #plt.title(f'Responder-cluster = {cluster}')
            #plt.show()
            #plot_event_rates(event_rate,timestamps,noise=None,conv_width=100,saveas=saveas+'Main_cand_ev', verbose=True, cluster=cluster)
            prel_results.append(np.array([MAIN_CANDIDATE, cytokine_stats]))
            print(f'CAP nr. {cluster} found to have a sufficient increase in firing rate for injection {injection}.')
    return prel_results
        

def __get_ev_stats__(event_rate,start_time=10*60, end_time=90*60, compare_to_theshold=None,conv_width=5):
    '''
    Called by "__evaluate_cytokine_candidates__()".

    Get event-rate mean and standard deviation for a specified time-period.
    If "compare_to_threshold" is not None, then the EV is compared to thresh to see if we have a "responder".

    Returns
    -------
    if "compare_to_threshold" is not None:
        returns: MU, SD, responder, time_above_thresh
    else: 
        returns: MU, SD
    
    MU : float
        Mean event rate of considered period
    SD : float
        Mean Standard deviation of considered period
    responder : booleon
        True is the event-rate of specified period is larger than thesh for 1/3 of the period. 
    time_above_thresh : float
        How much time in seconds that the EV is above thresh.
    '''
    T = end_time-start_time # Length of time interval in seconds

    event_rate_ = event_rate[start_time:end_time] # Event-rate under time-period of interest.
    SD = np.std(event_rate_) # Standart deviation (SD) of period of interest
    MU = np.mean(event_rate_) # Mean (MU) event-rate under period of interest
    if compare_to_theshold is not None:
        responder  = False
        conv_kernel = np.ones((conv_width))* 1/conv_width
        smoothed_EV = np.convolve(np.squeeze(event_rate_),conv_kernel,'same') # Smooth out event_rate
        EV_above_thres = smoothed_EV > compare_to_theshold # Find all Event-rates larger then threshold
        time_above_thresh = np.sum(EV_above_thres) # Total time, in seconds, above specified threshold. 
        if time_above_thresh > T/3: # If responder, The EV has to be higher than thresh for at least 1/3 of the time period.
            responder = True
        return MU, SD, responder, time_above_thresh
    else:
        return MU, SD


def find_reponders(directory, start_string='', end_string='',specify_recordings='R10',saveas=None,verbose=False, matlab_directory ='../matlab_files',return_main_candidates=False):
    '''
    Main function to evaluate the saved results given by "run_evaluation()".

    Looks through the stored numpy files, saved by main-file which is the output of "run_evaluation()", to 
    find responders to the cytokine injections. All calculations are done, this is just a file to process the results.
    The files under consideration are saved in "directory" with a common start- and/or end-string, specified by input.

    OBS, there is an added string e.g. 'R10', when disciminating between the files. 
    This is set manually depending on which files that are under consideration.
    '''
    responders = [] # We will ad a 1 if a recording in specified directory corresponds to a "responder", otherwise 0. 
                    # This is used to find how many, out of all considered recordings, that are showing promising results. 
    main_candidates = []
    for entry in scandir(directory):
        if entry.path.startswith(directory+start_string+specify_recordings) & entry.path.endswith(end_string): # Specify any uniquness in the name of the files to be considered. 
            result = np.load(entry.path, allow_pickle=True)
            responder_bool = False
            for injection_res in result: # "result" is a nested list where the first two elements are the results of the different injections.
                if len(injection_res)==0: # If no responders where found for this injection
                    pass
                else: # A responder was found
                    responder_bool=True
                    recording = entry.path[len(directory+start_string):-len(end_string)] # Get the matlab_file of recording
                    print('**********************************************************************')
                    print(f'The recording which was found to be a responder : {recording}')
                    print()
                    if injection_res.shape==(2,): # Only one cluster fullfilled the requerement to be defined as a responder
                        waveform = injection_res[0]
                        main_candidates.append(waveform)  
                        if verbose:
                            __evaluate_responder__(waveform, matlab_directory, recording,
                                        similarity_measure='ssq', similarity_thresh = 0.3, assumed_model_varaince = 0.5,saveas=saveas)
                            
                    else: #  >1 clusters where found with qualities as responders (most often corresponding to very similar CAPs) 
                        # Loop through the different candidate CAPs/clusters to find the one with the "clearest" result as responder.
                        times_above_thresh = np.zeros((len(injection_res,))) 
                        for ii,responder in enumerate(injection_res):
                            stats = responder[1]
                            times_above_thresh[ii] = stats[3] # time above threshold
                            #waveform = responder[0]
                        main_candidate = np.argmax(times_above_thresh)
                        waveform = injection_res[main_candidate][0]
                        main_candidates.append(waveform)
                        if verbose: # Show plots of "total event rate", "candidate-CAP event rate" and "candidate-CAP" of the main-candidate/responder.
                            __evaluate_responder__(waveform, matlab_directory, recording,
                                        similarity_measure='ssq', similarity_thresh = 0.3, assumed_model_varaince = 0.5,saveas=saveas)
            if responder_bool:
                responders.append(1)
            else: 
                responders.append(0)
    print(f'Number of responders: {np.sum(responders)} out of {len(responders)}')
    if return_main_candidates:
        return responders, main_candidates
    else:
        return responders

def __evaluate_responder__(cytokine_candidate, matlab_directory, file_name,
                    similarity_measure='ssq', similarity_thresh = 0.3, assumed_model_varaince = 0.5,saveas=None):
    '''
    Called by "find_responers()" 

    Plots event rate for "cytokine_candidate"-waveform. This needs the matlab-recording to be specified.

    Parameters
    ----------

    '''
    for entry in scandir(matlab_directory):
        if entry.path.startswith(matlab_directory+'\\ts' +  file_name): #"\\tsR10"): # Find unique recording string
            matlab_file = entry.path[len(matlab_directory)+3:-4] # Find unique recording string'
            print('************* PLOTTING RESPONDER RESULTS ***********************')
            print('*******************************************************************************')
            print(f'Responder-Recording : {matlab_file}')
            print()
            print(f'Using {similarity_measure} to evaluate final result')
            path_to_wf = matlab_directory + '/wf'+matlab_file +'.mat'
            path_to_ts = matlab_directory + '/ts'+matlab_file +'.mat'

            savefig = saveas + matlab_file.replace('.','_') #file_names['figure_strings'][run_i]

            print()
            print(f'DATA FILE : {matlab_file}')
            load_data = True
            if load_data:
                wf0 = load_waveforms(path_to_wf,'waveforms', verbose=0) # Load the candidate's corresponding matlab file 
                ts0 = load_timestamps(path_to_ts,'timestamps',verbose=0)

            wf0,ts0 = preprocess_wf.get_desired_shape(wf0,ts0,start_time=10,end_time=90,dim_of_wf=141,desired_num_of_samples=None)
            wf0 = preprocess_wf.standardise_wf(wf0)
            
            added_main_candidate_wf = np.concatenate((cytokine_candidate.reshape((1,cytokine_candidate.shape[0])),wf0),axis=0)
            assert np.sum(cytokine_candidate) == np.sum(added_main_candidate_wf[0,:]), 'Something wrong in concatenate..'

            #print(f'Shape of test-dataset (now considers all observations): {added_main_candidate_wf.shape}')

            if similarity_measure=='corr':
                #print('Using "corr" to evaluate final result')
                correlations = wf_correlation(0,added_main_candidate_wf)
                bool_labels = label_from_corr(correlations,threshold=similarity_thresh,return_boolean=True)
            if similarity_measure=='ssq':
                #print('Using "ssq" to evaluate final result')
                added_main_candidate_wf = added_main_candidate_wf/assumed_model_varaince  # (0.7) Assumed var in ssq
                bool_labels,_ = similarity_SSQ(0,added_main_candidate_wf,epsilon=similarity_thresh)
            event_rates, _ = get_event_rates(ts0,bool_labels[1:],bin_width=1,consider_only=1)
            # Plot titles etc.
            wf_title = 'Candidate-CAP'
            overall_ev_title = 'Event-Rate for all Observed CAPs'
            cluster_ev_title = 'Event-Rate for Candidate-CAP'
            plt.figure(1)
            plot_similar_wf(0,added_main_candidate_wf,bool_labels,similarity_thresh,saveas=savefig+'Main_cand'+'_wf',
                                verbose=False, show_clustered=False,cluster='Mean',title=wf_title)
            plt.figure(2)
            plot_event_rates(event_rates,ts0,noise=None,conv_width=100,saveas=savefig+'Main_cand'+'_ev', verbose=False,title=cluster_ev_title) 
            plt.figure(3)
            event_rates, _ = get_event_rates(ts0,np.ones((ts0.shape[0],)),bin_width=1,consider_only=1)
            plot_event_rates(event_rates,ts0,noise=None,conv_width=100,saveas=savefig+'overall_EV', verbose=False,title=overall_ev_title ) 

            plt.show()

def eval_candidateCAP_on_multiple_recordings(candidate_CAP,matlab_directory,similarity_measure='ssq',
                                        similarity_thresh=0.4,assumed_model_varaince=0.5,saveas='Not_specified'):
    '''
    Use candidate CAP to find similar waveforms in different recordings. 
    
    '''
    __evaluate_responder__(candidate_CAP, matlab_directory, '', similarity_measure=similarity_measure, 
                        similarity_thresh = similarity_thresh, assumed_model_varaince=assumed_model_varaince,saveas=saveas)





def run_DBSCAN_evaluation(wf_ho,wf0,ts0,ev_label_ho,labels_to_evaluate=[0,1], saveas=None, np_saveas = None, 
                db_eps=7, db_min_sample=10,matlab_file=None,similarity_measure='ssq',
                similarity_thresh=0.1, assumed_model_varaince=0.5):
    '''
    Skipp everything after labeling and perform clustering using DBSCAN on labeled high-occurance waveforms. 
    '''
    cytokine_candidates = np.empty((2,wf_ho.shape[-1])) # To save the main candidates
    for label_on in labels_to_evaluate:
        waveforms_increase = wf_ho[ev_label_ho[:,label_on]==1]
        if waveforms_increase.shape[0] == 0:
            waveforms_increase = np.append(np.zeros((1,141)),waveforms_increase).reshape((1,141))
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

        possible_wf_candidates = evaluate_hpdp_candidates(wf0,ts0,waveforms_increase,labels,saveas=saveas,similarity_measure=similarity_measure,
                                similarity_thresh=similarity_thresh, assumed_model_varaince=assumed_model_varaince,verbose=True, return_candidates=True)
        
        k_candidate  = input('which CAP-cluster seems most likely to encode the cytokine? (integer or None) :')
        if k_candidate != 'None':
            cytokine_candidates[label_on,:] = possible_wf_candidates[int(k_candidate),:]
    if k_candidate is not None:
        np.save(np_saveas+'DBSCAN', cytokine_candidates)