'''
Running visualisations of different steps workflow. 
For model assessment and tuning hyper-params etc.

************************************************************
'''

import numpy as np
import matplotlib.pyplot as plt
import json
from os import path, scandir, sys

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

sys.path.insert(1,'src/')

import preprocess_wf 
from load_and_GD_funs import load_mat_file, get_pdf_model
from wf_similarity_measures import similarity_SSQ, wf_correlation
from event_rate_funs import get_event_rates
from plot_functions_wf import *
from evaluation import  marginal_log_likelihood

# **************** WHAT TO PLOT: ************************
verbose_main = True   # Wether to show or just save figures

plot_raw_CAPs = False        # Grid of 16 examples of observed CAPs and one of these as full figure.
plot_similar_CAPs = False    # Plot example of event-rate and similarity cluster (both normalised and in microV) for a CAP.
plot_ev_stats = False        # Histogram of Mean and standard-deviation of event-rates for a recording.
plot_TOT_EV = False          # Total Event-Rate for recordings.
plot_ho_TOT_EVs = False      # Total Event-Rate for recordings after ev-threshold. 
plot_ho_EVs = False          # Plot example of event-rate and similarity cluster (normalised) for a CAP after EV-threshold.
view_encoded_latent = False   # Plot CAPs encoded as points in latent-space.
view_decoded_latent = False   # Takes (6 x 6) samples from grid in latent space and plots the decoded x-mean.
plot_simulatated_path_from_model = False     # Plots example waveform together with the corresponding predicted mean using the cvae 
                                             # and a samples from the model distribution: x_sim ~ N(mean, cvae_var*I)
plot_wf_and_ev_for_the_different_ev_labels = False   # Example of similarity clusters and corresponding ev for CAPs that has been labeled
                                                    # as "increase_after_first-" and "increase_after_second-" injection. 
plot_acf_pacf = False         # ACF and PACF for CVAE-model noise some example-CAPs.
evaluate_probabilities = False  # Estimation of log-likelihood for responders or random samples using importance sampling.

# ************************************************
directory = '../matlab_files'
directory = 'MATLAB/preprocessed2'
figures_directory = 'figures_main_vis/'

training_start_title = 'all_chan_KI_10min' # 'HereWeGoAgain'   # Specify title of the "run" to use.

rec_start_string = '_final2Baseline' # Baseline_10min_LPS_10min_KCl_10min_210617_142447A-001' #_6.28.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_03' # Since each recording has two files in directory (waveforms and timestamps)-- this is solution to only get each recording once.
# rec_start_string = '\\ts_Baseline_10min_LPS_10min_KCl_10min_210617_142447A-011' # Baseline_10min_LPS_10min_KCl_10min_210617_142447A-001' #_6.28.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_03' # Since each recording has two files in directory (waveforms and timestamps)-- this is solution to only get each recording once.

# ************************************************************
# ******************** Hyperparameters ****************************
# ************************************************************
with open('hypes/' + training_start_title + '.json', 'r') as f:
    hypes = json.load(f)
# *****************************************************

responder_count = 0
responder_rec_list = hypes["Results"]["Responders"]   # The found responders from evaluation
for entry in scandir(directory):
    # Uncomment the next line if you only want to consider the found responders from the evaluation. (saved in .json file)
    # rec_start_string = '\\ts' + responder_rec_list[responder_count] 
    file_name = directory + '/ts' + rec_start_string
    if entry.name.startswith('ts' + rec_start_string):    # Find unique recording string. tsR10 for all cytokine injections, tsR12 for saline. 
        matlab_file = entry.path[len(directory+'\\ts'):-len('.mat')]    # extract only the matlab_file name from string.
        print('\n'+'*'*40)
        print(f'Starting analysis on recording : {matlab_file} \n')

        path_to_matlab_wf = directory + '/wf' + matlab_file + '.mat'    # This is how the waveforms are saved. path + wf+MATLAB_FILE_NAME.mat
        path_to_matlab_ts = directory + '/ts' + matlab_file + '.mat'    # This is how the timestamps are saved. path + ts+MATLAB_FILE_NAME.mat

        ################################################
        unique_string_for_run = training_start_title + matlab_file
        unique_for_figs = training_start_title + matlab_file.replace('.', '_')
        
        path_to_model_weights = 'models/'+unique_string_for_run

        # **************** Paths to save Figures************************
        saveas_raw_CAPs = figures_directory + 'raw_CAPs/' + unique_for_figs
        saveas_similar_CAPs = figures_directory + 'similar_CAPs/' + unique_for_figs
        saveas_ev_stats = figures_directory + 'event_rate_stats/' + unique_for_figs
        saveas_TOT_HO_EV = figures_directory + 'TOT_EV/' + unique_for_figs
        saveas_TOT_HO_EV = figures_directory + 'TOT_EV/HO_' + unique_for_figs
        saveas_ho_EVs = figures_directory + 'event_rate_labels/' + unique_for_figs
        saveas_vae_result = figures_directory + 'encoded_decoded/' + unique_for_figs
        saveas_simulatated_path_from_model = figures_directory + 'model_assessment/' + unique_for_figs
        saveas_wf_and_ev_for_the_different_ev_labels = figures_directory + 'event_rate_labels/' + unique_for_figs
        savefig_acf_pacf = figures_directory + 'acf_pacf/' + unique_for_figs
        savefig_log_likes = figures_directory + 'log_likes/' + unique_for_figs

        # ******* Numpy File Paths ***********
        path_to_hpdp = "../numpy_files/numpy_hpdp/" + unique_string_for_run
        path_to_EVlabels = "../numpy_files/EV_labels/" + unique_string_for_run

        # ************************************************************
        # ******************** Load Files ****************************
        # ************************************************************
        load_data = True
        if load_data:
            waveforms = load_mat_file(path_to_matlab_wf, 'waveforms', verbose=1)
            timestamps = load_mat_file(path_to_matlab_ts, 'timestamps', verbose=1)

        # ************************************************************
        # ******************** Plot "raw" CAPs ***********************
        # ************************************************************
        if plot_raw_CAPs:
            plt.figure(1)
            plot_waveforms_grid(waveforms, 4, saveas=saveas_raw_CAPs,
                                verbose=False, title=None)
            plt.figure(2)
            plot_waveforms(waveforms[3,:], labels=None, 
                           saveas=saveas_raw_CAPs+'_zoomed', 
                           verbose=False, title='CAP-waveform')
            if verbose_main:
                plt.show()
            else:
                plt.close()

        # ************************************************************
        # ******************** Preprocess ****************************
        # ************************************************************
        # Remove "extreme-amplitude" CAPs-- otherwise risk that pdf-GD diverges:
        waveforms, timestamps = preprocess_wf.apply_amplitude_thresh(waveforms, 
                                                                     timestamps, 
                                                                     hypes)
        wf0, ts0 = preprocess_wf.get_desired_shape(waveforms, 
                                                    timestamps, 
                                                    hypes, training=False)
        waveforms, timestamps = preprocess_wf.get_desired_shape(waveforms, 
                                                                timestamps, 
                                                                hypes, training=False)

        # Standardise wavefroms: 
        waveforms = preprocess_wf.standardise_wf(waveforms, hypes)

        # ************************************************************
        # ******************** Plot "similar" CAPs ***********************
        # ************************************************************
        if plot_similar_CAPs:
            title_similarity = 'Similarity Cluster Given Candidate'
            saveas = saveas_similar_CAPs
            assumed_variance = hypes["labeling"]["assumed_model_varaince"]
            epsilon = hypes["labeling"]["similarity_thresh"]
            similarity_measure = hypes["labeling"]["similarity_measure"]
            # assumed_variance = 1
            # epsilon = 10

            #try:
            for i in [10,400,800]:
                if similarity_measure == 'ssq':
                    wf_ssq = waveforms / assumed_variance
                    bool_labels, _ = similarity_SSQ(i, wf_ssq, epsilon=epsilon, standardised_input=True)
                elif similarity_measure == 'corr':
                    correlations = wf_correlation(i, waveforms)
                    bool_labels = correlations > epsilon
                event_rates = get_event_rates(timestamps, hypes, labels=bool_labels, consider_only=1)
                tot_event_rates = get_event_rates(timestamps, hypes, labels=None, consider_only=1)
                plt.figure(1) 
                plot_similar_wf(i, waveforms, bool_labels, 
                                saveas=saveas+'_wf'+str(i) + '_thresh_' + str(epsilon).replace('.', '_'), 
                                verbose=False,
                                title=title_similarity)
                if not verbose_main:
                    plt.close()
                plt.figure(2) 
                plot_event_rates(event_rates, timestamps, hypes,
                                 saveas=saveas+'_ev'+str(i) + '_thresh_' + str(epsilon).replace('.', '_'), 
                                 verbose=False) 
                plt.figure(3) 
                plot_similar_wf(i, wf0, bool_labels, 
                                saveas=saveas+'non_normalised_wf'+str(i) + '_thresh_' + str(epsilon).replace('.', '_'), 
                                verbose=False, 
                                title=title_similarity)
                
                if verbose_main:
                    plt.show()
                plt.close()
            #except:
            #    print('\nOBSOBS!\nNot enough waveforms to plot the specified indicies...')
        if plot_TOT_EV:
            
            tot_event_rates = get_event_rates(timestamps, hypes, labels=None, consider_only=1)
            saveas = saveas_TOT_HO_EV
            plt.figure(4) 
            plot_event_rates(tot_event_rates, timestamps, hypes,
                                saveas=saveas, 
                                verbose=False, title='Total Event-Rate') 
            if verbose_main:
                plt.show()
            plt.close()

        # ************************************************************
        # ******************** Event-rate Labeling *******************
        # ************************************************************

        if path.isfile(path_to_EVlabels+'.npy'):
            print(f'\n Loading saved EV-labels from {path_to_EVlabels} \n')
            ev_labels = np.load(path_to_EVlabels + '.npy')
            ev_stats_tot = np.load(path_to_EVlabels + 'stats_tot.npy') 
        else:
            print(f'OBS: did not find {path_to_EVlabels}... \n \
                    Visualisation Recuires saved files by "main_train.py" ')

        # ************************************************************
        # ******************** Event-rate-stats **********************
        # ************************************************************
        if plot_ev_stats:
            saveas = saveas_ev_stats
            plot_event_rate_stats_hist(ev_stats_tot, saveas=saveas, 
                                       verbose=verbose_main)
        
        # ************************************************************
        # ************* High occurance WF: **********************
        # ************************************************************
        # Consider only waveforms with high occurance event rates.
        print(f'Number of wf which ("icreased after first","increased after second", "constant") = {np.sum(ev_labels,axis=1)} ')
        # ho := High Occurance, ts := timestamps
        wf_ho, ts_ho, ev_label_ho = preprocess_wf.apply_mean_ev_threshold(waveforms, timestamps, ev_stats_tot[0], hypes, ev_labels=ev_labels)

        print(f'After EV threshold: ("icreased after first","increased after second", "constant") = {np.sum(ev_label_ho,axis=0)} ')

        if plot_ho_TOT_EVs:
            ev_ho = get_event_rates(ts_ho, hypes )
            saveas = saveas_TOT_HO_EV
            KI_channel = matlab_file[-5:]
            plot_event_rates(ev_ho, ts_ho, hypes, saveas=saveas, 
                             verbose=verbose_main)
            plt.close()

        # Plotting EV etc. done for High occurance CAPs... :
        if plot_ho_EVs:
            # threshold = 0.6
            # threshold = hypes["labeling"]["similarity_thresh"]
            title_similarity = 'Similarity Cluster Given Candidate'
            saveas = saveas_ho_EVs
            assumed_variance = hypes["labeling"]["assumed_model_varaince"]
            epsilon = hypes["labeling"]["similarity_thresh"]
            similarity_measure = hypes["labeling"]["similarity_measure"]
            # assumed_variance = 1
            # epsilon = 10

            try:
                for i in [10,400,800]:
                    if similarity_measure == 'ssq':
                        wf_ho = wf_ho / assumed_variance
                        bool_labels, _ = similarity_SSQ(i, wf_ho, epsilon=epsilon, standardised_input=True)
                    elif similarity_measure == 'corr':
                        correlations = wf_correlation(i, wf_ho)
                        bool_labels = correlations > epsilon
                        
                    event_rates = get_event_rates(ts_ho, hypes, labels=bool_labels, consider_only=1)
                    plt.figure(1) 
                    plot_similar_wf(i, wf_ho, bool_labels, saveas=saveas+'_wf'+str(i), verbose=False, title=title_similarity)

                    plt.figure(2) 
                    plot_event_rates(event_rates, ts_ho, hypes, saveas=saveas+'_ev'+str(i), 
                                     verbose=False) 
                    if verbose_main:
                        plt.show()
                    plt.close()
            except Exception as e:
                print('\n[plot_ho_EVs] \nOBSOBS!\nMost Likely not enough waveforms to plot the specified indicies...')
                print(f'Full error : {e}')
        # ************************************************************
        # ******************** Train/Load model **********************
        # ************************************************************
        print('\n *********************** Tensorflow Verbosity.. ************************************* \n')

        encoder,decoder,cvae = get_pdf_model(wf_ho, hypes,
                                             path_to_weights=path_to_model_weights,
                                             continue_train=False, verbose=1, 
                                             ev_label=ev_label_ho)
        print('\n ******************************************************************************\n')

        # ************************************************************
        # **** PLOT Simulated wf. from N(mu_x,I) (CVAE) *********************
        # ************************************************************

        if plot_simulatated_path_from_model:
            model_variance = hypes["cvae"]["model_variance"]
            dim_of_wf = hypes["preprocess"]["dim_of_wf"]
            print(f'\n Model variance is set to : {model_variance} \n')
            for jj in [10, 40, 80]:
                saveas = saveas_simulatated_path_from_model + str(jj)
                x = wf_ho[jj, :].reshape((1, dim_of_wf))
                label = ev_label_ho[jj, :].reshape((1, 3))
                plot_simulated(cvae, x, ev_label=label, n=1, var=model_variance, saveas=saveas, verbose=verbose_main)
                
                plt.close()
            print('Done.')


        # ************************************************************
        # ******** Plot examples of event-rates from EV_labeles ******
        # ************************************************************
        if view_encoded_latent:
            encoded_hpdp_title = 'Encoded Latent Variable Mean.'
            plot_encoded(encoder, wf_ho, saveas=saveas_vae_result+'_encoded_ho_wf',
                         verbose=verbose_main, ev_label=ev_label_ho, title=encoded_hpdp_title)
            plt.close()

        if view_decoded_latent:
            plot_decoded_latent(decoder, saveas=saveas_vae_result+'_decoded_constant', 
                                verbose=verbose_main, ev_label=np.array((0, 0, 1)).reshape((1, 3)))
            plot_decoded_latent(decoder, saveas=saveas_vae_result+'_decoded_increase_first', 
                                verbose=verbose_main, ev_label=np.array((1, 0, 0)).reshape((1, 3)))
            plot_decoded_latent(decoder, saveas=saveas_vae_result+'_decoded_increase_second', 
                                verbose=verbose_main, ev_label=np.array((0, 1, 0)).reshape((1, 3)))

        # ************************************************************
        # ******** Plot examples of event-rates from EV_labeles ******
        # ************************************************************
        if plot_wf_and_ev_for_the_different_ev_labels:
            similarity_measure = hypes["labeling"]["similarity_measure"]
            epsilon = hypes["labeling"]["similarity_thresh"]
            # threshold = 0.6
            idx_increase_after_first = np.where(ev_labels[0, :] == 1)
            idx_increase_after_second = np.where(ev_labels[1, :] == 1)
            idx_constant_throughout = np.where(ev_labels[2, :] == 1)
            
            for injection in [0, 1]:
                idx_increase = np.where(ev_labels[injection, :] == 1)[0]
                saveas = saveas_wf_and_ev_for_the_different_ev_labels + 'injection_' + str(injection)
                print(f'plotting wf and ev for injection : {injection}')    # 0="increase after first", 1="increase after second"
                assumed_variance = hypes["labeling"]["assumed_model_varaince"]
                try:
                    for i in idx_increase[[10, 40, 80]]:
                        if similarity_measure == 'ssq':
                            waveforms_ = waveforms/assumed_variance
                            bool_labels, _ = similarity_SSQ(i, waveforms_, epsilon=epsilon, standardised_input=True)
                        elif similarity_measure == 'corr':
                            waveforms_ = waveforms
                            correlations = wf_correlation(i, waveforms_)
                            bool_labels = correlations > epsilon
                        
                        event_rates = get_event_rates(timestamps[:, 0], hypes, labels=bool_labels, consider_only=1)
                        # delta_ev, ev_stats = __delta_ev_measure__(event_rates)
                        plt.figure(1)
                        plot_similar_wf(i, waveforms, bool_labels,
                                        saveas=saveas+'_wf_'+str(i),
                                        verbose=False)
                        
                        plt.figure(2)
                        plot_event_rates(event_rates, timestamps, hypes,
                                         saveas=saveas+'_wf_'+str(i)+'_ev', 
                                         verbose=False)
                        if verbose_main:
                            plt.show()
                        else:
                            plt.close()
                except Exception as e:
                    print('\n[plot_wf_and_ev_for_the_different_ev_labels] \nOBSOBS!\nMost Likely not enough waveforms to plot the specified indicies...')
                    print(f'Full error : {e}')
        # Assming we have loaded the standardised waveforms : std_waveforms
        # 
        # import matplotlib.gridspec as gridspec
        # ************************************************************
        # ******** Quick look at ACF/PACF  ***************************
        # ************************************************************
        if plot_acf_pacf:
            dim_of_wf = hypes["preprocess"]["dim_of_wf"]
            i = 0
            saveas = savefig_acf_pacf
            print(f'\n Plotting ACF and PACF...')
            for j in [10, 40, 80]:
                # wf_dim = waveforms.shape[-1]
                # t_axis = np.arange(0,3.5,3.5/wf_dim)
                fig, (ax0, ax1, ax3) = plt.subplots(ncols=3, 
                                                    constrained_layout=True, 
                                                    figsize=(12,3))
                reconstructed = cvae.predict([waveforms[j, :].reshape((1, dim_of_wf)),
                                              ev_labels[:,j].reshape(1,3)])
                noise = waveforms[j,:] - reconstructed    # Remove modeled mean results in model-noise.
                plot_acf(noise[0, :], ax=ax0)
                plot_pacf(noise[0, :], ax=ax1)
                ax3.plot(noise[0, :])
                # ax3.plot(t_axis.T, noise[0,:])
                plt.title('Model Noise: $x-\mu_x$')
                i += 1
                plt.savefig(saveas+'_wf_'+str(j)+'.png', dpi=150)
                if verbose_main:
                    plt.show()
                else:
                    plt.close()

        if evaluate_probabilities:
            use_sample_from_responder_rec = False
            use_responders = True
            saveas_responder_caps = '../numpy_files/responder_CAPs/' + training_start_title
            responder_CAPs = np.load(saveas_responder_caps + '.npy')
            responder_labels = np.asarray(hypes["Results"]["labels"])

            N_evals = 1000
            probs = []
            # Calculate marginal liklihood.
            if use_sample_from_responder_rec:
                use_label = responder_labels[responder_count,:]
                label_nr = np.where(use_label==1)
                idx_for_label_one = np.where(ev_label_ho[:,label_nr] == 1)
                for wf_idx in idx_for_label_one[0][0:N_evals*10:10]:
                    v_prob = []
                    for i in range(1):
                        log_prob_x = marginal_log_likelihood(wf_ho[wf_idx, :], use_label, encoder, decoder, hypes)
                        v_prob.append(log_prob_x)
                    probs.append(np.mean(v_prob))
                plt.hist(probs, bins=100)
                plt.savefig(savefig_log_likes + '.png', dpi=150)
                plt.close()
                if verbose_main:
                    plt.show()
                print(f'mean of likelihood = {np.mean(probs)}, using N = {len(probs)} samples.')
            if use_responders:            
                probs = []
                label_i = 0
                for responder_CAP in responder_CAPs:
                    v_prob = []
                    for i in range(10):
                        log_prob_x = marginal_log_likelihood(responder_CAP, responder_labels[label_i,:], encoder, decoder, hypes)
                        v_prob.append(log_prob_x)
                    probs.append(np.mean(v_prob))
                    label_i  += 1
                if verbose_main:
                    plt.hist(probs)
                    plt.show()
                print(probs)
                print(f'mean of likelihood = {np.mean(probs)}, for responder CAPs. (mean of 10 runs.)')
            
        responder_count += 1


print('Done...')