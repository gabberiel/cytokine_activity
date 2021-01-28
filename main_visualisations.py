# ************************************************************
# Running visualisations of workflow. 
# Results -- tuning hyper params etc.
# ************************************************************

import numpy as np
import matplotlib.pyplot as plt
import preprocess_wf 
import json
from os import path, scandir
from load_and_GD_funs import load_waveforms, load_timestamps, get_pdf_model
from wf_similarity_measures import similarity_SSQ
from event_rate_funs import get_event_rates, __delta_ev_measure__
from plot_functions_wf import *
from evaluation import run_DBSCAN_evaluation, run_evaluation, marginal_log_likelihood
from scipy.spatial.distance import cdist

from scipy import stats

from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# **************** WHAT TO PLOT: ************************
verbose_main = True   # Wether to show or just save figures

plot_raw_CAPs = False
plot_ev_stats = False
plot_ho_EVs = False
view_encoded_latent = False
view_decoded_latent = False   
plot_simulatated_path_from_model = True
plot_wf_and_ev_for_the_different_ev_labels = False
plot_acf_pacf = False
evaluate_probabilities = False
###############################
directory = '../matlab_files'
figures_directory = 'figures_tests/'

training_start_title = 'test_run2'   # Specify title of the "run" to use.

rec_start_string = '\\tsR10' #.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05' # Since each recording has two files in directory (waveforms and timestamps)-- this is solution to only get each recording once.
rec_start_string = '\\tsR10_6.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05' #.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05' # Since each recording has two files in directory (waveforms and timestamps)-- this is solution to only get each recording once.
#rec_start_string = '\\tsR10_6.30.16_BALBC_TNF(0.5ug)_IL1B(35ngperkg)_05' #.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05' # Since each recording has two files in directory (waveforms and timestamps)-- this is solution to only get each recording once.

# ************************************************************
# ******************** Hyperparameters ****************************
# ************************************************************
with open('hypes/'+training_start_title+'.json', 'r') as f:
    hypes = json.load(f)
# *****************************************************

for entry in scandir(directory):
    if entry.path.startswith(directory+rec_start_string):    # Find unique recording string. tsR10 for all cytokine injections, tsR12 for saline. 
        matlab_file = entry.path[len(directory+'\\ts'):-len('.mat')]    # extract only the matlab_file name from string.
        print(' \n *******************************************************************************')
        print(f'Starting analysis on recording : {matlab_file} \n')

        path_to_matlab_wf = directory + '/wf' + matlab_file + '.mat'    # This is how the waveforms are saved. path + wf+MATLAB_FILE_NAME.mat
        path_to_matlab_ts = directory + '/ts' + matlab_file + '.mat'    # This is how the timestamps are saved. path + ts+MATLAB_FILE_NAME.mat

        ################################################
        unique_string_for_run = training_start_title + matlab_file
        unique_for_figs = training_start_title + matlab_file.replace('.', '_')
        
        path_to_model_weights = 'models/'+unique_string_for_run

        # **************** Paths to save Figures************************
        saveas_raw_CAPs = figures_directory + 'raw_CAPs/' + unique_for_figs
        saveas_ev_stats = figures_directory + 'event_rate_stats/' + unique_for_figs
        saveas_ho_EVs = figures_directory + 'event_rate_labels/' + unique_for_figs
        saveas_vae_result = figures_directory + 'encoded_decoded/' + unique_for_figs
        saveas_simulatated_path_from_model = figures_directory + 'model_assessment/' + unique_for_figs
        saveas_wf_and_ev_for_the_different_ev_labels = figures_directory + 'event_rate_labels/' + unique_for_figs
        savefig_acf_pacf = figures_directory + 'acf_pacf/' + unique_for_figs

        # ******* Numpy File Paths ***********
        path_to_hpdp = "../numpy_files/numpy_hpdp/" + unique_string_for_run
        path_to_EVlabels = "../numpy_files/EV_labels/" + unique_string_for_run

        # ************************************************************
        # ******************** Load Files ****************************
        # ************************************************************
        load_data = True
        if load_data:
            waveforms = load_waveforms(path_to_matlab_wf, 'waveforms', verbose=1)
            timestamps = load_timestamps(path_to_matlab_ts, 'timestamps', verbose=1)
        # ************************************************************
        # ******************** Plot "raw" CAPs ***********************
        # ************************************************************
        if plot_raw_CAPs:
            plot_waveforms_grid(waveforms, 4, saveas=saveas_raw_CAPs,
                                verbose=verbose_main, title=None)
            plt.close()
            plot_waveforms(waveforms[3,:], labels=None, 
                           saveas=saveas_raw_CAPs+'_zoomed', 
                           verbose=verbose_main, title='CAP-waveform')
            plt.close()
        # ************************************************************
        # ******************** Preprocess ****************************
        # ************************************************************
        # Remove "extreme-amplitude" CAPs-- otherwise risk that pdf-GD diverges:
        waveforms, timestamps = preprocess_wf.apply_amplitude_thresh(waveforms, 
                                                                     timestamps, 
                                                                     hypes)
        waveforms, timestamps = preprocess_wf.get_desired_shape(waveforms, 
                                                                timestamps, 
                                                                hypes)
        # Standardise wavefroms: 
        waveforms = preprocess_wf.standardise_wf(waveforms)

        # ************************************************************
        # ******************** Event-rate Labeling *******************
        # ************************************************************
        if path.isfile(path_to_EVlabels+'.npy'):
            print(f'\n Loading saved EV-labels from {path_to_EVlabels} \n')
            ev_labels = np.load(path_to_EVlabels+'.npy')
            ev_stats_tot = np.load(path_to_EVlabels+'tests_tot.npy') 
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

        # Plotting EV etc. done for High occurance CAPs... :
        if plot_ho_EVs:
            threshold = 0.6
            title_similarity = 'Similarity Cluster Given Candidate'
            saveas = saveas_ho_EVs
            for i in [10,40,80]:
                bool_labels, _ = similarity_SSQ(i, wf_ho, epsilon=0.1, var=0.7, standardised_input=True)
                event_rates = get_event_rates(ts_ho, bool_labels, bin_width=1, consider_only=1)
                delta_ev, ev_stats = __delta_ev_measure__(event_rates, timestamps=ts_ho)
                plot_similar_wf(i, wf_ho, bool_labels, threshold, saveas=saveas+'_wf'+str(i), verbose=verbose_main, title=title_similarity)
                plot_event_rates(event_rates, ts_ho, noise=None, conv_width=100, saveas=saveas+'_ev'+str(i), verbose=verbose_main) 
                plt.close()

        # ************************************************************
        # ******************** Train/Load model **********************
        # ************************************************************
        print('\n *********************** Tensorflow Blaj ************************************* \n')

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
            print(f'\n Model variance is set to : {model_variance} \n')
            for jj in [10, 40, 80]:
                saveas = saveas_simulatated_path_from_model + str(jj)
                x = wf_ho[jj, :].reshape((1, 141))
                label = ev_label_ho[jj, :].reshape((1, 3))
                plot_simulated(cvae, x, ev_label=label, n=1, var=model_variance, saveas=saveas, verbose=verbose_main)
                
                plt.close()
            print('Done.')


        # ************************************************************
        # ******** Plot examples of event-rates from EV_labeles ******
        # ************************************************************
        if view_encoded_latent:
            encoded_hpdp_title = 'Encoded Latent Variable Mean.'
            plot_encoded(encoder, wf_ho, saveas=saveas_vae_result+'_encoded_ho_wf', verbose=verbose_main,ev_label=ev_label_ho,title=encoded_hpdp_title)
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
            threshold = 0.6
            idx_increase_after_first = np.where(ev_labels[0, :] == 1)
            idx_increase_after_second = np.where(ev_labels[1, :] == 1)
            idx_constant_throughout = np.where(ev_labels[2, :] == 1)
            
            for cluster in [0, 1]:
                idx_increase = np.where(ev_labels[cluster, :] == 1)
                saveas = saveas_wf_and_ev_for_the_different_ev_labels + 'cluster_' + str(cluster)
                print(f'plotting wf and ev for cluster : {cluster}')    # 0="increase after first", 1="increase after second"
                for i in idx_increase[0][10, 40, 80]:
                    bool_labels, _ = similarity_SSQ(i, waveforms, epsilon=0.1, var=0.7, standardised_input=True)
                    event_rates = get_event_rates(timestamps[:, 0], bool_labels, bin_width=1, consider_only=1)
                    delta_ev, ev_stats = __delta_ev_measure__(event_rates)
                    plot_similar_wf(i, waveforms, bool_labels,
                                    threshold, saveas=saveas+'_wf_'+str(i),
                                    verbose=verbose_main)
                    plot_event_rates(event_rates, timestamps,
                                     noise=None, conv_width=100,
                                     saveas=saveas+'_wf_'+str(i)+'_ev', 
                                     verbose=verbose_main)
            
        # Assming we have loaded the standardised waveforms : std_waveforms
        # 
        # import matplotlib.gridspec as gridspec
        # ************************************************************
        # ******** Quick look at ACF/PACF  ***************************
        # ************************************************************
        if plot_acf_pacf:
            i = 0
            saveas = savefig_acf_pacf
            print(f'\n Plotting ACF and PACF...')
            for j in [10, 40, 80]:
                # wf_dim = waveforms.shape[-1]
                # t_axis = np.arange(0,3.5,3.5/wf_dim)
                fig, (ax0, ax1, ax3) = plt.subplots(ncols=3, 
                                                    constrained_layout=True, 
                                                    figsize=(12,3))
                reconstructed = cvae.predict([waveforms[j, :].reshape((1, 141)),
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

        if evaluate_probabilities:
            saveas_responder_caps = '../numpy_files/responder_CAPs/' + training_start_title
            responder_CAPs = np.load(saveas_responder_caps + '.npy')

            N_evals = 10
            probs = []
            # Calculate marginal liklihood.
            idx_for_label_one = np.where(ev_label_ho[:,2] == 1)
            for wf_idx in idx_for_label_one[0][0:N_evals*10:10]:
                v_prob = []
                for i in range(10):
                    log_prob_x = marginal_log_likelihood(wf_ho[wf_idx, :], np.array([0,0,1]), encoder, decoder, hypes)
                    v_prob.append(log_prob_x)
                probs.append(np.mean(v_prob))
            plt.hist(probs, bins=50)
            plt.show()
            print(f'mean of likelihood = {np.mean(probs)}, using N = {len(probs)} samples.')
            # Calculate marginal liklihood fr.
            # plt.plot(responder_CAPs[0])
            # plt.show()
            probs = []
            labels = np.array([[0,1,0],[1,0,0],[0,1,0],[1,0,0],[1,0,0],[0,1,0]])
            label_i = 0
            for responder_CAP in responder_CAPs:
                v_prob = []
                for i in range(10):
                    log_prob_x = marginal_log_likelihood(responder_CAP, labels[label_i,:], encoder, decoder, hypes)
                    v_prob.append(log_prob_x)
                probs.append(np.mean(v_prob))
                label_i  += 1
            plt.hist(probs)
            plt.show()
            print(probs)
            print(f'mean of likelihood = {np.mean(probs)}, for responder CAPs. (mean of 10 runs.)')
            
