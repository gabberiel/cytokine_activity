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
from wf_similarity_measures import wf_correlation, similarity_SSQ, label_from_corr
from event_rate_funs import get_ev_labels, get_event_rates,__delta_ev_measure__
from plot_functions_wf import *
from evaluation import run_DBSCAN_evaluation, run_evaluation
from scipy.spatial.distance import cdist

from scipy import stats

from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# **************** WHAT TO PLOT: ************************
verbose_main = True
plot_raw_CAPs = False
plot_ev_stats = False
plot_ho_EVs = True
view_encoded_latent = False
view_decoded_latent = False    # True => reqires user to give input if to continue the script to pdf-GD or not.. 
plot_simulatated_path_from_model = False
plot_wf_and_ev_for_the_different_ev_labels = False
plot_acf_pacf = False


# RESPONDERS:
# matlab_file = 'R10_6.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05' # 'R10'  / 'R12' for all case/control


###############################
directory = '../matlab_files'
training_start_title = 'finalrun_first'
rec_start_string = '\\tsR10' #.30.16_BALBC_IL1B(35ngperkg)_TNF(0.5ug)_05' # Since each recording has two files in directory (waveforms and timestamps)-- this is solution to only get each recording once.

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
        saveas_raw_CAPs = 'figures_tests/raw_CAPs/' + unique_for_figs
        saveas_ev_stats = 'figures_tests/event_rate_stats/' + unique_for_figs
        saveas_ho_EVs = 'figures_tests/event_rate_labels/' + unique_for_figs
        saveas_vae_result = 'figures_tests/encoded_decoded/' + unique_for_figs
        saveas_simulatated_path_from_model = 'figures_tests/model_assessment/' + unique_for_figs
        saveas_wf_and_ev_for_the_different_ev_labels = 'figures_tests/event_rate_labels/' + unique_for_figs
        savefig_acf_pacf = 'figures_tests/acf_pacf/' + unique_for_figs

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
                                       verbose=False)
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
                event_rates, real_clusters = get_event_rates(ts_ho, bool_labels, bin_width=1, consider_only=1)
                delta_ev, ev_stats = __delta_ev_measure__(event_rates, timestamps=ts_ho)
                #ev_labels = get_ev_labels(delta_ev,ev_stats,n_std=1)
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
        # plot_simulatated_path_from_model = True
        # saveas_simulatated_path_from_model = 'figures_tests/model_assessment/cvae_wf_'
        if plot_simulatated_path_from_model:
            for jj in [10, 40, 80]:
                saveas = saveas_simulatated_path_from_model + str(jj)
                x = wf_ho[jj, :].reshape((1, 141))
                label = ev_label_ho[jj, :].reshape((1, 3))
                plot_simulated(cvae, x, ev_label=label, n=1, var=0.5, saveas=saveas, verbose=verbose_main)
                
                # plot_simulated(cvae,x,ev_label=label,n=0,var=0.5, saveas=None, verbose=False)
                # bool_labels,_ = similarity_SSQ(jj, wf_ho, epsilon=0.1, var=0.7, standardised_input=True)
                # event_rates, real_clusters = get_event_rates(ts_ho,bool_labels,bin_width=1,consider_only=1)
                # delta_ev, ev_stats = __delta_ev_measure__(event_rates)
                # plot_similar_wf(jj,wf_ho,bool_labels,0.1,saveas='figures_tests/model_assessment/simulated_and_cluster_mean',cluster='mean',show_clustered=False, verbose=False,
                #          title='Comparison of Cluster- and CVAE Mean')
                #plt.show()
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
                    event_rates, real_clusters = get_event_rates(timestamps[:, 0], bool_labels, bin_width=1, consider_only=1)
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
        # plot_acf_pacf = False
        # savefig_acf_pacf = 'figures_tests/acf_pacf/26_nov'
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


        '''
        # TODO : -Fix plots of encoded/decoded Latent space.
        #        -Event-rate plots? ---Njaa, finns i main..
        # ************************************************************
        # ******** Cluster-Results using "Test-statistic"  ***********
        # ************************************************************
        #plot_test_of_test_statistic = False
        #savefig_test_of_test_statistic = 'figures_tests/test_statistic/26_nov_'
        if plot_test_of_test_statistic:
            print()
            print(f'Plotting old test')
            for c in [10,40,80]:
                saveas = savefig_test_of_test_statistic
                test_stat = waveforms - waveforms[c,:]
                print(test_stat.shape)
                mean = np.zeros((test_stat.shape[-1]))
                var = np.eye(test_stat.shape[-1])
                probs = stats.multivariate_normal.pdf(test_stat,mean,var)*1e57
                threshold = 1e-25
                bool_labels = probs>threshold
                #sum(bool_labels)
                plot_similar_wf(c,waveforms,bool_labels,threshold,saveas=saveas+'thres_'+str(threshold)+'_wf_'+str(c),verbose=verbose_main )
        '''