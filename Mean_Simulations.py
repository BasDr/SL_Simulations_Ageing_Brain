import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from StuartLandauModel import simulate_SL
from numba import njit
import concurrent.futures
import scipy.signal as sig
import scipy.stats as stat

os.chdir('Folder of code')


lowcut=8.0
highcut=13.0
order=2
b,a=sig.butter(order, [lowcut, highcut], btype='band', output='ba', fs=1000)

DMN = [8,10,12,14,21,23,24,42,44,46,48,55,57,58]

MeanFC_youth = np.load('location mean FC younger population')
MeanFC_elder = np.load('location mean FC older population')
MeanFC_youth_DMN = MeanFC_youth[DMN, :][:, DMN]
MeanFC_elder_DMN = MeanFC_elder[DMN, :][:, DMN]

# Load Connectivity Matrices
dk_dist_mat = np.load('Location lengths SC')
dk_con_mat = np.load('Location counts SC')


dk_con_mat[dk_con_mat < 10] = 0 # Removes negligible connections with less than 10 fibers
np.fill_diagonal(dk_con_mat, 0) # Ensures there are no self connections
dk_con_mat /= np.mean(dk_con_mat) # Normalizes connectivity matrix so that maximum value is 1



def compute_psd(data, nfft_factor=5):
    """Compute the Power Spectral Density (PSD) of the signal."""
    regions, time_p=data.shape

    frequencies, psd = sig.welch(data, fs=1000, nperseg=2 * 1000, nfft=nfft_factor * 1000)
    psd_mean         = np.mean(psd, axis=0)

    return frequencies, psd

def process_signals(signal, fs=1000):
    """
    Process signals by bandpass filtering and computing the Hilbert transform.

    Parameters:
    signal (np.ndarray): Signal array.
    fs (int): Sampling frequency.

    Returns:
    np.ndarray: Normalized Hilbert transformed signals.
    """
    # Bandpass the signal (uncomment if needed)
    #b, a = sig.butter(4, [8, 13], btype='band', fs=fs)
    #signal = sig.filtfilt(b, a, signal, axis=-1)
    
    hilbert_sigs = sig.hilbert(signal[0:68, :].real, axis=-1)
    return hilbert_sigs / np.abs(hilbert_sigs)

def compute_kuramoto_order(hilbert_sigs):
    """
    Compute the Kuramoto Order Parameter.

    Parameters:
    hilbert_sigs (np.ndarray): Hilbert transformed signals.

    Returns:
    np.ndarray: Kuramoto Order Parameter.
    """
    return np.abs(np.nanmean(hilbert_sigs, axis=0))

def compute_spectral_entropy(psd):
    """Compute the spectral entropy for each channel."""
    psd_normalized   = psd / np.sum(psd, axis=-1, keepdims=True)
    spectral_entropy = -np.sum(psd_normalized * np.log(psd_normalized + 1e-12), axis=-1)  # Avoid log(0)
    return spectral_entropy

def run_single_simulation(dk_dist_mat, dk_con_mat_SL, speed, gc_val, md_val, a_val):
    mean_corr_Y = []
    mean_corr_E = []
    mean_psd_peak = []
    mean_psd_power = []
    mean_meta=[]
    mean_sync=[]
    mean_SE=[]

    try:
        for interval in range(0,30):
            sm_ts = simulate_SL(dk_dist_mat, dk_con_mat_SL, 70000, 60000, 0.1, 1, a_val, speed)
            sm_ts = np.delete(sm_ts, [3, 38], axis=0)
            #DMN_network = sm_ts[DMN, :]
            sm_ts = np.real(sm_ts)
            #DMN_network = np.real(DMN_network)
            #if interval == 0:
                #np.save('Rerun_test/signal_C{:.2f}_md{:.1f}_a{:.2f}.npy'.format(gc_val,md_val,a_val,interval), sm_ts)
            
            normilized_smts = stat.zscore(sm_ts, axis=-1)
            #normilized_DMN=stat.zscore(DMN_network, axis=-1)
            frequencies, psd = compute_psd(normilized_smts)
            mean_psd=np.mean(psd, axis=0)
            #DMN_frequencies, DMN_psd = compute_psd(normilized_DMN)
            SE=compute_spectral_entropy(psd)
            #DMN_SE=compute_spectral_entropy(DMN_psd)

            sm_ts = sig.filtfilt(b, a, normilized_smts, axis=-1)
            #DMN_network = sig.filtfilt(b, a, DMN_network, axis=-1)

            hilbert_sigs = process_signals(sm_ts)
            #DMN_hilbert_sigs = process_signals(DMN_network)
            KOP          = compute_kuramoto_order(hilbert_sigs)
            #DMN_KOP      = compute_kuramoto_order(DMN_hilbert_sigs)
            if not np.isnan(KOP).all():
                synchrony    = np.nanmean(KOP)
                #DMN_synchrony= np.nanmean(DMN_KOP)
                metastability= np.nanstd(KOP)
                #DMN_metastability= np.nanstd(DMN_KOP)
            
            
            try:
                analytic_signal = sig.hilbert(sm_ts, axis=-1)
                #analytic_signal_DMN = sig.hilbert(DMN_network, axis=-1)
            except Exception as e:
                print(e)
            envelope = np.abs(analytic_signal)
            #envelope_DMN = np.abs(analytic_signal_DMN)
            corr = np.corrcoef(envelope)
            #corr_DMN = np.corrcoef(envelope_DMN)
            
            #np.save('Rerun_test/coh_matrix_YC{:.2f}_md{:.1f}_a{:.2f}_iter{:.0f}.npy'.format(gc_val,md_val,a_val,interval), corr)

        
            triu_indices = np.tril_indices_from(corr, k=-1)

            fc_vector1 = corr[triu_indices]
            fc_vector2 = MeanFC_youth[triu_indices]
            fc_vector3 = MeanFC_elder[triu_indices]
            # Calculate the Pearson correlation coefficient

            #DMN_triu_indices = np.tril_indices_from(corr_DMN, k=-1)
            #DMN_fc_vector1 = corr_DMN[DMN_triu_indices]
            #DMN_fc_vector2 = MeanFC_youth_DMN[DMN_triu_indices]
            #DMN_fc_vector3 = MeanFC_elder_DMN[DMN_triu_indices]

            
            youth_correlation= np.corrcoef(fc_vector2, fc_vector1)[0,1]
            #DMN_youth_correlation= np.corrcoef(DMN_fc_vector2, DMN_fc_vector1)[0,1]
            mean_corr_Y.append(youth_correlation)
            #mean_DMN_corr_Y.append(DMN_youth_correlation)
            
            elder_correlation= np.corrcoef(fc_vector3, fc_vector1)[0,1]
            #DMN_elder_correlation= np.corrcoef(DMN_fc_vector3, DMN_fc_vector1)[0,1]
            mean_corr_E.append(elder_correlation)
            #mean_DMN_corr_E.append(DMN_elder_correlation)

            
            mean_psd_peak.append(frequencies[np.argmax(10 * np.log10(mean_psd[:250]))])
            mean_psd_power.append(np.max(10 * np.log10(mean_psd[:250])))

            #mean_DMN_psd_peak.append(DMN_frequencies[np.argmax(10 * np.log10(DMN_psd[:250]))])
            #mean_DMN_psd_power.append(np.max(10 * np.log10(DMN_psd[:250])))

            mean_meta.append(metastability)
            mean_sync.append(synchrony)
            mean_SE.append(np.mean(SE))

            #mean_DMN_meta.append(DMN_metastability)
            #mean_DMN_sync.append(DMN_synchrony)
            #mean_DMN_SE.append(np.mean(DMN_SE))

        Simulation_Results= {
            'corr_info_Y': mean_corr_Y,
            'corr_info_E': mean_corr_E,
            'psd_peak': mean_psd_peak,
            'psd_power': mean_psd_power,
            'meta_info': mean_meta,
            'sync_info': mean_sync,
            'SE_info': mean_SE,
        }

        np.savez('Mean_simulation_results/SL_simulation_results_C{:.2f}_md{:.1f}_a{:.2f}.npz'.format(gc_val,md_val,a_val), **Simulation_Results)
        #np.savez('Mean_simulation_results/DMN_SL_simulation_results_C{:.2f}_md{:.1f}_a{:.2f}.npz'.format(gc_val,md_val,a_val), **DMN_Simulation_Results)
        

    except Exception as e:
        print(e)
    #max_corr_Y = np.max(mean_corr_Y)
    #max_corr_E = np.max(mean_corr_E)
    #mean_corr_Y = np.mean(mean_corr_Y)
    #mean_corr_E = np.mean(mean_corr_E)
    #result=np.array([max_corr_Y, max_corr_E, mean_corr_Y, mean_corr_E])
    #np.save('Mean_simulation_results/corr_info_C{:.2f}_md{:.1f}_a{:.2f}_result.npy'.format(gc_val,md_val,a_val), result)

        
    

simulate_SL(t_len = dk_dist_mat, W_mat = dk_con_mat, sim_time = 1000, store_time = 1000, dt = 1, dt_save = 1, a_bif = 0, speed = np.mean(dk_dist_mat[dk_con_mat > 0]) / 10)


# Structural Connectivity
gc = np.logspace(np.log10(1), np.log10(70), 40) 
dk_con_mat_SL_list = [dk_con_mat * i for i in gc]

# Conduction Speed
md = np.linspace(0, 20, 21)
speedSL = [np.mean(dk_dist_mat[dk_con_mat > 0]) / j if j > 0 else -1000 for j in md]

# Parameter 'a' range
a_range = [-50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50]  # Example range from -5 to 5 with 10 steps


# Run all simulations using concurrent.futures for parallel execution
def run_all_simulations(dk_dist_mat, dk_con_mat_SL_list, speedSL, gc, md, a_range):
    with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
        futures = []
        for i, dk_con_mat_SL in enumerate(dk_con_mat_SL_list):
            for j, speed in enumerate(speedSL):
                for a_val in a_range:
                    futures.append(executor.submit(run_single_simulation, dk_dist_mat, dk_con_mat_SL, speed, gc[i], md[j], a_val))
        concurrent.futures.wait(futures)
    print('all done')

# Run simulations
run_all_simulations(dk_dist_mat, dk_con_mat_SL_list, speedSL, gc, md, a_range)
