'''
Convert EEG move/rest files to xarray files.
Data is publicly available at:
http://bnci-horizon-2020.eu/database/data-sets
'''
import os

from data_load_utils import compute_xr_eeg_mr

# Set parameters
lp = '/data2/users/stepeter/cnn_hilb_datasets/EEG_arm/'
sp = '.../eeg_move_rest_xarray/'
tlims = [-2, 2] # seconds
tlims_handpos = [0,4]  # seconds
filt_freqs = [1, None] # Hz (low, high cutoffs)
n_chans = 61 # number of EEG channels
event_dict = {'elbow flexion':0x600, 'rest':0x606}
sfreq_new = 250 # Hz
sfreq_new_pose = 250 # Hz


# Create data files
if not os.path.exists(sp):
    os.mkdir(sp)
if not os.path.exists(sp+'/pose/'):
    os.mkdir(sp+'/pose/')

for id_num in range(1,16):
    sbj_id = 'S' + str(id_num).zfill(2)
    compute_xr_eeg_mr(sbj_id, lp, sp, tlims, sfreq_new, filt_freqs,
                      n_chans, event_dict, tlims_handpos, sfreq_new_pose)
    print('Finished '+sbj_id+'!')