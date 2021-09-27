import os

from data_load_utils import compute_xr_eeg_bal


lp = '/data1/users/stepeter/eeg_test/'
sp = '.../eeg_balance_perturbations_xarray/'
tlims = [-2, 2]  # seconds

chans_sel1 = 'eeg'  # 'eeg', 'emg', 'mocap', 'emg+mocap'
chans_sel2 = 'mocap'
chans_sel3 = 'emg'
decode_task = 'pull+rotate'
if not os.path.exists(sp):
    os.makedirs(sp)
if not os.path.exists(sp+'pose/'):
    os.makedirs(sp+'pose/')
if not os.path.exists(sp+'emg/'):
    os.makedirs(sp+'emg/')

for sbj_id in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08',
               'S09', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
               'S19', 'S20', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27',
               'S28', 'S29', 'S30', 'S31', 'S32', 'S33']:
    compute_xr_eeg_bal(sbj_id, lp, sp, tlims, chans_sel1,
                       chans_sel2, chans_sel3, decode_task)

    