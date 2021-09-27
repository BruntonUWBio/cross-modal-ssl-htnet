import mne
import glob
import natsort
import numpy as np
import xarray as xr
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
mne.set_log_level('error')


def mne_apply(func, raw, verbose="WARNING"):
    """
    Apply function to data of `mne.io.RawArray`.
    
    Parameters
    ----------
    func: function
        Should accept 2d-array (channels x time) and return modified 2d-array
    raw: `mne.io.RawArray`
    verbose: bool
        Whether to log creation of new `mne.io.RawArray`.
    Returns
    -------
    transformed_set: Copy of `raw` with data transformed by given function.
    """
    new_data = func(raw.get_data())
    return mne.io.RawArray(new_data, raw.info, verbose=verbose)

def balance_classes(X, y, random_state=0):
    '''Balances classes'''
    rus = RandomUnderSampler(random_state=random_state)
    n_eps, n_ecog2, n_ts = X.shape
    X_rsh = X.reshape((n_eps,-1))
    X_rs, y_rs = rus.fit_resample(X_rsh, y)
    n_eps = X_rs.shape[0]
    X_rs = X_rs.reshape((n_eps,n_ecog2,n_ts))
    return X_rs, y_rs


def compute_xr_eeg_mr(sbj_id,lp,sp,tlims,sfreq_new,filt_freqs,n_chans,event_dict,
                      tlims_handpos,sfreq_new_pose,n_splits=4):
    
    eeg_ch_names = ['F3','F1','Fz','F2','F4',
                    'FFC5h','FFC3h','FFC1h','FFC2h','FFC4h','FFC6h',
                    'FC5','FC3','FC1','FCz','FC2','FC4','FC6','FTT7h',
                    'FCC5h','FCC3h','FCC1h','FCC2h','FCC4h','FCC6h','FTT8h',
                    'C5','C3','C1','Cz','C2','C4','C6','TTP7h',
                    'CCP5h','CCP3h','CCP1h','CCP2h','CCP4h','CCP6h','TTP8h',
                    'CP5','CP3','CP1','CPz','CP2','CP4','CP6',
                    'CPP5h','CPP3h','CPP1h','CPP2h','CPP4h','CPP6h',
                    'P3','P1','Pz','P2','P4','PPO1h','PPO2h']
    
    fnames_all = natsort.natsorted(glob.glob(lp+sbj_id+'_ME/*.gdf'))

    for s,fname_curr in enumerate(fnames_all):
#         print(fname_curr)
        # Load datafile
        dat_load = mne.io.read_raw_edf(fname_curr,preload=True)
        dat_hand_pos = dat_load.copy()
        dat_hand_pos_ev = dat_load.copy()

        ch_labels = dat_load.info['ch_names']
        dat = dat_load.drop_channels(ch_labels[n_chans:])
        assert len(dat.ch_names) == n_chans

        # Convert to millvolt for numerical stability of next operations
        dat = mne_apply(lambda a: a * 1e3, dat)

        # High-pass filter
        dat.filter(filt_freqs[0], filt_freqs[1])
        
        # Common average reference
        dat.set_eeg_reference(ref_channels='average')

        # Find events (769, 770, 771, 772)
        events,ev_dic_orig = mne.events_from_annotations(dat_load)
        
    #     ev_dic_orig[str(int(event_dict['move']))]

        # Epoch data around events
        key_lst = list(event_dict.keys())
        event_id = {str(int(event_dict[key])):ev_dic_orig[str(int(event_dict[key]))] for key in key_lst}

        # Drop EEG/EOG electrodes for pose data
        drop_chan_pos_pose = [val for val in ch_labels if (val[:3] == 'eeg') or (val[:3] == 'eog')]
        drop_chan_pos_pose += [val for val in ch_labels if val in eeg_ch_names]
        drop_chan_pos_pose += ['wrist_bend', 'roll', 'pitch', 'gesture', 'GripPressure']
        dat_hand_pos.drop_channels(drop_chan_pos_pose)
        
        # Remove other pose data as well (except wrist/elbow XYZ)
        ch_names = dat_hand_pos.info['ch_names']
        diff_chs = ['handPosX', 'handPosY', 'handPosZ', 'elbowPosX', 'elbowPosY', 'elbowPosZ']
        non_diff_chs = ['thumb_near', 'thumb_far', 'thumb_index', 'index_near', 'index_far', 
                        'index_middle', 'middle_near', 'middle_far', 'middle_ring', 'ring_near', 
                        'ring_far', 'ring_little', 'litte_near', 'litte_far', 'thumb_palm', 'ShoulderAdducti', 
                        'ShoulderFlexion', 'ShoulderRotatio', 'Elbow', 'ProSupination', 'Wrist']
        dat_hand_pos.drop_channels(non_diff_chs)
        
        # Find movement event onset based on hand radial position
        ch_labels_pos = dat_hand_pos_ev.info['ch_names']
        drop_chan_pos = [val for val in ch_labels_pos if val not in ['handPosX', 'handPosY', 'handPosZ']]
        dat_hand_pos_ev.drop_channels(drop_chan_pos)
        radial_dist = np.sqrt(np.square(dat_hand_pos_ev._data).sum(axis=0))
        dat_hand_pos_ev.drop_channels(['handPosY', 'handPosZ'])
        dat_hand_pos_ev._data[0,:] = radial_dist
        ep_hand_pos = mne.Epochs(dat_hand_pos_ev, events, event_id, tlims_handpos[0],
                                 tlims_handpos[1], baseline=None, preload=True)
        ev_lab = list(event_id.keys())[0]
        move_ev_inds = np.nonzero(events[:,2]==event_id[ev_lab])[0]
        for i in range(ep_hand_pos[ev_lab]._data.shape[0]):
            curr_trace = ep_hand_pos[ev_lab]._data[i,...].squeeze()
            curr_trace = np.abs(curr_trace-curr_trace[0])
            thresh=min(curr_trace.max()*.75,1)
            events[move_ev_inds[i],0] += np.nonzero(curr_trace>thresh)[0][0]
        
        if s==0:
            ep_eeg = mne.Epochs(dat, events, event_id, tlims[0], tlims[1], baseline=None, preload=True)
            ep_pose = mne.Epochs(dat_hand_pos, events, event_id, tlims[0],
                                 tlims[1], baseline=None, preload=True)
        else:
            ep_eeg_tmp = mne.Epochs(dat, events, event_id, tlims[0], tlims[1], baseline=None,
                                    preload=True)
            ep_eeg = mne.concatenate_epochs([ep_eeg,ep_eeg_tmp])
            ep_pose_tmp = mne.Epochs(dat_hand_pos, events, event_id, tlims[0], tlims[1],
                                     baseline=None, preload=True)
            ep_pose = mne.concatenate_epochs([ep_pose,ep_pose_tmp])


    # Resample epochs to match ECoG inputs
    ep_eeg.resample(sfreq_new)
    ep_pose.resample(sfreq_new_pose)
    
    # Compute diff for pose data
    ch_names = ep_pose.info['ch_names']
    ep_pose_np = ep_pose.get_data().copy()
    ep_pose_np = np.append(np.diff(ep_pose_np,axis=-1), np.diff(ep_pose_np,axis=-1)[...,-1:], axis=-1)
#     ep_pose.pick([0,1])
    ep_pose._data = ep_pose_np
    
    # Remove NaN events
    nan_evs = np.sum(np.isnan(ep_eeg.get_data()).reshape(ep_eeg.get_data().shape[0],-1),axis=1)
    ep_eeg = ep_eeg.drop(np.nonzero(nan_evs)[0])
    ep_pose = ep_pose.drop(np.nonzero(nan_evs)[0])

    # Labels in consecutive integer order
    y = ep_eeg.events[:,-1]
    for i, uni_label in enumerate(np.unique(y).tolist()):
        y[y==uni_label] = i
    
    # Balance classes
    X_ecog = ep_eeg.get_data().copy()
    X_pose = ep_pose.get_data().copy()
    X_ecog_rs, y_ecog_rs = balance_classes(X_ecog, y)
    X_pose_rs, y_pose_rs = balance_classes(X_pose, y)
    assert all(y_ecog_rs == y_pose_rs)
    
    # Create recording day variable
    skf = StratifiedKFold(n_splits=n_splits)
    recording_day = np.zeros_like(y_ecog_rs)
    for i, (_, test_index) in enumerate(skf.split(X_ecog_rs, y_ecog_rs)):
        recording_day[test_index] = i
    print(np.unique(recording_day, return_counts=True))

    # Add labels to EEG data
    labels_arr = np.tile(y_ecog_rs[:, np.newaxis], (1,X_ecog_rs.shape[2]))
    ecog_dat_sbj = np.concatenate((X_ecog_rs,labels_arr[:, np.newaxis, :]), axis=1)

    # Add labels to pose data
    labels_arr_pose = np.tile(y_pose_rs[:, np.newaxis], (1,X_pose_rs.shape[2]))
    pose_dat_sbj = np.concatenate((X_pose_rs,labels_arr_pose[:, np.newaxis, :]),axis=1)

    # Randomize epoch order
    order_inds = np.arange(ecog_dat_sbj.shape[0])
    np.random.shuffle(order_inds)
    ecog_dat_sbj = ecog_dat_sbj[order_inds,...]
    pose_dat_sbj = pose_dat_sbj[order_inds,...]
    recording_day = (recording_day[order_inds]).tolist()
    
    # Convert EEG to xarray and save
    da_ecog = xr.DataArray(ecog_dat_sbj,
                      [('events', recording_day),
                       ('channels', np.arange(ecog_dat_sbj.shape[1])),
                       ('time', ep_eeg.times)])
    da_ecog.to_netcdf(sp+sbj_id+'_eeg_data.nc')

    # Convert EEG to xarray and save
    da_pose = xr.DataArray(pose_dat_sbj,
                      [('events', recording_day),
                       ('channels', np.arange(pose_dat_sbj.shape[1])),
                       ('time', ep_pose.times)])
    da_pose.to_netcdf(sp+'pose/'+sbj_id+'_pose_data.nc')
    print('Finished ' + sbj_id +'!')


def flatx_with_labels(evs_in):
    X, y = [], []
    for i in range(len(evs_in)):
        X.extend(evs_in[i])
        y.extend([i]*len(evs_in[i]))
    return np.array(X), np.array(y)

def align_evs_ff(move_evs, cue_evs):
    ind_move, ind_cue = 0, 0
    move_evs_out, cue_evs_out = [], []
    while (ind_move < len(move_evs)) and (ind_cue < len(cue_evs)):
        diff_val = move_evs[ind_move]-cue_evs[ind_cue]
        if abs(diff_val) < 3000:
            move_evs_out.append(move_evs[ind_move])
            cue_evs_out.append(cue_evs[ind_cue])
            ind_move += 1
            ind_cue += 1
        elif diff_val < 0:
            # No move event for given cue
            ind_move += 1
        elif diff_val > 0:
            # No cue event for given move
            ind_cue += 1
    return move_evs_out, cue_evs_out

def ev_ts_ff(in_dat, evs_good):
    evs = [[] for i in range(len(evs_good))]
    prev_val = 1
    for i, val in enumerate(in_dat.flatten().tolist()):
        if prev_val <= 0:
            if val in evs_good:
                evs[val-1].append(i)
        prev_val = val
    return evs

def compute_xr_ecog_ff(sbj_id, lp, sp, tlims, tlims_handpos,
                       filt_freqs, sfreq_new, out_sbj_d,
                       raw_sfreq=1000, n_splits=4):
    # Load data
    ff_dat = loadmat(lp + sbj_id + '/' + sbj_id + '_fingerflex.mat')
    pose = ff_dat['flex'].T
    ecog = ff_dat['data'].T
    cue = ff_dat['cue'].T

    stim_dat = loadmat(lp + sbj_id + '/' + sbj_id + '_stim.mat')
    move = stim_dat['stim'].T

    # Normalize pose (as done in "Decoding flexion of individual fingers using electrocorticographic signals in humans" section 2.2)
    ave_pose = np.tile(np.mean(pose,axis=-1,keepdims=True),[1,pose.shape[1]])
    std_pose = np.tile(np.std(pose,axis=-1,keepdims=True),[1,pose.shape[1]])
    pose = np.divide((pose - ave_pose), std_pose)

    # Identify event times (transition from 0 or negative value to positive value)
    evs_good = [1,2,3,4,5]
    evs_cue = ev_ts_ff(cue, evs_good)
    evs_move = ev_ts_ff(move, evs_good)

    # Identify good events (remove non-overlapping events between cue and behavior)
    move_evs_final, cue_evs_final = [],[]
    for curr_ev in range(len(evs_good)):
        move_evs_out, cue_evs_out = align_evs_ff(evs_move[curr_ev],
                                                 evs_cue[curr_ev])
        move_evs_final.append(move_evs_out)
        cue_evs_final.append(cue_evs_out)

    # Shuffle events and balance classes
    X_move, y_move = flatx_with_labels(move_evs_final)
    X_cue, y_cue = flatx_with_labels(cue_evs_final)

    assert (y_move == y_cue).all()
    rus = RandomUnderSampler(random_state=0)
    X_all = np.concatenate((X_move[:, np.newaxis], X_cue[:, np.newaxis]), axis=1)
    X_all_res, y_all_res = rus.fit_resample(X_all, y_move)

    # Check that events are balanced
    print(np.unique(y_all_res, return_counts=True))

    # Create raw for ECoG data
    ch_names = ['ECoG'+str(val) for val in range(ecog.shape[0])]
    ecog_info = mne.create_info(ch_names, raw_sfreq, ch_types='ecog')
    raw_ecog = mne.io.RawArray(ecog, ecog_info)

    # High-pass filter
    raw_ecog.filter(filt_freqs[0], filt_freqs[1])

    # Notch filter
    raw_ecog.notch_filter(np.arange(60, 301, 60), picks='all')

    # Common average reference
    raw_ecog.set_eeg_reference(ref_channels='average')

    # Create event struct
    events = [[], [], []]
    event_id = dict(zip(np.unique(y_all_res).astype('str').tolist(),
                        np.unique(y_all_res).tolist()))
    n_evs = X_all_res.shape[0]
    events = np.zeros((n_evs, 3))
    events[:, 0] = X_all_res[:, 0]
    events[:, 2] = y_all_res
    events = events.astype('int')

    # Create raw for pose data
    ch_names = ['Pose'+str(val) for val in range(pose.shape[0])]
    pose_info = mne.create_info(ch_names, raw_sfreq, ch_types='misc')
    raw_pose = mne.io.RawArray(pose, pose_info)

    # Epoch data
    ep_ecog = mne.Epochs(raw_ecog, events, event_id, tlims[0], tlims[1], baseline=None, preload=True)
    ep_pose = mne.Epochs(raw_pose, events, event_id, tlims[0], tlims[1], baseline=None, preload=True)

    # Resample epochs to match ECoG inputs
    ep_ecog.resample(sfreq_new)
    ep_pose.resample(sfreq_new)

    # Add labels to data
    event_id_labs = list(event_id.keys())
    days_start = (np.arange(n_splits)+1).tolist()
    recording_day,labels = [],[]
    for i,lab_curr in enumerate(event_id_labs):
        ep_tmp = ep_ecog[lab_curr]
        ep_tmp_pose = ep_pose[lab_curr]
        n_tmp = int(ep_tmp._data.shape[0])//n_splits + 1
        days_curr = np.asarray(days_start * n_tmp)[:ep_tmp._data.shape[0]]
        np.random.shuffle(days_curr)
        recording_day.extend(days_curr.tolist()) 
        if i==0:
            ecog_dat_sbj = ep_tmp.get_data().copy()
            pose_dat_sbj = ep_tmp_pose.get_data().copy()
        else:
            ecog_dat_sbj = np.concatenate((ecog_dat_sbj,ep_tmp.get_data().copy()),axis=0)
            pose_dat_sbj = np.concatenate((pose_dat_sbj,ep_tmp_pose.get_data().copy()),axis=0)
        labels.extend([i+1]*ep_tmp.get_data().shape[0])

    # Add labels to EEG data
    labels_arr = np.tile(np.asarray(labels)[:, np.newaxis],(1,ecog_dat_sbj.shape[2]))
    ecog_dat_sbj = np.concatenate((ecog_dat_sbj,labels_arr[:, np.newaxis]),axis=1)

    # Add labels to pose data
    labels_arr_pose = np.tile(np.asarray(labels)[:, np.newaxis],(1,pose_dat_sbj.shape[2]))
    pose_dat_sbj = np.concatenate((pose_dat_sbj,labels_arr_pose[:, np.newaxis]),axis=1)

    # Randomize epoch order
    order_inds = np.arange(ecog_dat_sbj.shape[0])
    np.random.shuffle(order_inds)
    ecog_dat_sbj = ecog_dat_sbj[order_inds,...]
    pose_dat_sbj = pose_dat_sbj[order_inds,...]
    recording_day = (np.asarray(recording_day)[order_inds]).tolist()

    # Check if labels are still the same between ECoG and pose data
    assert (ecog_dat_sbj[:, -1, 0].squeeze() == pose_dat_sbj[:, -1, 0].squeeze()).all()

    # Convert EEG to xarray and save
    da_ecog = xr.DataArray(ecog_dat_sbj,
                      [('events', recording_day),
                       ('channels', np.arange(ecog_dat_sbj.shape[1])),
                       ('time', ep_ecog.times)])
    da_ecog.to_netcdf(sp+out_sbj_d[sbj_id]+'_ec_data.nc')

    # Convert EEG to xarray and save
    da_pose = xr.DataArray(pose_dat_sbj,
                      [('events', recording_day),
                       ('channels', np.arange(pose_dat_sbj.shape[1])),
                       ('time', ep_pose.times)])
    da_pose.to_netcdf(sp+'pose/'+out_sbj_d[sbj_id]+'_pose_data.nc')
    print('Finished '+out_sbj_d[sbj_id]+'!')

def chans_keep(chans_sel, eeg_chs, non_eeg_chs):
        if chans_sel == 'eeg':
            keep_chans = eeg_chs
        elif chans_sel == 'emg+mocap':
            keep_chans = non_eeg_chs[5:-2]
        elif chans_sel == 'emg':
            keep_chans = non_eeg_chs[5:13]
        elif chans_sel == 'mocap':
            keep_chans = non_eeg_chs[13:-2]
        return keep_chans

def compute_xr_eeg_bal(sbj_id, lp, sp, tlims, chans_sel1,
                       chans_sel2, chans_sel3, decode_task,
                       raw_sfreq=1000, n_splits=4, scale_fact=1e6):
    raw = mne.io.read_raw_eeglab(lp+sbj_id+'.set')

    eeg_chs = [val for i,val in enumerate(raw.info['ch_names']) if i<128]
    non_eeg_chs = [val for i,val in enumerate(raw.info['ch_names']) if i>=128]

    # Epoch the data
    _, _ = mne.events_from_annotations(raw)

    if decode_task == 'pull':
        event_id = {'L_pull_Stn': 1, 'L_pull_Wlk': 3,
                'R_pull_Stn': 2, 'R_pull_Wlk': 4}
    elif decode_task == 'rotate':
        event_id = {'M_on_CCW_SVZ': 1, 'M_on_CW_SVZ': 2,
                    'M_on_CCW_WVZ': 3, 'M_on_CW_WVZ': 4}
    elif decode_task == 'pull+rotate':
        event_id = {'L_pull_Stn': 1, 'R_pull_Stn': 1, 'L_pull_Wlk': 2, 'R_pull_Wlk': 2,
                    'M_on_CCW_SVZ': 3, 'M_on_CW_SVZ': 3, 'M_on_CCW_WVZ': 4, 'M_on_CW_WVZ': 4}

    events_from_annot, event_dict = mne.events_from_annotations(raw, event_id=event_id)

    epochs_ecog = mne.Epochs(raw, events_from_annot, event_dict,
                             tmin=tlims[0], tmax=tlims[1], baseline=None, preload=True)

    # Identify channels to drop from each dataset
    keep_chans1 = chans_keep(chans_sel1, eeg_chs, non_eeg_chs)
    keep_chans2 = chans_keep(chans_sel2, eeg_chs, non_eeg_chs)
    keep_chans3 = chans_keep(chans_sel3, eeg_chs, non_eeg_chs)

    # Balance classes
    y = epochs_ecog.events[:,-1]
    X_ecog = epochs_ecog.get_data(picks=keep_chans1).copy()*scale_fact
    X_pose = epochs_ecog.get_data(picks=keep_chans2).copy()*scale_fact
    X_emg = epochs_ecog.get_data(picks=keep_chans3).copy()*scale_fact

    # Balance classes
    X_ecog_rs, y_ecog_rs = balance_classes(X_ecog, y)
    X_pose_rs, y_pose_rs = balance_classes(X_pose, y)
    X_emg_rs, y_emg_rs = balance_classes(X_emg, y)

    assert all(y_ecog_rs == y_pose_rs)
    assert all(y_ecog_rs == y_emg_rs)

    # Create recording day variable
    skf = StratifiedKFold(n_splits=n_splits)
    recording_day = np.zeros_like(y_ecog_rs)
    for i, (_, test_index) in enumerate(skf.split(X_ecog_rs, y_ecog_rs)):
        recording_day[test_index] = i
    print(np.unique(recording_day, return_counts=True))

    # Add labels to EEG data
    labels_arr = np.tile(y_ecog_rs[:, np.newaxis], (1,X_ecog_rs.shape[2]))
    ecog_dat_sbj = np.concatenate((X_ecog_rs,labels_arr[:, np.newaxis, :]), axis=1)

    # Add labels to pose data
    labels_arr_pose = np.tile(y_pose_rs[:, np.newaxis], (1,X_pose_rs.shape[2]))
    pose_dat_sbj = np.concatenate((X_pose_rs,labels_arr_pose[:, np.newaxis, :]),axis=1)

    # Add labels to EMG data
    labels_arr_emg = np.tile(y_emg_rs[:, np.newaxis], (1,X_emg_rs.shape[2]))
    emg_dat_sbj = np.concatenate((X_emg_rs,labels_arr_emg[:, np.newaxis, :]),axis=1)

    # Randomize epoch order
    order_inds = np.arange(ecog_dat_sbj.shape[0])
    np.random.shuffle(order_inds)
    ecog_dat_sbj = ecog_dat_sbj[order_inds,...]
    pose_dat_sbj = pose_dat_sbj[order_inds,...]
    emg_dat_sbj = emg_dat_sbj[order_inds,...]
    recording_day = (recording_day[order_inds]).tolist()

    # Convert EEG to xarray and save
    da_ecog = xr.DataArray(ecog_dat_sbj,
                      [('events', recording_day),
                       ('channels', np.arange(ecog_dat_sbj.shape[1])),
                       ('time', epochs_ecog.times)])
    da_ecog.to_netcdf(sp+sbj_id+'_eeg_data.nc')

    # Convert pose to xarray and save
    da_pose = xr.DataArray(pose_dat_sbj,
                      [('events', recording_day),
                       ('channels', np.arange(pose_dat_sbj.shape[1])),
                       ('time', epochs_ecog.times)])
    da_pose.to_netcdf(sp+'pose/'+sbj_id+'_pose_data.nc')

    # Convert emg to xarray and save
    da_emg = xr.DataArray(emg_dat_sbj,
                      [('events', recording_day),
                       ('channels', np.arange(emg_dat_sbj.shape[1])),
                       ('time', epochs_ecog.times)])
    da_emg.to_netcdf(sp+'emg/'+sbj_id+'_emg_data.nc')
    print('Finished '+sbj_id+'!')