''' Data loading functions during training
(modified from from https://github.com/yukimasano/self-label) '''

import torchvision
import torch
import torchvision.transforms as tfs
import models
import os, glob, natsort, pdb
import numpy as np
import util
import xarray as xr
from scipy.stats import mode
import pickle
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from datetime import datetime as dt
import h5py
import mne

class DataSet(torch.utils.data.Dataset):
    """ pytorch Dataset that return image index too"""
    def __init__(self, dt):
        self.dt = dt

    def __getitem__(self, index):
        data, target = self.dt[index]
        return data, target, index

    def __len__(self):
        return len(self.dt)


def get_aug_dataloader(image_dir, is_validation=False,
                       batch_size=256, image_size=256, crop_size=224,
                       mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                       num_workers=8,
                       augs=1, shuffle=True):

    print(image_dir)
    if image_dir is None:
        return None

    print("imagesize: ", image_size, "cropsize: ", crop_size)
    normalize = tfs.Normalize(mean=mean, std=std)
    if augs == 0:
        _transforms = tfs.Compose([
                                    tfs.Resize(image_size),
                                    tfs.CenterCrop(crop_size),
                                    tfs.ToTensor(),
                                    normalize
                                ])
    elif augs == 1:
        _transforms = tfs.Compose([
                                    tfs.Resize(image_size),
                                    tfs.CenterCrop(crop_size),
                                    tfs.RandomHorizontalFlip(),
                                    tfs.ToTensor(),
                                    normalize
                                ])
    elif augs == 2:
        _transforms = tfs.Compose([
                                    tfs.Resize(image_size),
                                    tfs.RandomResizedCrop(crop_size),
                                    tfs.RandomHorizontalFlip(),
                                    tfs.ToTensor(),
                                    normalize
                                ])
    elif augs == 3:
        _transforms = tfs.Compose([
                                    tfs.RandomResizedCrop(crop_size),
                                    tfs.RandomGrayscale(p=0.2),
                                    tfs.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                    tfs.RandomHorizontalFlip(),
                                    tfs.ToTensor(),
                                    normalize
                                ])

    if is_validation:
        dataset = DataSet(torchvision.datasets.ImageFolder(image_dir + '/val', _transforms))
    else:
        dataset = DataSet(torchvision.datasets.ImageFolder(image_dir + '/train', _transforms))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return loader


def return_model_loader(args, return_loader=True):
    outs = [args.ncl]*args.hc
    assert args.arch in ['alexnet','resnetv2','resnetv1','htnet','htnet_pose','htnet_rnn']
    
    if not hasattr(args, 't_max'):
        args.t_max = 1
    if not hasattr(args, 't_min'):
        args.t_min = -1
    if not hasattr(args, 'rescale_rate'):
        args.rescale_rate = 0
    if not hasattr(args, 'cont_data'):
        args.cont_data = False
    train_loader = get_htnet_data_loader(data_dir=args.imagenet_path,
                                         batch_size=args.batch_size,
                                         num_workers=args.workers,
                                         pat_id = args.pat_id,
                                         n_states = args.n_states,
                                         data_srate = args.data_srate,
                                         len_eps = args.len_eps,
                                         curr_fold = args.curr_fold,
                                         tlim = [args.t_min, args.t_max],
                                         rescale_rate = args.rescale_rate,
                                         cont_data = args.cont_data,
                                         use_ecog = args.use_ecog) # SP 12/22/2020: switched to HTNet subfunc and removed n_augs argument
    
    if (args.arch == 'alexnet'):
        model = models.__dict__[args.arch](num_classes=outs)
    elif args.arch in ['htnet','htnet_pose','htnet_rnn']:
        Chans,Samples = train_loader.dataset.__getitem__(0)[0].shape
        model, params = models.__dict__[args.arch](num_classes=outs, Chans=Chans, Samples=Samples,
                                           use_ecog = args.use_ecog, is_supervised = args.is_supervised,
                                           cont_data = args.cont_data, param_lp=args.param_lp)
    elif args.arch == 'resnetv2':  # resnet
        model = models.__dict__[args.arch](num_classes=outs, nlayers=50, expansion=1)
    else:
        model = models.__dict__[args.arch](num_classes=outs)
    if not return_loader:
        return model

    return model, train_loader, Chans, Samples, params

def get_standard_data_loader(image_dir, is_validation=False,
                             batch_size=192, image_size=256, crop_size=224,
                             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                             num_workers=8,no_random_crops=False, tencrops=True):
    """Get a standard data loader for evaluating AlexNet representations in a standard way.
    """
    if image_dir is None:
        return None
    normalize = tfs.Normalize(mean=mean, std=std)
    if is_validation:
        if tencrops:
            transforms = tfs.Compose([
                tfs.Resize(image_size),
                tfs.TenCrop(crop_size),
                tfs.Lambda(lambda crops: torch.stack([normalize(tfs.ToTensor()(crop)) for crop in crops]))
            ])
            batch_size = int(batch_size/10)
        else:
            transforms = tfs.Compose([
                tfs.Resize(image_size),
                tfs.CenterCrop(crop_size),
                tfs.ToTensor(),
                normalize
            ])
    else:
        if not no_random_crops:
            transforms = tfs.Compose([
                tfs.RandomResizedCrop(crop_size),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                normalize
            ])
        else:
            transforms = tfs.Compose([
                tfs.Resize(image_size),
                tfs.CenterCrop(crop_size),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                normalize
            ])

    dataset = torchvision.datasets.ImageFolder(image_dir, transforms)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None
    )
    return loader

def get_standard_data_loader_pairs(dir_path, **kargs):
    """Get a pair of data loaders for training and validation.
         This is only used for the representation EVALUATION part.
    """
    train = get_standard_data_loader(os.path.join(dir_path, "train"), is_validation=False, **kargs)
    val = get_standard_data_loader(os.path.join(dir_path, "val"), is_validation=True, **kargs)
    return train, val


### HTNet-specific code ###
def get_htnet_data_loader(data_dir, dat_type='train',
                          batch_size=192,
                          num_workers=1, shuffle=False,
                          pat_id = 'EC01', n_states = 2,
                          data_srate = 250, len_eps = 3, curr_fold = 0,
                          tlim = [-1,1], rescale_rate = 0, cont_data = False,
                          use_ecog=True):
    """Get a data loader for evaluating HTNet representations in a standard way.
    """
    shuffle = dat_type == 'train'
    
    if data_dir is None:
        return None
    
    if cont_data:
        is_h5 = True if data_dir[-2:] == 'h5' else False
        if is_h5:
            dataset = ContNeuralDataset(data_dir)
        else:
            dataset = NeuralDataset(data_dir, dat_type=dat_type,
                                    pat_id=pat_id, n_states=n_states,
                                    curr_fold=curr_fold, tlim = tlim,
                                    rescale_rate = rescale_rate)
#         win_len = 2
#         win_spacing = 0
#         t_ind_lims = [12,16] #[8,16]
#         dataset = ContNeuralDataset(data_dir, win_len = win_len,
#                                     win_spacing = win_spacing, t_ind_lims = t_ind_lims)
    else:
        dataset = NeuralDataset(data_dir, dat_type=dat_type,
                                pat_id=pat_id, n_states=n_states,
                                curr_fold=curr_fold, tlim = tlim,
                                rescale_rate = rescale_rate,
                                use_ecog=use_ecog)
#     if is_validation:
#         dataset = NeuralDataset(data_dir + '/val')
#     else:
#         dataset = NeuralDataset(data_dir + '/train')
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None
    )
    return loader


class NeuralDataset(torch.utils.data.Dataset):
    '''Data loader class (currently loads in all data at once; may want to change this for larger datasets)'''
    def __init__(self, lp, dat_type='train', rand_seed = 1337, pat_id = 'EC01',
                 tlim = [-1,1], n_chans_all = 140, n_folds = 3,
                 curr_fold = 0, n_states = 2, data_srate = 250, len_eps = 3,
                 rescale_rate = 0, use_ecog=True):
        self.dat_type = dat_type
        self.rescale_rate = rescale_rate
        np.random.seed(rand_seed)

        # Load ECoG data
        X,y,X_test,y_test,sbj_order,sbj_order_test = load_data(pat_id, lp+'/',
                                                               n_chans_all=n_chans_all,
                                                               test_day=None, tlim=tlim,
                                                               n_states=n_states,
                                                               data_srate = data_srate, len_eps = len_eps)  # test_day='last'
        X[np.isnan(X)] = 0 # set all NaN's to 0

        # Make sure labels start at 0 and are consecutive
        miny = y.min()
        y -= miny
        y = y.astype('int')

        # Create splits for train/val and fit model
        n_folds = 5
        sss = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        for i, inds in enumerate(sss.split(X, y)):
            if i == curr_fold:
                train_inds, test_inds = inds
        
        # Split data and labels into train/val sets
        scalings = 'median'  # 'mean' if use_ecog else 'median'
        ss_dat = mne.decoding.Scaler(scalings=scalings)
        ss_dat.fit(X[train_inds,...])
        if self.dat_type=='train':
            # Standardize data
            x_train = ss_dat.transform(X[train_inds,...])  # X[train_inds,...]
            self.x_data = torch.tensor(x_train, dtype=torch.float32)
            self.y_data = torch.tensor(y[train_inds], dtype=torch.long)
        elif self.dat_type=='test':
            
            # Standardize data
            X_test = ss_dat.transform(X[test_inds,...])
            self.x_data = torch.tensor(X_test, dtype=torch.float32)
            self.y_data = torch.tensor(y[test_inds], dtype=torch.long)
        else:
            # Standardize data
            x_val = ss_dat.transform(X[test_inds,...])  # X[val_inds,...]
            self.x_data = torch.tensor(x_val, dtype=torch.float32)
            self.y_data = torch.tensor(y[test_inds], dtype=torch.long)

    def __len__(self):
        return len(self.x_data)  # required

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.x_data[idx, ...]
        y = self.y_data[idx]
        return X, y, idx


def load_data(pats_ids_in, lp, n_chans_all=64, test_day=None, tlim=[-1,1], event_types=['rest','move'],
              n_states = 2, data_srate = 250, len_eps = 3):
    '''
    Load ECoG data from all subjects and combine (uses xarray variables)
    
    If len(pats_ids_in)>1, the number of electrodes will be padded or cut to match n_chans_all
    If test_day is not None, a variable with test data will be generated for the day specified
        If test_day = 'last', the last day will be set as the test day.
    '''
    if not isinstance(pats_ids_in, list):
        pats_ids_in = [pats_ids_in]
    sbj_order,sbj_order_test = [],[]
    X_test_subj,y_test_subj = [],[] #placeholder vals
        
    #Gather each subjects data, and concatenate all days
    fID = natsort.natsorted(glob.glob(lp+pats_ids_in[0]+'*_data.nc'))[0]
    ep_data_in = xr.open_dataset(fID)
    if 'events' not in ep_data_in['__xarray_dataarray_variable__'].dims:
        # Case for loading Kai's data
        for j in range(len(pats_ids_in)):
            pat_curr = pats_ids_in[j]
            
            fID = natsort.natsorted(glob.glob(lp+pat_curr+'*_data.nc'))[0]
            ep_data_in = xr.open_dataset(fID).to_array().values
            ep_data_in = ep_data_in.reshape(ep_data_in.shape[0],ep_data_in.shape[1],-1,int(len_eps*data_srate))
            ep_data_in = np.moveaxis(ep_data_in,2,0).squeeze() # events, 1, channels, time
            labels = ep_data_in[...,-1,:].squeeze()
            labels = mode(labels,axis=1)[0].squeeze() # 1 state per event
            ep_data_in = ep_data_in[...,:-1,:]
            n_ecog_chans = ep_data_in.shape[-2]

            if n_chans_all < n_ecog_chans:
                n_chans_curr = n_chans_all
            else:
                n_chans_curr = n_ecog_chans

            #Remove events with greater than n_states
            if n_states==3:
                labels_cp = labels.copy()
                labels = np.delete(labels,np.nonzero(labels_cp>2)[0])
                ep_data_in = np.delete(ep_data_in,np.nonzero(labels_cp>2)[0],axis=0)
            elif n_states==2:
                state_rem = 0
                labels_cp = labels.copy()
                labels = np.delete(labels,np.nonzero(labels_cp==state_rem)[0])
                ep_data_in = np.delete(ep_data_in,np.nonzero(labels_cp==state_rem)[0],axis=0)
                labels[labels>state_rem] -= 1

            #Note participant order
            sbj_order += [j]*ep_data_in.shape[0]

            #Pad data in electrode dimension if necessary
            if (len(pats_ids_in) > 1) and (n_chans_all > n_ecog_chans):
                dat_sh = list(ep_data_in.shape)
                dat_sh[1] = n_chans_all
                #Create dataset padded with zeros if less than n_chans_all, or cut down to n_chans_all
                X_pad = np.zeros(dat_sh)
                X_pad[:,:n_ecog_chans,...] = ep_data_in
                ep_data_in = X_pad.copy()

            #Concatenate across subjects
            if j==0:
                X_subj = ep_data_in.copy()
                y_subj = labels.copy()
            else:
                X_subj = np.concatenate((X_subj,ep_data_in.copy()),axis=0)
                y_subj = np.concatenate((y_subj,labels.copy()),axis=0)

        sbj_order = np.asarray(sbj_order)
        
        # Dummy variables for consistent output
        sbj_order_test = np.zeros_like(sbj_order)
        X_test_subj = np.zeros_like(X_subj)
        y_test_subj = np.zeros_like(y_subj)
    else:
        for j in range(len(pats_ids_in)):
            pat_curr = pats_ids_in[j]
            fID = natsort.natsorted(glob.glob(lp+pat_curr+'*_data.nc'))[0] # Take first matching file
            ep_data_in = xr.open_dataset(fID)
            ep_times = np.asarray(ep_data_in.time)
            time_inds = np.nonzero(np.logical_and(ep_times>=tlim[0],ep_times<=tlim[1]))[0]
            n_ecog_chans = (len(ep_data_in.channels)-1)

            if n_chans_all < n_ecog_chans:
                n_chans_curr = n_chans_all
            else:
                n_chans_curr = n_ecog_chans

            # Perform different actions depending if events dimension included
            if test_day == 'last':
                test_day_curr = np.unique(ep_data_in.events)[-1] #select last day
            else:
                test_day_curr = test_day

            days_all_in = np.asarray(ep_data_in.events)

            if test_day is None:
                #No test output here
                days_train = np.unique(days_all_in)
                test_day_curr = None
            else:
                days_train = np.unique(days_all_in)[:-1]
                day_test_curr = np.unique(days_all_in)[-1]
                days_test_inds = np.nonzero(days_all_in==day_test_curr)[0]

            #Compute indices of days_train in xarray dataset
            days_train_inds = []
            for day_tmp in list(days_train):
                days_train_inds.extend(np.nonzero(days_all_in==day_tmp)[0])

            #Extract data and labels
            dat_train = ep_data_in[dict(events=days_train_inds,channels=slice(0,n_chans_curr),
                                        time=time_inds)].to_array().values.squeeze()
            labels_train = ep_data_in[dict(events=days_train_inds,channels=ep_data_in.channels[-1],
                                           time=0)].to_array().values.squeeze()
            sbj_order += [j]*dat_train.shape[0]

            if test_day is not None:
                dat_test = ep_data_in[dict(events=days_test_inds,channels=slice(0,n_chans_curr),
                                           time=time_inds)].to_array().values.squeeze()
                labels_test = ep_data_in[dict(events=days_test_inds,channels=ep_data_in.channels[-1],
                                              time=0)].to_array().values.squeeze()
                sbj_order_test += [j]*dat_test.shape[0]

            # Pad data in electrode dimension if necessary
            if (len(pats_ids_in) > 1) and (n_chans_all > n_ecog_chans):
                dat_sh = list(dat_train.shape)
                dat_sh[1] = n_chans_all
                # Create dataset padded with zeros if less than n_chans_all, or cut down to n_chans_all
                X_pad = np.zeros(dat_sh)
                X_pad[:,:n_ecog_chans,...] = dat_train
                dat_train = X_pad.copy()

                if test_day is not None:
                    dat_sh = list(dat_test.shape)
                    dat_sh[1] = n_chans_all
                    #Create dataset padded with zeros if less than n_chans_all, or cut down to n_chans_all
                    X_pad = np.zeros(dat_sh)
                    X_pad[:,:n_ecog_chans,...] = dat_test
                    dat_test = X_pad.copy()
            
            # Remove bad electrodes for single subjects
            if (len(pats_ids_in) == 1) and (fID.split('_')[-2] == 'ecog'):
                # Load param file from pre-trained model
                file_pkl = open(lp+'/proj_mat/bad_ecog_electrodes.pkl', 'rb')
                bad_elecs_ecog = pickle.load(file_pkl)
                file_pkl.close()
                pat_ind = int(pat_curr[-2:])-1
                inds2drop = bad_elecs_ecog[pat_ind]
                
                indskeep = np.setdiff1d(np.arange(n_chans_curr),inds2drop)
                dat_train = dat_train[:,indskeep,:]
                if test_day is not None:
                    dat_test = dat_test[:,indskeep,:]
            
            #Concatenate across subjects
            if j==0:
                X_subj = dat_train.copy()
                y_subj = labels_train.copy()
                if test_day is not None:
                    X_test_subj = dat_test.copy()
                    y_test_subj = labels_test.copy()
            else:
                X_subj = np.concatenate((X_subj,dat_train.copy()),axis=0)
                y_subj = np.concatenate((y_subj,labels_train.copy()),axis=0)
                if test_day is not None:
                    X_test_subj = np.concatenate((X_test_subj,dat_test.copy()),axis=0)
                    y_test_subj = np.concatenate((y_test_subj,labels_test.copy()),axis=0)

        sbj_order = np.asarray(sbj_order)
        sbj_order_test = np.asarray(sbj_order_test)
    
    return X_subj,y_subj,X_test_subj,y_test_subj,sbj_order,sbj_order_test
    
    
    
class ContNeuralDataset(torch.utils.data.Dataset):
    '''Data loader class (loads in data to memory as needed)
       Note that H5 file must be opened/closed in __getitem__ to avoid spurious NaNs.'''
    def __init__(self, in_file):
        super(ContNeuralDataset, self).__init__()
        
        with h5py.File(in_file,'r') as fin:
            self.n_eps, self.n_chs, self.n_ts = fin['dataset'].shape
        self.in_file = in_file
        fin.close()
        del fin  

    def __getitem__(self, index):
        with h5py.File(self.in_file,'r') as fin:
            dat_val = fin['dataset'][index,:,:]
        y = np.random.rand(1) if dat_val.ndim < 3 else np.random.rand(dat_val.shape[0])
        return torch.tensor(dat_val, dtype=torch.float32), y, index

    def __len__(self):
        return self.n_eps
    
    
class SimpleDataset(torch.utils.data.Dataset):
    '''Data loader class (currently loads in all data at once; may want to change this for larger datasets)'''
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.long)

    def __len__(self):
        return len(self.x_data)  # required

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.x_data[idx, ...]
        y = self.y_data[idx]
        return X, y, idx