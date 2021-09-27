''' Main model training script
(modified from from https://github.com/yukimasano/self-label) '''

import argparse
import warnings
import os, pdb, pickle
import time
import numpy as np

import mne
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchsummary import summary
try:
    from tensorboardX import SummaryWriter
except:
    pass
from scipy.signal import savgol_filter
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, accuracy_score,\
                            confusion_matrix, adjusted_mutual_info_score, adjusted_rand_score
from sklearn import svm
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

import files
import util
import sinkhornknopp_multi as sk
from data import return_model_loader, get_htnet_data_loader, SimpleDataset, load_data
from util import EarlyStopping, set_n_mod_args
warnings.simplefilter("ignore", UserWarning)

def str2bool(v):
    '''Allows True/False booleans in argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# class LabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self, smoothing=0.1):
#         super(LabelSmoothingCrossEntropy, self).__init__()
#         self.smoothing = smoothing
#     def forward(self, x, target):
#         confidence = 1. - self.smoothing
#         logprobs = F.log_softmax(x, dim=-1)
#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -logprobs.mean(dim=-1)
#         loss = confidence * nll_loss + self.smoothing * smooth_loss
#         return loss.mean()
        
class Optimizer:
    def __init__(self, m, hc, ncl, t_loader, n_epochs, lr, arch, weight_decay=1e-5):
        self.num_epochs = n_epochs
        self.lr = lr
        self.lr_schedule = lambda epoch: ((epoch < 350) * (self.lr * (0.1 ** (epoch // args.lrdrop)))
                                          + (epoch >= 350) * self.lr * 0.1 ** 3)

        self.momentum = 0.9
        self.weight_decay = weight_decay
        self.resume = True
        self.writer = None

        # model stuff
        self.hc = hc
        self.K = ncl
        self.model = m
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nmodel_gpus = len(args.modeldevice)
        self.pseudo_loader = t_loader # can also be DataLoader with less aug.
        self.train_loader = t_loader
        self.lamb = args.lamb # the parameter lambda in the SK algorithm
        self.dtype = torch.float64 if not args.cpu else np.float64

        self.outs = [self.K]*args.hc
        # activations of previous to last layer to be saved if using multiple heads.
        self.presize = 4096 if arch == 'alexnet' else 2048
        self.arch = arch
        self.is_supervised = args.is_supervised
        self.is_xdc = args.is_xdc

def o_check(o_lst, field, subfield=None, checklen=False):
    """Check consistency of a field within list of objects"""
    assert isinstance(field, str)
    if subfield:
        assert isinstance(subfield, str)
        fld_lst = [getattr(getattr(o_curr,field),subfield) for o_curr in o_lst]
    else:
        fld_lst = [getattr(o_curr,field) for o_curr in o_lst]
        
    if checklen:
        fld_lst = [len(val) for val in fld_lst]
    assert fld_lst[:-1] == fld_lst[1:]
        
def optimize(o_lst):
    """Perform full optimization.
    o_lst : list of optimizers"""
    n_o = len(o_lst)
    N = len(o_lst[0].pseudo_loader.dataset)

    # Parameter consistency checks
    o_check(o_lst, 'pseudo_loader', 'dataset', checklen=True)
    o_check(o_lst, 'dev')
    o_check(o_lst, 'num_epochs')
    o_check(o_lst, 'hc')
    o_check(o_lst, 'is_xdc')
    o_check(o_lst, 'is_supervised')

    # Set up model ###############################################################
    optimizers = []
    for o_curr in o_lst:
        # Put model onto GPU
        o_curr.model = o_curr.model.to(o_curr.dev)
        
        # Set optimization times (spread exponentially, can also just be linearly spaced)
        rem_add = args.batch_size - (N % args.batch_size) # SP 12/23/2020 accounts for remainder if data loader adds extra batch
        o_curr.optimize_times = [(o_curr.num_epochs+2)*(N+rem_add)] + \
                                 ((o_curr.num_epochs+1.01)*(N+rem_add)*(np.linspace(0, 1, args.nopts)**2)\
                                  [::-1]).tolist()

        # Set optimizer
        if o_lst[0].is_supervised:
            optimizers.append(torch.optim.Adam(o_curr.model.parameters(),
                                               weight_decay=o_curr.weight_decay))
        else:
            optimizers.append(torch.optim.SGD(filter(lambda p: p.requires_grad, o_curr.model.parameters()),
                                              weight_decay=o_curr.weight_decay,
                                              momentum=o_curr.momentum,
                                              lr=o_curr.lr))

        print('We will optimize L at epochs:', [np.round(1.0 * t / N, 2) for t in o_curr.optimize_times],
              flush=True)

        # Shuffle initial labels
        o_curr.L = np.zeros((o_curr.hc, N), dtype=np.int32)
        for nh in range(o_curr.hc):
            for _i in range(N):
                o_curr.L[nh, _i] = _i % o_curr.outs[nh]
            o_curr.L[nh] = np.random.permutation(o_curr.L[nh])
        o_curr.L = torch.LongTensor(o_curr.L).to(o_curr.dev)

    # Perform optmization ###############################################################
    lowest_loss = [1e9 for _i in range(n_o)]
    epoch = 0
    thresh_div = 1/(n_o-1)
    while epoch < (o_lst[0].num_epochs+1):
        o_L = [deepcopy(o_curr.L) for o_curr in o_lst]
        
        for i, o_curr in enumerate(o_lst):
            if o_curr.is_xdc:
                inds_L = np.random.rand(N)//thresh_div  # randomly select cross-modal labels to use
                inds_L = inds_L.astype('int')
                inds_L[inds_L >= i] += 1  # avoid current model's labels
                o_L_curr = np.array([int(o_L[ind_i][0, _j]) for _j, ind_i in enumerate(inds_L)])
                o_L_curr = torch.LongTensor(o_L_curr[np.newaxis, :]).to(o_curr.dev)
            else:
                o_L_curr = o_L[i]  # use own labels for unimodal

            m, o_lst[i] = optimize_epoch(o_curr, o_L_curr, optimizers[i], o_curr.train_loader, epoch,
                                         validation=False)
            if m['loss'] < lowest_loss[i]:
                lowest_loss[i] = m['loss']
        epoch += 1

    return [o_curr.model for o_curr in o_lst]


def optimize_labels(o1, niter):
    if not args.cpu and torch.cuda.device_count() > 1:
        o1 = sk.gpu_sk(o1)
    else:
        o1.dtype = np.float64
        o1 = sk.cpu_sk(o1)
    return o1


def optimize_epoch(o_curr, o_otherL, optimizer, loader, epoch,
                   validation=False):
    print(f"Starting epoch {epoch}, validation: {validation} " + "="*30,flush=True)

    loss_value = util.AverageMeter()
    o_curr.model.train()
    
    if not o_curr.is_supervised:
        lr = o_curr.lr_schedule(epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
    XE = torch.nn.CrossEntropyLoss()  # LabelSmoothingCrossEntropy(smoothing=0.3) #

    for iter, (data, label, selected) in enumerate(loader):
        now = time.time()
        niter = epoch * len(loader) + iter
        if not o_curr.is_supervised:
            # only optimize labels for unsupervised condition
            if niter*args.batch_size >= o_curr.optimize_times[-1]:
                ############ optimize labels (done for o_curr SP 1/22/2021) #########################################
                o_curr.model.headcount = 1
                print('Optimizaton starting', flush=True)
                with torch.no_grad():
                    _ = o_curr.optimize_times.pop()
                    # Optimize training labels
                    o_curr = optimize_labels(o_curr, niter)

        data = data.to(o_curr.dev)
        mass = data.size(0)
        final = o_curr.model(data)

        #################### train CNN ####################################################
        if o_curr.is_supervised:
            # Use loss from label instead of cluster
            loss = XE(F.softmax(final, dim=1), label.to(o_curr.dev).long())
        else:
            o_other_hc = o_otherL.shape[0]
            if o_other_hc == 1:
                loss = XE(final, o_otherL[0, selected])
            else:
                loss = torch.mean(torch.stack([XE(final[h],
                                               o_otherL[h, selected]) for h in range(o_other_hc)]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not o_curr.is_supervised:
            loss_value.update(loss.item(), mass)
        data = 0

        # some logging stuff ##############################################################
        if iter % args.log_iter == 0:
            print(niter, " Loss: {0:.3f}".format(loss.item()), flush=True)
            print(niter, " Freq: {0:.2f}".format(mass/(time.time() - now)), flush=True)

    return {'loss': loss_value.avg}, o_curr

def model_acc(model_in, data_loader, sklsvm_in=None, meas='acc', D=None):
    '''Compute loss on validation set'''
    with torch.no_grad():
        model_in.eval()
        
        preds_all, true_all = [], []
        for X, Y, ind in data_loader:
            X_curr = X.to("cuda")
            predictions = F.softmax(model_in(X_curr), dim=1)
            _, pred = predictions.topk(1, 1, True, True)
            p_curr = pred.t().cpu().detach().numpy().tolist()[0]
            preds_all.extend(p_curr)
            true_all.extend(Y.detach().numpy().tolist())
        preds_all = np.asarray(preds_all)
        true_all = np.asarray(true_all)
    
    if meas == 'acc':
        acc, sklsvm_in, D_out = clust_acc(true_all,preds_all,sklsvm_in, D)
        return acc, D_out, sklsvm_in
    elif meas == 'homogsc':
        return homogeneity_score(true_all,preds_all), None, None
    elif meas == 'completesc':
        return homogeneity_score(true_all,preds_all), None, None
    elif meas == 'vscore':
        return v_measure_score(true_all,preds_all), None, None
    elif meas == 'adjmi':
        return adjusted_mutual_info_score(true_all,preds_all), None, None
    elif meas == 'adjrandsc':
        return adjusted_rand_score(true_all,preds_all), None, None
        
def clust_acc(y_true, y_pred, ind=None, D=None):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    if not D:
        D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    if not ind:
        ind = linear_sum_assignment(w.max() - w)
    acc = w[ind[0], ind[1]].sum()* 1.0 / y_pred.size
    return acc, ind, D

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')
    # optimizer
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=256, type=int, help='batch size (default: 256)')
    parser.add_argument('--lr', default=0.05, type=float, help='initial learning rate (default: 0.05)')
    parser.add_argument('--lrdrop', default=150, type=int, help='multiply LR by 0.1 every (default: 150 epochs)')
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: (-5)')
    parser.add_argument('--dtype', default='f64',choices=['f64','f32'], type=str, help='SK-algo dtype (default: f64)')

    # SK algo
    parser.add_argument('--nopts', default=100, type=int, help='number of pseudo-opts (default: 100)')
    parser.add_argument('--augs', default=3, type=int, help='augmentation level (default: 3)')
    parser.add_argument('--lamb', default=25, type=int, help='for pseudoopt: lambda (default:25) ')
    parser.add_argument('--cpu', default=False, action='store_true', help='use CPU variant (slow) (default: off)')


    # architecture
    parser.add_argument('--arch', nargs='+', type=str, help='model name')
    parser.add_argument('--archspec', default='big', choices=['big','small'], type=str, help='alexnet variant (default:big)')
    parser.add_argument('--ncl', default=3000, type=int, help='number of clusters per head (default: 3000)')
    parser.add_argument('--hc', default=1, type=int, help='number of heads (default: 1)')

    # housekeeping
    parser.add_argument('--device', default='0', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0', type=str, help='GPU numbers on which the CNN runs')
    parser.add_argument('--workers', default=6, type=int,help='number workers (default: 6)')
    parser.add_argument('--imagenet-path', nargs='+', default='', help='path to folder that contains `train` and `val`', type=str)
    parser.add_argument('--log-intv', default=1, type=int, help='save stuff every x epochs (default: 1)')
    parser.add_argument('--log-iter', default=200, type=int, help='log every x-th batch (default: 200)')

    parser.add_argument('--pat_id', default='EC01', type=str, help='Partcipant ID to run on')
    parser.add_argument('--n_states', default=2, type=int, help='Number of behavioral states to use (Kai data)')
    parser.add_argument('--data_srate', default=2, type=float, help='ECoG sampling rate (Kai data only)')
    parser.add_argument('--len_eps', default=3, type=float, help='Epoch length (Kai data only)')
    parser.add_argument('--curr_fold', default=0, type=int, help='Fold number to use (< 3)')
    
    parser.add_argument('--use_ecog', nargs='+', type=str2bool, required=False,
                        help='parameter switch for model (default: True)')
    parser.add_argument('--t_min', default=-1, type=int, help='Start time of epoch')
    parser.add_argument('--t_max', default=1, type=int, help='End time of epoch')
    parser.add_argument('--rescale_rate1', default=0, type=float, help='Percentage of data to randomly rescale (model 1)')
    parser.add_argument('--rescale_rate2', default=0, type=float, help='Percentage of data to randomly rescale (model 2)')
    parser.add_argument('--cont_data', type=str2bool, nargs='?', required=False,
                        default='False', help='Switch for loading continuous data (default: False)')
    parser.add_argument('--n_folds', default='3', type=str, help='Number of folds to use')
    parser.add_argument('--is_supervised', type=str2bool, nargs='?', required=False, default='False',
                        help='switch to perform supervised or unsupervised learning (default: False)')
    parser.add_argument('--is_xdc', type=str2bool, nargs='?', required=False, default='True',
                        help='switch to perform XDC or separate deep clustering (default: True)')
    parser.add_argument('--savepath', default='', help='path to save folder', type=str)
    parser.add_argument('--param_lp', default='', help='path to save folder', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()
    try:
        args.device = [int(item) for item in args.device.split(',')]
    except AttributeError:
        args.device = [int(args.device)]
    args.modeldevice = args.device
    
    assert (args.n_folds == 'loocv') or (isinstance(int(args.n_folds), int))
    util.setup_runtime(seed=42, cuda_dev_id=list(np.unique(args.modeldevice + args.device)))
    print(args, flush=True)
    
    
    args_lst = set_n_mod_args(args)

    # Load datasets
    X, y = [], []
    n_modalities = len(args_lst)
    for i in range(n_modalities):
        X1, y1, _, _, _, _ = load_data(args_lst[i].pat_id, args_lst[i].imagenet_path+'/',
                                       n_chans_all=140, test_day=None,
                                       tlim=[args_lst[i].t_min, args_lst[i].t_max])
        X1[np.isnan(X1)] = 0 # set all NaN's to 0
        y1 -= y1.min()
        y1 = y1.astype('int')
        
        X.append(X1)
        y.append(y1)
        del X1, y1

    # Consistency checks across modalities
    X_sh = [val.shape[0] for val in X]
    assert X_sh[:-1] == X_sh[1:]
    for i in range(n_modalities-1):
        assert (y[i] == y[i+1]).all()
        
    # Determine random fold splits
    n_evs = X_sh[0]
    n_folds = n_evs if args.n_folds == 'loocv' else int(args.n_folds)
    if args.n_folds == 'loocv':
        sss = LeaveOneOut()
        splits = sss.split(X[0])
    else:
        sss = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        splits = sss.split(X[0], y[0])
    accs = np.zeros((n_modalities, n_folds, 2))  # n_modalities x n_folds x [train, val]
    n_cls = len(np.unique(y[0]))
    
    homogsc, completesc, vscore, adjmi, adjrandsc = accs.copy(), accs.copy(), accs.copy(), accs.copy(), accs.copy()
    for i, inds in enumerate(splits):
        train_inds, test_inds = inds
        
        # Standardize data and create dataloader
        scalings = 'median'
        train_loader, test_loader = [], []
        params, o_lst = [], []
        for j in range(n_modalities):
#             ss_dat = mne.decoding.Scaler(scalings=scalings)
            x_train = X[j][train_inds,...] #ss_dat.fit_transform(X[j][train_inds,...])
            x_test = X[j][test_inds,...] #ss_dat.transform(X[j][test_inds,...])

            train_set = SimpleDataset(x_train, y[j][train_inds])
            test_set = SimpleDataset(x_test, y[j][test_inds])
            train_loader.append(torch.utils.data.DataLoader(train_set,
                                                            batch_size=args.batch_size, shuffle=True,
                                                            num_workers=args.workers))
            test_loader.append(torch.utils.data.DataLoader(test_set,
                                                           batch_size=1, shuffle=True,
                                                           num_workers=args.workers))
            del x_train, x_test, train_set, test_set
        
            # Setup models
            model_curr, _, Chans1, Samples1, params_curr = return_model_loader(args_lst[j])
            model_curr.to(torch.device("cuda"))
            summary(model_curr, input_size=(1, Chans1, Samples1))
            params.append(params_curr)

            # Setup optimizer
            o_lst.append(Optimizer(m=model_curr, hc=args.hc, ncl=args.ncl, t_loader=train_loader[j],
                                   n_epochs=args.epochs, lr=args.lr,
                                   arch = args.arch[j], weight_decay=10**args.wd,))
            del model_curr, params_curr

        # Optimize
        o_models = optimize(o_lst)

        # Compute model train/test accuracy
        for j, o_curr in enumerate(o_lst):
            accs[j, i, 0], D_curr, sklsvm_in_curr = model_acc(o_curr.model, train_loader[j])
            accs[j, i, 1], _, _ = model_acc(o_curr.model, test_loader[j], sklsvm_in_curr, D=D_curr)

            homogsc[j, i, 0], _, _ = model_acc(o_curr.model, train_loader[j], meas='homogsc')
            homogsc[j, i, 1], _, _ = model_acc(o_curr.model, test_loader[j], meas='homogsc')
            completesc[j, i, 0], _, _ = model_acc(o_curr.model, train_loader[j], meas='completesc')
            completesc[j, i, 1], _, _ = model_acc(o_curr.model, test_loader[j], meas='completesc')
            vscore[j, i, 0], _, _ = model_acc(o_curr.model, train_loader[j], meas='vscore')
            vscore[j, i, 1], _, _ = model_acc(o_curr.model, test_loader[j], meas='vscore')
            adjmi[j, i, 0], _, _ = model_acc(o_curr.model, train_loader[j], meas='adjmi')
            adjmi[j, i, 1], _, _ = model_acc(o_curr.model, test_loader[j], meas='adjmi')
            adjrandsc[j, i, 0], _, _ = model_acc(o_curr.model, train_loader[j], meas='adjrandsc')
            adjrandsc[j, i, 1], _, _ = model_acc(o_curr.model, test_loader[j], meas='adjrandsc')
            del sklsvm_in_curr
        del o_lst

    if not os.path.exists(args.savepath):
        os.mkdir(args.savepath)
    np.save(args.savepath + '/' + args.pat_id+'_acc.npy', accs.mean(1))
    np.save(args.savepath + '/' + args.pat_id+'_homogsc.npy', homogsc.mean(1))
    np.save(args.savepath + '/' + args.pat_id+'_completesc.npy', completesc.mean(1))
    np.save(args.savepath + '/' + args.pat_id+'_vscore.npy', vscore.mean(1))
    np.save(args.savepath + '/' + args.pat_id+'_adjmi.npy', adjmi.mean(1))
    np.save(args.savepath + '/' + args.pat_id+'_adjrandsc.npy', adjrandsc.mean(1))
    
    for i in range(n_modalities):
        pickle.dump(params[i], open(args.savepath + '/' + args.pat_id+'_params{}.pkl'.format(i), 'wb'))
    print(accs.mean(1))