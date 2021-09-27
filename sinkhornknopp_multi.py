import torch
import torch.nn as nn
import time, pdb
import numpy as np

from util import  py_softmax,MovingAverage
from multigpu import gpu_mul_Ax, gpu_mul_xA, aggreg_multi_gpu, gpu_mul_AB

def cpu_sk(o1):
    """ Sinkhorn Knopp optimization on CPU
        * stores activations to RAM
        * does matrix-vector multiplies on CPU
        * slower than GPU
    """
    # 1. aggregate inputs:
    N = len(o1.pseudo_loader.dataset)
    if o1.hc == 1:
        o1.PS = np.zeros((N, o1.K), dtype=o1.dtype)
    else:
        o1.PS_pre = np.zeros((N, o1.presize), dtype=o1.dtype)
    now = time.time()
    l_dl = len(o1.pseudo_loader)
    time.time()
    batch_time = MovingAverage(intertia=0.9)
    o1.model.headcount = 1
    for batch_idx, (data, _, _selected) in enumerate(o1.pseudo_loader):
        data = data.to(o1.dev)
        mass = data.size(0)
        if o1.hc == 1:
            p = nn.functional.softmax(o1.model(data), 1)
            o1.PS[_selected, :] = p.detach().cpu().numpy().astype(o1.dtype)
        else:
            p = o1.model(data)
            o1.PS_pre[_selected, :] = p.detach().cpu().numpy().astype(o1.dtype)
        batch_time.update(time.time() - now)
        now = time.time()
        if batch_idx % 50 == 0:
            print(f"Aggregating batch {batch_idx:03}/{l_dl}, speed: {mass / batch_time.avg:04.1f}Hz",
                  end='\r', flush=True)
    o1.model.headcount = o1.hc
    print("Aggreg of outputs  took {0:.2f} min".format((time.time() - now) / 60.), flush=True)
    
    # 2. solve label assignment via sinkhorn-knopp:
    if o1.hc == 1:
        o1 = optimize_L_sk(o1, nh=0)
    else:
        for nh in range(o1.hc):
            print("computing head %s " % nh, end="\r", flush=True)
            tl = getattr(o1.model, "top_layer%d" % nh)
            time_mat = time.time()

            # clear memory
            try:
                del o1.PS
            except:
                pass

            # apply last FC layer (a matmul and adding of bias)
            o1.PS = (o1.PS_pre @ tl.weight.cpu().numpy().T.astype(o1.dtype)
                       + tl.bias.cpu().numpy().astype(o1.dtype))
            print(f"matmul took {(time.time() - time_mat)/60:.2f}min", flush=True)
            o1.PS = py_softmax(o1.PS, 1)
            o1 = optimize_L_sk(o1, nh=nh)
    return o1

def gpu_sk(o1):
    """ Sinkhorn Knopp optimization on GPU
            * stores activations on multiple GPUs (needed when dataset is large)
            * does matrix-vector multiplies on GPU (extremely fast)
            * recommended variant
            * due to multi-GPU use, it's a bit harder to understand what's happening -> see CPU variant to understand
    """
    # 1. aggregate inputs:
    start_t = time.time()
    if o1.hc == 1:
        o1.PS, indices = aggreg_multi_gpu(o1.model, o1.pseudo_loader,
                                            hc=o1.hc, dim=o1.outs[0], TYPE=o1.dtype)

    else:
        try: # just in case stuff
            del o1.PS_pre
        except:
            pass
        torch.cuda.empty_cache()
        time.sleep(1)
        o1.PS_pre, indices = aggreg_multi_gpu(o1.model, o1.pseudo_loader,
                                                hc=o1.hc, dim=o1.presize, TYPE=torch.float32)
        o1.model.headcount = o1.hc
    print("Aggreg of outputs  took {0:.2f} min".format((time.time() - start_t) / 60.), flush=True)
    # 2. solve label assignment via sinkhorn-knopp:
    if o1.hc == 1:
        o1 = optimize_L_sk_multi(o1, nh=0)
        o1.L[0,indices] = o1.L[0,:]
    else:
        for nh in range(o1.hc):
            tl = getattr(o1.model, "top_layer%d" % nh)
            time_mat = time.time()
            try:
                del o1.PS
                torch.cuda.empty_cache()
            except:
                pass

            # apply last FC layer (a matmul and adding of bias)
            o1.PS = gpu_mul_AB(o1.PS_pre, tl.weight.t(),
                                 c=tl.bias, dim=o1.outs[nh], TYPE=o1.dtype)
            print("matmul took %smin" % ((time.time() - time_mat) / 60.), flush=True)
            o1 = optimize_L_sk_multi(o1, nh=nh)
            o1.L[nh][indices] = o1.L[nh]
    return o1

def optimize_L_sk(o1, nh=0):
    N = max(o1.L.size())
    tt = time.time()
    o1.PS = o1.PS.T # now it is K x N
    r = np.ones((o1.outs[nh], 1), dtype=o1.dtype) / o1.outs[nh]
    c = np.ones((N, 1), dtype=o1.dtype) / N
    o1.PS **= o1.lamb  # K x N
    inv_K = o1.dtype(1./o1.outs[nh])
    inv_N = o1.dtype(1./N)
    err = 1e6
    _counter = 0
    while err > 1e-1:
        r = inv_K / (o1.PS @ c)          # (KxN)@(N,1) = K x 1
        c_new = inv_N / (r.T @ o1.PS).T  # ((1,K)@(KxN)).t() = N x 1
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1
    print("error: ", err, 'step ', _counter, flush=True)  # " nonneg: ", sum(I), flush=True)
    # inplace calculations.
    o1.PS *= np.squeeze(c)
    o1.PS = o1.PS.T
    o1.PS *= np.squeeze(r)
    o1.PS = o1.PS.T
    argmaxes = np.nanargmax(o1.PS, 0) # size N
    newL = torch.LongTensor(argmaxes)
    o1.L[nh] = newL.to(o1.dev)
    print('opt took {0:.2f}min, {1:4d}iters'.format(((time.time() - tt) / 60.), _counter), flush=True)
    return o1

def optimize_L_sk_multi(o1, nh=0):
    """ optimizes label assignment via Sinkhorn-Knopp.

         this implementation uses multiple GPUs to store the activations which allow fast matrix multiplies

         Parameters:
             nh (int) number of the head that is being optimized.

    """
    N = max(o1.L.size())
    tt = time.time()
    r = torch.ones((o1.outs[nh], 1), device='cuda:0', dtype=o1.dtype) / o1.outs[nh]
    c = torch.ones((N, 1), device='cuda:0', dtype=o1.dtype) / N
    ones = torch.ones(N, device='cuda:0', dtype=o1.dtype)
    inv_K = 1. / o1.outs[nh]
    inv_N = 1. / N

    # inplace power of softmax activations:
    [qq.pow_(o1.lamb) for qq in o1.PS]  # K x N

    err = 1e6
    _counter = 0
    ngpu = torch.cuda.device_count()
    splits = np.cumsum([0] + [a.size(0) for a in o1.PS])
    while err > 1e-1:
        r = inv_K / (gpu_mul_xA(c.t(), o1.PS,
                                ngpu=ngpu, splits=splits, TYPE=o1.dtype)).t()  # ((1xN)@(NxK)).T = Kx1
        c_new = inv_N / (gpu_mul_Ax(o1.PS, r,
                                    ngpu=ngpu, splits=splits, TYPE=o1.dtype))  # (NxK)@(K,1) = N x 1
        torch.cuda.synchronize()  # just in case
        if _counter % 10 == 0:
            err = torch.sum(torch.abs((c.squeeze() / c_new.squeeze()) - ones)).cpu().item()
        c = c_new
        _counter += 1
    print("error: ", err, 'step ', _counter, flush=True)

    # getting the final tranportation matrix #####################
    for i, qq in enumerate(o1.PS):
        torch.mul(qq, c[splits[i]:splits[i + 1], :].to('cuda:' + str(i + 1)), out=qq)
    [torch.mul(r.to('cuda:' + str(i + 1)).t(), qq, out=qq) for i, qq in enumerate(o1.PS)]
    argmaxes = torch.empty(N, dtype=torch.int64, device='cuda:0')

    start_idx = 0
    for i, qq in enumerate(o1.PS):
        amax = torch.argmax(qq, 1)
        argmaxes[start_idx:start_idx + len(qq)].copy_(amax)
        start_idx += len(qq)
    newL = argmaxes
    print('opt took {0:.2f}min, {1:4d}iters'.format(((time.time() - tt) / 60.), _counter), flush=True)
    # finally, assign the new labels ########################
    o1.L[nh] = newL
    return o1

