import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.fft
import numpy as np

def hilbert_tf(x):
    #, N=None, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform in PyTorch.
    The transformation is done along the last axis by default.
    Adapted from scipy: https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py#L2012-L2120
    Parameters
    ----------
    x : tensor
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.
    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`
    """
    if x.dtype.is_complex:
        raise ValueError("x must be real.")
    N = x.size()[-1]
    
    Xf = torch.fft.fft(x)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    
    if len(x.size()) > 1:
        ind = [np.newaxis] * len(x.size())
        ind[-1] = slice(None)
        h = h[tuple(ind)]
    
    h = h.astype('complex64')
    use_cuda = torch.cuda.is_available()
    gpu_ind = torch.device("cuda:"+str(torch.cuda.current_device()) if use_cuda else "cpu")
    X_conv = Xf * torch.from_numpy(h).to(gpu_ind)
    X_ifft = torch.fft.ifft(X_conv)
    return X_ifft

def apply_hilbert_tf(x, do_log=False, compute_val='power', data_srate=250):
    """Compute Hilbert transform of signals w/ zero padding in PyTorch.
    Adapted from MNE function
    Parameters
    ----------
    x : tensor, shape (n_times)
        The signal to convert
    n_fft : int
        Size of the FFT to perform, must be at least ``len(x)``.
        The signal will be cut back to original length.
    Returns
    -------
    out : array, shape (n_times)
        The power, phase, or frequency from the hilbert transform of the signal.
    """
    hilb_sig = hilbert_tf(x)
    
    if compute_val=='power':
        out = torch.abs(hilb_sig)
        if do_log:
            out = torch.log1p(out)
#     elif compute_val=='phase':
#         out = unwrap(angle_custom(hilb_sig))
#     elif compute_val=='freqslide':
#         ang = angle_custom(hilb_sig) #tf.math.angle(hilb_sig)
#         ang = data_srate*diff(unwrap(ang))/(2*np.pi)
#         paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, 1]])
#         out = tf.pad(ang, paddings, "CONSTANT") # pad time dimension because of difference function
    return out


class ConstrainedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding = 0, bias = False, groups = 1, stride = 1,
                 dilation = 1, padding_mode = 'zeros', max_norm = 1):
        super(ConstrainedConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                                padding=padding, bias=bias, groups=groups,
                                                stride = stride, dilation = dilation,
                                                padding_mode = padding_mode)
        self.max_norm = max_norm
    
    def forward(self, input):
        '''Forward pass with norm constrain (from @kevinzakka)'''
        eps = 1e-8
        norm = torch.norm(self.weight, p=2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self.max_norm)
        w = self.weight * (desired / (eps + norm))
        return F.conv2d(input, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
class ConstrainedDense(nn.Linear):
    def __init__(self, in_features, out_features, max_norm = 1):
        super(ConstrainedDense, self).__init__(in_features, out_features)
        self.max_norm = max_norm
        
    def forward(self, input):
        '''Forward pass with norm constrain (from @kevinzakka)'''
        eps = 1e-8
        norm = torch.norm(self.weight, p=2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self.max_norm)
        w = self.weight * (desired / (eps + norm))
        return F.linear(input, w, self.bias)