import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math, pdb

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def sinc(band,t_right):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    y=torch.cat([y_left,Variable(torch.ones(1)).cuda(),y_right])

    return y
    

class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=250, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=1,
                 min_band_hz=5, logspace=False, max_band_hz=20):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.max_band_hz = max_band_hz
        self.logspace = logspace

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = self.min_low_hz #1 #
        high_hz = self.sample_rate / 2 #- (self.min_low_hz + self.min_band_hz)
        
        if self.logspace:	
            hz = np.logspace(np.log10(low_hz), np.log10(high_hz), self.out_channels + 1)	
        else:
            hz = np.linspace(low_hz, high_hz, self.out_channels + 1)
    
#         mel = np.linspace(self.to_mel(low_hz),
#                           self.to_mel(high_hz),
#                           self.out_channels + 1)
#         hz = self.to_hz(mel)
        

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1)) # /self.sample_rate

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1)) # /self.sample_rate

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

 


    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_elecs, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_elecs, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

#         low = torch.clamp(self.low_hz_*self.sample_rate,
#                           self.min_low_hz, (self.sample_rate/2 - self.min_band_hz))# self.min_low_hz  + 
        
#         band_hz = torch.clamp(self.band_hz_*self.sample_rate,
#                               self.min_band_hz,self.max_band_hz)
        
#         high = torch.clamp(low + band_hz, self.min_low_hz,self.sample_rate/2)

#         low = torch.clamp(self.min_low_hz  + torch.abs(self.low_hz_*self.sample_rate), self.min_low_hz, (self.sample_rate/2 - self.min_band_hz))
        
#         high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_*self.sample_rate),self.min_low_hz+self.min_band_hz,self.sample_rate/2)
        low = self.min_low_hz  + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]
#         self.low_hz_ = nn.Parameter(low/self.sample_rate)
#         self.band_hz_ = nn.Parameter((high-low)/self.sample_rate)
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])
        
        
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        
        band_pass = band_pass / (2*band[:,None])
        

        self.filters = (band_pass).view(self.out_channels, 1, 1, self.kernel_size)  # SP 12/31/2020: reshape filters to match 4D data shape
        
        return F.conv2d(waveforms, self.filters,
                        stride=self.stride, padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1) #SP 1/3/2020: changed F.conv1d to F.conv2d

    def constrain_weights(self):
        '''Constrain the weights after model optimization'''
        low = torch.clamp(self.low_hz_*self.sample_rate,
                          self.min_low_hz,
                          (self.sample_rate/2 - self.min_band_hz))
        
        self.low_hz_ = nn.Parameter(low/self.sample_rate)
        band = torch.clamp(self.band_hz_,
                           self.min_band_hz,
                           self.max_band_hz)
        hi = torch.clamp(low + band, self.min_low_hz, self.sample_rate/2)
        self.band_hz_ = nn.Parameter((hi-low)/self.sample_rate)
        
class sinc_conv(nn.Module):
    
#     def __init__(self,N_filt,Filt_dim,fs,pos_fs_order=True,min_freq=1.0,**kwargs):
#         super(sinc_conv,self).__init__(**kwargs)
#         self.N_filt=N_filt
#         self.Filt_dim=Filt_dim
#         self.fs=fs
#         self.pos_fs_order=pos_fs_order
#         self.min_freq = min_freq
        
        
    def __init__(self, N_filt,Filt_dim,fs):
        super(sinc_conv,self).__init__()
        
        self.min_freq = 1.0
        self.N_filt=N_filt
        self.Filt_dim=Filt_dim
        self.fs=fs
        # Mel Initialization of the filterbanks
#         low_freq_mel = 80
#         high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
#         mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
#         f_cos = (700 * (10**(mel_points / 2595) - 1)) # Convert Mel to Hz
        f_cos = np.linspace(self.min_freq, self.fs/2, self.N_filt+1)
#         f_cos = np.logspace(np.log10(self.min_freq), np.log10(self.fs/2), self.N_filt+1)
    
        b1=np.roll(f_cos,1)[1:]
        b2=np.roll(f_cos,-1)[:-1] # SP 12/31/2020: added 1 to n_filt, then remove extra value here to have non-redundant filters
#         b1[0]=1 #30
#         b2[-1]=(fs/2) #-100
                
        self.freq_scale=fs*1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1/self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2-b1)/self.freq_scale))

        
#         self.N_filt=N_filt
#         self.Filt_dim=Filt_dim
#         self.fs=fs
        

    def forward(self, x):
        
        filters=Variable(torch.zeros((self.N_filt,self.Filt_dim))).cuda()
        N=self.Filt_dim
        t_right=Variable(torch.linspace(1, (N-1)/2, steps=int((N-1)/2))/self.fs).cuda()
        
        
#         min_freq=50.0;
        min_band= 5.0 #50.0;
        
        filt_beg_freq=torch.abs(self.filt_b1)+self.min_freq/self.freq_scale
        filt_end_freq=filt_beg_freq+(torch.abs(self.filt_band)+min_band/self.freq_scale)
       
        n=torch.linspace(0, N, steps=N)

        # Filter window (hamming)
        window=0.54-0.46*torch.cos(2*math.pi*n/N);
        window=Variable(window.float().cuda())

        
        for i in range(self.N_filt):
                        
            low_pass1 = 2*filt_beg_freq[i].float()*sinc(filt_beg_freq[i].float()*self.freq_scale,t_right)
            low_pass2 = 2*filt_end_freq[i].float()*sinc(filt_end_freq[i].float()*self.freq_scale,t_right)
            band_pass=(low_pass2-low_pass1)

            band_pass=band_pass/torch.max(band_pass)

            filters[i,:]=band_pass.cuda()*window

        out=F.conv1d(x, filters.view(self.N_filt,1,1,self.Filt_dim))
    
        return out