# adapted from DeepCluster repo: https://github.com/facebookresearch/deepcluster
import math, pdb, pickle
import torch.nn as nn
import torch.nn.functional as F
from .htnet_utils import apply_hilbert_tf, ConstrainedConv2d, ConstrainedDense
from sincnet_utils import sinc_conv, SincConv_fast

__all__ = [ 'HTNet', 'htnet']

# Use HTNet model
class HTNet(nn.Module):
    def __init__(self, features, num_classes, is_supervised, rem_classifier=True):
        super(HTNet, self).__init__()
        self.features = features
        self.is_supervised = is_supervised
        self.rem_classifier = rem_classifier
        
        if self.is_supervised:
            # If supervised training, use softmax activation at end
            self.classifier = nn.Sequential(ConstrainedDense(self.features.last_lay_dim, num_classes[0],
                                                             max_norm = self.features.norm_rate),
                                            nn.Softmax(dim=1))
        else:
            out_dim = self.features.last_lay_dim//2
            if not self.rem_classifier:
                self.classifier = nn.Sequential(ConstrainedDense(self.features.last_lay_dim, out_dim,
                                                                 max_norm = self.features.norm_rate),
                                                nn.ELU(inplace=True))
            self.headcount = len(num_classes)
            self.return_features = False
            if len(num_classes) == 1:
                if self.rem_classifier:
                    self.top_layer = ConstrainedDense(self.features.last_lay_dim,
                                                      num_classes[0],
                                                      max_norm = self.features.norm_rate)
                else:
                    self.top_layer = nn.Linear(out_dim, num_classes[0])
            else:
                for a,i in enumerate(num_classes):
                    setattr(self, "top_layer%d" % a, nn.Linear(out_dim, i))
                self.top_layer = None  # this way headcount can act as switch.
    
    def forward(self, x):
        if not hasattr(self, 'is_supervised'):
            # for backwards compatibility
            self.is_supervised = False
        
        x = self.features(x)
        if (self.is_supervised) or (not self.rem_classifier):
            x = self.classifier(x)
    
        if self.is_supervised:
            return x
        else:
            if self.return_features: # switch only used for CIFAR-experiments
                return x
            if self.headcount == 1:
                if self.top_layer: # this way headcount can act as switch.
                    x = self.top_layer(x)
                return x
            else:
                outp = []
                for i in range(self.headcount):
                    outp.append(getattr(self, "top_layer%d" % i)(x))
                return outp


class ht_submodel(nn.Module):
    def __init__(self, args_in):
        super(ht_submodel, self).__init__()
        
        # Initialize class based on dictionary value/keys
        for key, value in args_in.items():
            setattr(self, key, value)
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if self.kernLength%2==0:
            self.kernLength=self.kernLength+1
        
        # Layer 1
        if self.useSincnet:
            self.conv1 = SincConv_fast(self.F1,self.kernLength,sample_rate = self.data_srate,
                                       min_low_hz=1, min_band_hz=5)
        else:
            self.conv1 = nn.Conv2d(1, self.F1, (1, self.kernLength),
                                   padding = 0, bias = False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1)
        
        # Layer 2
        self.depthconv = ConstrainedConv2d(self.F1, self.F1*self.D, (self.Chans, 1),
                                           groups=self.F1, bias = False, max_norm = 1)
        self.batchnorm2 = nn.BatchNorm2d(self.F1*self.D)
        if self.cont_data:
            self.pooling1 = nn.AvgPool2d((1, self.Samples))
            self.flatten = nn.Flatten(start_dim=1)
            self.last_lay_dim = self.F2
        else:
            self.pooling1 = nn.AvgPool2d((1, self.avg_1))
            self.drop_layer1 = nn.Dropout(p=self.dropoutRate)
            
            # Layer 3
            self.sepconv_depthwise = nn.Conv2d(self.F1*self.D, self.F1*self.D, 
                                               kernel_size=(1, self.kernLength_sep), 
                                               groups=self.F1*self.D, bias=False, padding=0)
            self.sepconv_pointwise = nn.Conv2d(self.F1*self.D, self.F2, 
                                               kernel_size=1, bias=False)
            self.batchnorm3 = nn.BatchNorm2d(self.F1*self.D)
            self.pooling2 = nn.AvgPool2d((1, self.avg_2))
            self.drop_layer2 = nn.Dropout(p=self.dropoutRate)
            self.flatten = nn.Flatten(start_dim=1)

            # FC Layer
            n_samps_avg1 = int((self.Samples-self.avg_1)/self.avg_1+1)
            n_samps_avg2 = int((n_samps_avg1-self.avg_2)/self.avg_2+1)
            self.last_lay_dim = self.F2*n_samps_avg2
        

    def forward(self, x):
        x = x.view(-1, 1, self.Chans, self.Samples)
        
        # Layer 1 (padding to equal Tensorflow 'same' padding)
        x = F.pad(x, (self.kernLength//2,
                      self.kernLength//2 + (self.kernLength%2) - 1, 0, 0))
        x = self.conv1(x)
        
        if self.useHilbert:
            x = apply_hilbert_tf(x, do_log=self.do_log, compute_val=self.compute_val)
        x = self.batchnorm1(x)
        
        # Layer 2
        x = self.depthconv(x)
        x = F.elu(self.batchnorm2(x))
        x = self.pooling1(x)
        
        if not self.cont_data:
            x = self.drop_layer1(x)

            # Layer 3
            x = F.pad(x, (self.kernLength_sep//2,
                          self.kernLength_sep//2 + (self.kernLength_sep%2) - 1, 0, 0))
            x = self.sepconv_depthwise(x) # 1st part of separable conv
            x = self.sepconv_pointwise(x) # 2nd part of separable conv

            x = F.elu(self.batchnorm3(x))
            x = self.pooling2(x)
            x = self.drop_layer2(x)
        x = self.flatten(x)
        return x

def htnet(num_classes=[1000], Chans=94, Samples=501, use_ecog = True, is_supervised = False,
          cont_data = False, param_lp=''):
    '''Build HTNet model and then add it into model with clustering'''
    if use_ecog:
        args_in = pickle.load(open(param_lp+'_neural.pkl', 'rb'))
    else:
        args_in = pickle.load(open(param_lp+'_pose.pkl', 'rb'))

    args_in['Chans'] = Chans
    args_in['Samples'] = Samples
    model = HTNet(ht_submodel(args_in), num_classes, is_supervised)
    return model, args_in
