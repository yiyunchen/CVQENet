from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision
from torch.nn import init
import numpy as np
import functools
import math
# from model_arch.common import PCD_Align, ResidualBlock_noBN, make_layer, QEnetwork, TSA_Fusion
from model_arch.basicmodule import PCD_Align, ResidualBlock_noBNSA, make_layer, QEnetwork, TSA_Fusion



# ==========
# VQE network
# ==========

class VQE(nn.Module):
    def __init__(self, nf, b1, b2):
        super(VQE, self).__init__()
        self.radius = 2
        self.input_len = 2 * self.radius + 1
        front_RBs = b1
        groups = 8
        back_RBs = b2
        nb = b2

        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBNSA, nf=nf)

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.pcd_align = PCD_Align(nf=nf, groups=groups)

        self.tsa_fusion = TSA_Fusion(nf=nf, nframes=5, center=2)  #nn.Conv2d(self.input_len * nf, nf, 1, 1, bias=True)  #
        self.qe = QEnetwork(nf, nf, nf, nb)
        #### reconstruction
        self.recon_trunk = make_layer(ResidualBlock_noBN_f, back_RBs)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, n, h, w = x.size()

        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        target = x[:, :, 2, :, :]
        x4 = x[:, :, 3, :, :]
        x5 = x[:, :, 4, :, :]

        x1_L1 = self.feature_extraction(self.lrelu(self.conv_first(x1)))
        x2_L1 = self.feature_extraction(self.lrelu(self.conv_first(x2)))
        x3_L1 = self.feature_extraction(self.lrelu(self.conv_first(target)))
        x4_L1 = self.feature_extraction(self.lrelu(self.conv_first(x4)))
        x5_L1 = self.feature_extraction(self.lrelu(self.conv_first(x5)))

        # L2
        x1_L2 = self.lrelu(self.fea_L2_conv1(x1_L1))
        x1_L2 = self.lrelu(self.fea_L2_conv2(x1_L2))
        x2_L2 = self.lrelu(self.fea_L2_conv1(x2_L1))
        x2_L2 = self.lrelu(self.fea_L2_conv2(x2_L2))
        x3_L2 = self.lrelu(self.fea_L2_conv1(x3_L1))
        x3_L2 = self.lrelu(self.fea_L2_conv2(x3_L2))
        x4_L2 = self.lrelu(self.fea_L2_conv1(x4_L1))
        x4_L2 = self.lrelu(self.fea_L2_conv2(x4_L2))
        x5_L2 = self.lrelu(self.fea_L2_conv1(x5_L1))
        x5_L2 = self.lrelu(self.fea_L2_conv2(x5_L2))

        # L3
        x1_L3 = self.lrelu(self.fea_L2_conv1(x1_L2))
        x1_L3 = self.lrelu(self.fea_L2_conv2(x1_L3))
        x2_L3 = self.lrelu(self.fea_L2_conv1(x2_L2))
        x2_L3 = self.lrelu(self.fea_L2_conv2(x2_L3))
        x3_L3 = self.lrelu(self.fea_L2_conv1(x3_L2))
        x3_L3 = self.lrelu(self.fea_L2_conv2(x3_L3))
        x4_L3 = self.lrelu(self.fea_L2_conv1(x4_L2))
        x4_L3 = self.lrelu(self.fea_L2_conv2(x4_L3))
        x5_L3 = self.lrelu(self.fea_L2_conv1(x5_L2))
        x5_L3 = self.lrelu(self.fea_L2_conv2(x5_L3))

        # pcd align
        ref_fea_l = [x3_L1.clone(), x3_L2.clone(), x3_L3.clone()]

        aligned_fea = []
        nbr_fea_l1 = [x1_L1.clone(), x1_L2.clone(), x1_L3.clone()]
        aligned_fea.append(self.pcd_align(nbr_fea_l1, ref_fea_l))
        nbr_fea_l2 = [x2_L1.clone(), x2_L2.clone(), x2_L3.clone()]
        aligned_fea.append(self.pcd_align(nbr_fea_l2, ref_fea_l))
        nbr_fea_l3 = [x3_L1.clone(), x3_L2.clone(), x3_L3.clone()]
        aligned_fea.append(self.pcd_align(nbr_fea_l3, ref_fea_l))
        nbr_fea_l4 = [x4_L1.clone(), x4_L2.clone(), x4_L3.clone()]
        aligned_fea.append(self.pcd_align(nbr_fea_l4, ref_fea_l))
        nbr_fea_l5 = [x5_L1.clone(), x5_L2.clone(), x5_L3.clone()]
        aligned_fea.append(self.pcd_align(nbr_fea_l5, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)
        # aligned_fea = aligned_fea.view(b, -1, h, w)
        fea = self.tsa_fusion(aligned_fea)

        out = self.qe(fea)
        out = self.recon_trunk(out)
        out = self.conv_last(out)
        out += target

        return out











