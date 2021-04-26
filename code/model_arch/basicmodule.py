from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_arch.ops.dcn.deform_conv import ModulatedDeformConv
from model_arch.ops.dcn.deform_conv import ModulatedDeformConvPack as DCN


def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


# 3x3 convolution with padding
def conv3x3(in_channels, out_channels, stride=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
		padding=1, bias=False)


class CBAM_channel(nn.Module):
	def __init__(self, out_channels, reduction=16, dilation=4):
		super().__init__()
		# 2D adaptive average pooling
		self.average_pool = nn.AdaptiveAvgPool2d(1)
		# 2D adaptive max pooling
		self.max_pool = nn.AdaptiveMaxPool2d(1)
		self.fc1  = nn.Linear(out_channels, out_channels//reduction, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.fc2  = nn.Linear(out_channels//reduction, out_channels, bias=False)
		self.sigmoid   = nn.Sigmoid()

	def forward(self, x):
		out = x

		out_c1 = self.average_pool(out)
		out_c1 = out_c1.view(out_c1.size(0), -1) # NxC
		out_c1 = self.fc1(out_c1)
		out_c1 = self.relu(out_c1)
		out_c1 = self.fc2(out_c1)

		out_c2 = self.max_pool(out)
		out_c2 = out_c2.view(out_c2.size(0), -1) # NxC
		out_c2 = self.fc1(out_c2)
		out_c2 = self.relu(out_c2)
		out_c2 = self.fc2(out_c2)

		out = out_c1 + out_c2
		out = self.sigmoid(out)
		out = out.view(out.size(0), out.size(1), 1, 1) # NxCx1x1
		out = out.expand_as(x) # NxCxHxW

		out = x * out
		return out


class CBAM_spatial(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
		# self.bn1   = nn.BatchNorm2d(1)
		self.relu  = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		out = x
		# max pooling along channel axis
		out_s1 = torch.max(out, 1)[0].unsqueeze(1) #Nx1xHxW
		# mean pooling along channel axis
		out_s2 = torch.mean(out, 1).unsqueeze(1) #Nx1xHxW
		# concatenate out_s1 and out_s2 along channel axis
		out = torch.cat((out_s1, out_s2), 1) #Nx2xHxW
		out = self.conv1(out)
		# out = self.bn1(out)
		out = self.relu(out)
		out = self.sigmoid(out) # Nx1xHxW
		out = out.expand_as(x) # NxCxHxW
		out = x * out
		return out


class ResidualBlock_noBNSA(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBNSA, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.cbam_spatial = CBAM_spatial()

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        out = self.cbam_spatial(out)
        return identity + out


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class QEnetwork(nn.Module):
    def __init__(self, in_nc, nf, out_nc, nb):
        super(QEnetwork, self).__init__()
        self.InputConv = InputConv(in_nc, nf, 2)
        self.ECA = nn.ModuleList([ECA(nf) for _ in range(21)])
        self.DownConv = nn.ModuleList([DownConv(nf, nf, nb) for _ in range(3)])
        self.UpConv_64 = nn.ModuleList([UpConv(nf*2, nf, nb) for _ in range(3)])
        self.UpConv_96 = nn.ModuleList([UpConv(nf*3, nf, nb) for _ in range(2)])
        self.UpConv_128 = UpConv(nf*4, nf, nb)
        self.OutConv = OutConv(nf, out_nc)

    def forward(self, x):
        x00 = self.InputConv(x)

        x10 = self.DownConv[0](self.ECA[0](x00))
        x01 = self.UpConv_64[0](self.ECA[1](x00),
                                self.ECA[2](x10))

        x20 = self.DownConv[1](self.ECA[4](x10))
        x11 = self.UpConv_64[1](self.ECA[5](x10),
                                self.ECA[6](x20))
        x02 = self.UpConv_96[0](torch.cat((self.ECA[7](x00),
                                           self.ECA[8](x01)), dim=1),
                                self.ECA[9](x11))

        x30 = self.DownConv[2](self.ECA[11](x20))
        x21 = self.UpConv_64[2](self.ECA[12](x20),
                                self.ECA[13](x30))
        x12 = self.UpConv_96[1](torch.cat((self.ECA[14](x10),
                                           self.ECA[15](x11)), dim=1),
                                self.ECA[16](x21))
        x03 = self.UpConv_128(torch.cat((self.ECA[17](x00),
                                         self.ECA[18](x01),
                                         self.ECA[19](x02)), dim=1),
                              self.ECA[20](x12))

        out = self.OutConv(x03)
        return out


class InputConv(nn.Module):
    def __init__(self, in_nc, outch, nb=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_nc, outch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        body = []
        for i in range(1, nb):
            body.append(ResidualBlock_noBN(outch))
        self.InputConv = nn.Sequential(*body)

    def forward(self, x):
        x = self.head(x)
        out = self.InputConv(x)
        out = out + x
        return out


class DownConv(nn.Module):
    def __init__(self, inch, outch, nb=5):
        super().__init__()
        body = []
        for i in range(1, nb):
            body.append(ResidualBlock_noBNSA(inch))
        self.body = nn.Sequential(*body)
        self.DownConv = nn.Sequential(
            nn.Conv2d(inch, outch, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.body(x)
        y = y + x
        return self.DownConv(y)


class UpConv(nn.Module):
    def __init__(self, inch, outch, nb=5):
        super().__init__()
        self.UpConv = nn.Sequential(
            # nn.ConvTranspose2d(outch, outch, kernel_size=2, stride=2),
            nn.Conv2d(outch, outch*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Conv2d(inch, outch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        body = []
        for i in range(1, nb):
            body.append(ResidualBlock_noBNSA(outch))
        self.conv = nn.Sequential(*body)

    def forward(self, x, x_l):
        x_l = self.UpConv(x_l)
        y = self.head(torch.cat((x, x_l), dim=1))
        out = self.conv(y)
        return out + y


class ECA(nn.Module):
    def __init__(self, inch):
        super().__init__()
        self.ave_pool = nn.AdaptiveAvgPool2d(1)  # B inch 1 1
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.size()
        y = self.ave_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class OutConv(nn.Module):
    def __init__(self, inch, out_nc):
        super(OutConv, self).__init__()
        self.OutConv = nn.Conv2d(inch, out_nc, 3, 1, 1)

    def forward(self, x):
        return self.OutConv(x)


# ==========
# Spatio-temporal deformable fusion module
# ==========

class STDF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF, self).__init__()

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True)
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    # nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.Conv2d(nf, 4 * nf, 3, padding=1),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True)
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )

        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        self.offset_mask = nn.Conv2d(
            nf, in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2
        )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks // 2, deformable_groups=in_nc
        )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # print(out.shape)
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            # print(out_lst[i].shape)
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
            )

        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        off_msk = self.offset_mask(self.out_conv(out))
        off = off_msk[:, :in_nc * 2 * n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc * 2 * n_off_msk:, ...]
        )

        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk),
            inplace=True
        )

        return fused_feat


class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8):
        super(PCD_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                               extra_offset_mask=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack([nbr_fea_l[2], L3_offset]))
        # L2
        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack([nbr_fea_l[1], L2_offset])
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack([nbr_fea_l[0], L1_offset])
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack([L1_fea, offset]))

        return L1_fea


class TSA_Fusion(nn.Module):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 3 pyramid levels.
    '''

    def __init__(self, nf=64, nframes=5, center=2):
        super(TSA_Fusion, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        # spatial attention (after fusion conv)
        self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea = aligned_fea.view(B, -1, H, W) * cor_prob

        #### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))

        #### spatial attention
        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea