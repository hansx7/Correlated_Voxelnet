import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple


# conv2d + bn + relu
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


# conv3d + bn + relu
class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, activation=True, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


class SpatioTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]

        # compute the number of intermediary channels
        intermed_channels = int(
            math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / \
                       (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = Conv3d(in_channels, intermed_channels,
                                   k=spatial_kernel_size, s=spatial_stride, p=spatial_padding)

        self.temporal_conv = Conv3d(intermed_channels, out_channels,
                                    k=temporal_kernel_size, s=temporal_stride, p=temporal_padding)

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        return x


class FeatureMixLayer(nn.Module):
    def __init__(self, in_channels, c_frames):
        super(FeatureMixLayer, self).__init__()
        self.in_channels = in_channels
        self.c_frames = c_frames

        self.features_mixed = nn.Sequential(
            Conv3d(self.in_channels, 16, k=1, s=1, p=0),
            Conv3d(16, 1, k=1, s=1, p=0)
        )

    def forward(self, input):
        '''
        :param input: (batch_size, c_frames, 5, D, H, W)
        :return: out: (batch_size, D, c_frames, H, W)
        '''
        # (batch_size, c_frames, 5, D, H, W)  ==> [(batch_size, 5, D, H, W)*c_frames]
        frames = [input[:, i, ...] for i in range(self.c_frames)]
        # [(batch_size, 5, D, H, W)*c_frames] ==> [(batch_size, D, H, W)*c_frames]
        out = [self.features_mixed(frame) for frame in frames]
        # [(batch_size, D, H, W)*c_frames] ==> (batch_size, c_frames, D, H, W)
        out = torch.cat(out, 1)
        # (batch_size, c_frames, D, H, W) ==> (batch_size, D, c_frames, H, W)
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        return out


class MiddleConvLayer(nn.Module):
    def __init__(self, in_channels):
        super(MiddleConvLayer, self).__init__()
        self.in_channels = in_channels

        self.conv2d_1 = nn.Sequential(
            Conv3d(self.in_channels, 32, k=(1, 3, 3), s=1, p=(0, 1, 1)),
            Conv3d(32, 64, k=(1, 3, 3), s=1, p=(0, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

        self.conv3d_1 = SpatioTemporalConv(64, 64, kernel_size=3, stride=1, padding=(0, 1, 1))

        self.conv2d_2 = nn.Sequential(
            Conv3d(64, 64, k=(1, 3, 3), s=1, p=(0, 1, 1)),
            Conv3d(64, 128, k=(1, 3, 3), s=1, p=(0, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

        self.conv3d_2 = SpatioTemporalConv(128, 128, kernel_size=3, stride=1, padding=(0, 1, 1))

        self.conv2d_3 = Conv2d(128, 128, 3, 1, 1)

    def forward(self, input):
        '''
        :param input: (batch_size, D, c_frames, H, W)
        :return: (batch_size, 128, H/4, W/4)
        '''
        out = self.conv2d_1(input)
        out = self.conv3d_1(out)
        out = self.conv2d_2(out)
        out = self.conv3d_2(out)
        out = out.squeeze()
        out = self.conv2d_3(out)
        return out


# Region Proposal Network
class RPN(nn.Module):
    def __init__(self, cfg):
        super(RPN, self).__init__()

        self.C_s = cfg.n_frame * cfg.anchors_per_position
        self.C_r = cfg.n_frame * cfg.anchors_per_position * 7

        self.block_1 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_1 += [Conv2d(128, 128, 3, 1, 1) for _ in range(3)]
        self.block_1 = nn.Sequential(*self.block_1)

        self.block_2 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_2 += [Conv2d(128, 128, 3, 1, 1) for _ in range(5)]
        self.block_2 = nn.Sequential(*self.block_2)

        self.block_3 = [Conv2d(128, 256, 3, 2, 1)]
        self.block_3 += [Conv2d(256, 256, 3, 1, 1) for _ in range(5)]
        self.block_3 = nn.Sequential(*self.block_3)

        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 4, 0),
                                      nn.BatchNorm2d(256))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(128, 256, 2, 2, 0),
                                      nn.BatchNorm2d(256))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(128, 256, 1, 1, 0),
                                      nn.BatchNorm2d(256))

        self.score_head = Conv2d(768, self.C_s, 1, 1, 0, activation=False, batch_norm=False)
        self.reg_head = Conv2d(768, self.C_r, 1, 1, 0, activation=False, batch_norm=False)

    def forward(self, input):
        out = self.block_1(input)
        x_skip_1 = out
        out = self.block_2(out)
        x_skip_2 = out
        out = self.block_3(out)
        x_0 = self.deconv_1(out)
        x_1 = self.deconv_2(x_skip_2)
        x_2 = self.deconv_3(x_skip_1)
        out = torch.cat((x_0, x_1, x_2), 1)
        return self.score_head(out), self.reg_head(out)


class PointS3D(nn.Module):
    def __init__(self, cfg):
        super(PointS3D, self).__init__()
        self.in_channels = cfg.feature_channels
        self.c_frames = cfg.n_frame
        self.middle_channels = cfg.D
        self.features_mixed = FeatureMixLayer(self.in_channels, self.c_frames)
        self.middle_conv = MiddleConvLayer(self.middle_channels)
        self.rpn = RPN(cfg)

    def forward(self, input):
        out = self.features_mixed(input)
        out = self.middle_conv(out)
        cls_head, reg_head = self.rpn(out)
        return cls_head, reg_head


if __name__ == '__main__':
    import sys
    sys.path.append('../../')
    from torch.autograd import Variable
    from project.config.pointS3D_config import config as cfg

    features = Variable(torch.rand((2, 5, 5, 20, 416, 352)))
    B, c_frames, feature_channels, *shape = features.size()
    net = PointS3D(cfg)
    cls_head, reg_head = net(features)
    print(cls_head.size(), reg_head.size())