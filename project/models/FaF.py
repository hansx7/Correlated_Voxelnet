import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if self.gn is not None:
            x = self.gn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


# conv3d + bn + relu
class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.gn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class EasyFusion(nn.Module):
    # features: (batch_size, n_frames, D, H, W) //(2,5,20,400,352)
    def __init__(self, cfg):
        super(EasyFusion, self).__init__()

        # (batch_size, n_frames, D, H, W) // (2,20,400,352)
        self.conv_1 = nn.Conv3d(cfg.D, 1, kernel_size=1, groups=1)

        self.block1 = nn.Sequential(
            Conv2d(cfg.D, 32, 3, 1, 1),
            Conv2d(32, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block2 = nn.Sequential(
            Conv2d(32, 64, 3, 1, 1),
            Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block3 = nn.Sequential(
            Conv2d(64, 128, 3, 1, 1),
            Conv2d(128, 128, 3, 1, 1),
            Conv2d(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block4 = nn.Sequential(
            Conv2d(128, 256, 3, 1, 1),
            Conv2d(256, 256, 3, 1, 1),
            Conv2d(256, 256, 3, 1, 1),
            # nn.ConvTranspose2d(256, 256, 4, 4, 0),
            # nn.BatchNorm2d(256)
        )

        self.cls_head = Conv2d(256, cfg.anchors_per_position * cfg.n_frame, 1, 1, 0,
                               activation=False, batch_norm=False)
        self.reg_head = Conv2d(256, self.c_frames * 7 * cfg.anchors_per_position, 1, 1, 0,
                               activation=False, batch_norm=False)

    def forward(self, x):
        # 3D convolution on temporal dimension with kernel 1x1 to reduce the temporal dimension from n to 1
        out = self.conv_1(x)
        out = out.squeeze(1)    # torch.Size([2, 20, 400, 352])
        out = self.block1(out)  # torch.Size([2, 32, 200, 176])
        out = self.block2(out)  # torch.Size([2, 64, 100, 88])
        out = self.block3(out)  # torch.Size([2, 128, 50, 44])
        out = self.block4(out)  # torch.Size([2, 256, 200, 176])

        cls_head = self.cls_head(out)  # torch.Size([2, 10, 200, 176])
        reg_head = self.reg_head(out)  # torch.Size([2, 70, 200, 176])
        return cls_head, reg_head


class LaterFusion(nn.Module):
    def __init__(self, cfg):
        self.c_frames = cfg.n_frame
        self.in_channels = cfg.D
        # features: (2,5,20,400,352)
        super(LaterFusion, self).__init__()

        self.block1_0 = nn.Sequential(
            Conv2d(self.in_channels, 32, 3, 1, 1),
            Conv2d(32, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block1_1 = nn.Sequential(
            Conv2d(self.in_channels, 32, 3, 1, 1),
            Conv2d(32, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block1_2 = nn.Sequential(
            Conv2d(self.in_channels, 32, 3, 1, 1),
            Conv2d(32, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block1_3 = nn.Sequential(
            Conv2d(self.in_channels, 32, 3, 1, 1),
            Conv2d(32, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block1_4 = nn.Sequential(
            Conv2d(self.in_channels, 32, 3, 1, 1),
            Conv2d(32, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block1 = [self.block1_0, self.block1_1, self.block1_2, self.block1_3, self.block1_4]

        self.conv3d_1 = Conv3d(self.c_frames, 3, 3, 1, 1)

        self.block2_0 = nn.Sequential(
            Conv2d(32, 64, 3, 1, 1),
            Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block2_1 = nn.Sequential(
            Conv2d(32, 64, 3, 1, 1),
            Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block2_2 = nn.Sequential(
            Conv2d(32, 64, 3, 1, 1),
            Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block2 = [self.block2_0, self.block2_1, self.block2_2]

        self.conv3d_2 = Conv3d(3, 1, 3, 1, 1)

        self.block3 = nn.Sequential(
            Conv2d(64, 128, 3, 1, 1),
            Conv2d(128, 128, 3, 1, 1),
            Conv2d(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block4 = nn.Sequential(
            Conv2d(128, 256, 3, 1, 1),
            Conv2d(256, 256, 3, 1, 1),
            Conv2d(256, 256, 3, 1, 1),
            # nn.ConvTranspose2d(256, 256, 4, 4, 0),
            # nn.BatchNorm2d(256)
        )

        self.cls_head = Conv2d(256, cfg.anchors_per_position * self.c_frames, 1, 1, 0,
                               activation=False, batch_norm=False)
        self.reg_head = Conv2d(256, 7 * cfg.anchors_per_position * self.c_frames, 1, 1, 0,
                               activation=False, batch_norm=False)

    def forward(self, x):
        frames = [x[:, i, ...] for i in range(self.c_frames)]  # [torch.Size([2, 20, 400, 352])*5]
        out = [self.block1[i](frames[i]).unsqueeze(1) for i in
               range(self.c_frames)]  # [torch.Size([2, 1, 32, 200, 176])*5]
        out = torch.cat(out, 1)  # torch.Size([2, 5, 32, 200, 176])

        out = self.conv3d_1(out)  # torch.Size([2, 3, 32, 200, 176])

        out = [out[:, i, ...] for i in range(3)]  # [torch.Size([2, 32, 200, 176])*3]
        out = [self.block2[i](out[i]).unsqueeze(1) for i in range(3)]  # [torch.Size([2, 1, 64, 100, 88])*3]
        out = torch.cat(out, 1)  # torch.Size([2, 3, 64, 100, 88])

        out = self.conv3d_2(out).squeeze(1)  # torch.Size([2, 64, 100, 88])

        out = self.block3(out)  # torch.Size([2, 128, 50, 44])
        out = self.block4(out)  # torch.Size([2, 256, 200, 176])

        cls_head = self.cls_head(out)  # torch.Size([2, 10, 200, 176])
        reg_head = self.reg_head(out)  # torch.Size([2, 70, 200, 176])

        return cls_head, reg_head

    def save_weights(self):
        pass

    def load_weight(self):
        pass


if __name__ == '__main__':
    import sys

    sys.path.append('../../')
    from torch.autograd import Variable
    from project.config.dFaF_config import config as cfg

    features = Variable(torch.ones((2, 5, 20, 400, 352)))
    B, c_frames, in_channels, *shape = features.size()
    net = LaterFusion(cfg)
    cls_head, reg_head = net(features)
    print(cls_head.size(), reg_head.size())