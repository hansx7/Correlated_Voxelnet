import torch
import torch.nn as nn

class VoxelLoss(nn.Module):
    def __init__(self, cfg):
        super(VoxelLoss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(reduction='sum')
        self.alpha = cfg.alpha
        self.beta = cfg.beta

    def forward(self, rm, psm, pos_equal_one, neg_equal_one, targets, corr, targets_diff):
        '''
        rm:                 torch.Size([batch_size, 14, h, w])
        psm:                torch.Size([batch_size, 2, h, w])
        pos_equal_one:      torch.Size([batch_size, h, w, 2])
        neg_equal_one:      torch.Size([batch_size, h, w, 2])
        targets:            torch.Size([batch_size, h, w, 14])
        '''
        # print('rm: ', rm.size(), 'psm ', psm.size())
        p_pos = torch.sigmoid(psm.permute(0, 2, 3, 1))
        rm = rm.permute(0, 2, 3, 1).contiguous()
        rm = rm.view(rm.size(0), rm.size(1), rm.size(2), -1, 7)
        targets = targets.view(targets.size(0), targets.size(1), targets.size(2), -1, 7)
        pos_equal_one_for_reg = pos_equal_one.unsqueeze(pos_equal_one.dim()).expand(-1, -1, -1, -1, 7)
        
        rm_pos = rm * pos_equal_one_for_reg
        targets_pos = targets * pos_equal_one_for_reg

        cls_pos_loss = -pos_equal_one * torch.log(p_pos + 1e-6)
        cls_pos_loss = cls_pos_loss.sum() / (pos_equal_one.sum() + 1e-6)

        cls_neg_loss = -neg_equal_one * torch.log(1 - p_pos + 1e-6)
        cls_neg_loss = cls_neg_loss.sum() / (neg_equal_one.sum() + 1e-6)

        reg_loss = self.smoothl1loss(rm_pos, targets_pos)
        reg_loss = reg_loss / (pos_equal_one.sum() + 1e-6)

        conf_loss = self.alpha * cls_pos_loss + self.beta * cls_neg_loss

        corr = corr.permute(0, 2, 3, 1).contiguous()
        targets_xyzr = targets[:, :, :, [0, 1, 2, 6, 7, 8, 9, 13]]
        corr_loss = self.smoothl1loss(corr, targets_xyzr)

        return conf_loss, reg_loss, cls_pos_loss, cls_neg_loss, corr_loss


class LRMloss(nn.Module):
    def __init__(self, cfg):
        super(LRMloss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(reduction='sum')
        self.neg_ratio = cfg.neg_ratio
        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.gamma = cfg.gamma

    def forward(self, rm, psm, pos_equal_one, neg_equal_one, targets):
        '''
        input:
            rm:                 torch.Size([2, 70, 50, 44]), reg_head
            psm:                torch.Size([2, 10, 50, 44]), cls_head
            pos_equal_one:      torch.Size([2, 50, 44, 10])
            neg_equal_one:      torch.Size([2, 50, 44, 10])
            targets:            torch.Size([2, 50, 44, 70])

        '''
        p_pos = torch.sigmoid(psm.permute(0, 2, 3, 1))                                      # (2, 50, 44, 10)
        targets = targets.view(targets.size(0), targets.size(1), targets.size(2), -1, 7)    # (2, 50, 44, 10, 7)
        rm = rm.permute(0, 2, 3, 1).contiguous()                                            # (2, 50, 44, 70)
        rm = rm.view(rm.size(0), rm.size(1), rm.size(2), -1, 7)
        pos_equal_one_for_reg = pos_equal_one.unsqueeze(pos_equal_one.dim()).\
                                expand(-1, -1, -1, -1, 7)                                   # (2, 50, 44, 10, 7)

        # cal reg_loss
        rm_pos = rm * pos_equal_one_for_reg
        targets_pos = targets * pos_equal_one_for_reg                                       # (2, 50, 44, 10, 7)
        reg_loss = self.smoothl1loss(rm_pos, targets_pos)
        reg_loss = reg_loss / (pos_equal_one.sum() + 1e-6)

        # cal cls_pos_loss
        cls_pos_loss = -pos_equal_one * torch.log(p_pos + 1e-6)
        cls_pos_loss = cls_pos_loss.sum() / (pos_equal_one.sum() + 1e-6)

        # cal cls_neg_loss
        # take k neg class
        k = self.neg_ratio * (pos_equal_one.sum() + 1)
        # filter top-k loss value position
        neg_log_loss = torch.log(1 - p_pos + 1e-6)
        neg_loss = -neg_equal_one * neg_log_loss                    # (2, 50, 44, 10)
        r_neg_loss = neg_loss.view(-1)                              # (2*50*44*10, )
        topk_cls_neg_loss, topk_idx = torch.topk(r_neg_loss, k.int())
        # reconstruction neg_equal_one
        r_neg_equal_one = neg_equal_one.view(-1).contiguous() * 0.0
        r_neg_equal_one[topk_idx] = 1.0
        r_neg_equal_one = r_neg_equal_one.view(neg_equal_one.size())
        # cal finally neg loss
        cls_neg_loss = -r_neg_equal_one * neg_log_loss
        cls_neg_loss = cls_neg_loss.sum() / (r_neg_equal_one.sum() + 1e-6)

        cls_pos_loss = self.alpha * cls_pos_loss
        cls_neg_loss = self.beta * cls_neg_loss
        reg_loss = self.gamma * reg_loss

        conf_loss = cls_pos_loss + cls_neg_loss

        return conf_loss, reg_loss, cls_pos_loss, cls_neg_loss


class LRMloss_v2(nn.Module):
    def __init__(self, cfg):
        super(LRMloss_v2, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(reduction='sum')
        self.cfg = cfg
        self.neg_ratio = self.cfg.neg_ratio
        self.alpha = self.cfg.alpha
        self.beta = self.cfg.beta
        self.gamma = self.cfg.gamma

    def forward(self, rm, psm, pos_equal_one, neg_equal_one, targets):
        '''
        input:
            rm:                 torch.Size([2, 70, 50, 44]), reg_head
            psm:                torch.Size([2, 10, 50, 44]), cls_head
            pos_equal_one:      torch.Size([2, 50, 44, 10])
            neg_equal_one:      torch.Size([2, 50, 44, 10])
            targets:            torch.Size([2, 50, 44, 70])

        '''
        p_pos = torch.sigmoid(psm.permute(0, 2, 3, 1))                                      # (2, 50, 44, 10)
        targets = targets.view(targets.size(0), targets.size(1), targets.size(2), -1, 7)    # (2, 50, 44, 10, 7)
        rm = rm.permute(0, 2, 3, 1).contiguous()                                            # (2, 50, 44, 70)
        rm = rm.view(rm.size(0), rm.size(1), rm.size(2), -1, 7)
        pos_equal_one_for_reg = pos_equal_one.unsqueeze(pos_equal_one.dim()).\
                                expand(-1, -1, -1, -1, 7)                                   # (2, 50, 44, 10, 7)

        # cal reg_loss
        rm_pos = rm * pos_equal_one_for_reg
        targets_pos = targets * pos_equal_one_for_reg                                       # (2, 50, 44, 10, 7)
        reg_loss = self.smoothl1loss(rm_pos, targets_pos)
        reg_loss = reg_loss / (pos_equal_one.sum() + 1e-6)

        # cal cls_pos_loss
        cls_pos_loss = -pos_equal_one * torch.log(p_pos + 1e-6)
        cls_pos_loss = cls_pos_loss.sum() / (pos_equal_one.sum() + 1e-6)

        # take k neg class at each frame
        batch_size = pos_equal_one.size()[0]
        for batch in range(batch_size):
            b_pos_equal_one = pos_equal_one[batch]          # (50, 44, 10)
            b_neg_equal_one = neg_equal_one[batch]          # (50, 44, 10)
            b_p_pos = p_pos[batch]                          # (50, 44, 10)
            n_anchors = self.cfg.anchors_per_position

            for i in range(self.cfg.n_frame):
                frame_pos_equal_one = b_pos_equal_one[:, :, i*n_anchors:(i+1)*n_anchors]    # (50, 44, 2)
                frame_neg_equal_one = b_neg_equal_one[:, :, i*n_anchors:(i+1)*n_anchors]    # (50, 44, 2)
                frame_p_pos = b_p_pos[:, :, i*n_anchors:(i+1)*n_anchors]                    # (50, 44, 2)

                k = self.neg_ratio * (frame_pos_equal_one.sum() + 1)

                # filter top-k loss value position
                frame_neg_log_loss = torch.log(1 - frame_p_pos + 1e-6)              # (50, 44, 2)
                frame_neg_loss = -frame_neg_equal_one * frame_neg_log_loss          # (50, 44, 2)
                r_neg_loss = frame_neg_loss.contiguous().view(-1)                   # (50*44*2, )
                topk_cls_neg_loss, topk_idx = torch.topk(r_neg_loss, k.int())

                # reconstruction neg_equal_one
                r_neg_equal_one = frame_neg_equal_one.contiguous().view(-1) * 0.0   # (50*44*2, )
                r_neg_equal_one[topk_idx] = 1.0
                r_neg_equal_one = r_neg_equal_one.view(frame_neg_equal_one.size())  # (50, 44, 2)

                b_neg_equal_one[:, :, i*n_anchors:(i+1)*n_anchors] = r_neg_equal_one

            # update neg_equal one
            neg_equal_one[batch] = b_neg_equal_one

        neg_log_loss = torch.log(1 - p_pos + 1e-6)
        cls_neg_loss = - neg_equal_one * neg_log_loss
        cls_neg_loss = cls_neg_loss.sum()/(neg_equal_one.sum() + 1e-6)

        cls_pos_loss = self.alpha * cls_pos_loss
        cls_neg_loss = self.beta * cls_neg_loss
        reg_loss = self.gamma * reg_loss

        conf_loss = cls_pos_loss + cls_neg_loss

        return conf_loss, reg_loss, cls_pos_loss, cls_neg_loss
