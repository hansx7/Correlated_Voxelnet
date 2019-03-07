import torch
import sys
import cv2
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

sys.path.append('..')
from project.config.tFaF_config import config as cfg
from project.models.FaF import EasyFusion, LaterFusion
from project.utils.eval_utils import generate_tracking_pre_label
from project.utils.test_utils import generate_summary_img
from project.kittiDataLoader.tFaF_data_loader import KittiTrackingDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = EasyFusion(cfg)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(model_path))
    return model


def draw_pred_result():
    data_root = cfg.KITTI_TRACKING_DATASET_ROOT
    datasets = KittiTrackingDataset(data_root, set_root='/kitti/', seq_len=5, stride=5,
                                    train_type='Car', set='val_test')
    print(len(datasets))
    dataloader = DataLoader(datasets, batch_size=1, num_workers=4)
    model = load_model('./checkpoints/dFaF/2018-12-17_20/19s.pkl')
    n_anchors = cfg.anchors_per_position
    for i_batch, (data_list, voxel_features_seq, images) in enumerate(dataloader, 0):
        voxel_features = Variable(voxel_features_seq).float().cuda()

        with torch.no_grad():
            cls_head, reg_head = model(voxel_features)

        cls_head = torch.sigmoid(cls_head.permute(0, 2, 3, 1))[0]            # (50, 44, 10)
        reg_head = reg_head.permute(0, 2, 3, 1).contiguous()[0]              # (50, 44, 70)
        reg_head = reg_head.view(reg_head.size(0), reg_head.size(1), -1, 7)  # (50, 44, 10, 7)

        cls_head = cls_head.cpu().numpy()
        reg_head = reg_head.cpu().numpy()

        data_list[0] = data_list[0][0]
        data_list[1:] = [int(i) for i in data_list[1:]]
        names, pre_labels, confs = generate_tracking_pre_label(cls_head, reg_head, data_list, data_root, cfg)
        for i in range(len(names)):
            i_cls_head = cls_head[:, :, i*n_anchors:(i+1)*n_anchors]
            image = images[i][0].numpy()
            print(names[i])
            front_image, bird_view, heatmap = generate_summary_img(names[i], image, i_cls_head, pre_labels[i],
                                             confs[i], data_root, cfg, datasets='tracking')

            bird_view = bird_view.transpose(1, 2, 0)
            print(bird_view.shape)
            bird_view = cv2.cvtColor(bird_view.astype(np.uint8), cv2.COLOR_RGB2BGR)
            bird_view = cv2.resize(bird_view, (800,670))
            cv2.imshow('bird_view', bird_view)
            cv2.waitKey()

if __name__=='__main__':
    draw_pred_result()

