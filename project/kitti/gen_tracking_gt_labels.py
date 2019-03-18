import os

gt_label_path = '/data/hk1/datasets/kitti/tracking/training/label_2/'
output_path = '../prediction/tFaF/label/'

os.makedirs(output_path, exist_ok=True)

labels_name = os.listdir(gt_label_path)

for name in labels_name:
    video_id = name.split('.')[0]
    with open(gt_label_path+name, 'r') as f:
        lines = f.readlines()
        frame_count = int(lines[-1].split()[0])
        labels = ['']*(frame_count+1)
        for line in lines:
            frame_id = int(line.split()[0])
            content = ' '.join(line.split()[2:]) + '\n'
            labels[frame_id] += content

        for frame in range(frame_count+1):
            new_name = video_id[2:] + str(frame).zfill(4) + '.txt'
            with open(output_path+new_name, 'w+') as w:
                w.write(labels[frame])

