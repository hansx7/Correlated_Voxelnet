import os
import shutil

input_root = '/data/hk1/datasets/kitti/tracking/'
output_root = '/data/hk1/guoslu/Datasets/Kitti/tracking/'


def copy_and_rename(fpath_input, fpath_output, vid):
    for file in os.listdir(fpath_input):
        oldname = os.path.join(fpath_input, file)
        [name, end] = file.split('.')
        nname = str(int(vid)).zfill(2) + str(int(name)).zfill(4) + '.' + end
        newname = os.path.join(fpath_output, nname)
        shutil.copyfile(oldname, newname)


def reconstraction_datasets(input_root, output_root, split='training'):
    datasets_root = input_root + split

    # velodyne
    velo_data_root = datasets_root + '/velodyne/'
    video_lists = os.listdir(velo_data_root)
    for vid in video_lists:
        velo_path_input = velo_data_root + vid
        velo_path_output = output_root + split + '/velodyne/'
        os.makedirs(velo_path_output,exist_ok=True)
        copy_and_rename(velo_path_input,velo_path_output, vid)

    # image_2, calib
    img_data_root = datasets_root + '/image_2/'
    calib_data_root = datasets_root + '/calib/'
    calib_output_name = output_root + split + '/calib/'
    os.makedirs(calib_output_name, exist_ok=True)
    video_lists = os.listdir(img_data_root)
    for vid in video_lists:
        img_path_input = img_data_root + vid
        img_path_output = output_root + split + '/image_2/'
        os.makedirs(img_path_output, exist_ok=True)
        copy_and_rename(img_path_input, img_path_output, vid)

        # calib
        calib_input_name = calib_data_root + vid + '.txt'
        for i in range(len(os.listdir(img_path_input))):
            newname = str(int(vid)).zfill(2) + str(i).zfill(4) + '.txt'
            shutil.copyfile(calib_input_name, calib_output_name + newname)

    # label_2
    if split == 'training':
        label_data_root = datasets_root + '/label_2/'
        label_output_root = output_root + split + '/label_2_with_tracking_id/'
        os.makedirs(label_output_root, exist_ok=True)
        label_list = os.listdir(label_data_root)
        for label_name in label_list:
            video_id = label_name.split('.')[0]
            with open(label_data_root + label_name, 'r') as f:
                lines = f.readlines()
                frame_count = int(lines[-1].split()[0])
                labels = [''] * (frame_count + 1)
                for line in lines:
                    frame_id = int(line.split()[0])
                    content = ' '.join(line.split()[1:]) + '\n'
                    labels[frame_id] += content

                for frame in range(frame_count + 1):
                    new_name = video_id[2:] + str(frame).zfill(4) + '.txt'
                    with open(label_output_root + new_name, 'w+') as w:
                        if labels[frame] is '':
                            w.write('DontCare -1 -1 -10.000000 534.660000 164.230000 558.910000 191.400000 '
                                    '-1000.000000 -1000.000000 -1000.000000 -10.000000 -1.000000 -1.000000 -1.00000')
                        else:
                            w.write(labels[frame])

if __name__ == '__main__':
    # reconstraction_datasets(input_root, output_root, split='training')
    # reconstraction_datasets(input_root, output_root, split='testing')

    output_root = '/data/hk1/guoslu/Datasets/Kitti/tracking/'

    import os

    files = os.listdir(output_root+'training/calib/')
    for name in files:
        file = open(output_root+'training/calib/'+name)
        lines = file.readlines()
        rlines = []
        for line in lines:
            rline = line.rstrip()
            rlines.append(rline)

        w = open(output_root+'training/calib/'+name, 'w+')
        for rline in rlines:
            w.write(rline+'\n')
        w.close()



