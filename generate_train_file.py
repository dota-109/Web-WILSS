import os
import glob
import numpy as np


path = r"./replay_data/10-10-ov"
dirs = sorted(os.listdir(path))
split_rate = 1
for dir in dirs:
    print(dir)
    file_path = os.path.join(path, dir)
    img_path = os.path.join(file_path, 'image')
    label_path = os.path.join(file_path, 'label')
    # label_path = os.path.join(file_path, 'label')
    img_files = os.listdir(img_path)

    with open(os.path.join(file_path, 'train_fullPath.txt'), 'w+') as f:
        for idx in range(int(len(img_files)*split_rate)):
            img_file = os.path.join('image',img_files[idx])
            img_file_fullPath = os.path.join( img_path, img_files[idx] )
            label_file_fullPath = os.path.join( label_path, img_files[idx][:-3]+"png" )
            label_file = os.path.join('label',img_files[idx][:-3]+"png")
            f.write( img_file_fullPath.replace("\\", "/") + " " + label_file_fullPath.replace("\\", "/") + "\n")
    print("finish writing train file.")

    if split_rate == 1:
        continue
    else:
        with open(os.path.join(file_path, 'val.txt'), 'w+') as f:
            for idx in range(int(len(img_files)*split_rate),len(img_files)):
                img_file = os.path.join('image',img_files[idx])
                label_file = os.path.join('label',img_files[idx])
                f.write( img_file.replace("\\", "/") + " " + label_file.replace("\\", "/") + "\n")
        print("finisn writing validation file.")

