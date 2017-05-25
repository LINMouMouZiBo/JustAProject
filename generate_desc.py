import sys
import os  
sys.path = sys.path + ['/usr/local/anaconda/lib/python27.zip', '/usr/local/anaconda/lib/python2.7', '/usr/local/anaconda/lib/python2.7/plat-linux2', '/usr/local/anaconda/lib/python2.7/lib-tk', '/usr/local/anaconda/lib/python2.7/lib-old', '/usr/local/anaconda/lib/python2.7/lib-dynload', '/usr/local/anaconda/lib/python2.7/site-packages', '/usr/local/anaconda/lib/python2.7/site-packages/Sphinx-1.5.1-py2.7.egg', '/usr/local/anaconda/lib/python2.7/site-packages/setuptools-27.2.0-py2.7.egg']

import scipy.misc
import numpy as np

import cv2

label_num = 249

def reshape_train_set(sample_size = 9, stride = 4, desc_file_name = 'sample_data_desc.txt'):
    print 'reading training set...'
    p = '/share/data/CodaLab/ConGD/ConGD_phase_1/'
    filename = 'train.txt'

    desc = open(desc_file_name, "w")

    with open(p+filename) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\n')[0].split(' ')
            video_name = line[0]
            video_rgb_name = line[0] + '.M.avi'
            video_deep_name = line[0] + '.K.avi'
            
            if video_rgb_name == '049/02444.M.avi': # bad data
                continue

            gray_frames = []
            deep_frames = []

            cap_len = int(line[-1].split(':')[0].split(',')[1])
            action_arr = []

            for j in range(1, len(line)):
                start, rest = line[j].split(',')
                end, label = rest.split(':')
                start, end, label = int(start), int(end), int(label)
                action_arr.append((start, end, label))
            
            for action in action_arr:
                start, end, label = action
                action_len = end - start + 1
                if action_len < sample_size:
                    desc.write(video_name + ':' + str(start) + '-' + str(end) + ' ' + str(label) + '\n')
                else:
                    for j in range(1, action_len, stride):
                        sample_start = j
                        sample_end = j + sample_size -1
                        should_stop = False
                        if sample_end > action_len:
                            sample_start = action_len - sample_size + 1
                            sample_end = action_len
                            should_stop = True
                        desc.write(video_name + ':' + str(sample_start) + '-' + str(sample_end) + ' ' + str(label) + '\n')
                        if should_stop:
                            break

    desc.close()

if __name__ == '__main__':
    reshape_train_set(sample_size = 32, stride = 10 ,desc_file_name = 'sample_32_10_desc.txt')
    reshape_train_set(sample_size = 64, stride = 20 ,desc_file_name = 'sample_64_20_desc.txt')
