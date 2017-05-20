import sys
import os  
sys.path = sys.path + ['/usr/local/anaconda/lib/python27.zip', '/usr/local/anaconda/lib/python2.7', '/usr/local/anaconda/lib/python2.7/plat-linux2', '/usr/local/anaconda/lib/python2.7/lib-tk', '/usr/local/anaconda/lib/python2.7/lib-old', '/usr/local/anaconda/lib/python2.7/lib-dynload', '/usr/local/anaconda/lib/python2.7/site-packages', '/usr/local/anaconda/lib/python2.7/site-packages/Sphinx-1.5.1-py2.7.egg', '/usr/local/anaconda/lib/python2.7/site-packages/setuptools-27.2.0-py2.7.egg']

import scipy.misc
import numpy as np

import cv2

label_num = 249

def make_label_arr(n):
    result = [0] * 249
    result[n-1] = 1
    return result

# [sample_start, sample_end)
def cal_sample(sample_start, sample_end, action_arr, gray_frames, deep_frames):
    sample = []
    for i in range(len(action_arr)):
        action_start, action_end, action_label = action_arr[i]
        if action_start <= sample_start and action_end >= sample_end:
            return sample, action_label
        elif action_start > sample_start and action_start < sample_end:
            return None, -1
        elif action_end > sample_start and action_end < sample_end:
            return None, -1
    return sample, 0

def save_sample(video_name, sample_start, sample_end, label, desc):
    img_path = video_name + ':' + str(sample_start) + '-' + str(sample_end) + '.txt ' + str(label)
    desc.write(img_path + '\n')

def reshape_train_set(sample_size = 9, desc_file_name = 'sample_data_desc.txt'):
    print 'reading training set...'
    p = '/share/data/CodaLab/ConGD/ConGD_phase_1/'
    filename = 'train.txt'

    desc = open(desc_file_name, "w")

    with open(p+filename) as f:
        lines = f.readlines()
        cnt = 0
        for line in lines:
            if cnt % 100 == 0:
                print ("%.2f%%"%(float(cnt)/len(lines)*100))
            cnt += 1
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
                action_arr.append((start-1, end, label))
            


            for j in range(0, cap_len, 4):
                sample_start = j
                sample_end = j + sample_size
                should_stop = False
                if sample_end > cap_len:
                    sample_start = cap_len - sample_size
                    sample_end = cap_len
                    should_stop = True
                    # [sample_start, sample_end)
                sample, label = cal_sample(sample_start, sample_end, action_arr, gray_frames, deep_frames)
                if sample != None:
                    # 2*240*320
                    save_sample(video_name, sample_start+1, sample_end, label, desc)
                if should_stop:
                    break
    desc.close()

def read_valid_data():
    print 'reading validation set...'
    p = '/share/data/CodaLab/ConGD/ConGD_phase_1/'
    filename = 'valid.txt'

    valid_gray_value = []
    valid_deep_value = []

    with open(p+filename) as f:
        lines = f.readlines()
        cnt = 0
        for line in lines:
            if cnt % 100 == 0:
                print ("%.2f%%"%(float(cnt)/len(lines)*100))
            cnt += 1
            line = line.split('\n')[0]
            video_rgb_name = line + '.M.avi'
            video_deep_name = line + '.K.avi'
            rgb_cap = cv2.VideoCapture(p + 'valid/' + video_rgb_name)
            deep_cap = cv2.VideoCapture(p + 'valid/' + video_deep_name)
            gray_frames = []
            deep_frames = []

            rgb_cap_len = int(rgb_cap.get(7))
            # deep_cap_len = int(rgb_cap.get(7))

            for j in range(rgb_cap_len):
                rgb_ret, rgb_frame = rgb_cap.read()
                deep_ret, deep_frame = deep_cap.read()
                gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
                gray_frames.append(gray_frame)
                deep_frames.append(deep_frame)

            valid_gray_value.append(gray_frames)
            valid_deep_value.append(deep_frames)
            
            rgb_cap.release()
            deep_cap.release()

    return valid_gray_value, valid_deep_value

if __name__ == '__main__':
    reshape_train_set(sample_size = 16, desc_file_name = 'sample_16_desc.txt')
