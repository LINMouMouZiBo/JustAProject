import numpy as np
import cv2
import random
import math

class SampleData:
    def __init__(self, filename):
        self.desc_filename = filename
        self.p = '/share/data/CodaLab/ConGD/ConGD_phase_1/'
        self.read_desc()

    def read_desc(self):
        with open(self.desc_filename) as f:
            lines = f.readlines() # len = 302271
            self.meta_data = []
            for line in lines:
                line = line.split('\n')[0]
                video_name, rest = line.split(':')
                rest, label = rest.split(' ')
                start, end = rest.split('-')
                start, end, label = int(start), int(end), int(label)
                self.meta_data.append({'video_name':video_name,
                                       'start':start, 
                                       'end':end, 
                                       'label':label})

    def rand(self):
        return int(math.floor(random.random() * len(self.meta_data)))
    
    def select(self, n):
        video_rgb_name = self.meta_data[n]['video_name'] + '.M.avi'
        video_deep_name = self.meta_data[n]['video_name'] + '.K.avi'

        start = self.meta_data[n]['start']
        end = self.meta_data[n]['end']
        label = self.meta_data[n]['label']

        rgb_cap = cv2.VideoCapture(self.p + 'train/' + video_rgb_name)
        deep_cap = cv2.VideoCapture(self.p + 'train/' + video_deep_name)
        two_channel_frames = []

        for j in range(1, end + 1):
            rgb_ret, rgb_frame = rgb_cap.read()
            deep_ret, deep_frame = deep_cap.read()
            if j < start:
                continue
            gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            deep_frame = cv2.cvtColor(deep_frame, cv2.COLOR_BGR2GRAY)
            two_channel_frames.append([gray_frame, deep_frame])

        rgb_cap.release()
        deep_cap.release()
        return two_channel_frames, label

    # batch item shape [2 * 240 * 320 * 1]
    def batch(self, size = 10):
        values = []
        labels = []
        for i in range(size):
            v, l = self.select(self.rand())
            values.append(v)
            labels.append(l)
        return values, labels

def read_valid_data():
    print 'reading validation set...'
    p = '/share/data/CodaLab/ConGD/ConGD_phase_1/'
    filename = 'valid.txt'

    valid_gray_value = []
    valid_deep_value = []

    with open(p+filename) as f:
        lines = f.readlines()
        print len(lines)
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
    sample = SampleData('sample_data_desc.txt')
    values, labels = sample.batch(size = 15)
    # valid_gray_value, valid_deep_value = read_valid_data()
