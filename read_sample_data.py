import numpy as np
import cv2
import random
import math

class SampleData:
    def __init__(self, filename, fstart = 0, fend = -1, height = 240, width = 320, start_cnt = 0):
        self.desc_filename = filename
        self.p = '/share/data/CodaLab/ConGD/ConGD_phase_1/'
        self.fstart = fstart
        self.fend = fend
        self.height = height
        self.width = width
        self.cnt = start_cnt
        self.read_desc()

    def read_desc(self):
        with open(self.desc_filename) as f:
            lines = f.readlines()[self.fstart: self.fend]
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
        result = self.cnt % len(self.meta_data)
        self.cnt += 1
        return result
        # return int(math.floor(random.random() * len(self.meta_data)))
    
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
            gray_frame = np.resize(gray_frame, (self.height, self.width))
            deep_frame = np.resize(deep_frame, (self.height, self.width))
            two_channel_frames.append(self.concat_image(gray_frame, deep_frame))

        rgb_cap.release()
        deep_cap.release()
        return two_channel_frames, label

    def concat_image(self, gray_frame, deep_frame):
        result = np.dstack((gray_frame, deep_frame))
        return result

    # batch shape [batch_size, height, width, 2]
    def batch(self, size = 10):
        values = []
        labels = []
        for i in range(size):
            v, l = self.select(self.rand())
            print l
            values.append(v)
            labels.append(l-1)
        return values, labels

if __name__ == '__main__':
    sample = SampleData('shuffle_sample_16_desc.txt')
    values, labels = sample.batch(size = 15)
