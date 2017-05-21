import numpy as np
import cv2

class ValidationData:
    def __init__(self, height = 240, width = 320):
        self.p = '/share/data/CodaLab/ConGD/ConGD_phase_1/'
        self.filename = 'valid.txt'
        self.sample_size = 9
        self.sample_stride = 4
        self.height = height
        self.width = width
        self.read_desc()
    def read_desc(self):
        self.meta_data = []
        with open(self.p+self.filename) as f:
            lines = f.readlines()
            self.length = len(lines)
            for line in lines:
                line = line.split('\n')[0]
                self.meta_data.append({"video_name": line})

    def concat_image(self, gray_frame, deep_frame):
        result = np.dstack((gray_frame, deep_frame))
        return result

    def select(self, n):
        video_name = self.meta_data[n]["video_name"]
        video_rgb_name = video_name + '.M.avi'
        video_deep_name = video_name + '.K.avi'
        
        rgb_cap = cv2.VideoCapture(self.p + 'valid/' + video_rgb_name)
        deep_cap = cv2.VideoCapture(self.p + 'valid/' + video_deep_name)
        rgb_cap_len = int(rgb_cap.get(7))
        # deep_cap_len = int(rgb_cap.get(7))
        two_channel_frames = []

        for j in range(rgb_cap_len):
            rgb_ret, rgb_frame = rgb_cap.read()
            deep_ret, deep_frame = deep_cap.read()
            gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            deep_frame = cv2.cvtColor(deep_frame, cv2.COLOR_BGR2GRAY)
            gray_frame = np.resize(gray_frame, (self.height, self.width))
            deep_frame = np.resize(deep_frame, (self.height, self.width))
            two_channel_frames.append(self.concat_image(gray_frame, deep_frame))
        
        while len(two_channel_frames) < 9:
            two_channel_frames.append(two_channel_frames[-1])

        rgb_cap.release()
        deep_cap.release()
        return two_channel_frames, video_name

    def select_with_sampling(self, n):
        two_channel_frames, video_name = self.select(n)
        return self.sampling(two_channel_frames), video_name

    def sampling(self, two_channel_frames):
        result = []
        cap_len = len(two_channel_frames)
        for j in range(0, cap_len, self.sample_stride):
            sample_start = j
            sample_end = j + self.sample_size
            if sample_end > cap_len:
                sample_start = cap_len - self.sample_size
                sample_end = cap_len
            # [sample_start, sample_end)
            sample = two_channel_frames[sample_start: sample_end]
            # index start from 1
            result.append({"sample": sample,
                           "start": sample_start + 1,
                           "end": sample_end})
        return result


if __name__ == '__main__':
    vd = ValidationData(height = 112, width = 112)
    for i in range(vd.length):
        samples = vd.select_with_sampling(i)
        print i 
        break