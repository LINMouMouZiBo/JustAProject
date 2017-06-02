import numpy as np
import cv2

classSize = 249
base = '/share/data/CodaLab/ConGD/ConGD_phase_1/train/'

def readSample(filename):
    classSample = [{'found': False, 'record': ''} for i in range(1, 250)]
    
    lines = [line.rstrip('\n').split(' ') for line in open(filename)]
    
    classFound = 0

    for line in lines:
        if classFound == classSize:
            print('done')
            break

        for i in range(1, len(line)):
            labelPair = line[i].split(':')
            label = int(labelPair[1])
            if not classSample[label - 1]['found']:
                classFound += 1
                classSample[label - 1] = {
                    'found': True,
                    'record': line[0] + ' ' + labelPair[0]
                }

    return classSample

def saveClassSample(sample):
    sampleRecord = ''
    for index, item in enumerate(sample):
        sampleRecord += '{0} {1}\n'.format(item['record'], index + 1)
    
    with open('class_sample.txt', 'w') as f:
        f.write(sampleRecord)

def generate_sample_video(vlist):
    lines = [line.rstrip('\n').split(' ') for line in open(vlist)]

    for line in lines:
        print(line)
        idx_pair = line[1].split(',')
        start_frame = int(idx_pair[0])
        end_frame = int(idx_pair[1])

        cap = cv2.VideoCapture('{0}{1}.M.avi'.format(base, line[0]))

        frame_width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        out = cv2.VideoWriter(line[2] + '.avi', fourcc, 10, (frame_width, frame_height))
        for i in range(1, end_frame + 1):
            ret, frame = cap.read()

            if i < start_frame or not ret:
                continue

            out.write(frame)

        cap.release()
        out.release()

if __name__ == "__main__":
    # sample = readSample('train.txt')
    generate_sample_video('class_sample.txt')
