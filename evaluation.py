"""
Evaluation the Chalearn LAP predict accuracy by Jaccard Index according to 
    https://competitions.codalab.org/competitions/16499#learn_the_details-evaluation

"""
NUM_CLASSES = 249

def getJaccard(gClips = {}, pClips = {}):
    J_s = 0.0
    for i in xrange(1, NUM_CLASSES + 1):
        if gClips.has_key(i) and pClips.has_key(i):
            g_s_i = set(xrange(gClips[i]['start'], gClips[i]['end'] + 1))
            p_s_i = set(xrange(pClips[i]['start'], pClips[i]['end'] + 1))
            J_s_i = float(len(g_s_i & p_s_i)) / float(len(g_s_i | p_s_i))
            J_s += J_s_i
            # print g_s_i
            # print p_s_i
            # print g_s_i & p_s_i
            # print g_s_i | p_s_i
            # print("%d %% %d = %f" % (len(g_s_i & p_s_i), len(g_s_i | p_s_i), J_s_i))
    J_s = float(J_s) / float(len(gClips))
    return J_s

def getDict(file_name):
    res = {}
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\n')[0]
            video_name = line.split(' ')[0]
            raw_clips = line.split(' ')[1:]
            clips = dict()
            for i in range(len(raw_clips)):
                rest, label = raw_clips[i].split(':')
                start, end = rest.split(',')
                start, end, label = int(start), int(end), int(label)
                clips[label] = {
                    'start': start,
                    'end': end
                }
            res[video_name] = clips
    return res

def eval(groundTrue_file_name, predict_file_name):
    groundDict = getDict(groundTrue_file_name)
    preDict = getDict(predict_file_name)
    Js = 0.0
    for name in preDict.keys():
        Js += getJaccard(groundDict[name], preDict[name])
    Js = Js / len(preDict)
    print("Evalution Jaccard Index : %.6f" % Js)


if __name__ == '__main__':
    eval(groundTrue_file_name='./predict/valid_result-9_frame_pretrain.txt', predict_file_name='./predict/valid_prediction.txt')
