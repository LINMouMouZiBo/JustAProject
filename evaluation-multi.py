"""
Evaluation the Chalearn LAP predict accuracy by Jaccard Index according to 
    https://competitions.codalab.org/competitions/16499#learn_the_details-evaluation
From the website above, we can not know which sitation is the official like, so we implement the sigle label here.
Assuming that every label of the prediction of the fusion result has only one continuous frame,for example
	only considering this situation: 271/13537 1,20:10
	but not                          271/13537 1,10:10 11,20:10 or 271/13537 1,10:10 11,20:12 21,30:10 
"""
NUM_CLASSES = 249

def getJaccard(gClips = {}, pClips = {}):
    J_s = 0.0
    for i in xrange(1, NUM_CLASSES + 1):
        if gClips.has_key(i) and pClips.has_key(i):
            g_s_i = set()
            p_s_i = set()
            for frames in gClips[i]:
                g_s_i |= set(xrange(frames['start'], frames['end'] + 1))
            for frames in pClips[i]:
                p_s_i |= set(xrange(frames['start'], frames['end'] + 1))
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
                if not clips.has_key(label):
                    clips[label] = list()
                clips[label].append({
                    'start': start,
                    'end': end
                })
            res[video_name] = clips
    return res

def eval(groundTrue_file_name, predict_file_name, result_save_name):
    groundDict = getDict(groundTrue_file_name)
    preDict = getDict(predict_file_name)
    Js = 0.0
    with open(result_save_name, 'w') as f:
        for name in preDict.keys():
            sJs = getJaccard(groundDict[name], preDict[name])
            f.write('{0} {1}\n'.format(name, sJs))
            Js += sJs
        Js = Js / len(preDict)
        f.write("Evalution Jaccard Index : %.6f" % Js)


if __name__ == '__main__':
    eval(groundTrue_file_name='/share/data/CodaLab/ConGD/ConGD_phase_1/train.txt', predict_file_name='./valid_result.txt', result_save_name='./evaluation_result.txt')
