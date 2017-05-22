def cal_gini(clips):
    result = 0.0
    p = [0.0] * 250
    for i in range(len(clips)):
        clip = clips[i]
        p[clip['label']] += 1.0
    for i in range(len(p)):
        p[i] /= len(clips)
        result += p[i] * p[i]
    return 1 - result

def clips_total_len(clips):
    return clips[-1]['end'] - clips[0]['start'] + 1

def concat(clips1, clips2):
    end = clips1[-1]['end']
    start = clips2[0]['start']
    mid = (start + end) / 2
    clips1[-1]['end'] = mid
    clips2[0]['start'] = mid + 1
    return clips1 + clips2

def top1_probability(clips):
    arr = [0] * 250
    for i in range(len(clips)):
        clip = clips[i]
        arr[clip['label']] += 1
    max_value = 0
    max_label = -1
    for i in range(len(arr)):
        if arr[i] > max_value:
            max_value = arr[i]
            max_label = i
    return max_value / float(len(clips))

def top1_fusion(clips):
    arr = [0] * 250
    for i in range(len(clips)):
        clip = clips[i]
        arr[clip['label']] += 1
    max_value = 0
    max_label = -1
    for i in range(len(arr)):
        if arr[i] > max_value:
            max_value = arr[i]
            max_label = i
    return [{
        'start': clips[0]['start'],
        'end': clips[-1]['end'],
        'label': max_label
    }]


def decision_tree(clips):
    if clips_total_len(clips) < 30: #param
        return top1_fusion(clips)
    if top1_probability(clips) > 0.5: #param
        return top1_fusion(clips)

    offset = 4 #param
    min_split_gini = 1
    min_index = -1
    cur_gini = cal_gini(clips)
    for i in range(offset,len(clips)-offset):
        split_gini = float(i)/len(clips) * cal_gini(clips[:i]) + (1-float(i)/len(clips)) * cal_gini(clips[i:])
        if split_gini < min_split_gini:
            min_split_gini = split_gini
            min_index = i
    
    if min_split_gini < cur_gini:
        return concat(decision_tree(clips[:min_index]), decision_tree(clips[min_index:]))
    else:
        return top1_fusion(clips)

def clips_fusion(clips):
    return decision_tree(clips)

def fusion(input_file_name, output_file_name):
    filehandle = open(output_file_name, 'w')

    with open(input_file_name) as f:
            lines = f.readlines()
            cnt = 0
            for line in lines:
                cnt += 1
                # print cnt
                line = line.split('\n')[0]
                video_name = line.split(' ')[0]
                raw_clips = line.split(' ')[1:]
                clips = []
                for i in range(len(raw_clips)):
                    rest, label = raw_clips[i].split(':')
                    start, end = rest.split(',')
                    start, end, label = int(start), int(end), int(label)
                    clips.append({
                            'start': start,
                            'end': end,
                            'label': label
                        })
                fusion_clips = clips_fusion(clips)
                string = video_name
                for i in range(len(fusion_clips)):
                    clip = fusion_clips[i]
                    string += ' ' + str(clip['start']) + ',' + str(clip['end']) + ':' + str(clip['label'])
                filehandle.write(string + '\n')

    filehandle.close()

if __name__ == '__main__':
    fusion(input_file_name = 'valid_result-9_frame_pretrain.txt', output_file_name = 'valid_prediction.txt')
