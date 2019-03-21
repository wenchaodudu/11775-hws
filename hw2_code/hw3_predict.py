import sys
import pdb
import yaml
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC



if __name__ == '__main__':
    '''
    if len(sys.argv) != 3:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        exit(1)
    '''
    TYPE = 'cnn'
    all_video_names = sys.argv[1]
    vocab_size = int(sys.argv[2])

    # Get parameters from config file

    #kmeans = KMeans(n_clusters=num_cluster, max_iter=50, verbose=3, n_init=1)
    kmeans = pickle.load(open('{}_kmeans'.format(TYPE), 'rb'))
    #svm = SVC(kernel='linear')
    svms = []
    '''
    for event_name in ['P001', 'P002', 'P003']:
        svm = pickle.load(open('hw3_{}_svm'.format(event_name), 'rb'))
        svms.append(svm)
    '''
    svms.append(pickle.load(open('hw3_lf_P001_svm', 'rb')))
    svms.append(pickle.load(open('hw3_lf_P002_svm', 'rb')))
    svms.append(pickle.load(open('hw3_P003_svm', 'rb')))

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes
    '''
    def get_labels(path):
        train_labels = open(path).readlines()
        labels = {}
        pos = []
        neg = []
        for line in train_labels:
            name, l = line.split()
            if l == event_name:
                pos.append(name)
                labels[name] = 1
            else:
                neg.append(name)
                labels[name] = 0
        return labels, pos, neg
    train_labels, pos_labels, neg_labels = get_labels('../all_test_fake.lst')
    '''

    fread = open(all_video_names, "r")
    data = []
    labels = []
    names = []
    id_lst = []
    for line in fread.readlines():
        video_name = line.replace('\n', '') 
        names.append(video_name)
        feat = np.load('{}/{}.npy'.format(TYPE, video_name))
        feat = [f for f in feat if f is not None]
        print(video_name)
        bow = np.zeros(vocab_size)
        if feat:
            feat = np.vstack(feat)
            words = kmeans.predict(feat)
            bow[words] += 1
            data.append(bow)
        else:
            data.append(bow)
        #labels.append(train_labels[video_name])
        id_lst.append(video_name)

    def read_hw1_data(feat_name):
        feature = np.load('../{}_feat.npy'.format(feat_name))
        ids = np.load('../{}_id.npy'.format(feat_name)).tolist()
        idx_select = [ids.index(i) for i in id_lst]
        train_features = feature[idx_select]
        return train_features
    asr_features = read_hw1_data('asr')
    mfcc_features = read_hw1_data('mfcc')
    data = np.concatenate((data, mfcc_features, asr_features), axis=1)

    data = np.vstack(data)
    mean = np.load('hw3_{}_mean.npy'.format('P001'))
    std = np.load('hw3_{}_std.npy'.format('P001'))
    data = (data - mean) / std
    #sign = np.load('{}_{}_sign.npy'.format(TYPE, event_name))
    def get_scores(x):
        cnn_scores = svms[x][0].decision_function(data[:, :50])
        mfcc_scores = svms[x][1].decision_function(data[:, 50:100])
        asr_scores = svms[x][2].decision_function(data[:, 100:])
        scores = np.concatenate([cnn_scores[:, np.newaxis], mfcc_scores[:, np.newaxis], asr_scores[:, np.newaxis]], axis=1)
        return svms[x][3].decision_function(scores)

    '''
    pred_1 = svms[0].decision_function(data)[np.newaxis, :]
    pred_2 = svms[1].decision_function(data)[np.newaxis, :]
    pred_3 = svms[2].decision_function(data)[np.newaxis, :]
    '''
    pred_1 = get_scores(0)[np.newaxis, :]
    pred_2 = get_scores(1)[np.newaxis, :]
    pred_3 = svms[2].decision_function(data[:, :101])[np.newaxis, :]
    pred = np.vstack((pred_1, pred_2, pred_3))
    with open('hw3_predictions.csv', 'w') as output:
        output.write('VideoID,Label\n')
        for x in range(pred.shape[1]):
            p = pred[:, x]
            if (p < -1).all():
                output.write('{},NULL\n'.format(names[x]))
            else:
                l = np.argmax(p) + 1
                output.write('{},{}\n'.format(names[x], l))

