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
    kmeans = pickle.load(open('cnn_kmeans'.format(TYPE), 'rb'))
    #svm = SVC(kernel='linear')
    f_svm = pickle.load(open('hw3_cf_svm', 'rb'))

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
    id_lst = []
    for line in fread.readlines():
        video_name = line.replace('\n', '') 
        feat = np.load('cnn/{}.npy'.format(video_name))
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

    scores = []
    for x in range(1, 4):
        cnn_svm, mfcc_svm, asr_svm, svm = pickle.load(open('hw3_lf_P00{}_svm'.format(x), 'rb'))
        cnn_scores = cnn_svm.decision_function(data[:, :50])
        mfcc_scores = mfcc_svm.decision_function(data[:, 50:100])
        asr_scores = asr_svm.decision_function(data[:, 100:])
        scores.append(np.concatenate([cnn_scores[:, np.newaxis], mfcc_scores[:, np.newaxis], asr_scores[:, np.newaxis]], axis=1))
    scores = np.concatenate(scores, axis=1)
    #sign = np.load('{}_{}_sign.npy'.format(TYPE, event_name))
    pred = f_svm.predict(scores)
    with open('hw3_predictions.csv', 'w') as output:
        output.write('VideoID,Label\n')
        for x in range(pred.shape[0]):
            if pred[x] == 0:
                output.write('{},NULL\n'.format(id_lst[x]))
            else:
                output.write('{},{}\n'.format(id_lst[x], pred[x]))

