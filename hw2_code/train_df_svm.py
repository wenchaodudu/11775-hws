import sys
import pdb
import yaml
import pickle
import scipy
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
    event_name = sys.argv[1]
    #TYPE = sys.argv[3]
    TYPE = 'cnn'
    vocab_size = int(sys.argv[2])

    # Get parameters from config file

    #kmeans = KMeans(n_clusters=num_cluster, max_iter=50, verbose=3, n_init=1)
    kmeans = pickle.load(open('{}_kmeans'.format(TYPE), 'rb'))
    cnn_svm = SVC(kernel='rbf')
    mfcc_svm = SVC(kernel='rbf')
    asr_svm = SVC(kernel='rbf')
    cnn_mfcc_svm = SVC(kernel='rbf')
    cnn_asr_svm = SVC(kernel='rbf')
    mfcc_asr_svm = SVC(kernel='rbf')
    all_svm = SVC(kernel='rbf')
    svm = SVC(kernel='rbf')

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes
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
    train_labels, pos_labels, neg_labels = get_labels('../all_trn.lst')
    valid_labels, pos_labels, neg_labels = get_labels('../all_val.lst')

    def read_data(all_video_names, _labels):
        fread = open(all_video_names, "r")
        id_lst = []
        data = []
        labels = []
        for line in fread.readlines():
            video_name = line.replace('\n', '') 
            feat = np.load('{}/{}.npy'.format(TYPE, video_name))
            feat = [f for f in feat if f is not None]
            print(video_name)
            id_lst.append(video_name)
            bow = np.zeros(vocab_size)
            if feat:
                feat = np.vstack(feat)
                words = kmeans.predict(feat)
                bow[words] += 1
                data.append(bow)
            else:
                data.append(bow)
            labels.append(_labels[video_name])
        return data, labels, id_lst
    t_data, t_labels, t_id_lst = read_data('list/train.video', train_labels)
    v_data, v_labels, v_id_lst = read_data('list/val.video', valid_labels)
    data = t_data + v_data
    #data = t_data
    labels = t_labels + v_labels
    #labels = t_labels
    data = np.vstack(data)
    id_lst = t_id_lst + v_id_lst
    #id_lst = t_id_lst
    
    def read_hw1_data(feat_name):
        feature = np.load('../{}_feat.npy'.format(feat_name))
        ids = np.load('../{}_id.npy'.format(feat_name)).tolist()
        idx_select = [ids.index(i) for i in id_lst]
        train_features = feature[idx_select]
        return train_features
    asr_features = read_hw1_data('asr')
    mfcc_features = read_hw1_data('mfcc')
    data = np.concatenate((data, mfcc_features, asr_features), axis=1)

    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1
    np.save('hw3_{}_mean'.format(event_name), mean)
    np.save('hw3_{}_std'.format(event_name), std)
    #data = np.log(data+1)
    data = (data - mean) / std

    pos_features = data[np.asarray(labels) == 1]
    neg_features = data[np.asarray(labels) == 0]
    '''
    statistic, pvalue = scipy.stats.ttest_ind(pos_features, neg_features)
    sign = np.where(pvalue < 0.1)[0]
    print(len(sign))
    np.save('{}_{}_sign'.format(TYPE, event_name), sign)
    '''

    num_repeat = neg_features.shape[0] // pos_features.shape[0]
    add_features = pos_features.repeat(num_repeat, axis=0)
    data = np.vstack([data, add_features])
    labels += [1] * add_features.shape[0]
    cnn_svm.fit(data[:, :50], labels)
    mfcc_svm.fit(data[:, 50:100], labels)
    asr_svm.fit(data[:, 100:], labels)
    cnn_mfcc_svm.fit(data[:, :100], labels)
    cnn_asr_svm.fit(np.concatenate((data[:, :50], data[:, 100:]), axis=1), labels)
    mfcc_asr_svm.fit(data[:, 50:], labels)
    all_svm.fit(data, labels)

    cnn_scores = cnn_svm.decision_function(data[:, :50])
    mfcc_scores = mfcc_svm.decision_function(data[:, 50:100])
    asr_scores = asr_svm.decision_function(data[:, 100:])
    cnn_mfcc_scores = cnn_mfcc_svm.decision_function(data[:, :100])
    cnn_asr_scores = cnn_asr_svm.decision_function(np.concatenate((data[:, :50], data[:, 100:]), axis=1))
    mfcc_asr_scores = mfcc_asr_svm.decision_function(data[:, 50:])
    all_scores = all_svm.decision_function(data)
    scores = np.concatenate([cnn_scores[:, np.newaxis], mfcc_scores[:, np.newaxis], asr_scores[:, np.newaxis], 
                             cnn_mfcc_scores[:, np.newaxis], cnn_asr_scores[:, np.newaxis], mfcc_asr_scores[:, np.newaxis], all_scores[:, np.newaxis]], axis=1)
    svm.fit(scores, labels)
    pickle.dump([cnn_svm, mfcc_svm, asr_svm, cnn_mfcc_svm, cnn_asr_svm, mfcc_asr_svm, all_svm, svm], open('hw3_df_{}_svm'.format(event_name), 'wb'))
