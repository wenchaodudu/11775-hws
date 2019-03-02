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
    TYPE = sys.argv[3]
    all_video_names = sys.argv[1]
    event_name = sys.argv[2]
    vocab_size = int(sys.argv[4])

    # Get parameters from config file

    #kmeans = KMeans(n_clusters=num_cluster, max_iter=50, verbose=3, n_init=1)
    kmeans = pickle.load(open('{}_kmeans'.format(TYPE), 'rb'))
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
        data = []
        labels = []
        for line in fread.readlines():
            video_name = line.replace('\n', '') 
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
            labels.append(_labels[video_name])
        return data, labels
    t_data, t_labels = read_data('list/train.video', train_labels)
    v_data, v_labels = read_data('list/val.video', valid_labels)
    data = t_data + v_data
    labels = t_labels + v_labels
    data = np.vstack(data)
    
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    np.save('{}_{}_mean'.format(TYPE, event_name), mean)
    np.save('{}_{}_std'.format(TYPE, event_name), std)
    #data = np.log(data+1)
    data = (data - mean) / std

    pos_features = data[np.asarray(labels) == 1]
    neg_features = data[np.asarray(labels) == 0]
    statistic, pvalue = scipy.stats.ttest_ind(pos_features, neg_features)
    sign = np.where(pvalue < 0.1)[0]
    print(len(sign))
    np.save('{}_{}_sign'.format(TYPE, event_name), sign)

    num_repeat = neg_features.shape[0] // pos_features.shape[0]
    add_features = pos_features.repeat(num_repeat, axis=0)
    data = np.vstack([data, add_features])
    labels += [1] * add_features.shape[0]
    svm.fit(data, labels)
    pickle.dump(svm, open('{}_{}_svm'.format(TYPE, event_name), 'wb'))

