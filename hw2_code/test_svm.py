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
    TYPE = sys.argv[3]
    all_video_names = sys.argv[1]
    event_name = sys.argv[2]
    vocab_size = int(sys.argv[4])

    # Get parameters from config file

    #kmeans = KMeans(n_clusters=num_cluster, max_iter=50, verbose=3, n_init=1)
    kmeans = pickle.load(open('{}_kmeans'.format(TYPE), 'rb'))
    #svm = SVC(kernel='linear')
    svm = pickle.load(open('{}_{}_svm'.format(TYPE, event_name), 'rb'))

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
        #labels.append(train_labels[video_name])

    data = np.vstack(data)
    mean = np.load('{}_{}_mean.npy'.format(TYPE, event_name))
    std = np.load('{}_{}_std.npy'.format(TYPE, event_name))
    data = (data - mean) / std
    sign = np.load('{}_{}_sign.npy'.format(TYPE, event_name))
    pred = svm.decision_function(data)
    with open('{}_{}_pred.txt'.format(TYPE, event_name), 'w') as output:
        for l in pred:
            output.write(str(l))
            output.write('\n')

