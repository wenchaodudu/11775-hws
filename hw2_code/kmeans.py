import sys
import pdb
import yaml
import pickle
import numpy as np
from sklearn.cluster import KMeans


TYPE = 'surf'

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")
        exit(1)

    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    my_params = yaml.load(open(config_file))

    # Get parameters from config file
    if TYPE == 'surf':
        num_cluster = my_params.get('kmeans_cluster_num')
    else:
        num_cluster = 50

    kmeans = KMeans(n_clusters=num_cluster, max_iter=50, verbose=3, n_init=1)

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes

    fread = open(all_video_names, "r")
    data = []
    for line in fread.readlines():
        video_name = line.replace('\n', '') 
        feat = np.load('{}/{}.npy'.format(TYPE, video_name))
        feat = [f for f in feat if f is not None]
        print(video_name)
        if feat:
            feat = np.vstack(feat)
            num = feat.shape[0]
            if TYPE == 'surf':
                feat = feat[np.random.choice(num, num // 10)]
            data.append(feat)

    data = np.vstack(data)
    print(data.shape[0])
    kmeans.fit(data)
    pickle.dump(kmeans, open('{}_kmeans'.format(TYPE), 'wb'))
