#!/usr/bin/env python3

import os
import sys
import threading
import cv2
import numpy as np
import yaml
import pickle
import pdb
import torch
from torchvision import models


hessian_threshold = 100
model = models.vgg16(pretrained=True).cuda()
classifier = list(model.classifier)
def transform(x):
    x = x.view(x.size(0), -1)
    for y in [0, 1, 3]:
        model = classifier[y]
        x = model(x)
    return x

def get_surf_features_from_video(downsampled_video_filename, surf_feat_video_filename, keyframe_interval):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."
    # TODO
    features = []
    for img in get_keyframes(downsampled_video_filename, keyframe_interval):
        features.append(img[np.newaxis, : ,: ,:])
    if features:
        features = np.vstack(features)
        x_cent = features.shape[1] // 2
        y_cent = features.shape[2] // 2
        if x_cent < 112:
            _features = np.zeros((features.shape[0], 224, features.shape[2], 3))
            _features[:, 112-x_cent:112+x_cent, :, :] = features
            features = _features
            x_cent = 112
        if y_cent < 112:
            _features = np.zeros((features.shape[0], features.shape[1], 224, 3))
            _features[:, :, 112-y_cent:112+y_cent, :] = features
            features = _features
            y_cent = 112
        features = features[:, x_cent-112:x_cent+112, y_cent-112:y_cent+112, :]
        batch_input = torch.from_numpy(features).permute(0, 3, 1, 2).float().cuda()
        batch_feat = model.features(batch_input)
        batch_feat = transform(batch_feat)
        return batch_feat.detach().cpu().numpy()
    else:
        return features


def get_keyframes(downsampled_video_filename, keyframe_interval):
    "Generator function which returns the next keyframe."

    # Create video capture object
    video_cap = cv2.VideoCapture(downsampled_video_filename)
    frame = 0
    while True:
        frame += 1
        ret, img = video_cap.read()
        if ret is False:
            break
        if frame % keyframe_interval == 0:
            yield img
    video_cap.release()


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
    keyframe_interval = my_params.get('keyframe_interval')
    hessian_threshold = my_params.get('hessian_threshold')
    surf_features_folderpath = my_params.get('surf_features')
    downsampled_videos = my_params.get('downsampled_videos')

    # TODO: Create SURF object

    # Check if folder for SURF features exists
    if not os.path.exists(surf_features_folderpath):
        os.mkdir(surf_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes

    fread = open(all_video_names, "r")
    for line in fread.readlines():
        video_name = line.replace('\n', '')
        downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')
        surf_feat_video_filename = os.path.join(surf_features_folderpath, video_name + '.surf')

        if not os.path.isfile(downsampled_video_filename):
            continue

        print(video_name)
        # Get SURF features for one video
        features = get_surf_features_from_video(downsampled_video_filename, surf_feat_video_filename, keyframe_interval)
        np.save('cnn/{}'.format(video_name), features)
