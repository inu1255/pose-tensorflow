#!/usr/bin/env python
# coding=utf-8

import os
import sys
import numpy as np
from scipy.misc import imread
from pose.config import load_config
from pose.dataset.factory import create as create_dataset
from pose.nnet import predict
from pose.dataset.pose_dataset import data_to_input
from pose.multiperson.detections import extract_detections
from pose.multiperson.predict import SpatialModel, eval_graph, get_person_conf_multicut
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
cfg = load_config(os.path.join(dir_path, "pose_cfg_multi.yaml"))

dataset = create_dataset(cfg)

sm = SpatialModel(cfg)
sm.load()

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

def detection(image):
    time_start=time.time()
    image_batch = data_to_input(image)

    # Compute prediction with the CNN
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref, pairwise_diff = predict.extract_cnn_output(outputs_np, cfg, dataset.pairwise_stats)

    detections = extract_detections(cfg, scmap, locref, pairwise_diff)
    unLab, pos_array, unary_array, pwidx_array, pw_array = eval_graph(sm, detections)
    person_conf_multi = get_person_conf_multicut(sm, unLab, unary_array, pos_array)

    time_end=time.time()
    print('totally cost', time_end-time_start)
    return person_conf_multi
