#!/usr/bin/env python
# coding=utf-8

import os
import sys
import numpy as np
sys.path.append(os.path.dirname(__file__) + "/../")
from scipy.misc import imread
from config import load_config
from dataset.factory import create as create_dataset
from nnet import predict
from dataset.pose_dataset import data_to_input
from multiperson.detections import extract_detections
from multiperson.predict import SpatialModel, eval_graph, get_person_conf_multicut
import time
from flask import Flask, request, jsonify

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
app = Flask(__name__)


cfg = load_config("demo/pose_cfg_multi.yaml")

dataset = create_dataset(cfg)

sm = SpatialModel(cfg)
sm.load()

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

@app.route('/detection', methods=['POST','GET'])
def detection():
    f = None
    if request.method == 'POST':
        f = request.files.get('f')
    else:
        f = request.args.get('f')
    if f is None:
        return jsonify({'no': 400, "msg": "缺少文件参数"})
    try:
        image = imread(f, mode='RGB')
    except Exception as e:
        print(e)
        return jsonify({'no': 404, 'msg': '找不到文件'})
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
    return jsonify({'no': 200, 'data': person_conf_multi.tolist()})

if __name__ == '__main__':
    app.run('127.0.0.1', 3008)
