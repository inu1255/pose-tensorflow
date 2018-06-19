import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread, imsave

from config import load_config
from dataset.factory import create as create_dataset
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

from multiperson.detections import extract_detections
from multiperson.predict import SpatialModel, eval_graph, get_person_conf_multicut
from multiperson.visualize import PersonDraw, visualize_detections

import matplotlib.pyplot as plt
import time
import cv2
from flask import send_file, Flask, request, jsonify
from io import BytesIO
from threading import Lock

mutex = Lock()
app = Flask(__name__)


cfg = load_config("demo/pose_cfg_multi.yaml")

dataset = create_dataset(cfg)

sm = SpatialModel(cfg)
sm.load()

draw_multi = PersonDraw()

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

image = None
person_conf_multi = None
running = False

@app.route('/load', methods=['POST','GET'])
def load():
    global image
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
    return jsonify({'no': 200})

@app.route('/detection')
def detection():
    global person_conf_multi
    global running
    if image is None:
        return jsonify({'no':404,'msg': '找不到文件'})
    if running:
        return jsonify({'no':405,'msg': '正在执行'})
    else:
        running = True
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
        running = False
        return jsonify({'no': 200, 'persons': person_conf_multi.tolist()})

@app.route('/image')
def image_render():
    if image is None:
        return jsonify({"no": 404, "msg": "尚未上传图片"})
    if person_conf_multi is None:
        return jsonify({"no": 404, "msg": "尚未执行检测"})
    visim_multi = np.copy(image)
    plt.cla()
    fig = plt.imshow(visim_multi)
    draw_multi.draw(visim_multi, dataset, person_conf_multi)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    img = BytesIO()
    plt.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run('127.0.0.1', 3008)
