# coding=utf-8
import sys
sys.path.append('.')
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf

from shapely.geometry import Polygon, MultiPoint
from nets import model
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import graph_util
from six.moves import xrange


def quad_iou(_gt_bbox, _pre_bbox):

    gt_poly = Polygon(_gt_bbox).convex_hull
    pre_poly = Polygon(_pre_bbox).convex_hull

    union_poly = np.concatenate((_gt_bbox, _pre_bbox))

    if not gt_poly.intersects(pre_poly):
        iou = 0
        return iou
    else:
        inter_area = gt_poly.intersection(pre_poly).area
        union_area = MultiPoint(union_poly).convex_hull.area

        if union_area == 0:
            iou = 0
        else:
            iou = float(inter_area) / union_area

        return iou

def polygon_riou(pred_box, gt_box):
    """
    :param pred_box: list [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :param gt_box: list [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :return:
    """
    pred_polygon_points = np.array(pred_box).reshape(-1, 2)
    pred_poly = Polygon(pred_polygon_points).convex_hull

    gt_polygon_points = np.array(gt_box).reshape(-1, 2)

    gt_poly = Polygon(gt_polygon_points).convex_hull
    if not pred_poly.intersects(gt_poly):
        iou = 0
    else:
        inter_area = pred_poly.intersection(gt_poly).area
        union_area = gt_poly.area
        if union_area == 0:
            iou = 0
        else:
            iou = float(inter_area) / union_area
    return iou

def compute_f1_score(precision, recall):
    if precision == 0 or recall == 0:
        return 0.0
    else:
        return 2.0 * (precision * recall) / (precision + recall)

def load_ctw1500_labels(path):
    """
    load pts
    :param path:
    :return: polys shape [N, 14, 2]
    """
    assert os.path.exists(path), '{} is not exits'.format(path)
    polys = []
    tags = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(',')
            x = float(parts[0])
            y = float(parts[1])
            pts = [float(i) for i in parts[4:32]]
            poly = np.array(pts) + [x, y] * 14
            polys.append(poly.reshape([-1, 2]))
            tags.append(False)
    return np.array(polys, np.float), tags

def load_icdar_labels(path):
    """
        load pts
        :param path:
        :return:
        """
    assert os.path.exists(path), '{} is not exits'.format(path)
    polys = []
    tags = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(',')
            pts = [float(i) for i in parts[0:8]]
            poly = np.array(pts)
            polys.append(poly.reshape([-1, 2]))
            tags.append(False)
    return np.array(polys, np.float), np.array(tags)

def make_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def resize_img(img, max_size=736):
    h, w, _ = img.shape

    if max(h, w) > max_size:
        ratio = float(max_size) / h if h > w else float(max_size) / w
    else:
        ratio = 1.

    resize_h = int(ratio * h)
    resize_w = int(ratio * w)

    resize_h = resize_h if resize_h % 32 == 0 else abs(resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else abs(resize_w // 32 + 1) * 32
    resized_img = cv2.resize(img, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return resized_img, (ratio_h, ratio_w)

def ckpt2pb(ckptpath):

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.reset_default_graph()
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    binarize_map, threshold_map, thresh_binary = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=gpu_config)
    saver.restore(sess, ckptpath)
    input_graph_def = sess.graph.as_graph_def()

    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        ['feature_fusion/binarize_branch/Conv2d_transpose_1/Sigmoid'])

    output_graph_def = optimize_for_inference(input_graph_def=constant_graph,
                                              input_node_names=['input_images'],
                                              output_node_names=['feature_fusion/binarize_branch/Conv2d_transpose_1/Sigmoid'],
                                              placeholder_type_enum=[tf.float32.as_datatype_enum]
                                              )
    # 转化为tlite文件
    #converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(args.output_file, ['image_batch'],
    #                                                              ['pfld_inference/fc/BiasAdd'],
    #                                                              {"image_batch": [1, 112, 112, 3]
    #                                                               }
    #                                                              )
    # converter.allow_custom_ops = True
    # converter.inference_type = _types_pb2.QUANTIZED_UINT8
    #converter.post_training_quantize = True
    #tflite_model = converter.convert()




    with tf.gfile.FastGFile('db.pb', mode='wb') as f:
        f.write(output_graph_def.SerializeToString())
