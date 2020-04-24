import os
import sys
sys.path.append('.')
import cv2
import time
import tqdm
import argparse
import numpy as np
import tensorflow as tf

from config.config import cfg
from shapely.geometry import Polygon
from utils.post_process import SegDetectorRepresenter
from nets import model


def get_args():
    parser = argparse.ArgumentParser(description='DB-tf')
    parser.add_argument('--ckptpath', default='model/ckpt/DB_resnet_v1_50_aspp_model.ckpt-632263',
                        type=str,
                        help='load model')
    parser.add_argument('--imgpath', default='/home/lishuanzhu/Downloads/results_summary/3fc3daf3-b7b7-4a3e-ba72-deefd56f7a82.jpg',
                        type=str)
    parser.add_argument('--gpuid', default='0',
                        type=str)
    parser.add_argument('--ispoly', default=False,
                        type=bool)
    parser.add_argument('--show_res', default=True,
                        type=bool)

    args = parser.parse_args()

    return args

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

class DB():

    def __init__(self, ckpt_path, gpuid='0'):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        tf.reset_default_graph()
        self._input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        self._binarize_map, self._threshold_map, self._thresh_binary = model.model(self._input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=gpu_config)
        saver.restore(self.sess, ckpt_path)
        self.decoder = SegDetectorRepresenter()
        print('restore model from:', ckpt_path)

    def __del__(self):
        self.sess.close()

    def detect_img(self, img_path, ispoly=True, show_res=True):
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        resized_img, ratio, size = self._resize_img(img)

        s = time.time()
        binarize_map, threshold_map, thresh_binary = self.sess.run([self._binarize_map, self._threshold_map, self._thresh_binary],
                                                                   feed_dict={self._input_images: [resized_img]})
        net_time = time.time()-s

        s = time.time()
        boxes, scores = self.decoder([resized_img], thresh_binary, ispoly)
        boxes = boxes[0]
        area = h * w
        res_boxes = []
        res_scores = []
        for i, box in enumerate(boxes):
            box[:, 0] *= ratio[1]
            box[:, 1] *= ratio[0]
            if Polygon(box).convex_hull.area > cfg.FILTER_MIN_AREA*area:
                res_boxes.append(box)
                res_scores.append(scores[0][i])
        post_time = time.time()-s

        if show_res:
            img_name = os.path.splitext(os.path.split(img_path)[-1])[0]
            make_dir('./show')
            #cv2.imwrite('show/' + img_name + '_binarize_map.jpg', binarize_map[0][0:size[0], 0:size[1], :]*255)
            #cv2.imwrite('show/' + img_name + '_threshold_map.jpg', threshold_map[0][0:size[0], 0:size[1], :]*255)
            #cv2.imwrite('show/' + img_name + '_thresh_binary.jpg', thresh_binary[0][0:size[0], 0:size[1], :]*255)
            for box in res_boxes:
                cv2.polylines(img, [box.astype(np.int).reshape([-1, 1, 2])], True, (0, 255, 0), 5)
                # print(Polygon(box).convex_hull.area, Polygon(box).convex_hull.area/area)
            #cv2.namedWindow("a", 0);
            #cv2.resizeWindow("a", 1024, 768);
            #cv2.imshow('a', img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            cv2.imwrite('show/' + img_name + '_show.jpg', img)

        return res_boxes, res_scores, (net_time, post_time)


    def detect_batch(self, batch):
        pass

    def _resize_img(self, img, max_size=1024):
        h, w, _ = img.shape

        ratio = float(max(h, w)) / max_size

        new_h = int((h / ratio // 32) * 32)
        new_w = int((w / ratio // 32) * 32)

        resized_img = cv2.resize(img, dsize=(new_w, new_h))

        input_img = np.zeros([max_size, max_size, 3])
        input_img[0:new_h, 0:new_w, :] = resized_img

        ratio_w = w / new_w
        ratio_h = h / new_h

        return input_img, (ratio_h, ratio_w), (new_h, new_w)


if __name__ == "__main__":
    args = get_args()

    db = DB(args.ckptpath, args.gpuid)

    db.detect_img(args.imgpath, args.ispoly, args.show_res)

    img_list = os.listdir('/home/lishuanzhu/Downloads/ktp2')

    net_all = 0
    post_all = 0
    pipe_all = 0

    for i in tqdm.tqdm(img_list):
        #print(i)
        _, _, (net_time, post_time) = db.detect_img(os.path.join('/home/lishuanzhu/Downloads/ktp2',i), args.ispoly, show_res=True)
        net_all += net_time
        post_all += post_time
        pipe_all += (net_time + post_time)

    print('net:', net_all/len(img_list))
    print('post:', post_all/len(img_list))
    print('pipe:', pipe_all/len(img_list))