"""
Created by: SonNN27
Created at: 28/05/2021
"""

from __future__ import print_function

import argparse
import collections
import glob
import os
import time

import torch
import numpy as np
import cv2
from tqdm import tqdm

from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from data import cfg_mnet, cfg_re50

def load_weights(model, weight_path):
    """
    Loading weights to network
    :return:
    """
    checkpoint = torch.load(weight_path, map_location=lambda storage, loc: storage)
    source_state_ = checkpoint
    source_state = {}

    target_state = model.state_dict()
    new_target_state = collections.OrderedDict()

    for k in source_state_:
        if k.startswith('module') and not k.startswith('module_list'):
            source_state[k[7:]] = source_state_[k]
        else:
            source_state[k] = source_state_[k]

    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] {} Not found pre-trained parameters for {}'
                  .format(weight_path.split('/')[-1], target_key))

    model.load_state_dict(new_target_state, strict=False)

    model.eval()
    if torch.cuda.is_available():
        print("Model {} runs on cuda".format(weight_path.split('/')[-1]))
        model = model.cuda()

    return model


class Detection:
    def __init__(self, weight, model_type='mobile0.25', confident=0.8, mns_thresh=0.15, input_w=640):
        if model_type == 'mobile0.25':
            self._cfg = cfg_mnet
        elif model_type == 'resnet50':
            self._cfg = cfg_re50
        self._model = RetinaFace(self._cfg, phase="test")

        self._model = load_weights(self._model, weight)
        self._scale = None
        self._input_w = input_w
        self._confident = confident
        self._mns_thresh = mns_thresh

    def pre_process(self, frame):
        """
        pre-process before feed to model
        @frame: original opencv frame
        :return: processed image
        """
        img_raw = frame.copy()
        self._scale = self._input_w / img_raw.shape[1]
        resized_img = cv2.resize(img_raw, None, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_LINEAR)
        img_h, img_w = resized_img.shape[:2]
        img = np.float32(resized_img)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        if torch.cuda.is_available():
            img = img.cuda()
        return img, img_h, img_w

    def predict(self, frame):
        """
        predict coordinate of face in image
        @frame: original opencv frame
        :return: list of bounding boxes and confident scores in format of [x1, y1, x2, y2, score]
        """
        img, img_h, img_w = self.pre_process(frame)
        loc, conf, landmarks = self._model(img)
        prior_box = PriorBox(cfg=self._cfg, image_size=(img_h, img_w))
        priors = prior_box.forward()
        resized_size = torch.Tensor([img_w, img_h, img_w, img_h])
        if torch.cuda.is_available():
            priors = priors.cuda()
            resized_size = resized_size.cuda()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self._cfg['variance'])
        boxes = boxes * resized_size / self._scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landmarks = decode_landm(landmarks.data.squeeze(0), prior_data, self._cfg['variance'])
        lm_scale = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                 img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                 img.shape[3], img.shape[2]])
        if torch.cuda.is_available():
            lm_scale = lm_scale.cuda()
        landmarks = landmarks * lm_scale / self._scale
        landmarks = landmarks.cpu().numpy()
        # ignore low scores

        indexes = np.where(scores > self._confident)[0]
        boxes = boxes[indexes]
        scores = scores[indexes]
        landmarks = landmarks[indexes]

        # keep top-K before NMS
        order = scores.argsort()[::-1][: 100]
        boxes = boxes[order]
        scores = scores[order]
        landmarks = landmarks[order]

        # do NMS
        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(detections, self._mns_thresh)
        detections = detections[keep, :]
        landmarks = landmarks[keep]

        # keep top-K faster NMS
        detections = detections[: 50, :]
        landmarks = landmarks[:50, :]
        b_outputs = []
        for det in detections:
            x1, y1, x2, y2, score = det
            b_outputs.append([int(x1), int(y1), int(x2), int(y2), score])

        l_outputs = []
        for l in landmarks:
            l_outputs.append(list(map(int, l)))
        return b_outputs, l_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--trained_model_folder', default='./selected_weights_mobilenet/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--dataset_folder', default='/home/sonnn27/WorkSpace/Datasets/mgv_images/Gallery/',
                        help='path to test images')
    parser.add_argument('--save_folder', default='./mgv_txt_evaluate/mgv_txt_mobilenet/gallery//',
                        help='path to saved txt file')
    parser.add_argument('--backbone', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--confidence', default=0.8, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=500, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.2, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--input_w', default=640, type=int, help='keep_top_k')
    args = parser.parse_args()

    for weight_path in glob.glob(os.path.join(args.trained_model_folder, "*.pth")):
        print(weight_path)
        epoch = weight_path.split("_")[-1].replace(".pth", "")
        detector = Detection(weight=weight_path, model_type=args.backbone,
                             confident=args.confidence, mns_thresh=args.nms_threshold, input_w=args.input_w)
        save_path = os.path.join(args.save_folder, epoch)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for img_path in tqdm(glob.glob(os.path.join(args.dataset_folder, "*.jpeg"))):
            txt_name = img_path.split("/")[-1].replace(".jpeg", '.txt')
            img = cv2.imread(img_path)
            b_outputs, _ = detector.predict(img)
            with open(os.path.join(save_path, txt_name), 'a') as f:
                for box in b_outputs:
                    x1, y1, x2, y2, score = box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    f.write("face {} {} {} {} {}\n".format(score, x1, y1, x2-x1, y2-y1))
            cv2.imshow("a", img)
            cv2.waitKey(0)