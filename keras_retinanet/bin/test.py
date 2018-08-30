#!/usr/bin/env python3

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from keras_retinanet.models import load_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import os

def main():
    backbone_name, model_path = 'mobilenet128', './model/ad/ad.mobilenet128_1_csv_20.h5'
    # backbone_name, model_path = 'mobilenet128', './model/mobilenet128_1_csv_10.h5'
    # backbone_name, model_path = 'resnet50', './model/resnet50_csv_20.h5'

    model = load_model(model_path, backbone_name, convert=True)
    # print(model.summary())
    # labels_to_names = {0:'aipai', 1:'bilibili', 2:'douyin', 3:'douyu', 4:'huya', 5:'qq', 6:'qr', 7:'taobao', 8:'toutiao', 9:'wangyi', 10:'weibo', 11:'weixin', 12:'xigua', 13:'yy', }
    labels_to_names = {0:'ad'}
    paths = []
    for root, dirs, files in os.walk(f'../FLVGetter/ad_test/images'):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpg'):
                paths.append(str(Path(root) / file))
    paths = sorted(paths)

    # cap = cv2.VideoCapture('../FLVGetter/logo_test/1446750313-201808172105.flv')
    # images = []
    # while True:
    #     succ, image = cap.read()
    #     if not succ: break
    #     images.append(image)

    for path in paths:
        image = read_image_bgr(path)
        draw = image.copy()

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale

        # visualize detections
        captions = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.2:
                break
                
            color = label_color(label)
            
            b = box.astype(int)
            draw_box(draw, b, color=color)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            captions.append(caption)
            draw_caption(draw, b, caption)
        
        print(path, ','.join(captions))

        winname = '(q)uit'
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 100, 100)
        cv2.imshow(winname, resize_image(draw)[0])
        key = cv2.waitKey(0)

        if key == ord('q'):
            return

if __name__ == '__main__':
    main()