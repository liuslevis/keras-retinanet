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

import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

def main():
    backbone_name, model_path = 'mobilenet128', './snapshots/mobilenet128_1_csv_20.h5'
    # backbone_name, model_path = 'mobilenet128', './snapshots/mobilenet128_0.5_csv_20.h5'
    # backbone_name, model_path = 'resnet50', './snapshots/resnet50_csv_19.h5'
    model = load_model(model_path, backbone_name, convert=True)
    # print(model.summary())
    labels_to_names = {0:'bilibili', 1:'douyin', 2:'qq', 3:'qr', 4:'toutiao', 5:'weibo', 6:'weixin', 7:'xigua', 8:'yy'}
    
    images = [
        read_image_bgr('../FLVGetter/logo_train/images/2197526474-201808241033.sec64.jpg'),
        read_image_bgr('../FLVGetter/logo_train/images/129841-201808241034.sec2.jpg'),
    ] + [
        read_image_bgr(f'../FLVGetter/logo_test/{i}.jpg') for i in range(1, 14)
    ]

    # cap = cv2.VideoCapture('../FLVGetter/logo_test/1446750313-201808172105.flv')
    # images = []
    # while True:
    #     succ, image = cap.read()
    #     if not succ: break
    #     images.append(image)

    for image in images:
        # copy to draw on    
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
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break
                
            color = label_color(label)
            
            b = box.astype(int)
            draw_box(draw, b, color=color)
            
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)
        

        cv2.imshow('(q)uit', cv2.resize(draw, (800, 450)))
        key = cv2.waitKey(0)
        if key == ord('q'):
            return

if __name__ == '__main__':
    main()
