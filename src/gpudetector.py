import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from loader.trtloader import load_model as trt_loader
from loader.tfloader import load_model as tf_loader
import time
import click

import tensorflow as tf
import cv2
import numpy as np


@click.command()
@click.option('--model-name', default='ssd_inception_v2_coco',
              help='The name of the pre-trained TF model to load. See: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md')
@click.option('--camera-id', default=1,
              help='The id of the camera to use.. You can discover the connected cameras by runnimg: ls -ltrh /dev/video*.')
@click.option('--trt-optimize', default=False,
              help='Setting this to True, the downloaded TF model will be converted to TensorRT model.', is_flag=True)
def detector(model_name, camera_id, trt_optimize):

    trt_graph = None

    if trt_optimize:
        trt_graph = trt_loader(model_name)
    else:
        trt_graph = tf_loader(model_name)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(trt_graph, name='')

    tf_input = tf_sess.graph.get_tensor_by_name('image_tensor:0')
    tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
    tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
    tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
    tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

    # video_capture = cv2.VideoCapture(camera_id)
    video_capture = cv2.VideoCapture("nvcamerasrc ! \
                                      video/x-raw(memory:NVMM), \
                                      width=(int)640, height=(int)480, \
                                      format=(string)I420, framerate=(fraction)30/1 \
                                      ! nvvidconv ! video/x-raw, format=(string)BGRx \
                                      ! videoconvert ! video/x-raw, format=(string)BGR \
                                      ! appsink")
    
    video_capture_result, frame = video_capture.read()
    camera_height, camera_width, channels = frame.shape

    start_time = time.time()

    while(video_capture_result):
        # Capture frame-by-frame
        video_capture_result, frame = video_capture.read()

        if video_capture_result == False:
            raise ValueError(
                'Error reading the frame from camera {camera_id}')

        # face detection and other logic goes here
        image_resized = cv2.resize(frame, (300, 300))

        scores, boxes, classes, num_detections = tf_sess.run(
            [tf_scores, tf_boxes, tf_classes, tf_num_detections],
            feed_dict={tf_input: image_resized[None, ...]})

        boxes = boxes[0]  # index by 0 to remove batch dimension
        scores = scores[0]
        classes = classes[0]
        num_detections = num_detections[0]

        for i in range(int(num_detections)):
            box = boxes[i] * np.array([camera_height,
                                       camera_width, camera_height, camera_width])
            box = box.astype(int)

            cv2.rectangle(frame, (box[1], box[0]), (box[3],
                                                   box[2]), color=(0, 255, 0), thickness=1)
            text = "{:.0f} | {}".format(scores[i]*100, str(int(classes[i])))
            cv2.putText(frame, text, (box[3]+10, box[2]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.putText(frame, "FPS:{:0.1f}".format(1.0 / (time.time() - start_time)),
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        start_time = time.time()

        cv2.imshow('Input', frame)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    detector()
