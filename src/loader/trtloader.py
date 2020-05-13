import numpy as np
import os
import cv2 as cv
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf

def load_model(model_name):
    
    trt_output_file = './models/{}_trt.pb'.format(model_name)

    # trt_graph = tf.compat.v1.GraphDef()
    trt_graph = tf.GraphDef()

    if os.path.exists(trt_output_file):
        print('Loading model {}...'.format(trt_output_file))
        # with tf.io.gfile.GFile(trt_output_file, 'rb') as f:
        with tf.gfile.GFile(trt_output_file, 'rb') as f:
            trt_graph.ParseFromString(f.read())
            print('{} loaded.'.format(trt_output_file))
    else:
        # Lazy load these dependencies
        import sys
        sys.path.insert(1, '/')
        from tf_trt_models.detection import download_detection_model
        from tf_trt_models.detection import build_detection_graph
        
        config_path, checkpoint_path = download_detection_model(
            model_name, './models/')

        frozen_graph, input_names, output_names = build_detection_graph(
            config=config_path,
            checkpoint=checkpoint_path
        )

        print('Converting {} to trt..'.format(model_name))
        trt_graph = trt.create_inference_graph(
            input_graph_def=frozen_graph,
            outputs=output_names,
            max_batch_size=1,
            max_workspace_size_bytes=1 << 25,
            precision_mode='FP16',
            minimum_segment_size=50
        )
        with open(trt_output_file, 'wb') as f:
            f.write(trt_graph.SerializeToString())
            print('{} saved.'.format(trt_output_file    ))

    return trt_graph