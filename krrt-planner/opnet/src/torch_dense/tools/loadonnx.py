# import onnx

# # Load the ONNX model
# model = onnx.load("alexnet.onnx")

# # Check that the IR is well formed
# onnx.checker.check_model(model)

# # Print a human readable representation of the graph
# onnx.helper.printable_graph(model.graph)

# onnx.checker.check_model(model)

# # ...continuing from above
# import caffe2.python.onnx.backend as backend
# import numpy as np

# rep = backend.prepare(model, device="CUDA:0") # or "CPU"
# # For the Caffe2 backend:
# #     rep.predict_net is the Caffe2 protobuf for the network
# #     rep.workspace is the Caffe2 workspace for the network
# #       (see the class caffe2.python.onnx.backend.Workspace)
# outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))
# # To run networks with more than one input, pass a tuple
# # rather than a single numpy ndarray.
# print(outputs[0])

######################## 
from __future__ import division
from __future__ import print_function

from IPython import embed

import numpy as np
# import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys, os
import common

# import gc
import time

import data_util

import sys, os
# sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

TRT_LOGGER = trt.Logger()

INPUTS = OUTPUTS = BINDINGS = STREAM = CONTEXT = None
SHAPE = [1,80,80,32]

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = SHAPE
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    # if os.path.exists(engine_file_path):
    #     # If a serialized engine exists, use it instead of building an engine.
    #     print("Reading engine from file {}".format(engine_file_path))
    #     with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    #         return runtime.deserialize_cuda_engine(f.read())
    # else:
    return build_engine()

def  trt_inference(input):
    global INPUTS, OUTPUTS, BINDINGS, STREAM, CONTEXT
    INPUTS[0].host = input
    trt_outputs = common.do_inference_v2(CONTEXT, bindings=BINDINGS, inputs=INPUTS, outputs=OUTPUTS, stream=STREAM)

    output = trt_outputs[0].reshape(SHAPE[1], SHAPE[2], SHAPE[3])
    return output

def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""
    global INPUTS, OUTPUTS, BINDINGS, STREAM, CONTEXT

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:

    onnx_file_path = os.path.join(sys.path[0], '/home/wlz/catkin_ws/src/krrt-planner/opnet/models/no_surf_80_32.onnx')
    engine_file_path = os.path.join(sys.path[0], '/home/wlz/catkin_ws/src/krrt-planner/opnet/models/no_surf_80_32.trt')

    input = np.random.randn(1, SHAPE[1], SHAPE[2], SHAPE[3]).astype(np.float32)
    # Output shapes expected by the post-processor
    output_shapes = [SHAPE]
    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        print("GET ENGINE SUCCEED")
        
        CONTEXT = context
        INPUTS, OUTPUTS, BINDINGS, STREAM = common.allocate_buffers(engine)
        # inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        time0 = time.time()
        for i in range(10):
            # print('Running inference')
            output = trt_inference(input)
        print("prepocess time: %fs"%(time.time() - time0))
        
    print('done')

if __name__ == '__main__':
    main()
