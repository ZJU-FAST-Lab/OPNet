from __future__ import division
from __future__ import print_function

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys, os
import common
import numpy as np

TRT_LOGGER = trt.Logger()

class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class trt_runner:
    
    def __init__(self, onnx_path, engine_path, dimx=80, dimy=80, dimz=48):

        self.output_shape = [1, dimx, dimy, dimz]
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.cuda_ctx = cuda.Device(0).make_context()
        self.engine = self.get_engine(onnx_path, engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        print("BUIND ENGINE SUCCEED")

    def get_engine(self, onnx_file_path, engine_file_path=""):
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
        def build_engine():
            """Takes an ONNX file and creates a TensorRT engine to run inference with"""
            with trt.Builder(self.trt_logger) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
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
            network.get_input(0).shape = self.output_shape
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

        if os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine()

    def inference(self, input):

        trt_outputs = []

        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing.
        self.inputs[0].host = input

        if self.cuda_ctx:
            self.cuda_ctx.push()

        self.bindings = [int(i) for i in self.bindings]

        context = self.context
        bindings = self.bindings
        inputs = self.inputs
        outputs = self.outputs
        stream = self.stream

        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
            # Run inference.
        context.execute_async_v2(bindings=self.bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        trt_outputs = [out.host for out in outputs]
        output = trt_outputs[0].reshape(self.output_shape[1:]) # [x,y,z]
        print("GOT OUTPUT")
        del context

        if self.cuda_ctx:
            self.cuda_ctx.pop()

        # self.inputs[0].host = input
        # trt_outputs = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs,\
        #      outputs=self.outputs, stream=self.stream)
        return output
    
    def __del__(self):
        """Free CUDA memories."""
        if self.cuda_ctx:
            self.cuda_ctx.pop()
        del self.outputs
        del self.inputs
        del self.stream
        

# def get_engine(onnx_file_path, engine_file_path=""):
#     """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
#     def build_engine():
#         """Takes an ONNX file and creates a TensorRT engine to run inference with"""
#         with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
#             builder.max_workspace_size = 1 << 28 # 256MiB
#             builder.max_batch_size = 1
#             # Parse model file
#             if not os.path.exists(onnx_file_path):
#                 print('ONNX file {} not found'.format(onnx_file_path))
#                 exit(0)
#             print('Loading ONNX file from path {}...'.format(onnx_file_path))
#             with open(onnx_file_path, 'rb') as model:
#                 print('Beginning ONNX file parsing')
#                 if not parser.parse(model.read()):
#                     print ('ERROR: Failed to parse the ONNX file.')
#                     for error in range(parser.num_errors):
#                         print (parser.get_error(error))
#                     return None
#             # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
#             network.get_input(0).shape = [1, 80, 80, 48]
#             print('Completed parsing of ONNX file')
#             print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
#             engine = builder.build_cuda_engine(network)
#             print("Completed creating Engine")
#             with open(engine_file_path, "wb") as f:
#                 f.write(engine.serialize())
#             return engine

#     if os.path.exists(engine_file_path):
#         # If a serialized engine exists, use it instead of building an engine.
#         print("Reading engine from file {}".format(engine_file_path))
#         with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#             return runtime.deserialize_cuda_engine(f.read())
#     else:
#         return build_engine()

# def  trt_inference(input):

#     global INPUTS, OUTPUTS, BINDINGS, STREAM, CONTEXT
#     INPUTS[0].host = input
#     trt_outputs = common.do_inference_v2(CONTEXT, bindings=BINDINGS, inputs=INPUTS, outputs=OUTPUTS, stream=STREAM)

#     output = trt_outputs[0].reshape(args.dimx, args.dimx, args.dimz)
#     return output

