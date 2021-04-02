import numpy as np
import cv2
import caffe2.python.onnx.backend as backend
import onnx



def load_and_run(file_path = "/home/wlz/catkin_ws/src/opnet/models/simple_80_40.onnx"):
    # Load the ONNX model
    model = onnx.load(file_path)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)

    # rep = backend.prepare(model, device="CUDA:0") # or "CPU"
    # outputs = rep.run(np.random.randn(1,80,80,40).astype(np.float32))

    # print(outputs[0].shape)

if __name__ == "__main__":
    load_and_run()

