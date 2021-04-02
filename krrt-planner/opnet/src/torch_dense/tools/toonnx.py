import torch
import torchvision
import numpy as np
import os
import onnx

import caffe2.python.onnx.backend as backend
import numpy as np
dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
model = torchvision.models.alexnet(pretrained=True).cuda()

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = [ "input1" ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)


# # Load the ONNX model
# print(os.getcwd())
# model = onnx.load("alexnet.onnx")

# # Check that the IR is well formed
# onnx.checker.check_model(model)

# # Print a human readable representation of the graph
# onnx.helper.printable_graph(model.graph)

# rep = backend.prepare(model, device="CPU") # or "CPU"
# # For the Caffe2 backend:
# #     rep.predict_net is the Caffe2 protobuf for the network
# #     rep.workspace is the Caffe2 workspace for the network
# #       (see the class caffe2.python.onnx.backend.Workspace)
# outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))
# # To run networks with more than one input, pass a tuple
# # rather than a single numpy ndarray.
# print(outputs[0])