#include <torch/torch.h>
#include <iostream>

int main() {
	torch::Device device = torch::kCPU;
	if (torch::cuda::is_available()) {
  	std::cout << "CUDA is available! Training on GPU." << std::endl;
  	device = torch::kCUDA;
	}
  torch::Tensor tensor = torch::eye(3).to(device);
  std::cout << tensor << std::endl;
}
