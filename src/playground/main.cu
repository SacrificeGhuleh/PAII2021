#include <cudaDefs.h>
#include <stdio.h>
#include <timer.h>
#include <rng.h>
#include <vector>

struct testStruct {
  char *devPtr;
  int len;
  
  void free() {
    printf("Destructor");
    SAFE_DELETE_CUDA(devPtr);
  }
};

__global__ void myKernel(testStruct ts) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  printf("Hello, world from the device %d %d !\n", ts.len, idx);
  ts.devPtr[idx] = idx;
}

int main() {
  cudaError_t err = cudaError_t::cudaSuccess;
  cudaDeviceProp deviceProp = cudaDeviceProp();
  initializeCUDA(deviceProp);
  
  testStruct ts{};
  ts.devPtr = nullptr;
  ts.len = 9;
  
  cudaMalloc(&ts.devPtr, ts.len);
  
  myKernel<<<3, 3>>>(ts);
  printLastCudaError("ERROR: ");
  checkCudaErrors(cudaDeviceSynchronize());
  
  checkDeviceMatrix(ts.devPtr, ts.len, 1, ts.len, "%d ");
  
  ts.free();
//  SAFE_DELETE_CUDA(ts.devPtr);
  
}