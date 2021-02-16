#include <cudaDefs.h>
#include <vector>
#include <rng.h>

cudaError_t err = cudaError_t::cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

/// \brief Threads per block
constexpr unsigned int TPB = 256;

/// \brief Number of blocks
constexpr unsigned int NOB = 16;

/// \brief Memory block per thread block
constexpr unsigned int MBPTB = 2;


/// \brief
/// \param a
/// \param b
/// \param length
/// \param c
/// \note  __restrict__  Avoids pointers aliasing
__global__ void add(const int *__restrict__ a, const int *__restrict__ b, const unsigned int length, int *__restrict__ c) {
  //TODO c[i] = a[i] + b[i]
  
  const unsigned int offset = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (offset < length) {
    c[offset] = a[offset] + b[offset];
  }
}

int main() {
  initializeCUDA(deviceProp);

//  constexpr unsigned int length = 1U << 20U;
  constexpr unsigned int length = 100U;
  constexpr unsigned int sizeInBytes = length * sizeof(int);
  
  /// Allocate host data (on CPU)
  int *host_a = static_cast<int *>(::operator new(sizeInBytes));
  int *host_b = static_cast<int *>(::operator new(sizeInBytes));
  int *host_c = static_cast<int *>(::operator new(sizeInBytes));

//  std::vector<unsigned int> host_a(length);
//  std::vector<unsigned int> host_b(length);
//  std::vector<unsigned int> host_c(length);
  
  /// Initialize data

//  constexpr float maxNum = INT_MAX / 2;
  constexpr float maxNum = 255;
  
  for (size_t i = 0; i < length; i++) {
    host_a[i] = static_cast<int>(rng(0.f, maxNum));
    host_b[i] = static_cast<int>(rng(0.f, maxNum));
  }
  
  /// Allocate device data (on GPU)
  int *device_a = nullptr;
  int *device_b = nullptr;
  int *device_c = nullptr;
  
  checkCudaErrors(cudaMalloc((void **) &device_a, sizeInBytes));
  checkCudaErrors(cudaMalloc((void **) &device_b, sizeInBytes));
  checkCudaErrors(cudaMalloc((void **) &device_c, sizeInBytes));
  
  /// Copy data host -> device
  checkCudaErrors(cudaMemcpy(device_a, host_a, sizeInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_b, host_b, sizeInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
  
  checkDeviceMatrix(device_a, sizeInBytes, 1, length, "%d ", "Device A");
  checkDeviceMatrix(device_b, sizeInBytes, 1, length, "%d ", "Device B");
  checkDeviceMatrix(device_c, sizeInBytes, 1, length, "%d ", "Device C");
  
  /// Prepare grid and blocks
  dim3 dimBlock(TPB, 1, 1);

//  dim3 dimGrid(NOB, 1, 1);                           // What about data?
//  dim3 dimGrid(getNumberOfParts(length, TPB), 1, 1); // Great number of blocks
  dim3 dimGrid(getNumberOfParts(length, TPB * MBPTB), 1, 1);
  
  /// Call kernel
  add<<<dimGrid, dimBlock>>>(device_a, device_b, length, device_c); // ATTENTION, always pass device pointers
  printLastCudaError("ERROR: ");
  checkCudaErrors(cudaDeviceSynchronize());
  checkDeviceMatrix(device_c, sizeInBytes, 1, length, "%d ", "Device C");
  
  
  /// Copy data device -> host
  checkCudaErrors(cudaMemcpy(host_c, device_c, sizeInBytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
  checkHostMatrix(host_c, sizeInBytes, 1, length, "%d ", "Host C");
  
  
  /// Free memory
  SAFE_DELETE_ARRAY(host_a);
  SAFE_DELETE_ARRAY(host_b);
  SAFE_DELETE_ARRAY(host_c);
  
  SAFE_DELETE_CUDA(device_a);
  SAFE_DELETE_CUDA(device_b);
  SAFE_DELETE_CUDA(device_c);
}