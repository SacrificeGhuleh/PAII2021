#include <cudaDefs.h>
#include <vector>
#include <rng.h>
#include <timer.h>
#include <type_traits>
#include <benchmark.h>

cudaError_t err = cudaError_t::cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

/// \brief Threads per block
constexpr unsigned int TPB = 256;

/// \brief Number of blocks
constexpr unsigned int NOB = 16;

/// \brief Memory block per thread block
constexpr unsigned int MBPTB = 2;

constexpr size_t length = 1U << 28U;


/// \brief
/// \param a
/// \param b
/// \param length
/// \param c
/// \note  __restrict__  Avoids pointers aliasing
template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__global__ void add(const T *__restrict__ a, const T *__restrict__ b, const size_t length, T *__restrict__ c) {
////TODO c[i] = a[i] + b[i]
//
//  const size_t offset = (blockDim.x * blockIdx.x) + threadIdx.x;
////  const size_t offset = (threadIdx.x + blockIdx.x * TPB);
//
//  if (offset < length) {
//    c[offset] = a[offset] + b[offset];
//  }
  
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t skip = blockDim.x * gridDim.x;
  while (tid < length) {
    c[tid] = a[tid] + b[tid];
    tid += skip;
  }
}

class CudaTimer {
public:
  CudaTimer() : elapsed_{0}, running_{false} {
    cudaEventCreate(&startEvent_);
    cudaEventCreate(&stopEvent_);
  }
  
  ~CudaTimer() {
    cudaEventDestroy(startEvent_);
    cudaEventDestroy(stopEvent_);
  }
  
  void start(cudaStream_t stream = 0) {
    if (running_) {
      throw std::runtime_error("Timer is already running");
    }
    running_ = true;
    cudaEventRecord(startEvent_, stream);
  }
  
  float stop(cudaStream_t stream = 0) {
    if (!running_) {
      throw std::runtime_error("Timer is not running");
    }
    
    running_ = false;
    cudaEventRecord(stopEvent_, stream);
    cudaEventSynchronize(stopEvent_);
    cudaEventElapsedTime(&elapsed_, startEvent_, stopEvent_);
    return elapsed_;
  }

private:
  cudaEvent_t startEvent_;
  cudaEvent_t stopEvent_;
  bool running_;
  float elapsed_;
};

int main() {
  initializeCUDA(deviceProp);
  
  
  #define COMP_TYPE int

//  constexpr unsigned int length = 1000U;
  constexpr size_t sizeInBytes = length * sizeof(COMP_TYPE);
  
  /// Allocate host data (on CPU)
  auto *host_a = static_cast<COMP_TYPE *>(::operator new(sizeInBytes));
  auto *host_b = static_cast<COMP_TYPE *>(::operator new(sizeInBytes));
  auto *host_c = static_cast<COMP_TYPE *>(::operator new(sizeInBytes));
  
  /// Initialize data

//  constexpr float maxNum = INT_MAX / 2;
  constexpr float maxNum = 255.f;
  
  for (size_t i = 0; i < length; i++) {
    host_a[i] = static_cast<COMP_TYPE>(rng(0.f, maxNum));
    host_b[i] = static_cast<COMP_TYPE>(rng(0.f, maxNum));
  }
  
  /// Allocate device data (on GPU)
  COMP_TYPE *device_a = nullptr;
  COMP_TYPE *device_b = nullptr;
  COMP_TYPE *device_c = nullptr;
  
  checkCudaErrors(cudaMalloc((void **) &device_a, sizeInBytes));
  checkCudaErrors(cudaMalloc((void **) &device_b, sizeInBytes));
  checkCudaErrors(cudaMalloc((void **) &device_c, sizeInBytes));
  
  /// Copy data host -> device
  checkCudaErrors(cudaMemcpy(device_a, host_a, sizeInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_b, host_b, sizeInBytes, cudaMemcpyKind::cudaMemcpyHostToDevice));

//  checkDeviceMatrix(device_a, sizeInBytes, 1, length, "%d ", "Device A");
//  checkDeviceMatrix(device_b, sizeInBytes, 1, length, "%d ", "Device B");
//  checkDeviceMatrix(device_c, sizeInBytes, 1, length, "%d ", "Device C");
  
  /// Prepare grid and blocks
  dim3 dimBlock(TPB, 1, 1);

//  dim3 dimGrid(NOB, 1, 1);                           // What about data?
  dim3 dimGrid(getNumberOfParts(length, TPB), 1, 1); // Great number of blocks
//  dim3 dimGrid(getNumberOfParts(length, TPB * MBPTB), 1, 1);
  /// Call kernel

//  for (int i = 0; i < 10; i++) {
//    CudaTimer cuTimer;
//    cuTimer.start();
//    add<<<dimGrid, dimBlock>>>(device_a, device_b, length, device_c); // ATTENTION, always pass device pointers
//    const auto cuTimerElapsed = cuTimer.stop();
//    printf("Iteration: %d, Cuda Timer elapsed: %f\n", i, cuTimerElapsed);
//
//  }
  
  gpubenchmark::print_time("Add kernel", [&]() { add<<<dimGrid, dimBlock>>>(device_a, device_b, length, device_c); }, 10);
  
  printLastCudaError("ERROR: ");
  checkCudaErrors(cudaDeviceSynchronize());
//  checkDeviceMatrix(device_c, sizeInBytes, 1, length, "%d ", "Device C");
  
  
  /// Copy data device -> host
  checkCudaErrors(cudaMemcpy(host_c, device_c, sizeInBytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
//  checkHostMatrix(host_c, sizeInBytes, 1, length, "%d ", "Host C");
  
  
  /// Free memory
  SAFE_DELETE_ARRAY(host_a);
  SAFE_DELETE_ARRAY(host_b);
  SAFE_DELETE_ARRAY(host_c);
  
  SAFE_DELETE_CUDA(device_a);
  SAFE_DELETE_CUDA(device_b);
  SAFE_DELETE_CUDA(device_c);
}