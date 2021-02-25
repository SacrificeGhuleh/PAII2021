#include <cudaDefs.h>
#include <timer.h>
#include <type_traits>
#include <benchmark.h>
#include <mat.cuh>

#define COMP_TYPE int

constexpr size_t width = 10; //cols
constexpr size_t height = 10; // rows
constexpr size_t numberOfTests = 10;


constexpr size_t widthInBytes = width * sizeof(COMP_TYPE); //cols
constexpr size_t heightInBytes = height * sizeof(COMP_TYPE); // rows
constexpr size_t length = width * height;
constexpr size_t lengthInBytes = length * sizeof(COMP_TYPE);

constexpr size_t TPB = 8;

// Print matrices, if matrices are small
constexpr bool printMatrices = (length <= 15 * 15);

cudaError_t err = cudaError_t::cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();


template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__global__ void fill(T *__restrict__ arr, const size_t pitchInElements) {
  const size_t col = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t row = blockDim.x * blockIdx.x + threadIdx.x;
  
  const size_t idx = row * pitchInElements + col;

//  printf("--------------------------\n");
//  printf("col: %d, row: %d idx: %d\n", col, row, idx);
//  printf("bd: %d, %d %d\n", blockDim.x, blockDim.y, blockDim.z);
//  printf("bi: %d, %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
//  printf("ti: %d, %d %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
//  printf("--------------------------\n\n");

//  printf("--------------------------\ncol: %d, row: %d idx: %d\nbd: %d, %d %d\nbi: %d, %d %d\nti: %d, %d %d\n--------------------------\n\n",
//         col, row, idx,
//         blockDim.x, blockDim.y, blockDim.z,
//         blockIdx.x, blockIdx.y, blockIdx.z,
//         threadIdx.x, threadIdx.y, threadIdx.z);
//
  
  if ((row < height) && (col < width)) {
    arr[idx] = col * height + row;
  }
}


template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__global__ void increment(T *__restrict__ arr, const size_t pitchInElements) {
  const size_t col = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t row = blockDim.x * blockIdx.x + threadIdx.x;
  
  const size_t idx = row * pitchInElements + col;
  
  if ((row < height) && (col < width)) {
//    arr[idx] = arr[idx] << 1;
    arr[idx] += 1;
  }
}

template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__global__ void fillMat(Mat<T> arr) {
  const size_t row = arr.getRowIdx(blockDim, blockIdx, threadIdx);
  const size_t col = arr.getColIdx(blockDim, blockIdx, threadIdx);
  if (arr.inBounds(row, col)) {
    arr.atDevice(row, col) = col * arr.getRows() + row;
  }
}


template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
__global__ void incrementMat(Mat<T> arr) {
  const size_t row = arr.getRowIdx(blockDim, blockIdx, threadIdx);
  const size_t col = arr.getColIdx(blockDim, blockIdx, threadIdx);
  if (arr.inBounds(row, col)) {
    arr.atDevice(row, col)++;
//    arr.atDevice(row, col) = arr.atDevice(row, col) << 1;
  }
}

void testMat(int nrOfTests) {
  printf(" --- \n MATRIX\n --- \n");
  Mat<COMP_TYPE> myMat(height, width);
  
  printf("Pitch: %zu B (%zu items)\n", myMat.getPitchInBytes(), myMat.getPitchInElements());
  
  dim3 dimBlock(TPB, TPB, 1);
  dim3 dimGrid(getNumberOfParts(height, TPB), getNumberOfParts(width, TPB), 1);
  
  printf("Dim block: %d, %d, %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
  printf("Dim  grid: %d, %d, %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
  
  if (printMatrices) {
    //If printing is enabled, print filled result before benchmark
    
    fillMat<<<dimGrid, dimBlock>>>(myMat);
    printLastCudaError("ERROR: ");
    checkCudaErrors(cudaDeviceSynchronize());
    
    myMat.checkDeviceMatrix("%4d ", "Matrix filled data");
  }
  
  auto testFunc = [&]() {
    
    fillMat<<<dimGrid, dimBlock>>>(myMat);
    printLastCudaError("ERROR: ");
    checkCudaErrors(cudaDeviceSynchronize());
    
    incrementMat<<<dimGrid, dimBlock>>>(myMat);
    printLastCudaError("ERROR: ");
    checkCudaErrors(cudaDeviceSynchronize());
    
  };
  
  gpubenchmark::print_time("My mat test", testFunc, nrOfTests);
  
  if (printMatrices) {
    myMat.checkHostMatrix(true, "%4d ", "Matrix incremented data");
  }
  
  myMat.free();
}

void testNative(int nrOfTests) {
  printf(" --- \n NATIVE\n --- \n");
  
  COMP_TYPE *deviceArray = nullptr;
  size_t pitchInBytes = 0;
  checkCudaErrors(cudaMallocPitch((void **) &deviceArray, &pitchInBytes, widthInBytes, height));
  const size_t pitchInElements = pitchInBytes / sizeof(COMP_TYPE);
  
  printf("Pitch: %zu B (%zu items)\n", pitchInBytes, pitchInElements);
  
  dim3 dimBlock(TPB, TPB, 1);
  dim3 dimGrid(getNumberOfParts(height, TPB), getNumberOfParts(width, TPB), 1);
  
  printf("Dim block: %d, %d, %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
  printf("Dim  grid: %d, %d, %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
  
  if (printMatrices) {
    //If printing is enabled, print filled result before benchmark
    
    fill<<<dimGrid, dimBlock>>>(deviceArray, pitchInElements);
    printLastCudaError("ERROR: ");
    checkCudaErrors(cudaDeviceSynchronize());
    
    checkDeviceMatrix(deviceArray, pitchInBytes, height, width, "%4d ", "Native filled data");
  }
  
  auto testFunc = [&]() {
    fill<<<dimGrid, dimBlock>>>(deviceArray, pitchInElements);
    printLastCudaError("ERROR: ");
    checkCudaErrors(cudaDeviceSynchronize());
    
    increment<<<dimGrid, dimBlock>>>(deviceArray, pitchInElements);
    printLastCudaError("ERROR: ");
    checkCudaErrors(cudaDeviceSynchronize());
    
  };
  
  gpubenchmark::print_time("Native test", testFunc, nrOfTests);
  
  COMP_TYPE *hostArray = static_cast<COMP_TYPE *>(::operator new(pitchInBytes * height));
  if (printMatrices) {
    cudaMemcpy2D(hostArray, pitchInBytes, deviceArray, pitchInBytes, widthInBytes, height, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    checkHostMatrix(hostArray, pitchInBytes, height, width, "%4d ", "Native incremented data");
  }
  
  SAFE_DELETE_CUDA(deviceArray);
  SAFE_DELETE_ARRAY(hostArray);
}

int main() {
  initializeCUDA(deviceProp);
  
  printf("testing %zu x %zu matrices\n", width, height);
//  printSize(length * sizeof(lengthInBytes));
  
  testNative(numberOfTests);
  testMat(numberOfTests);
}
