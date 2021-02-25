/*
* This is a personal academic project. Dear PVS-Studio, please check it.
* PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com
*/
#ifndef MAT_CUH
#define MAT_CUH

#include <cudaDefs.h>

template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
struct Mat {
public:
  typedef T Type;
  
  Mat(size_t rows, size_t cols) :
      rows_(rows),
      cols_(cols),
      rowsInBytes_(rows * sizeof(Type)),
      colsInBytes_(cols * sizeof(Type)),
      deviceArray_(nullptr),
      hostArray_(nullptr) {
    // Allocate device data
    checkCudaErrors(cudaMallocPitch((void **) &deviceArray_, &pitchInBytes_, colsInBytes_, rows_));
    pitchInElements_ = pitchInBytes_ / sizeof(Type);
    
    printf("Pitch: %zu B (%zu items)\n", pitchInBytes_, pitchInElements_);
    
    // Allocate host data
    hostArray_ = static_cast<Type *>(::operator new(pitchInBytes_ * rows_));
  }
  
  void free() {
    SAFE_DELETE_CUDA(deviceArray_);
    SAFE_DELETE_ARRAY(hostArray_);
  }
  
  __device__ inline size_t getRowIdx(const dim3 &bd, const dim3 &bi, const dim3 &ti) const { return bd.x * bi.x + ti.x; }
  
  __device__ inline size_t getColIdx(const dim3 &bd, const dim3 &bi, const dim3 &ti) const { return bd.y * bi.y + ti.y; }
  
  __device__ __host__ inline size_t getIdx(const size_t row, const size_t col) const { return row * pitchInElements_ + col; }
  
  __device__ __host__  bool inBounds(const size_t row, const size_t col) const { return (row < rows_) && (col < cols_); }
  
  __device__ inline Type &atDevice(const dim3 &bd, const dim3 &bi, const dim3 &ti) {
//    const size_t row = bd.x * bi.x + ti.x;
//    const size_t col = bd.y * bi.y + ti.y;
//    return (deviceArray_ + (row * pitchInElements_))[col];
    const size_t row = getRowIdx(bd, bi, ti);
    const size_t col = getRowIdx(bd, bi, ti);
    return atDevice(row, col);
  }
  
  __device__ inline Type &atDevice(const size_t row, const size_t col) {
//      assert((row < rows_) && (col < cols_))
//    return (deviceArray_ + (row * pitchInElements_))[col];
    const size_t idx = row * pitchInElements_ + col;
    return atDevice(idx);
  }
  
  __device__ inline Type &atDevice(const size_t idx) {
    return deviceArray_[idx];
  }
  
  __host__ inline Type &atHost(const size_t row, const size_t col) {
//      assert((row < rows_) && (col < cols_))
    
    const size_t idx = row * pitchInElements_ + col;
    return atHost(idx);
  }
  
  __host__ inline Type &atHost(const size_t idx) {
    return hostArray_[idx];
  }
  
  // Getters
  __device__ __host__ inline size_t getRows() const {
    return rows_;
  }
  
  __host__ inline void download() {
    cudaMemcpy2D(hostArray_, pitchInBytes_, deviceArray_, pitchInBytes_, colsInBytes_, rows_, cudaMemcpyKind::cudaMemcpyDeviceToHost);
  }
  
  __host__ inline void upload() {
    cudaMemcpy2D(deviceArray_, pitchInBytes_, hostArray_, pitchInBytes_, colsInBytes_, rows_, cudaMemcpyKind::cudaMemcpyHostToDevice);
  }
  
  __host__ inline void checkDeviceMatrix(const char *format = "%f ", const char *message = "") {
    ::checkDeviceMatrix(deviceArray_, pitchInBytes_, rows_, cols_, format, message);
  }
  
  __host__ inline void checkDeviceMatrix(const bool uploadToDevice, const char *format = "%f ", const char *message = "") {
    if (uploadToDevice)
      upload();
    checkDeviceMatrix(format, message);
  }
  
  __host__ inline void checkHostMatrix(const char *format = "%f ", const char *message = "") {
    ::checkHostMatrix(hostArray_, pitchInBytes_, rows_, cols_, format, message);
  }
  
  __host__ inline void checkHostMatrix(const bool downloadFromDevice, const char *format = "%f ", const char *message = "") {
    if (downloadFromDevice)
      download();
    checkHostMatrix(format, message);
  }
  
  __device__ __host__
  inline size_t getCols() const {
    return cols_;
  }
  
  __device__ __host__
  inline size_t getRowsInBytes() const {
    return rowsInBytes_;
  }
  
  __device__ __host__
  inline size_t getColsInBytes() const {
    return colsInBytes_;
  }
  
  __device__ inline Type *getDeviceArray() {
    return deviceArray_;
  }
  
  __host__ inline Type *getHostArray() {
    return hostArray_;
  }
  
  __device__ __host__ inline size_t getPitchInBytes() const {
    return pitchInBytes_;
  }
  
  __device__ __host__ inline size_t getPitchInElements() const {
    return pitchInElements_;
  }

private:
  const size_t rows_;
  const size_t cols_;
  
  const size_t rowsInBytes_;
  const size_t colsInBytes_;
  
  Type *deviceArray_;
  Type *hostArray_;
  
  size_t pitchInBytes_;
  size_t pitchInElements_;
};


#endif //MAT_CUH
