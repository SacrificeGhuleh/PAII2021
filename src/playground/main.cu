#include <cudaDefs.h>
#include <stdio.h>
#include <timer.h>
#include <rng.h>
#include <vector>
//__global__ void myKernel() {
//  printf("Hello, world from the device!\n");
//}

int main() {
//  myKernel<<<1, 10>>>();
//  cudaDeviceSynchronize();
  
  constexpr size_t length = 1U << 20U;
  constexpr size_t sizeInBytes = length * sizeof(int);
  
  printf("Allocating %zu bytes\n", sizeInBytes);
  
  Timer t1;
  int *aArr = static_cast<int *>(::operator new(sizeInBytes));
  int *bArr = static_cast<int *>(::operator new(sizeInBytes));
  int *cArr = static_cast<int *>(::operator new(sizeInBytes));
//  memset(aArr, 0, sizeInBytes);
//  memset(bArr, 0, sizeInBytes);
//  memset(cArr, 0, sizeInBytes);
  const auto t1Elapsed = t1.elapsed();
  
  Timer t2;
  std::vector<int> aVec(length);
  std::vector<int> bVec(length);
  std::vector<int> cVec(length);
  const auto t2Elapsed = t2.elapsed();
  
  
  Timer t3;
  int *dArr = new int[sizeInBytes];
  int *eArr = new int[sizeInBytes];
  int *fArr = new int[sizeInBytes];
  const auto t3Elapsed = t1.elapsed();
  
  
  for (int i = 0; i < length; i++) {
    if (static_cast<int>(rng(0, 10000)) == 0) {
      printf("[%d][%d][%d][%d][%d][%d]\n", aArr[i], bArr[i], cArr[i], aVec[i], bVec[i], cVec[i]);
    }
  }
  printf("Operator new allocation: %f\n", t1Elapsed);
  printf("   Basic new allocation: %f\n", t3Elapsed);
  printf("      Vector allocation: %f\n", t2Elapsed);
  
  SAFE_DELETE_ARRAY(aArr);
  SAFE_DELETE_ARRAY(bArr);
  SAFE_DELETE_ARRAY(cArr);
  SAFE_DELETE_ARRAY(dArr);
  SAFE_DELETE_ARRAY(eArr);
  SAFE_DELETE_ARRAY(fArr);
}