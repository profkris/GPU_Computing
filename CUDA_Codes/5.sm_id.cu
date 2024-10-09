#include <stdio.h>
#include <stdint.h>

static __device__ __inline__ uint32_t __mysmid(){
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;}

static __device__ __inline__ uint32_t __mywarpid(){
  uint32_t warpid;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;}

static __device__ __inline__ uint32_t __mylaneid(){
  uint32_t laneid;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;}


__global__ void mykernel(){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  printf("I am thread %d | my SM ID is %d | my warp ID is %d | and my warp lane is %d\n", idx, __mysmid(), __mywarpid(), __mylaneid());
}

int main(){

  mykernel<<<41,2>>>();
  cudaDeviceSynchronize();
  return 0;
}
