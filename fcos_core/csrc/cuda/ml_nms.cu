// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>
#define BOX_DIM 8
/*
a[0 ~ 3]  : regressed box left, top, right, bottom
a[4]      : confidence of box
a[5]      : label
a[6 ~ 7]  : loction of regressed posiiton (x, y)
a[8]      : centerness of box
*/

int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
  if (a[5] != b[5]) {
    return 0.0;
  }
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}

__device__ inline int devPredLoc(float const * const a, float const * const b) {
  if (a[5] != b[5]){
    return 0;
  }
  /*
  //a box's center and b box's center
  //check centerness of each other and if they are same they are detecting same object
  float a_hor = (a[0] + a[2])/2, a_ver = (a[1] + a[3])/2;
  float b_hor = (b[0] + b[2])/2, b_ver = (b[1] + b[3])/2;

  float center_dist = (a_hor - b_hor)*(a_hor - b_hor) + (a_ver - b_ver) * (a_ver - b_ver);
  center_dist = sqrt(center_dist);
  
  return 0;
  */

  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  int retA=1, retB = 1;
  if(a[6] > left && a[6] < right && a[7] < bottom && a[7] > top){
    retA = 0;
  }
  if(b[6] > left && b[6] < right && b[7] < bottom && b[7] > top){
    retB = 0;
  }
  //return 0 -> nms out, return 1 -> nms stay ???
  return 1;
  return (retA + retB) == 0;
}

__global__ void ml_nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * BOX_DIM];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * BOX_DIM + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * BOX_DIM + 0];
    block_boxes[threadIdx.x * BOX_DIM + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * BOX_DIM + 1];
    block_boxes[threadIdx.x * BOX_DIM + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * BOX_DIM + 2];
    block_boxes[threadIdx.x * BOX_DIM + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * BOX_DIM + 3];
    block_boxes[threadIdx.x * BOX_DIM + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * BOX_DIM + 4];
    block_boxes[threadIdx.x * BOX_DIM + 5] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * BOX_DIM + 5];
    block_boxes[threadIdx.x * BOX_DIM + 6] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * BOX_DIM + 6];
    block_boxes[threadIdx.x * BOX_DIM + 7] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * BOX_DIM + 7];


  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * BOX_DIM;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if ((devIoU(cur_box, block_boxes + i * BOX_DIM) > nms_overlap_thresh) && devPredLoc(cur_box, block_boxes + i* BOX_DIM)) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

// boxes is a N x BOX_DIM tensor
at::Tensor ml_nms_cuda(const at::Tensor boxes, float nms_overlap_thresh) {
  using scalar_t = float;
  AT_ASSERTM(boxes.type().is_cuda(), "boxes must be a CUDA tensor");
  auto scores = boxes.select(1, 4);
  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t);

  int boxes_num = boxes.size(0);

  const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

  scalar_t* boxes_dev = boxes_sorted.data<scalar_t>();

  THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState

  unsigned long long* mask_dev = NULL;
  //THCudaCheck(THCudaMalloc(state, (void**) &mask_dev,
  //                      boxes_num * col_blocks * sizeof(unsigned long long)));

  mask_dev = (unsigned long long*) THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));

  dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
              THCCeilDiv(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  ml_nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  THCudaCheck(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  at::Tensor keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data<int64_t>();

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  THCudaFree(state, mask_dev);
  // TODO improve this part
  return std::get<0>(order_t.index({
                       keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(
                         order_t.device(), keep.scalar_type())
                     }).sort(0, false));
}
