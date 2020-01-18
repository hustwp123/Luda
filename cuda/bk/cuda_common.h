#ifndef LEVELDB_CUDA_COMMON_H
#define LEVELDB_CUDA_COMMON_H

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

//#define __device__
//#define __host__
//#define __global__
//#define __CUDA_DEBUG

#define CUDA_MAX_COMPACTION_FILES (25)
#define __SST_SIZE (4 * 1024 * 1024)
#define __MIN_KEY_SIZE (256) // 这里假设Value最小为256字节，当然可以更改
#define CUDA_MAX_KEY_PER_SST (__SST_SIZE / __MIN_KEY_SIZE)
#define CUDA_MAX_GDI_PER_SST (CUDA_MAX_KEY_PER_SST / 16)
#define CUDA_MAX_KEYS_COMPACTION (CUDA_MAX_KEY_PER_SST * CUDA_MAX_COMPACTION_FILES)

namespace leveldb {
namespace gpu {

/* fake API */
//__host__ void cudaMalloc(void** devPtr, size_t size);
//__host__ void cudaFree(void* devPtr);
//__host__ void cudaMemcpy(void *dst, const void *src, size_t cnt);

}
}

#endif //LEVELDB_CUDA_COMMON_H
