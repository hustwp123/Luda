//
// Created by crabo on 2019/11/1.
//

#ifndef LEVELDB_CUDA_UTIL_H
#define LEVELDB_CUDA_UTIL_H

#include "cuda/cuda_common.h"

namespace leveldb {
namespace gpu {

enum {
    kKeyBufferSize = 32,    // KEY 占用的最大Size（包括8字节）
    kSharedKeys = 16,       // 每个SharedBlock中Key的数目
    kDataSharedCnt = 4,     // 每个DataBlock中SharedBlock的数目
    kBitsPerKey = 10,       // filter时候，每个Key所用bit位数
    kSharedPerSST = 256,    // 每个SST中有多少个SharedBlock
};

class Buffer;
class Slice;

__host__ __device__ void Memcpy(char *dst, const char *src, size_t n);

__host__ __device__ void EncodeFixed32(char *dst, uint32_t value);

__host__ __device__ void EncodeFixed64(char *dst, uint64_t value);

__host__ __device__ void PutFixed32(Buffer *dst, uint32_t value);

__host__ __device__ void PutFixed64(Buffer *dst, uint64_t value);

__host__ __device__ char *EncodeVarint32(char *dst, uint32_t v);

__host__ __device__ void PutVarint32(Buffer *dst, uint32_t v);

__host__ __device__ char *EncodeVarint64(char *dst, uint64_t v);

__host__ __device__ void PutVarint64(Buffer *dst, uint64_t v);

__host__ __device__ void PutLengthPrefixedSlice(Buffer *dst, const Slice &value);

__host__ __device__ int VarintLength(uint64_t v);

__host__ __device__ uint32_t DecodeFixed32(const char *ptr);

__host__ __device__ uint64_t DecodeFixed64(const char *ptr);

__host__ __device__
const char *GetVarint32PtrFallback(const char *p, const char *limit, uint32_t *value);

__host__ __device__
inline const char *GetVarint32Ptr(const char *p, const char *limit, uint32_t *value);

__host__ __device__ bool GetVarint32(Slice *input, uint32_t *value);

__host__ __device__
const char *GetVarint64Ptr(const char *p, const char *limit, uint64_t *value);

__host__ __device__ bool GetVarint64(Slice *input, uint64_t *value);

__host__ __device__
const char *GetLengthPrefixedSlice(const char *p, const char *limit, Slice *result);

__host__ __device__ bool GetLengthPrefixedSlice(Slice *input, Slice *result);

__host__ __device__ void EncodeValueOffset(uint32_t *offset, int Idx);
__host__ __device__ void DecodeValueOffset(uint32_t *offset, int *Idx);

__host__ __device__
const char* DecodeEntry(const char* p, const char* limit,
                        uint32_t* shared, uint32_t* non_shared,
                        uint32_t* value_length);
__host__ __device__
uint32_t Hash(const char* data, size_t n, uint32_t seed = 0xbc9f1d34);

}  // namespace gpu
}  // namespace leveldb

#endif //LEVELDB_UTIL_H
