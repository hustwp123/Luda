//
// Created by crabo on 2019/11/6.
//

#ifndef LEVELDB_CUDA_FORMAT_H
#define LEVELDB_CUDA_FORMAT_H

#include "cuda/cuda_common.h"
#include "cuda/data.h"
#include "cuda/util.h"
#include "table/format.h"

namespace leveldb {
namespace gpu {
class BlockHandle { // the same as leveldb::BlockHandle
public:
    __host__ __device__ BlockHandle() :offset_(0), size_(0) {}
    __host__ __device__ ~BlockHandle() = default;

    __host__ __device__
    void EncodeTo(Buffer* dst) {
        assert(offset_ && size_);
        PutVarint64(dst, offset_);
        PutVarint64(dst, size_);
    }

    __host__ __device__
    bool DecodeFrom(Slice *input) {
        assert(GetVarint64(input, &offset_));
        assert(GetVarint64(input, &size_));
        return true;
    }

    uint64_t offset_;
    uint64_t size_;
};

class Footer {
public:
    __host__ __device__ Footer() {}
    __host__ __device__ ~Footer() = default;

    __host__ __device__
    void EncodeTo(Buffer *dst) {
        assert(dst->total_ == leveldb::Footer::kEncodedLength); // 20 + 20 + 8 = 48
        metaindex_handle_.EncodeTo(dst);
        index_handle_.EncodeTo(dst);

        dst->advance(dst->total_ - 8 - dst->size_);
        PutFixed32(dst, static_cast<uint32_t>(leveldb::kTableMagicNumber & 0xffffffffu));
        PutFixed32(dst, static_cast<uint32_t>(leveldb::kTableMagicNumber >> 32));
    }

    __host__ __device__
    bool DecodeFrom(Slice *input) {
        const char* magic_ptr = input->data() + leveldb::Footer::kEncodedLength - 8;
        const uint32_t magic_lo = DecodeFixed32(magic_ptr);
        const uint32_t magic_hi = DecodeFixed32(magic_ptr + 4);
        const uint64_t magic = ((static_cast<uint64_t>(magic_hi) << 32) |
                                (static_cast<uint64_t>(magic_lo)));
        if (magic != leveldb::kTableMagicNumber) {
            return false;
        }

        assert(metaindex_handle_.DecodeFrom(input));
        assert(index_handle_.DecodeFrom(input));

        return true;
    }

    BlockHandle metaindex_handle_;
    BlockHandle index_handle_;
};

}
}
#endif //LEVELDB_CUDA_FORMAT_H
