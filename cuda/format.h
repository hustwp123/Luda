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

void BlockHandle::EncodeTo(Buffer* dst) {
    bool tmpb = (offset_ && size_);
    if(!tmpb) {
      fprintf(stderr, "ERROR! %s:%d:%s\n", __FILE__, __LINE__, __func__);
      int* purpose_crash = NULL; *purpose_crash = 1;
    }
    PutVarint64(dst, offset_);
    PutVarint64(dst, size_);
}

     
bool BlockHandle::DecodeFrom(Slice *input) {
    bool tmpb = GetVarint64(input, &offset_);
    if(!tmpb) {
      fprintf(stderr, "ERROR! %s:%d:%s\n", __FILE__, __LINE__, __func__);
      int* purpose_crash = NULL; *purpose_crash = 1;
    }
    tmpb = (GetVarint64(input, &size_));
    if(!tmpb) {
      fprintf(stderr, "ERROR! %s:%d:%s\n", __FILE__, __LINE__, __func__);
      int* purpose_crash = NULL; *purpose_crash = 1;
    }
    return true;
}

void Footer::EncodeTo(Buffer *dst) {
    bool tmpb = (dst->total_ == leveldb::Footer::kEncodedLength); // 20 + 20 + 8 = 48
    if(!tmpb) {
      fprintf(stderr, "ERROR! %s:%d:%s\n", __FILE__, __LINE__, __func__);
      int* purpose_crash = NULL; *purpose_crash = 1;
    }
    metaindex_handle_.EncodeTo(dst);
    index_handle_.EncodeTo(dst);

    dst->advance(dst->total_ - 8 - dst->size_);
    PutFixed32(dst, static_cast<uint32_t>(leveldb::kTableMagicNumber & 0xffffffffu));
    PutFixed32(dst, static_cast<uint32_t>(leveldb::kTableMagicNumber >> 32));
}

 
bool Footer::DecodeFrom(Slice *input) {
    const char* magic_ptr = input->data() + leveldb::Footer::kEncodedLength - 8;
    const uint32_t magic_lo = DecodeFixed32(magic_ptr);
    const uint32_t magic_hi = DecodeFixed32(magic_ptr + 4);
    const uint64_t magic = ((static_cast<uint64_t>(magic_hi) << 32) |
                            (static_cast<uint64_t>(magic_lo)));
    if (magic != leveldb::kTableMagicNumber) {
        return false;
    }

    bool tmpb = (metaindex_handle_.DecodeFrom(input));
    if(!tmpb) {
      fprintf(stderr, "ERROR! %s:%d:%s\n", __FILE__, __LINE__, __func__);
      int* purpose_crash = NULL; *purpose_crash = 1;
    }
    tmpb = (index_handle_.DecodeFrom(input));
    if(!tmpb) {
      fprintf(stderr, "ERROR! %s:%d:%s\n", __FILE__, __LINE__, __func__);
      int* purpose_crash = NULL; *purpose_crash = 1;
    }
    // fprintf(stderr, "XXXDBG cuda/format.h Footer::DecodeFrom() meta/idx_handle_ <off, size>: <%ld, %ld> <%ld, %ld>\n",
    //         metaindex_handle_.offset_, metaindex_handle_.size_, index_handle_.offset_, index_handle_.size_);

    return true;
}

}
}
#endif //LEVELDB_CUDA_FORMAT_H
