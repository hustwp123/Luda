//
// Created by crabo on 2019/11/1.
//

#ifndef LEVELDB_CUDA_DATA_H
#define LEVELDB_CUDA_DATA_H

#include "cuda/util.h"

namespace leveldb {
namespace gpu {
// std::string can not use in GPU-Device, so we use class Buffer
class Buffer {
public:
    __host__ __device__ Buffer(char *buf, size_t size) : base_(buf), total_(size), size_(0) {}

    __host__ __device__ char *now() {return base_ + size_;}
    __host__ __device__ char *data() { return base_; }
    __host__ __device__ void reset() { size_ = 0;}

    __host__ __device__
    inline void advance(int n) {
        assert(size_ + n <= total_);
        size_ += n;
    }

    __host__ __device__
    void append(const char *data, size_t size) {
        assert(size_ + size <= total_);
        Memcpy(base_ + size_, data, size);
        advance(size);
    }

    __host__ __device__ Buffer& operator=(const Buffer&) = default;

    char  *base_;
    size_t total_;
    size_t size_;
};

class Slice {
public:
    // Create an empty slice.
    __host__ __device__
    Slice() : data_(""), size_(0) {}

    // Create a slice that refers to d[0,n-1].
    __host__ __device__
    Slice(const char *d, size_t n) : data_(d), size_(n) {}

    __host__ __device__
    Slice(const char *d, size_t n, uint32_t off, int len) : data_(d), size_(n), offset_(off), value_len_(len) {}

    // Create a slice that refers to s[0,strlen(s)-1]
    __host__ __device__
    Slice(const char *s) : data_(s), size_(strlen(s)) {}

    // Intentionally copyable.
    __host__ __device__ Slice(const Slice &) = default;

    __host__ __device__ Slice &operator=(const Slice &) = default;

    // Return a pointer to the beginning of the referenced data
    __host__ __device__
    const char *data() const { return data_; }

    // Return the length (in bytes) of the referenced data
    __host__ __device__
    size_t size() const { return size_; }

    // Return true iff the length of the referenced data is zero
    __host__ __device__
    bool empty() const { return size_ == 0; }

    // Return the ith byte in the referenced data.
    // REQUIRES: n < size()
    __host__ __device__
    char operator[](size_t n) const {
        assert(n < size());
        return data_[n];
    }

    // Change this slice to refer to an empty array
    __host__ __device__
    void clear() {
        data_ = "";
        size_ = 0;
    }

    // Drop the first "n" bytes from this slice.
    __host__ __device__
    void remove_prefix(size_t n) {
        assert(n <= size());
        data_ += n;
        size_ -= n;
    }

    // Three-way comparison.  Returns value:
    //   <  0 iff "*this" <  "b",
    //   == 0 iff "*this" == "b",
    //   >  0 iff "*this" >  "b"
    __host__ __device__
    int compare(const Slice &b) const;
    __host__ __device__
    int internal_compare(const Slice &b) const;

    // Return true iff "x" is a prefix of "*this"
    __host__ __device__
    bool starts_with(const Slice &x) const {
        return ((size_ >= x.size_) && (memcmp(data_, x.data_, x.size_) == 0));
    }

public:
    uint32_t offset_;
    int value_len_;

private:
    const char *data_;
    size_t size_;
};

__host__ __device__
inline int Slice::compare(const Slice &b) const {
    assert(data_ && b.size());

    const size_t min_len = (size_ < b.size_) ? size_ : b.size_;
    int r = memcmp(data_, b.data_, min_len);
    if (r == 0) {
        if (size_ < b.size_)
            r = -1;
        else if (size_ > b.size_)
            r = +1;
    }
    return r;
}

__host__ __device__
inline int Slice::internal_compare(const leveldb::gpu::Slice &b) const {
    Slice user_key_a(data_, size_ - 8);
    Slice user_key_b(b.data(), b.size() - 8);

    int r = user_key_a.compare(user_key_b);
    if (r == 0) {
        const uint64_t anum = gpu::DecodeFixed64((const char *)user_key_a.data() + user_key_a.size());
        const uint64_t bnum = gpu::DecodeFixed64((const char *)user_key_b.data() + user_key_b.size());
        if (anum > bnum) {
            r = -1;
        } else if (anum < bnum) {
            r = +1;
        }
    }
    return r;
}

__host__ __device__
inline bool operator==(const Slice &x, const Slice &y) {
    return ((x.size() == y.size()) &&
            (memcmp(x.data(), y.data(), x.size()) == 0));
}

__host__ __device__
inline bool operator!=(const Slice &x, const Slice &y) { return !(x == y); }

struct GDI {      // Gpu Decode Info
    uint32_t offset;      // 考虑到SST的大小，使用int已经足够了，相对一个SST内存的偏移
    uint32_t kv_base_idx; // SST_kv[]中一个GPU线程添加数据的起始槽下标，解码出来的多个共享前缀KV依次往后放
    uint32_t limit;
};

struct SST_kv {          // SST sorted KV pair
    char ikey[kKeyBufferSize];       // 解码出来的Key，[key + Seq + Type]
                         // GPU Encode后的 [shared + non_shared + value_size + partial_key ] 也放在这个地方
    uint32_t key_size;       // Key的大小

    uint32_t value_offset;   // SST中value的偏移，这是包含前面的Varint size
    uint32_t value_size;     // Value的大小(包含前面的Varint size)
};

// GPU使用
struct filter_meta {
    int start;      // 从哪个kv开始计算filter SST_kv[]
    int cnt;        // 有多少个kv一起计算filter

    uint32_t offset;    // 此filter从哪些开始写
    int filter_size;    // 此filter所占用大小(包括最后的k_的一字节)
};

// CPU使用，计算后面的index_block
struct block_meta {
    uint64_t offset; // 每个DataBlock开始的offset
    uint64_t size;        // 每个DataBlock的大小，不包括（type + CRC）
};

struct gpu_thread_meta {
    int x;
};

extern gpu_thread_meta threadIdx;
extern gpu_thread_meta blockIdx;
extern gpu_thread_meta blockDim;
}
}
#endif //LEVELDB_DATA_H
