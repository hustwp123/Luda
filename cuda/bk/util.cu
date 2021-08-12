// here we put some basic util function

#include "cuda/data.h"
#include "cuda/util.h"

namespace leveldb {
namespace gpu {
// faker function
__host__ void cudaMalloc(void** devPtr, size_t size) {*devPtr = (void *)malloc(size);}
__host__ void cudaFree(void* devPtr) {free(devPtr);}
__host__ void cudaMemcpy(void *dst, const void *src, size_t cnt) { memcpy(dst, src, cnt);}

__host__ __device__
void Memcpy(char *dst, const char *src, size_t n) {
    memcpy(dst, src, n);
}

__host__ __device__
void EncodeFixed32(char *dst, uint32_t value) {
    Memcpy(dst, (char *) &value, sizeof(value));
}

__host__ __device__
void EncodeFixed64(char *dst, uint64_t value) {
    Memcpy(dst, (char *) &value, sizeof(value));
}

__host__ __device__
void PutFixed32(Buffer *dst, uint32_t value) {
    char buf[sizeof(value)];
    EncodeFixed32(buf, value);
    dst->append(buf, sizeof(buf));
}

__host__ __device__
void PutFixed64(Buffer *dst, uint64_t value) {
    char buf[sizeof(value)];
    EncodeFixed64(buf, value);
    dst->append(buf, sizeof(buf));
}

__host__ __device__
char *EncodeVarint32(char *dst, uint32_t v) {
    // Operate on characters as unsigneds
    unsigned char *ptr = reinterpret_cast<unsigned char *>(dst);
    static const int B = 128;
    if (v < (1 << 7)) {
        *(ptr++) = v;
    } else if (v < (1 << 14)) {
        *(ptr++) = v | B;
        *(ptr++) = v >> 7;
    } else if (v < (1 << 21)) {
        *(ptr++) = v | B;
        *(ptr++) = (v >> 7) | B;
        *(ptr++) = v >> 14;
    } else if (v < (1 << 28)) {
        *(ptr++) = v | B;
        *(ptr++) = (v >> 7) | B;
        *(ptr++) = (v >> 14) | B;
        *(ptr++) = v >> 21;
    } else {
        *(ptr++) = v | B;
        *(ptr++) = (v >> 7) | B;
        *(ptr++) = (v >> 14) | B;
        *(ptr++) = (v >> 21) | B;
        *(ptr++) = v >> 28;
    }
    return reinterpret_cast<char *>(ptr);
}

__host__ __device__
void PutVarint32(Buffer *dst, uint32_t v) {
    char buf[5];
    char *ptr = EncodeVarint32(buf, v);
    dst->append(buf, ptr - buf);
}

__host__ __device__
char *EncodeVarint64(char *dst, uint64_t v) {
    static const int B = 128;
    unsigned char *ptr = reinterpret_cast<unsigned char *>(dst);
    while (v >= B) {
        *(ptr++) = v | B;
        v >>= 7;
    }
    *(ptr++) = static_cast<unsigned char>(v);
    return reinterpret_cast<char *>(ptr);
}

__host__ __device__
void PutVarint64(Buffer *dst, uint64_t v) {
    char buf[10];
    char *ptr = EncodeVarint64(buf, v);
    dst->append(buf, ptr - buf);
}

__host__ __device__
void PutLengthPrefixedSlice(Buffer *dst, const Slice &value) {
    PutVarint32(dst, value.size());
    dst->append(value.data(), value.size());
}

__host__ __device__
int VarintLength(uint64_t v) {
    int len = 1;
    while (v >= 128) {
        v >>= 7;
        len++;
    }
    return len;
}

__host__ __device__
uint32_t DecodeFixed32(const char *ptr) {
    uint32_t result;
    Memcpy((char *) &result, ptr, sizeof(result));
    return result;
}

__host__ __device__
uint64_t DecodeFixed64(const char *ptr) {
    // Load the raw bytes
    uint64_t result;
    Memcpy((char *) &result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
}

__host__ __device__
const char *GetVarint32PtrFallback(const char *p, const char *limit,
                                   uint32_t *value) {
    uint32_t result = 0;
    for (uint32_t shift = 0; shift <= 28 && p < limit; shift += 7) {
        uint32_t byte = *(reinterpret_cast<const unsigned char *>(p));
        p++;
        if (byte & 128) {
            // More bytes are present
            result |= ((byte & 127) << shift);
        } else {
            result |= (byte << shift);
            *value = result;
            return reinterpret_cast<const char *>(p);
        }
    }
    return nullptr;
}

__host__ __device__
inline const char *GetVarint32Ptr(const char *p, const char *limit,
                                  uint32_t *value) {
    if (p < limit) {
        uint32_t result = *(reinterpret_cast<const unsigned char *>(p));
        if ((result & 128) == 0) {
            *value = result;
            return p + 1;
        }
    }
    return GetVarint32PtrFallback(p, limit, value);
}


__host__ __device__
bool GetVarint32(Slice *input, uint32_t *value) {
    const char *p = input->data();
    const char *limit = p + input->size();
    const char *q = GetVarint32Ptr(p, limit, value);
    if (q == nullptr) {
        return false;
    } else {
        *input = Slice(q, limit - q);
        return true;
    }
}

__host__ __device__
const char *GetVarint64Ptr(const char *p, const char *limit, uint64_t *value) {
    uint64_t result = 0;
    for (uint32_t shift = 0; shift <= 63 && p < limit; shift += 7) {
        uint64_t byte = *(reinterpret_cast<const unsigned char *>(p));
        p++;
        if (byte & 128) {
            // More bytes are present
            result |= ((byte & 127) << shift);
        } else {
            result |= (byte << shift);
            *value = result;
            return reinterpret_cast<const char *>(p);
        }
    }
    return nullptr;
}

__host__ __device__
bool GetVarint64(Slice *input, uint64_t *value) {
    const char *p = input->data();
    const char *limit = p + input->size();
    const char *q = GetVarint64Ptr(p, limit, value);
    if (q == nullptr) {
        return false;
    } else {
        *input = Slice(q, limit - q);
        return true;
    }
}

__host__ __device__
const char *GetLengthPrefixedSlice(const char *p, const char *limit,
                                   Slice *result) {
    uint32_t len;
    p = GetVarint32Ptr(p, limit, &len);
    if (p == nullptr) return nullptr;
    if (p + len > limit) return nullptr;
    *result = Slice(p, len);
    return p + len;
}

__host__ __device__
bool GetLengthPrefixedSlice(Slice *input, Slice *result) {
    uint32_t len;
    if (GetVarint32(input, &len) && input->size() >= len) {
        *result = Slice(input->data(), len);
        input->remove_prefix(len);
        return true;
    } else {
        return false;
    }
}
__host__ __device__ void EncodeValueOffset(uint32_t *offset, int Idx) {
    *offset = (*offset & 0x00FFFFFF) | (Idx << 24);
}

__host__ __device__ void DecodeValueOffset(uint32_t *offset, int *Idx) {
    *Idx = (*offset) >> 24;
    *offset &= 0x00FFFFFF;
}

__host__ __device__
const char* DecodeEntry(const char* p, const char* limit,
                                      uint32_t* shared, uint32_t* non_shared,
                                      uint32_t* value_length) {
    if (limit - p < 3) {
        return nullptr;
    }
    *shared = reinterpret_cast<const unsigned char*>(p)[0];
    *non_shared = reinterpret_cast<const unsigned char*>(p)[1];
    *value_length = reinterpret_cast<const unsigned char*>(p)[2];
    if ((*shared | *non_shared | *value_length) < 128) {
        // Fast path: all three values are encoded in one byte each
        p += 3;
    } else {
        if ((p = GetVarint32Ptr(p, limit, shared)) == nullptr) return nullptr;
        if ((p = GetVarint32Ptr(p, limit, non_shared)) == nullptr) return nullptr;
        if ((p = GetVarint32Ptr(p, limit, value_length)) == nullptr) return nullptr;
    }

    if (static_cast<uint32_t>(limit - p) < (*non_shared + *value_length)) {
        return nullptr;
    }
    return p;
}

__host__ __device__
uint32_t Hash(const char* data, size_t n, uint32_t seed) {
    // Similar to murmur hash
    const uint32_t m = 0xc6a4a793;
    const uint32_t r = 24;
    const char* limit = data + n;
    uint32_t h = seed ^ (n * m);

    // Pick up four bytes at a time
    while (data + 4 <= limit) {
        uint32_t w = DecodeFixed32(data);
        data += 4;
        h += w;
        h *= m;
        h ^= (h >> 16);
    }

    // Pick up remaining bytes
    switch (limit - data) {
        case 3:
            h += static_cast<unsigned char>(data[2]) << 16;
        case 2:
            h += static_cast<unsigned char>(data[1]) << 8;
        case 1:
            h += static_cast<unsigned char>(data[0]);
            h *= m;
            h ^= (h >> r);
            break;
    }
    return h;
}


}  // namespace leveldb
}

