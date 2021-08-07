#ifndef LEVELDB_CUDA_COMMON_H
#define LEVELDB_CUDA_COMMON_H

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>

#define K_SHARED_KEYS (4)
#define CUDA_MAX_COMPACTION_FILES (40)
#define __SST_SIZE (16 * 1024 * 1024)
#define __MIN_KEY_SIZE (256 + 32)       // 这里假设Value最小为256字节，当然可以更改
#define CUDA_MAX_KEY_PER_SST (__SST_SIZE / __MIN_KEY_SIZE + 4096)
#define CUDA_MAX_GDI_PER_SST (CUDA_MAX_KEY_PER_SST / K_SHARED_KEYS + 100)
#define CUDA_MAX_KEYS_COMPACTION (CUDA_MAX_KEY_PER_SST * CUDA_MAX_COMPACTION_FILES)


#include "db/version_set.h"

#include "cuda/util.h"

namespace leveldb {
namespace gpu {


enum {
    kKeyBufferSize = 32,    // KEY 占用的最大Size（包括8字节）
    kSharedKeys = K_SHARED_KEYS,       // 每个SharedBlock中Key的数目
    kDataSharedCnt = 4,     // 每个DataBlock中SharedBlock的数目
    kBitsPerKey = 10,       // filter时候，每个Key所用bit位数
    kSharedPerSST = __SST_SIZE / __MIN_KEY_SIZE / kSharedKeys - 100,    // 每个SST中有多少个SharedBlock
};
struct SST_kv;
class WpSlice {
public:  
    bool operator <(WpSlice& b);

public:
    // uint32_t offset_;
    // int value_len_;
    // char *data_;

    // size_t size_;

    SST_kv* skv;

    // bool drop;   
    // uint64_t seq_;
};

class Stream {
public:
    Stream();
    ~Stream();
    void Sync();
    unsigned long data() { return s_; }

    unsigned long s_;
};

class Buffer {
public:
    Buffer(char *buf, size_t size);

    char *now();
    char *data();
    void reset();

    
    inline void advance(int n);
   
    void append(const char *data, size_t size);

    Buffer& operator=(const Buffer&) = default;

    char  *base_;
    size_t total_;
    size_t size_;
};

class Slice {
public:
    // Create an empty slice.
     
    Slice() : data_(""), size_(0) {}

    // Create a slice that refers to d[0,n-1].
     
    Slice(const char *d, size_t n) : data_(d), size_(n) {}

     
    Slice(const char *d, size_t n, uint32_t off, int len) : data_(d), size_(n), offset_(off), value_len_(len) {}

    // Create a slice that refers to s[0,strlen(s)-1]
     
    Slice(const char *s) : data_(s), size_(strlen(s)) {}

    // Intentionally copyable.
    Slice(const Slice &) = default;

    Slice &operator=(const Slice &) = default;

    // Return a pointer to the beginning of the referenced data
     
    const char *data() const { return data_; }

    // Return the length (in bytes) of the referenced data
     
    size_t size() const { return size_; }

    // Return true iff the length of the referenced data is zero
     
    bool empty() const { return size_ == 0; }

    // Return the ith byte in the referenced data.
    // REQUIRES: n < size()
     
    char operator[](size_t n) const {
        assert(n < size());
        return data_[n];
    }

    // Change this slice to refer to an empty array
     
    void clear() {
        data_ = "";
        size_ = 0;
    }

    // Drop the first "n" bytes from this slice.
     
    void remove_prefix(size_t n) {
        assert(n <= size());
        data_ += n;
        size_ -= n;
    }

    // Three-way comparison.  Returns value:
    //   <  0 iff "*this" <  "b",
    //   == 0 iff "*this" == "b",
    //   >  0 iff "*this" >  "b"
     
    int compare(const Slice &b) const {
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

     
    int internal_compare(const gpu::Slice &b) const;

    

    // Return true iff "x" is a prefix of "*this"
     
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

class BlockHandle { // the same as leveldb::BlockHandle
public:
      BlockHandle() :offset_(0), size_(0) {}
      ~BlockHandle() = default;

     
    void EncodeTo(Buffer* dst);
    bool DecodeFrom(Slice *input);

    uint64_t offset_;
    uint64_t size_;
};

class Footer {
public:
      Footer() {}
      ~Footer() = default;

     
    void EncodeTo(Buffer *dst);
     
    bool DecodeFrom(Slice *input);

    BlockHandle metaindex_handle_;
    BlockHandle index_handle_;
};

// Basic DataStructure
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

// 统一负责主机与设备的内存的申请与释放
// 大致内存最多等于 SST_SIZE * 25 / SST_SIZE * 25 * 2
class HostAndDeviceMemory {
public:
     HostAndDeviceMemory();
     ~HostAndDeviceMemory();

    // vector 中每个成员只代表一个SST中decode、endcode等等的数据
    std::vector<char *> h_SST;    // 保存读取上来的SST，也保存Decode生成的新SST
    std::vector<char *> d_SST;
    std::vector<char *> d_SST_new;

    std::vector<GDI *> d_gdi;

    std::vector<SST_kv *> d_skv;

    // Decode后Sorted排序好的
    SST_kv *d_skv_sorted;
    SST_kv *d_skv_sorted_shared;

    // Device 端访问SST的数组
    char **d_SST_ptr;

    std::vector<uint32_t *> d_shared_size;

    std::vector<uint32_t *> d_shared_offset;

    std::vector<filter_meta *> d_fmeta;

    WpSlice* lowSlices;
    int low_size;
    WpSlice* highSlices;
    int high_size;
    WpSlice* resultSlice;
    int result_size;
};

class SSTDecode {
public:
    
    SSTDecode(const char* filename, int filesize, char *SST) :
           all_kv_(0), shared_cnt_(0), h_SST_(SST), file_size_(filesize) {
        //TODO: open file and read it to h_SST_
        FILE* file = ::fopen(filename, "rb");
        assert(file);
        size_t n = ::fread(h_SST_, sizeof(char), file_size_, file);
        assert(n == file_size_);
        ::fclose(file);
    }

     void SetMemory(int idx, HostAndDeviceMemory *m) {
        d_SST_ptr_ = m->d_SST_ptr;
        SST_idx_ = idx;

        d_SST_ = m->d_SST[idx];

        d_skv_ = m->d_skv[idx];

        d_gdi_ = m->d_gdi[idx];
    }

     ~SSTDecode() = default;
     void DoDecode();
     void DoGPUDecode();

     // Async Decode
     void DoGPUDecode_1(WpSlice* slices=nullptr,int index=0);
     void DoGPUDecode_2();

     void Sync();

    int all_kv_;

//private:
    char* h_SST_;
    char* d_SST_;

    int   file_size_;
    GDI*  d_gdi_;

    SST_kv* d_skv_;

    char **d_SST_ptr_;
    int SST_idx_;
    int shared_cnt_;  // the count of all restarts
    Stream s_;
};

class SSTCompactionUtil{
public:
     SSTCompactionUtil(leveldb::Version *input, int level) : input_version_(input), level_(level) {}
     ~SSTCompactionUtil() {}

    // 确保标志位Delete的key可以安全删除
     bool IsBaseLevelForKey(const Slice& __user_key) {
        leveldb::Slice user_key(__user_key.data(), __user_key.size());
        //const Comparator* user_cmp = input_version_->vset_->icmp_.user_comparator();
        const Comparator* user_cmp = BytewiseComparator();

        for (int lvl = level_ + 2; lvl < config::kNumLevels; lvl++) {
            const std::vector<FileMetaData*>& files = input_version_->files_[lvl];
            for (; level_ptrs_[lvl] < files.size();) {
                FileMetaData* f = files[level_ptrs_[lvl]];
                if (user_cmp->Compare(user_key, f->largest.user_key()) <= 0) {
                    // We've advanced far enough
                    if (user_cmp->Compare(user_key, f->smallest.user_key()) >= 0) {
                        // Key falls in this file's range, so definitely not base level
                        return false;
                    }
                    break;
                }
                level_ptrs_[lvl]++;
            }
        }
        return true;
    }

private:
    leveldb::Version *input_version_;
    size_t level_ptrs_[config::kNumLevels];
    int level_;
};

class SSTSort {
public:
    enum SSTSortType {enNULL = -1, enL0 = 5, enLow = 8, enHigh = 10};
     SSTSort(uint64_t seq, SST_kv* out, SSTCompactionUtil* util,SST_kv *d_kv=nullptr):
            seq_(seq), witch_(enNULL), l0_sst_index_(-1), util_(util),
            out_(out), out_size_(0), key_(),
            low_sst_index_(0), high_sst_index_(0),d_kvs_(d_kv) {}

     ~SSTSort() {}

     void AddL0(int size, SST_kv* skv) {
        l0_sizes_.push_back(size);
        l0_idx_.push_back(0);
        l0_skvs_.push_back(skv);
    }

     void AddLow(int size, SST_kv* skv) {
        low_sizes_.push_back(size);
        low_idx_.push_back(0);
        low_skvs_.push_back(skv);
        
    }

     void AddHigh(int size, SST_kv* skv) {
        high_sizes_.push_back(size);
        high_idx_.push_back(0);
        high_skvs_.push_back(skv);
        
    }

     void WpSort();
     void Sort();


private:
     Slice FindLowSmallest();
     Slice FindHighSmallest();
     Slice FindL0Smallest();
     Slice GetCurrent(std::vector<SST_kv*> &skvs, std::vector<int> &idxs, std::vector<int> &sizes, int &sst_idx);
     void Next(std::vector<SST_kv*> &skvs, std::vector<int> &idxs, std::vector<int> &sizes, int &sst_idx);

    Slice key_;         // 当前的key_，witch_表示从哪一层取出来的
    SSTSortType witch_;         // -1:null, 0:l0, 1:low, 2:high
    uint64_t seq_;      // 当前最小的序列号

    // xx_sizes_ : 表示每个SST中解析出来kv的数目
    // xx_idx_   : 表示每个SST已经遍历到第几个kv了
    std::vector<int>     l0_sizes_;
    std::vector<int>     l0_idx_;
    std::vector<SST_kv*> l0_skvs_;

    std::vector<int>     low_sizes_;
    std::vector<int>     low_idx_;
    std::vector<SST_kv*> low_skvs_;
    int low_kvs=0;

    std::vector<SST_kv*> low_skvs_2;

    std::vector<int>     high_sizes_;
    std::vector<int>     high_idx_;
    std::vector<SST_kv*> high_skvs_;
    int high_kvs=0;

    std::vector<SST_kv*> high_skvs_2;

    int low_sst_index_;     // low level first SST
    int high_sst_index_;    // high level first SST
    int l0_sst_index_;      // Level 0 current minimum SST

    SSTCompactionUtil* util_;

public:
    SST_kv *out_;           // 最后输出已经排好序的KV
    int out_size_;          // KV总个数

    SST_kv *d_kvs_;
    WpSlice* low_slices=nullptr;
    WpSlice* high_slices=nullptr;
    WpSlice* result_slices=nullptr;
    int num;
    int low_num;
    int high_num;
    int low_index=0;
    int high_index=0;
    void AddLowSlice(int size, SST_kv* skv);
    void AddHighSlice(int size, SST_kv* skv);
    void AllocLow(int size,HostAndDeviceMemory* m);
    void AllocHigh(int size,HostAndDeviceMemory* m);
    void AllocResult(int size,HostAndDeviceMemory* m);
};

class SSTEncode {
public:
     SSTEncode(char *SST, int kv_cnt, int SST_idx)
            :cur_(0), h_SST_(SST) , SST_idx_(SST_idx), kv_count_(kv_cnt) {
        shared_count_ = (kv_cnt + kSharedKeys - 1) / kSharedKeys;
    }
     ~SSTEncode() {}

     void SetMemory(HostAndDeviceMemory *m, int base) {
        base_ = base;

        h_skv_ = m->d_skv_sorted + base;
        d_skv_ = m->d_skv_sorted;
        d_skv_new_ = m->d_skv_sorted_shared;

        d_SST_ptr = m->d_SST_ptr;
        d_SST_new_ = m->d_SST_new[SST_idx_];

        d_shared_size_ = m->d_shared_size[SST_idx_];

        d_shared_offset_ = m->d_shared_offset[SST_idx_];

        d_fmeta_ = m->d_fmeta[SST_idx_];
    }

    void ComputeDataBlockOffset(int sc = kDataSharedCnt);     // 每个DataBlock 默认有多少个共享前缀块
    void ComputeFilter();          // 计算每个DataBlock对应的filter的下标，数据长度，等等
    void WriteIndexAndFooter();    // 填写剩下的部分

    void DoEncode();

    // AsyncEncode
    void DoEncode_1();
    void DoEncode_2();
    void DoEncode_3();
    void DoEncode_4();

    char *h_SST_;

    char *d_SST_new_;
    uint32_t cur_;              // 指示当前写到什么地方了

    char **d_SST_ptr;
    int SST_idx_;

    SST_kv *h_skv_;
    SST_kv *d_skv_;               // Full Key-Value Not the shared-KV
    SST_kv *d_skv_new_;
    int kv_count_;
    int base_;

    int datablock_count_;       //  一共有多少个KV DataBlock

    /* 共享前缀的Shared块的一些信息 */
    uint32_t *d_shared_size_;
    int shared_count_;

  // 每个shared从哪个地方开始写
    uint32_t *d_shared_offset_;

    filter_meta *d_fmeta_;        // 每个DataBlock对应的Filter元数据，包括多少个KV，从第几个开始等

    std::vector<block_meta> bmeta_;          // 每个KV DataBlock的元数据，offset、size

    BlockHandle  filter_handle_;
    int filter_end_;
    Footer footer;

    int data_blocks_size_;

    Stream s1_, s2_;
};

// CUDA function
void cudaMemHtD(void *dst, void *src, size_t size);
void cudaMemDtH(void *dst, void *src, size_t size);

class Debug {
public:
	void Test(const char *src, size_t cnt);
};

}
}
#endif //LEVELDB_CUDA_DECODE_KV_H
