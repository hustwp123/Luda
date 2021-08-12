//
// Created by crabo on 2019/11/6.
//

#ifndef LEVELDB_CUDA_DECODE_KV_H
#define LEVELDB_CUDA_DECODE_KV_H

#include <vector>

#include "cuda/util.h"
#include "cuda/data.h"
#include "cuda/cuda_common.h"
#include "cuda/format.h"
#include "db/version_set.h"

namespace leveldb {
namespace gpu {

// 统一负责主机与设备的内存的申请与释放
// 大致内存最多等于 SST_SIZE * 25 / SST_SIZE * 25 * 2
class HostAndDeviceMemory {
public:
    __host__ HostAndDeviceMemory();
    __host__ ~HostAndDeviceMemory();

    // vector 中每个成员只代表一个SST中decode、endcode等等的数据
    std::vector<char *> h_SST;    // 保存读取上来的SST，也保存Decode生成的新SST
    std::vector<char *> d_SST;
    std::vector<char *> d_SST_new;

    std::vector<GDI *> h_gdi;
    std::vector<GDI *> d_gdi;

    std::vector<SST_kv *> h_skv;
    std::vector<SST_kv *> d_skv;

    // Decode后Sorted排序好的
    SST_kv *h_skv_sorted;
    SST_kv *d_skv_sorted;
    SST_kv *d_skv_sorted_shared;

    SST_kv *L0_d_skv_sorted;

    // Device 端访问SST的数组
    char **d_SST_ptr;

    std::vector<uint32_t *> h_shared_size;
    std::vector<uint32_t *> d_shared_size;

    std::vector<uint32_t *> h_shared_offset;
    std::vector<uint32_t *> d_shared_offset;

    std::vector<filter_meta *> h_fmeta;
    std::vector<filter_meta *> d_fmeta;
};

__global__
void GPUDecodeKernel(char **SST, int SSTIdx, GDI *gdi, int gdi_cnt, SST_kv *skv);

class SSTDecode {
public:
    __host__
    SSTDecode(const char* filename, int filesize, char *SST) :
           all_kv_(0), shared_cnt_(0), h_SST_(SST), file_size_(filesize) {
        //TODO: open file and read it to h_SST_
        FILE* file = ::fopen(filename, "rb");
        assert(file);
        size_t n = ::fread(h_SST_, sizeof(char), file_size_, file);
        assert(n == file_size_);
        ::fclose(file);
    }

    //__host__ void SetMemory(char **SST, int idx, SST_kv *d_skv) {
    __host__ void SetMemory(int idx, HostAndDeviceMemory *m) {
        d_SST_ptr_ = m->d_SST_ptr;
        SST_idx_ = idx;

        //h_SST_ = m->h_SST[idx];
        d_SST_ = m->d_SST[idx];

        h_skv_ = m->h_skv[idx];
        d_skv_ = m->d_skv[idx];

        h_gdi_ = m->h_gdi[idx];
        d_gdi_ = m->d_gdi[idx];
    }

    __host__ ~SSTDecode() = default;
    __host__ void DoDecode();
    __host__ void DoGPUDecode();

    int all_kv_;

private:
    char* h_SST_;
    char* d_SST_;

    int   file_size_;
    GDI*  h_gdi_;
    GDI*  d_gdi_;

    SST_kv* h_skv_;
    SST_kv* d_skv_;

    char **d_SST_ptr_;
    int SST_idx_;
    int shared_cnt_;  // the count of all restarts
};

class SSTCompactionUtil{
public:
    __host__ SSTCompactionUtil(leveldb::Version *input, int level) : input_version_(input), level_(level) {}
    __host__ ~SSTCompactionUtil() {}

    // 确保标志位Delete的key可以安全删除
    __host__ bool IsBaseLevelForKey(const Slice& __user_key) {
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
    __host__ SSTSort(uint64_t seq, SST_kv* out, SSTCompactionUtil* util):
            seq_(seq), witch_(enNULL), l0_sst_index_(-1), util_(util),
            out_(out), out_size_(0), key_(),
            low_sst_index_(0), high_sst_index_(0) {}

    __host__ ~SSTSort() {}

    __host__ void AddL0(int size, SST_kv* skv) {
        l0_sizes_.push_back(size);
        l0_idx_.push_back(0);
        l0_skvs_.push_back(skv);
    }

    __host__ void AddLow(int size, SST_kv* skv) {
        low_sizes_.push_back(size);
        low_idx_.push_back(0);
        low_skvs_.push_back(skv);
    }

    __host__ void AddHigh(int size, SST_kv* skv) {
        high_sizes_.push_back(size);
        high_idx_.push_back(0);
        high_skvs_.push_back(skv);
    }

    __host__ void Sort();


private:
    __host__ Slice FindLowSmallest();
    __host__ Slice FindHighSmallest();
    __host__ Slice FindL0Smallest();
    __host__ Slice GetCurrent(std::vector<SST_kv*> &skvs, std::vector<int> &idxs, std::vector<int> &sizes, int &sst_idx);
    __host__ void Next(std::vector<SST_kv*> &skvs, std::vector<int> &idxs, std::vector<int> &sizes, int &sst_idx);

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

    std::vector<int>     high_sizes_;
    std::vector<int>     high_idx_;
    std::vector<SST_kv*> high_skvs_;

    int low_sst_index_;     // low level first SST
    int high_sst_index_;    // high level first SST
    int l0_sst_index_;      // Level 0 current minimum SST

    SSTCompactionUtil* util_;

public:
    SST_kv *out_;           // 最后输出已经排好序的KV
    int out_size_;          // KV总个数
};

// 对生成的KeyValue进行每16(kv_count)个一组进行前缀压缩
// kv_start 可以使用 GPU的两维度坐标去计算
// <<<x, y>>>
// SharedIdx = x * dim_y + y
// kv_start = (x * dim_y + y) * kSharedKeys
// SharedBlock kv_count = distance[kv_start, min(kv_start + kSharedKeys, kv_end)]
__global__
void GPUEncodeSharedKernel(SST_kv *skv, SST_kv *skv_new, int base, int skv_cnt, uint32_t *shared_size);  // default that Per-Shared have 16 KVs

// Copy the DataBlock and the SharedBlock
// <<<x, y>>>
// SharedIdx = x * dim_y + y
// kv_start = (x * dim_y + y) * kSharedKeys
// SharedBlock kv_count = distance[kv_start, min(kv_start + kSharedKeys, kv_end)]
// DataBlock restarts[] Write
__global__
void GPUEncodeCopyShared(char **SST, char *SST_new, SST_kv *skv, int base_idx, int skv_cnt,
        uint32_t *shared_offset, int shared_cnt);

// <<<x, y>>>
// idx = x * dim_y + y
__global__
void GPUEncodeFilter(char *SST_new, SST_kv *skv, filter_meta *fmeta, int f_cnt, int k);

class SSTEncode {
public:
    __host__ SSTEncode(char *SST, int kv_cnt, int SST_idx)
            :cur_(0), h_SST_(SST) , SST_idx_(SST_idx), kv_count_(kv_cnt) {
        shared_count_ = (kv_cnt + kSharedKeys - 1) / kSharedKeys;
    }
    __host__ ~SSTEncode() {}

    __host__ void SetMemory(HostAndDeviceMemory *m, int base) {
        base_ = base;

        h_skv_ = m->h_skv_sorted + base;
        d_skv_ = m->d_skv_sorted;
        d_skv_new_ = m->d_skv_sorted_shared;

        d_SST_ptr = m->d_SST_ptr;
        d_SST_new_ = m->d_SST_new[SST_idx_];

        h_shared_size_ = m->h_shared_size[SST_idx_];
        d_shared_size_ = m->d_shared_size[SST_idx_];

        h_shared_offset_ = m->h_shared_offset[SST_idx_];
        d_shared_offset_ = m->h_shared_offset[SST_idx_];

        h_fmeta_ = m->h_fmeta[SST_idx_];
        d_fmeta_ = m->d_fmeta[SST_idx_];
    }

    __host__ void ComputeDataBlockOffset(int sc = kDataSharedCnt);     // 每个DataBlock 默认有多少个共享前缀块
    __host__ void ComputeFilter();          // 计算每个DataBlock对应的filter的下标，数据长度，等等
    __host__ void WriteIndexAndFooter();    // 填写剩下的部分

    __host__ void DoEncode();

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
    uint32_t *h_shared_size_;
    uint32_t *d_shared_size_;
    int shared_count_;

    uint32_t *h_shared_offset_;   // 每个shared从哪个地方开始写
    uint32_t *d_shared_offset_;


    filter_meta *h_fmeta_;        // 每个DataBlock对应的Filter元数据，包括多少个KV，从第几个开始等
    filter_meta *d_fmeta_;        // 每个DataBlock对应的Filter元数据，包括多少个KV，从第几个开始等

    std::vector<block_meta> bmeta_;          // 每个KV DataBlock的元数据，offset、size

    BlockHandle  filter_handle_;
    Footer footer;
};


}
}
#endif //LEVELDB_CUDA_DECODE_KV_H
