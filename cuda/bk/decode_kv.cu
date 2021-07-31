//
// Created by crabo on 2019/11/4.
//

#include "cuda/decode_kv.h"
#define M 128
#define N 32

namespace leveldb {
namespace gpu {
//gpu_thread_meta threadIdx;
//gpu_thread_meta blockIdx;
//gpu_thread_meta blockDim;
// 统一负责主机与设备的内存的申请与释放
// 大致内存最多等于 SST_SIZE * 25 / SST_SIZE * 25 * 2
__host__
HostAndDeviceMemory::HostAndDeviceMemory() {
    char *ptr[CUDA_MAX_COMPACTION_FILES];
    // 每个SST对应的空间申请
    for (int i = 0; i < CUDA_MAX_COMPACTION_FILES; ++i) {
        char *ph_SST, *pd_SST, *pd_SST_new;
        GDI  *ph_gdi, *pd_gdi;
        SST_kv *ph_skv, *pd_skv;
        uint32_t *ph_shared_size, *pd_shared_size;
        uint32_t *ph_so, *pd_so;
        filter_meta *ph_fm, *pd_fm;

        ph_SST = (char *)malloc(__SST_SIZE + 100 * 1024); // 比设置的SST大100KB
        ph_gdi = (GDI *)malloc(sizeof(GDI) * CUDA_MAX_GDI_PER_SST);
        ph_skv = (SST_kv*)malloc(sizeof(SST_kv) * CUDA_MAX_KEY_PER_SST);
        ph_shared_size = (uint32_t *)malloc(sizeof(uint32_t) * kSharedPerSST);
        ph_so = (uint32_t *)malloc(sizeof(uint32_t) * kSharedPerSST);
        ph_fm = (filter_meta *)malloc(sizeof(filter_meta) * kSharedPerSST / kDataSharedCnt);
        assert(ph_SST && ph_gdi && ph_skv && ph_shared_size && ph_so && ph_fm);

        cudaMalloc((void **)&pd_SST, __SST_SIZE + 100 * 1024);
        cudaMalloc((void **)&pd_SST_new, __SST_SIZE + 100 * 1024);
        cudaMalloc((void **)&pd_gdi, sizeof(GDI) * CUDA_MAX_GDI_PER_SST);
        cudaMalloc((void **)&pd_skv, sizeof(SST_kv) * CUDA_MAX_KEY_PER_SST);
        cudaMalloc((void **)&pd_shared_size, sizeof(uint32_t) * kSharedPerSST);
        cudaMalloc((void **)&pd_so, sizeof(uint32_t) * kSharedPerSST);
        cudaMalloc((void **)&pd_fm, sizeof(filter_meta) * kSharedPerSST / kDataSharedCnt);
        assert(pd_SST && pd_gdi && pd_skv && pd_shared_size && pd_so);

        h_SST.push_back(ph_SST);
        h_gdi.push_back(ph_gdi);
        h_skv.push_back(ph_skv);
        h_shared_size.push_back(ph_shared_size);
        h_shared_offset.push_back(ph_so);
        h_fmeta.push_back(ph_fm);

        d_SST.push_back(pd_SST);
        ptr[i] = pd_SST;
        d_SST_new.push_back(pd_SST_new);
        d_gdi.push_back(pd_gdi);
        d_skv.push_back(pd_skv);
        d_shared_size.push_back(pd_shared_size);
        d_shared_offset.push_back(pd_so);
        d_fmeta.push_back(pd_fm);
    }

    // 排序好的空间申请
    h_skv_sorted = (SST_kv *)malloc(sizeof(SST_kv) * CUDA_MAX_KEYS_COMPACTION);
    cudaMalloc((void **)&d_skv_sorted, sizeof(SST_kv) * CUDA_MAX_KEYS_COMPACTION);
    cudaMalloc((void **)&d_skv_sorted_shared, sizeof(SST_kv) * CUDA_MAX_KEYS_COMPACTION);
    assert(h_skv_sorted && d_skv_sorted && d_skv_sorted_shared);


    // d_SST_ptr 申请
    cudaMalloc((void **)&d_SST_ptr, sizeof(char *) * CUDA_MAX_COMPACTION_FILES);
    cudaMemcpy((void *)d_SST_ptr, (void *)ptr, sizeof(char *) * CUDA_MAX_COMPACTION_FILES, cudaMemcpyHostToDevice);
}

__host__
HostAndDeviceMemory::~HostAndDeviceMemory() {
    for (int i = 0; i < CUDA_MAX_COMPACTION_FILES; ++i) {
        free(h_SST[i]);
        free(h_gdi[i]);
        free(h_skv[i]);
        free(h_shared_size[i]);
        free(h_shared_offset[i]);
        free(h_fmeta[i]);

        cudaFree(d_SST[i]);
        cudaFree(d_SST_new[i]);
        cudaFree(d_gdi[i]);
        cudaFree(d_skv[i]);
        cudaFree(d_shared_size[i]);
        cudaFree(d_shared_offset[i]);
        cudaFree(d_fmeta[i]);
    }

    free(h_skv_sorted);
    cudaFree(d_skv_sorted);
    cudaFree(d_skv_sorted_shared);
}

//////////// Decodde /////////////////////////////
__host__
void SSTDecode::DoDecode() {
    // 1. Read the footer to find index-block
    Slice footer_slice(h_SST_ + file_size_ - leveldb::Footer::kEncodedLength,
            leveldb::Footer::kEncodedLength);
    Footer footer;
    footer.DecodeFrom(&footer_slice);

    // 2. Iterator index-block and decode it to GDI
    char *contents = h_SST_ + footer.index_handle_.offset_;
    size_t contents_size = footer.index_handle_.size_;
    //printf("%llu %llu\n", footer.metaindex_handle_.offset_, footer.metaindex_handle_.size_);
    //printf("%u\n", DecodeFixed32(h_SST_ + footer.metaindex_handle_.offset_ + footer.metaindex_handle_.size_ - 4));
    //printf("%u\n", ((uint32_t *)(h_SST_ + footer.metaindex_handle_.offset_))[0]);

    // TODO: crc32c checksums. No Compression

    uint32_t index_num = DecodeFixed32((const char *)(contents + contents_size - sizeof(uint32_t)));
    uint32_t *index_restart = (uint32_t *)(contents + contents_size - sizeof(uint32_t) * (1 + index_num));
    //printf("num : %u\n", index_num);
    //int restarts_idx = 0;

    // 2.1 Iterate all the restart array to get all DataBlock offset and size(don't contain type+CRC32)
    for (uint32_t i = 0; i < index_num; ++i) {
        const char* p = contents + (index_restart[i] >> 8);
        const char* limit = (const char *)index_restart;
        uint32_t shared, non_shared, value_length;

        p = DecodeEntry(p, limit, &shared, &non_shared, &value_length);
        assert(shared == 0);            // index-block中没有使用共享前缀

        Slice key(p, non_shared);       // the minimal key in this DataBlock
        Slice value(p + non_shared, value_length);
        BlockHandle data;
        data.DecodeFrom(&value);

        uint32_t data_restart_num = DecodeFixed32(h_SST_ + data.offset_ + data.size_ - sizeof(uint32_t));
        uint32_t *array = (uint32_t *) (h_SST_ + data.offset_ + data.size_ - sizeof(uint32_t) * (data_restart_num + 1));

        for (uint32_t j = 0; j < data_restart_num; ++j) {
            //printf("data restoff:%llu size:%llu num:%u\n", data.offset_, data.size_, data_restart_num);
            uint32_t cnt = array[j] & 0xff;
            uint32_t off = array[j] >> 8;

            h_gdi_[shared_cnt_ + j].offset = data.offset_ + off;
            h_gdi_[shared_cnt_ + j].kv_base_idx = all_kv_;

            if (j + 1 < data_restart_num) {
                h_gdi_[shared_cnt_ + j].limit = data.offset_ + (array[j + 1] >> 8);
            } else {
                h_gdi_[shared_cnt_ + j].limit = data.offset_ + data.size_ - sizeof(uint32_t) * (data_restart_num + 1);
            }

            all_kv_ += cnt;
        }
        shared_cnt_ += data_restart_num;
    }
}

__host__
void SSTDecode::DoGPUDecode() {
    // TODO: copy DATA from CPU to GPU
    // It can all be ASYNC
    cudaMemcpy(d_SST_, h_SST_, file_size_, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gdi_, h_gdi_, sizeof(GDI) * shared_cnt_, cudaMemcpyHostToDevice);

#ifdef __CUDA_DEBUG
    for (int i = 0; i < shared_cnt_; ++i) {
        threadIdx.x = i;
        GPUDecodeKernel(d_SST_ptr_, SST_idx_, d_gdi_, shared_cnt_, d_skv_);
    }
#else
    // cudaMemcpy: h_SST, h_gdi,  == > GPU
    GPUDecodeKernel<<<M, N>>>(d_SST_ptr_, SST_idx_, d_gdi_, shared_cnt_, d_skv_);
    // cudaMemcpy  h_skv_         <=== GPU
#endif

    cudaMemcpy(h_skv_, d_skv_, sizeof(SST_kv) * all_kv_, cudaMemcpyDeviceToHost);
}

// Gpu kernel <<<x, y>>>

__global__
void GPUDecodeKernel(char **SST, int SSTIdx, GDI *gdi, int gdi_cnt, SST_kv *skv) {
    int v_gdi_index = ::blockIdx.x + ::threadIdx.x;
    if (v_gdi_index >= gdi_cnt) {
        return ;
    }

    uint32_t kv_idx = 1;
    GDI *cur = &gdi[v_gdi_index];
    char *d_SST = SST[SSTIdx];
    const char *p = d_SST + cur->offset;
    const char *limit = d_SST + cur->limit;
    SST_kv *pskv = &skv[cur->kv_base_idx];
    uint32_t shared, non_shared, value_length;

    // 1. Decode first KeyValue
    p = DecodeEntry(p, limit, &shared, &non_shared, &value_length);
    assert(p && !shared);
    Memcpy(pskv->ikey, p, non_shared);
    pskv->key_size = shared + non_shared;
    pskv->value_offset = p + non_shared - d_SST;
    EncodeValueOffset(&pskv->value_offset, SSTIdx);
    pskv->value_size = value_length;
    p += non_shared + value_length;

    // 2. Decode the last keys
    while (p < limit) {
        pskv = &skv[cur->kv_base_idx + kv_idx];

        p = DecodeEntry(p, limit, &shared, &non_shared, &value_length);
        assert(p);

        Memcpy(pskv->ikey, (pskv - 1)->ikey, shared);   // copy the last Shared-Key
        Memcpy(pskv->ikey + shared, p, non_shared);
        pskv->key_size = shared + non_shared;
        pskv->value_offset = p + non_shared - d_SST;
        EncodeValueOffset(&pskv->value_offset, SSTIdx);
        pskv->value_size = value_length;

        p = p + non_shared + value_length;
        ++ kv_idx;
    }
}

/*
 * @skv: the SORTED KVs put in here, full key
 * @skv_new: the shared KVs, partial key
 *
 * <<<x, y>>>
 * kv_start = base + (x * Dim_y + y) * 16;
 * kv_end = min(kv_start + 16, kv_max_cnt);
 */
__global__
void GPUEncodeSharedKernel(SST_kv *skv, SST_kv *skv_new, int base, int skv_cnt, uint32_t *shared_size) {
    int kv_start = base + (::blockIdx.x * ::blockDim.x + ::threadIdx.x) * kSharedKeys;
    int kv_count = kSharedKeys <= skv_cnt - kv_start ? kSharedKeys : skv_cnt - kv_start;

    SST_kv *pf = &skv_new[kv_start];
    int fkey_size = skv[kv_start] .key_size;    // The first or the last
    Buffer fbuf(pf->ikey, kKeyBufferSize); // the First-Key, and the Last-Key
    int total_size = 0;

    if (kv_start >= skv_cnt)
        return ;

    // 1. Encode the First-Key
    PutVarint32(&fbuf, 0);
    PutVarint32(&fbuf, fkey_size);
    PutVarint32(&fbuf, skv[kv_start].value_size);
    Memcpy(fbuf.now(), skv[kv_start].ikey, fkey_size);
    pf->key_size = fbuf.size_ + fkey_size;
    pf->value_offset = skv[kv_start].value_offset;
    pf->value_size = skv[kv_start].value_size;

    total_size += pf->key_size + pf->value_size ;

    // 2. Encode the last keys.
    // Odd idx use key_buf[0] for LAST, and use key_buf[1] for NOW
    // even idx use key_buf[1] for LAST, and so on
    for (int i = 1; i < kv_count; ++i) {
        SST_kv *pskv = &skv[kv_start + i];
        char   *key_last = skv[kv_start + i - 1].ikey;

        SST_kv *pskv_new = &skv_new[kv_start + i];
        Buffer buf(pskv_new->ikey, kKeyBufferSize);

        int key_size = pskv->key_size;
        int value_size = pskv->value_size;
        int shared = 0, non_shared = 0;

        while (shared < key_size && shared < fkey_size && pskv->ikey[shared] == key_last[shared]) {
            shared ++;
        }

        non_shared = key_size - shared;
        PutVarint32(&buf, shared);
        PutVarint32(&buf, non_shared);
        PutVarint32(&buf, value_size);
        Memcpy(buf.now(), pskv->ikey + shared, non_shared);
        pskv_new->key_size = buf.size_ + non_shared;
        pskv_new->value_size = pskv->value_size;
        pskv_new->value_offset = pskv->value_offset;

        total_size += pskv_new->key_size + pskv_new->value_size;

        fkey_size = key_size;   // save the LAST-KEY size
    }

    shared_size[kv_start / kSharedKeys] = total_size;
}

/*
 * Every DataBlock : kSharedKeys * kSharedCnt = 16 * 3 = 48, so, the 2, 5, 8 and so on will write the restarts[]
 * 0 1 2 3 4 5      SharedIdx
 *   0     1        DataBlockIdx
 *
 */
__global__
void GPUEncodeCopyShared(char **SST, char *SST_new, SST_kv *skv, int base_idx,
        int skv_cnt, uint32_t *shared_offset, int shared_cnt) {
    int shared_idx = ::blockIdx.x * ::blockDim.x + ::threadIdx.x;
    int kv_start = base_idx + shared_idx * kSharedKeys;
    int kv_cnt = kSharedKeys <= skv_cnt - kv_start ? kSharedKeys : skv_cnt - kv_start;

    if (shared_idx >= shared_cnt) {
        return ;
    }

    char *base = SST_new + (shared_offset[shared_idx] >> 8);
    uint32_t cur = 0;

    // copy KV from OLDPlace to NEWPlace
    for (int i = 0; i < kv_cnt; ++i) {
        SST_kv *pskv = &skv[kv_start + i];
        int idx;
        Memcpy(base + cur, pskv->ikey, pskv->key_size);
        cur += pskv->key_size;

        DecodeValueOffset(&pskv->value_offset, &idx);
        char *value = SST[idx] + pskv->value_offset;
        Memcpy(base + cur, value, pskv->value_size);
        cur += pskv->value_size;
    }

    // 每个DataBlock中的最后一个SharedBlock做最后的Reastart[]落盘操作
    if ((shared_idx + 1) % kDataSharedCnt == 0 || shared_idx == shared_cnt - 1) {
        Buffer buf(base + cur, sizeof(uint32_t) * (kDataSharedCnt + 1));
        int shared_start = shared_idx - shared_idx % kDataSharedCnt;

        for (int i = 0; i < kDataSharedCnt && shared_start + i < shared_cnt; ++i) {
            uint32_t boffset = shared_offset[shared_start] >> 8;
            uint32_t tmp = ((shared_offset[shared_start + i] >> 8) - boffset) << 8;
            tmp |= shared_offset[shared_start + i] & 0xff;
            //PutFixed32(&buf, shared_offset[shared_start + i]);
            PutFixed32(&buf, tmp);
        }
        PutFixed32(&buf, shared_idx - shared_start + 1);
    }
}

/*
 * @skv : FULL key
 */
__global__
void GPUEncodeFilter(char *SST_new, SST_kv *skv, filter_meta *fmeta, int f_cnt, int k) {
    int idx = ::blockIdx.x * ::blockDim.x + ::threadIdx.x;
    if (idx >= f_cnt) return ;

    char *base = SST_new + fmeta[idx].offset;
    size_t bits = fmeta[idx].cnt * kBitsPerKey;

    if (bits < 64) bits = 64;

    size_t bytes = (bits + 7) / 8;
    bits = bytes * 8;
    assert(bytes == fmeta[idx].filter_size - 1);

    base[bytes] = static_cast<char>(k);
    char* array = base;
    for (int i = 0; i < fmeta[idx].cnt; i++) {
        int kv_idx = fmeta[idx].start + i;
        uint32_t h = Hash(skv[kv_idx].ikey, skv[kv_idx].key_size - 8);    // Use User-Key
        const uint32_t delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
        for (size_t j = 0; j < k; j++) {
            const uint32_t bitpos = h % bits;
            array[bitpos / 8] |= (1 << (bitpos % 8));
            h += delta;
        }
    }
}

///////////// Sort /////////////////////////
__host__
Slice SSTSort::GetCurrent(std::vector<SST_kv*> &skvs, std::vector<int> &idxs, std::vector<int> &sizes, int &sst_idx) {
    if (sst_idx >= skvs.size()) {
        return Slice(NULL, 0);
    }

    SST_kv *pskv = skvs[sst_idx];
    int skv_idx = idxs[sst_idx];
    assert(skv_idx < sizes[sst_idx]);

    return Slice(pskv[skv_idx].ikey, pskv[skv_idx].key_size, pskv[skv_idx].value_offset, pskv[skv_idx].value_size);
}
__host__
void SSTSort::Next(std::vector<SST_kv*> &skvs, std::vector<int> &idxs, std::vector<int> &sizes, int &sst_idx) {
    if (sst_idx >= skvs.size()) {
        return ;
    }

    ++ idxs[sst_idx];
    if (idxs[sst_idx] >= sizes[sst_idx]) {
        ++ sst_idx;
    }
}

__host__
Slice SSTSort::FindLowSmallest() {
    if (witch_ == enLow) {
        Next(low_skvs_, low_idx_, low_sizes_, low_sst_index_);
    }
    return GetCurrent(low_skvs_, low_idx_, low_sizes_, low_sst_index_);
}

__host__
Slice SSTSort::FindHighSmallest() {
    if (witch_ == enHigh) {
        Next(high_skvs_, high_idx_, high_sizes_, high_sst_index_);
    }
    return GetCurrent(high_skvs_, high_idx_, high_sizes_, high_sst_index_);
}

__host__
Slice SSTSort::FindL0Smallest() {
    Slice min_key(NULL, 0);

    if (witch_ == enL0 || witch_ == enLow) {
        ++ l0_idx_[l0_sst_index_];
    }

    l0_sst_index_ = -1;     // init the it to -1

    for (int i = 0; i < l0_skvs_.size(); ++i) {
        SST_kv *pskv = l0_skvs_[i];
        int skv_idx  = l0_idx_[i];

        if (skv_idx >= l0_sizes_[i]) {
            continue;
        }

        if (l0_sst_index_ == -1) {
            min_key = Slice(pskv[skv_idx].ikey, pskv[skv_idx].key_size,
                    pskv[skv_idx].value_offset, pskv[skv_idx].value_size);
            l0_sst_index_ = i;
        } else {
            Slice key_cur(pskv[skv_idx].ikey, pskv[skv_idx].key_size,
                pskv[skv_idx].value_offset, pskv[skv_idx].value_size);
            int r = key_cur.internal_compare(min_key);
            if (r < 0) {
                min_key = key_cur;
                l0_sst_index_ = i;
            }
        }
    }

    return min_key;
}

__host__
void SSTSort::Sort() {
    printf(stderr,"sort2\n");
    Slice low_key, high_key, last_user_key;
    uint64_t last_seq = kMaxSequenceNumber;

    witch_ = enNULL;

    while (true) {
        bool drop = false;

        // 1. Iterator the TWO-LEVEL and get the minimum KV
        if (l0_skvs_.empty()) {  // low level not Level0
            low_key = FindLowSmallest();
        } else {
            low_key = FindL0Smallest();
        }
        high_key = FindHighSmallest();

        if (high_key.empty() && low_key.empty()) {
            break;
        }

        if (low_key.empty()) {
            low_key = high_key;
            witch_ = enHigh;
        } else if (high_key.empty()) {
            witch_ = enLow;
        } else if (low_key.internal_compare(high_key) >= 0) {
            low_key = high_key;
            witch_ = enHigh;
        } else {
            witch_ = enLow;
        }

        // 2. Check the key
        Slice min_user_key(low_key.data(), low_key.size() - 8);
        if (!last_user_key.empty() && last_user_key.compare(min_user_key) != 0) {
            last_user_key = min_user_key;
            last_seq = kMaxSequenceNumber;
        }

        uint64_t inum = DecodeFixed64(low_key.data() + low_key.size() - 8);
        uint64_t iseq = inum >> 8;
        uint8_t  itype = inum & 0xff;

        if (last_seq <= seq_) {
            drop = true;
        } else if (itype == kTypeDeletion &&
#ifdef __CUDA_DEBUG
                   iseq <= seq_) {
#else
                   iseq <= seq_ && util_->IsBaseLevelForKey(min_user_key)) {
#endif
            drop = true;
        }

        last_seq = iseq;

        // 3. Write KV to out_
        if (!drop) {
           Memcpy(out_[out_size_].ikey, low_key.data(), low_key.size());
           out_[out_size_].key_size = low_key.size();
           out_[out_size_].value_size = low_key.value_len_;
           out_[out_size_].value_offset = low_key.offset_;

           ++ out_size_;
        }
    }
}

/////////// Encode //////////////////////

/*
 * 计算每个DataBlock
 */
__host__
void SSTEncode::ComputeDataBlockOffset(int SC) { // SC: shared_count
    int kv_cnt_last = kv_count_;

    datablock_count_ = (shared_count_ + SC - 1) / SC;
    bmeta_.resize(datablock_count_);

    for (int i = 0; i < datablock_count_; ++i) {
        int cnt = 0, sc = 0;
        uint32_t boffset = cur_;
        h_fmeta_[i].start = kv_count_ - kv_cnt_last;

        for (int j = 0; j < SC; ++j) {  // 遍历一个DataBlock中的所有SharedBlock
            int idx = i * SC + j;
            int sc_cnt = (kSharedKeys <= kv_cnt_last - cnt) ? kSharedKeys : kv_cnt_last - cnt;
            if (idx >= shared_count_) break;

            ++ sc;
            cnt += sc_cnt;
            h_shared_offset_[idx] = (cur_ << 8) | sc_cnt;
            cur_ += h_shared_size_[idx];
        }

        h_fmeta_[i].cnt = cnt;
        kv_cnt_last -= h_fmeta_[i].cnt;

        // write the restart_[] TO SST
        // Because it will be overwrite by GPU memory, so here we don't write that
        /*
        Buffer buf(SST_ + cur_, sizeof(uint32_t) * (sc + 1));
        for (int j = 0; j < sc; ++j) {
            int idx = i * SC + j;
            PutFixed32(&buf, shared_offset_[idx]);
        }
        PutFixed32(&buf, sc);
        */
        cur_ += sizeof(uint32_t) * (sc + 1);

        bmeta_[i].offset = boffset;
        bmeta_[i].size = cur_ - boffset;

        cur_ += 5;                   //Type, CRC
    }

    assert(kv_cnt_last == 0);
}

/*
 * 计算出每个DataBlock对应的FilterMap的长度跟对应要填写的位置
 */
__host__
void SSTEncode::ComputeFilter() {
    std::vector<uint32_t> offsets(cur_ / 2048 + 1, 0);
    int foffset = cur_;

    filter_handle_.offset_ = foffset;

    // 遍历所有的Filter
    for (int i = 0; i < datablock_count_; ++i) {
        filter_meta *pfm = &h_fmeta_[i];

        pfm->offset = cur_;
        pfm->filter_size = (pfm->cnt * kBitsPerKey + 7) / 8 + 1;
        offsets[bmeta_[i].offset / 2048] = cur_ - foffset;

        cur_ += pfm->filter_size;
    }

    // Finish Filter
    Buffer buf(h_SST_ + cur_, sizeof(uint32_t) * (offsets.size() + 1 + 1));
    for (int i = 0; i < offsets.size(); ++i) {
        PutFixed32(&buf, offsets[i]);
    }
    PutFixed32(&buf, offsets.size());
    PutFixed32(&buf, 11);           // Put kFilterBaseLg 11 2^11 = 2048

    cur_ += sizeof(uint32_t) * (offsets.size() + 1);
    filter_handle_.size_ = cur_ - foffset;

    cur_ += 5;                            // Type + CRC
}

__host__
void SSTEncode::WriteIndexAndFooter() {
    footer.metaindex_handle_.offset_ = cur_;
    const char *filter_name = "filter.leveldb.BuiltinBloomFilter2";
    char cbuf[128];

    // meta_handler
    {
        int name_len = strlen(filter_name);
        Buffer data(cbuf, 128);
        filter_handle_.EncodeTo(&data);

        Buffer fbuf(h_SST_ + cur_, 128);
        PutFixed32(&fbuf, 0);
        PutFixed32(&fbuf, name_len);
        PutFixed32(&fbuf, data.size_);
        cur_ += fbuf.size_;

        memcpy(h_SST_ + cur_, filter_name, name_len);
        cur_ += name_len;

        memcpy(h_SST_ + cur_, data.data(), data.size_);
        cur_ += data.size_;

        Buffer buf(h_SST_ + cur_, sizeof(uint32_t) * 2);
        PutFixed32(&buf, footer.metaindex_handle_.offset_);
        PutFixed32(&buf, 1);
        cur_ += sizeof(uint32_t) * 2;

        footer.metaindex_handle_.size_ = cur_ - footer.metaindex_handle_.offset_;

        cur_ += 5;          // Type and CRC
    }

    // index_block
    {
        Buffer kv(cbuf, 128);
        std::vector<uint32_t> restarts;
        footer.index_handle_.offset_ = cur_;

        for (int i = 0; i < datablock_count_; ++i) {
            SST_kv *min_kv = &h_skv_[h_fmeta_[i].start];
            Buffer ibuf(h_SST_ + cur_, 64);

            restarts.push_back(cur_ - footer.index_handle_.offset_);
            // Encode offset_size as VALUE, the min_key as KEY
            kv.reset();
            PutVarint64(&kv, bmeta_[i].offset);
            PutVarint64(&kv, bmeta_[i].size);

            PutVarint32(&ibuf, 0);
            PutVarint32(&ibuf, min_kv->key_size);
            PutVarint32(&ibuf, kv.size_);
            cur_ += ibuf.size_;

            memcpy(h_SST_ + cur_, min_kv->ikey, min_kv->key_size);
            cur_ += min_kv->key_size;

            memcpy(h_SST_ + cur_, kv.data(), kv.size_);
            cur_ += kv.size_;
        }


        Buffer buf(h_SST_ + cur_, sizeof(uint32_t) * (datablock_count_ + 1));
        for (int i = 0; i < datablock_count_; ++i) {
            PutFixed32(&buf, restarts[i] << 8 | 0x01);
        }
        PutFixed32(&buf, datablock_count_);
        cur_ += sizeof(uint32_t) * (datablock_count_ + 1);

        footer.index_handle_.size_ = cur_ - footer.index_handle_.offset_;

        cur_ += 5;          // Type CRC
    }

    // footer
    {
        Buffer buf(h_SST_ + cur_, 2 * leveldb::BlockHandle::kMaxEncodedLength + sizeof(uint32_t) * 2);
        footer.EncodeTo(&buf);
        cur_ += 48;
    }
}

__host__ void SSTEncode::DoEncode() {
    int k = 7;
    /*
    for (int i = 0; i < shared_count_; ++i) {
        threadIdx.x = i;
        GPUEncodeSharedKernel(d_skv_, d_skv_new_, base_, kv_count_, d_shared_size_);
    }
    */
    GPUEncodeSharedKernel<<<M,N>>>(d_skv_, d_skv_new_, base_, kv_count_, d_shared_size_);
    cudaMemcpy(h_shared_size_, d_shared_size_, sizeof(uint32_t ) * shared_count_, cudaMemcpyDeviceToHost);

    ComputeDataBlockOffset();
    cudaMemcpy(d_shared_offset_, h_shared_offset_, sizeof(uint32_t ) * shared_count_, cudaMemcpyHostToDevice);


    /*
    for (int i = 0; i < shared_count_; ++i) {
        threadIdx.x = i;
        GPUEncodeCopyShared(d_SST_ptr, d_SST_new_, d_skv_new_, base_, kv_count_, d_shared_offset_, shared_count_);
    }
    */
    GPUEncodeCopyShared<<<M, N>>>(d_SST_ptr, d_SST_new_, d_skv_new_, base_, kv_count_, d_shared_offset_, shared_count_);
    cudaMemcpy(h_SST_, d_SST_new_, cur_, cudaMemcpyDeviceToHost);

    ComputeFilter();
    cudaMemcpy(d_fmeta_, h_fmeta_, sizeof(filter_meta) * datablock_count_, cudaMemcpyHostToDevice);
    /*
    for (int i = 0; i < datablock_count_; ++i) {
        threadIdx.x = i;
        GPUEncodeFilter(d_SST_new_, d_skv_, d_fmeta_, datablock_count_, k);
    }
    */
    GPUEncodeFilter<<<M, N>>>(d_SST_new_, d_skv_, d_fmeta_, datablock_count_, k);

    cudaMemcpy(h_SST_ + filter_handle_.offset_, d_SST_new_ + filter_handle_.offset_, filter_handle_.size_, cudaMemcpyDeviceToHost);
    WriteIndexAndFooter();
}

}
}
