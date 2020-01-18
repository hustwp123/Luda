### 做测试的前提
1. Key都是固定字节的
    - 32字节级别的(包括8字节的Seq+Type)，那么Varint32就只占一个字节
2. Value的字节也都是固定的
    - 256、1024、4096、8096等等，需要其偏小，那么Varint32占2个字节
3. 申请的内存假设都是常驻的，也就是只申请一次，后面一直复用，当系统退出后释放对应的内存
4. 不使用Snappy等等各种压缩算法
5. 不同SST之间可以完全并行，所以下面只列出一个SST的处理情况，每个SST各个操作之间都可以使用Stream来异步创建

### 需要申请的内存
0. LevelDB中默认的合并最多的SST数目为**25**
1. CPU端
    - SST
    - SST_gdi[], SST_kv[]，数目跟SST相同
2. GPU端
    - SST
    - SST_gdi[], SST_kv[]，数目跟SST相同

### Decode

1. 据结构设计
``` C++
// 首先需要注意的是，我们从IndexBlock中可以获取所有的DataBlock的(offset,size)
// 然后根据(offset,size)，我们可以算出每个DataBlock中对应的restart_数组个数信息
// 也就是能够获取每个共享前缀的的数据的开始offset(当然提取的操作可以交由GPU线程去做)
// 因为每个SST的处理之间可以完全并行，那么下面就只列出处理一个SST时候的情况
// GPU线程调用  gpu_decode_sst<<<DataBlockNum, MaxRestartNum>>>()

// SST CPU ---> GPU
struct SST_gdi { // Gpu Decode Info
    int  offset;      // 考虑到SST的大小，使用int已经足够了，相对一个SST内存的偏移
    int  size;        // 每个GPU thread处理的一个DataBlock的大小
    int  restart_num; // restart数组的大小，考虑CPU已经解码出来了，就不让GPU去解码了，直接保存下来
    int  kv_base_idx; // SST_kv[]中一个GPU线程添加数据的起始槽下标，解码出来的多个共享前缀KV依次往后放
};

struct SST_kv { // SST sorted KV pair
    /* 下面的这两个其实可以使用Slice */
    char ikey[32];       // 解码出来的Key， [key + Seq + Type]
    int  key_size;       // Key的大小

    int  value_offset;   // SST中value的偏移，这是包含前面的Varint size 
    int  value_size;     // Value的大小(包含前面的Varint size)
};

```
2. 部分核心伪代码
``` C++
 for (now = start, cnt = 0; now != end; ++i, now ++) {
    S[i].block_start = now.start;
    S[i].size        = now.size;
    S[i].num         = GetSize(base_ + now.start + now.size - 4);
    S[i].d_sort_id   = cnt;
    cnt             += [].num;
 }

 // **假设**：不考虑内存占用
 // 需要了解一下异步操作，这个地方如果SST特别多的话，每个SST的单独处理都需要异步
 // 可以创建n个异步流去真正的并行decode
 cudaMalloc(d_SST, d_S, d_sort);    // for every SST, d_sort : {char K[32]; value; seq; valid; ...}
 cudaMemcpyasync(d_S, S, HtD);
 cudaMemcpyasync(d_SST, SST, HtD);

 gpu_decode<<<block_cnt, restart_cnt>>>(d_SST, d_S) {
    int block_idx = blockIdx.x;
    int restart_idx = threadIdx.x;

    if (restart_idx >= d_S[block_idx].size) {
        return;
    }

    int* restart =(int*) d_SST + d_S[block_idx].block_start + [].size - 4 * ([].num + 1);
    char* base = d_SST + GetSize(restart[restart_idx]);

    for () {
        d_sort[ [].d_sort_id ] = {K, V, seq, valid, ...};
    }
 }
```


### Sort
1. 由GPU解析出来的数据已经copy回到SST_kv[]中，每个SST都有一个对应的数组
2. 对多个SST_kv[]进行排序，且删除过时、Type为delete的KV，重新填充到SST_final_kv[]中
3. 对生成的SST_final_kv[]直接等分生成多个SST、DataBlock、Restart等个数(因为KV都是定长的，所以可以这样分)
```
 ////////// sort ///////////
 // 让主机去排序？因为涉及到N个数组K的排序，感觉主机排序更容易点？
 // 如果想要并行排序的话，如何并行排序？？？？
 finall_sort[]; // 40K * 25 = 1M 个 直接等分
 /////////////////////////////////////////////////////////////////
 SST_final_kv
 -----------
  D  D  D       FileMeta
 -----------
  D  D  D       FileMeta
 -----------
  D  D  D       FileMeta
 -----------
```

### Encode
0. 这里不使用Snappy压缩，CRC校验最后再做，空出对应的位置即可
1. 第一遍计算Encode之后每个 Restart/Block等等的占用大小等情况，其实主要就是计算出每个Block的大小，压缩计算大小，??过滤filter计算??等
 因为这样就可以知道每个Block的位置了，restart_array, index_block等等等都会求出来的
``` C++
struct {
    non, shared,
struct SST_ei { // encode info
    int restart_block_size;
};
```
2. 第二遍直接根据计算结果进行并行 copy
3. 最后一遍扫尾计算CRC

- 暂定做法
    1. GPU并行计算出每个Shared大小， 并生成SST中存储的Key(s + n + v + key)
    2. GPU并行计算出所有DataBlock中的size并行生成对应的FilterBlock、IndexBlock
    3. GPU并行复制DataBlock、FilterBlock
    4. CPU串行添加MateIndex、IndexBlock、Footer（这一部分可以与上面的第三步一起并行执行）
    并最后落盘
- 上述操作中可能出现的问题
    1. 由于这个地方我们是预先设定好的方式，也就是每16(kv_count)进行一个共享前缀的SharedPartial，所以当Key的大小不均匀时候，
    生成的DataBlock的大小可能不受我们的控制，可能会导致后续的查找效率变低

```
  [DataBlock]
  [DataBlock]
  [...]
  // 这一部分能够并行生成吗？暂时还没有看，如果能的话就好处理一点，如果不能的话，那就麻烦了。。只能让CPU端串行生成了
  // 追代码时候没有发现这部分代码有跑？
  // 这个filter就需要改变一下策略，之前是每2KB就会产生一个filter，而这里我们就控制每个data-block产生一个(或者类似)的规则去产生，这样就可以并行
  [filter]
  [meta]
  [IndexBlock]
  [footer]

 gpu_encode<<<newSSTcnt, block_cnt>>>(newSST) {
    int seg = blockIdx.x;
    int block = threadIdx.x;

    finall_sort(seg * per_SST , seg * per_SST + fdfd);  //
    for () {
        shared;
        non

        // offset ? filter?
    }
 }
```

## 原版代码部分的修改
1. ReadBlock() 函数中取出CRC校验部分，以及kNocompression部分的检验;
2. 在Restart[] 数组低8位增加SharedBlock中的KV个数
