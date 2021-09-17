// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// Must not be included from any .h files to avoid polluting the namespace
// with macros.

#ifndef STORAGE_LEVELDB_UTIL_MONITOR_H_
#define STORAGE_LEVELDB_UTIL_MONITOR_H_

#include <stdint.h>
#include <stdio.h>

#include <string>
#include <fstream>
#include <sstream>

#include "port/port.h"

#include "leveldb/env.h"
#include <hdr/hdr_histogram.h>

namespace leveldb {

class Env;

class IOStats { //xp
 public:
  IOStats() {
    int ret;
    
    ret = hdr_init(1, INT64_C(3600000000), 2, &hdr_time_read_file_micros);
    ret |= hdr_init(1, INT64_C(3600000000), 2, &hdr_time_uncrc_micros);
    ret |= hdr_init(1, INT64_C(3600000000), 2, &hdr_time_decompress_micros);
    ret |= hdr_init(1, INT64_C(3600000000), 2, &hdr_time_merge_cpu_micros);
    ret |= hdr_init(1, INT64_C(3600000000), 2, &hdr_time_compress_micros);
    ret |= hdr_init(1, INT64_C(3600000000), 2, &hdr_time_crc_micros);
    ret |= hdr_init(1, INT64_C(3600000000), 2, &hdr_time_write_file_micros);
    ret |= hdr_init(1, INT64_C(3600000000), 2, &hdr_time_decode_micros);
    ret |= hdr_init(1, INT64_C(3600000000), 2, &hdr_time_encode_micros);
    ret |= hdr_init(1, INT64_C(3600000000), 2, &hdr_time_get_micros);
    ret |= hdr_init(1, INT64_C(3600000000), 2, &hdr_time_put_micros);
    ret |= hdr_init(1, INT64_C(3600000000), 2, &hdr_time_get_hiccup_micros);
    ret |= hdr_init(1, INT64_C(3600000000), 2, &hdr_time_put_hiccup_micros);
    if((0 != ret) ||
        (NULL == hdr_time_read_file_micros) || (NULL == hdr_time_uncrc_micros) ||
        (NULL == hdr_time_decompress_micros) || (NULL == hdr_time_merge_cpu_micros) ||
        (NULL == hdr_time_compress_micros) || (NULL == hdr_time_crc_micros) || 
        (NULL == hdr_time_write_file_micros) || (NULL == hdr_time_decode_micros) || 
        (NULL == hdr_time_encode_micros) || (NULL == hdr_time_get_micros) ||
        (NULL == hdr_time_put_micros) || (NULL == hdr_time_get_hiccup_micros) ||
        (NULL == hdr_time_put_hiccup_micros)) {
      fprintf(stderr, "ERROR! %s:%d: %s hdr initailization failed %d\n", 
        __FILE__, __LINE__, __func__, ret);
    }

    OpenReportFile();
  }

  ~IOStats() {
    Clear();
  }

  void Add(int entry, int64_t val);

  int64_t Percentile(const hdr_histogram* h, double p);
  // int64_t Percentile(int entry, const hdr_histogram* h, double p);

  int64_t Median(const hdr_histogram* h);

  int64_t Min(const hdr_histogram* h);

  int64_t Max(const hdr_histogram* h);

  int64_t HdrSum(const struct hdr_histogram* h);

  int64_t Sum(int entry);

  int64_t Count(int entry);

  int64_t HdrCount(const struct hdr_histogram* h);

  int64_t Stddev(const hdr_histogram* h);
  
  // TODO hdr_percentiles_print
  // Print out a percentile based histogram to the supplied stream.

  size_t MemSize(hdr_histogram* h);

  size_t MemSumSize();

  int64_t Access(int entry, int op);

  void Reset(int entry);
  
  void Clear();

  int OpenReportFile();
  void AppendToReportFile(const char* contents); // TODO
  void CloseReportFile();
  std::string GetFormattedTimeStamp();

  std::string report_fname;
  FILE* fp_report_file;

  // correspond to enum in include/leveldb/env.h
  struct hdr_histogram* hdr_time_read_file_micros; //read file

  struct hdr_histogram* hdr_time_uncrc_micros; // crc32::Unmask()
  struct hdr_histogram* hdr_time_decompress_micros; // Snappy_Uncompress()

  struct hdr_histogram* hdr_time_merge_cpu_micros; // input->Next() in DoCompactionWork()

  struct hdr_histogram* hdr_time_compress_micros; // Snappy_Compress()
  struct hdr_histogram* hdr_time_crc_micros; // crc32c::Value()

  struct hdr_histogram* hdr_time_write_file_micros; //read file

  struct hdr_histogram* hdr_time_decode_micros; // DecodeFrom()
  struct hdr_histogram* hdr_time_encode_micros; // EncodeTo()

  struct hdr_histogram* hdr_time_get_micros; // VersionSet::Get()
  struct hdr_histogram* hdr_time_get_hiccup_micros; // db_bench.cc
  struct hdr_histogram* hdr_time_put_micros; 
  struct hdr_histogram* hdr_time_put_hiccup_micros; // DoWrite() db_bench.cc
};


}  // namespace leveldb

#endif  // STORAGE_LEVELDB_UTIL_MONITOR_H_
