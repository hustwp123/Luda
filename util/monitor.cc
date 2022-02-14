// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "util/monitor.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>

#include <limits>

#include "leveldb/env.h"
#include "leveldb/slice.h"

#include <hdr/hdr_histogram.h>

namespace leveldb {

  // corresponding to enum in include/leveldb/env.h
  void IOStats::Add(int entry, int64_t val) {
    // TODO wrap this translation
    switch(entry) {
      case kReadFile:
        hdr_record_value(hdr_time_read_file_micros, val);
        return ;
      case kUncrc:
        hdr_record_value(hdr_time_uncrc_micros, val);
        return ;
      case kDecomp:
        hdr_record_value(hdr_time_decompress_micros, val);
        return ;
      case kMerge:
        hdr_record_value(hdr_time_merge_cpu_micros, val);
        return ;
      case kComp:
        hdr_record_value(hdr_time_compress_micros, val);
        return ;
      case kCrc:
        hdr_record_value(hdr_time_crc_micros, val);
        return ;
      case kWrite:
        hdr_record_value(hdr_time_write_file_micros, val);
        return ;
      case kDecodeFrom:
        hdr_record_value(hdr_time_decode_micros, val);
        return ;
      case kEncodeTo:
        hdr_record_value(hdr_time_encode_micros, val);
        return ;
      case kGet:
        hdr_record_value(hdr_time_get_micros, val);
        return ;
      case kPut:
        hdr_record_value(hdr_time_put_micros, val);
        return ;
      case kGetHiccup:
        hdr_record_value(hdr_time_get_hiccup_micros, val);
        return ;
      case kPutHiccup:
        hdr_record_value(hdr_time_put_hiccup_micros, val);
        return ;
      default:
        fprintf(stderr, "ERROR! %s:%d: %s undefined IOStats entry: %d\n",
        __FILE__, __LINE__, __func__, entry);
        return ;
    }
  }

  int64_t IOStats::Percentile(const hdr_histogram* h, double p) {
    return hdr_value_at_percentile(h, p);
  }

  int64_t IOStats::Median(const hdr_histogram* h) {
    return Percentile(h, 50.0);
  }

  int64_t IOStats::Min(const hdr_histogram* h) {
    return hdr_min(h);
  }

  int64_t IOStats::Max(const hdr_histogram* h) {
    return hdr_max(h);
  }

  // borrow from hdr_histogram.c
  int64_t IOStats::HdrSum(const struct hdr_histogram* h)
  {
      struct hdr_iter iter;
      int64_t total = 0;
  
      hdr_iter_init(&iter, h);
      while (hdr_iter_next(&iter)) {
          if (0 != iter.count) {
              total += iter.count * hdr_median_equivalent_value(h, iter.value);
          }
      }
      return total;
  }

  int64_t IOStats::HdrCount(const struct hdr_histogram* h)
  {
    return h->total_count;
  }

  int64_t IOStats::Count(int entry) {
    switch(entry) {
      case kReadFile:
        return HdrCount(hdr_time_read_file_micros);
      case kUncrc:
        return HdrCount(hdr_time_uncrc_micros);
      case kDecomp:
        return HdrCount(hdr_time_decompress_micros);
      case kMerge:
        return HdrCount(hdr_time_merge_cpu_micros);
      case kComp:
        return HdrCount(hdr_time_compress_micros);
      case kCrc:
        return HdrCount(hdr_time_crc_micros);
      case kWrite:
        return HdrCount(hdr_time_write_file_micros);
      case kDecodeFrom:
        return HdrCount(hdr_time_decode_micros);
      case kEncodeTo:
        return HdrCount(hdr_time_encode_micros);
      case kGet:
        return HdrCount(hdr_time_get_micros);
      case kPut:
        return HdrCount(hdr_time_put_micros);
      case kGetHiccup:
        return HdrCount(hdr_time_get_hiccup_micros);
      case kPutHiccup:
        return HdrCount(hdr_time_put_hiccup_micros);
      default:
        fprintf(stderr, "ERROR! %s:%d: %s undefined IOStats entry: %d\n",
        __FILE__, __LINE__, __func__, entry);
        return 0;
    }
  }

  int64_t IOStats::Sum(int entry) {
    switch(entry) {
      case kReadFile:
        return HdrSum(hdr_time_read_file_micros);
      case kUncrc:
        return HdrSum(hdr_time_uncrc_micros);
      case kDecomp:
        return HdrSum(hdr_time_decompress_micros);
      case kMerge:
        return HdrSum(hdr_time_merge_cpu_micros);
      case kComp:
        return HdrSum(hdr_time_compress_micros);
      case kCrc:
        return HdrSum(hdr_time_crc_micros);
      case kWrite:
        return HdrSum(hdr_time_write_file_micros);
      case kDecodeFrom:
        return HdrSum(hdr_time_decode_micros);
      case kEncodeTo:
        return HdrSum(hdr_time_encode_micros);
      case kGet:
        return HdrSum(hdr_time_get_micros);
      case kPut:
        return HdrSum(hdr_time_put_micros);
      case kGetHiccup:
        return HdrSum(hdr_time_get_hiccup_micros);
      case kPutHiccup:
        return HdrSum(hdr_time_put_hiccup_micros);
      default:
        fprintf(stderr, "ERROR! %s:%d: %s undefined IOStats entry: %d\n",
        __FILE__, __LINE__, __func__, entry);
        return 0;
    }
  }

  int64_t IOStats::Stddev(const hdr_histogram* h) {
    return hdr_stddev(h);
  }

  // TODO hdr_percentiles_print
  // Print out a percentile based histogram to the supplied stream.

  size_t IOStats::MemSize(hdr_histogram* h) {
    return hdr_get_memory_size(h);
  }

  size_t IOStats::MemSumSize() {
    return hdr_get_memory_size(hdr_time_read_file_micros)+
      hdr_get_memory_size(hdr_time_uncrc_micros)+ 
      hdr_get_memory_size(hdr_time_decompress_micros)+ 
      hdr_get_memory_size(hdr_time_merge_cpu_micros)+ 
      hdr_get_memory_size(hdr_time_compress_micros)+ 
      hdr_get_memory_size(hdr_time_crc_micros)+ 
      hdr_get_memory_size(hdr_time_write_file_micros)+ 
      hdr_get_memory_size(hdr_time_decode_micros)+ 
      hdr_get_memory_size(hdr_time_encode_micros)+
      hdr_get_memory_size(hdr_time_get_micros)+
      hdr_get_memory_size(hdr_time_put_micros)+
      hdr_get_memory_size(hdr_time_get_hiccup_micros)+
      hdr_get_memory_size(hdr_time_put_hiccup_micros);
  }

  void IOStats::Reset(int entry) {
    switch(entry) {
      case kGetHiccup:
        hdr_reset(hdr_time_get_hiccup_micros);
        return ;
      case kPutHiccup:
        hdr_reset(hdr_time_put_hiccup_micros);
        return ;
      default:
        fprintf(stderr, "ERROR! %s:%d: %s undefined IOStats entry: %d\n",
        __FILE__, __LINE__, __func__, entry);
        return ;
    }
  }

  int64_t IOStats::Access(int entry, int op) {
    switch(entry) {
      case kGetHiccup:
        if(0 == op) { return hdr_mean(hdr_time_get_hiccup_micros); }
        else if(1 == op) { return Percentile(hdr_time_get_hiccup_micros, 99.0); }
        else if(2 == op) { return Min(hdr_time_get_hiccup_micros); }
        else if(3 == op) { return Max(hdr_time_get_hiccup_micros); }
      case kPutHiccup:
        if(0 == op) { return hdr_mean(hdr_time_put_hiccup_micros); }
        else if(1 == op) { return Percentile(hdr_time_put_hiccup_micros, 99.0); }
        else if(2 == op) { return Min(hdr_time_put_hiccup_micros); }
        else if(3 == op) { return Max(hdr_time_put_hiccup_micros); }
      default:
        fprintf(stderr, "ERROR! %s:%d: %s undefined IOStats entry: %d\n",
        __FILE__, __LINE__, __func__, entry);
        return -1;
    }
  }

  void IOStats::Clear() {
    hdr_close(hdr_time_read_file_micros);
    hdr_close(hdr_time_uncrc_micros);
    hdr_close(hdr_time_decompress_micros);
    hdr_close(hdr_time_merge_cpu_micros);
    hdr_close(hdr_time_compress_micros);
    hdr_close(hdr_time_crc_micros);
    hdr_close(hdr_time_write_file_micros);
    hdr_close(hdr_time_decode_micros);
    hdr_close(hdr_time_encode_micros);
    hdr_close(hdr_time_get_micros);
    hdr_close(hdr_time_put_micros);
    hdr_close(hdr_time_get_hiccup_micros);
    hdr_close(hdr_time_put_hiccup_micros);
    
    CloseReportFile();
  }
  
  // globally available
  // borrow from RocksDB perf_context
  IOStats iostats_;

  // globally available
  // std::string report_fname = "report.csv";
  // std::ofstream report_file(report_fname);

  std::string IOStats::GetFormattedTimeStamp() {
    std::stringstream raw_timestamp;
    auto in_time_t = 
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    raw_timestamp << std::put_time(std::localtime(&in_time_t), "%Y%m%d%H%M%S");
    return raw_timestamp.str();
  }
  
  int IOStats::OpenReportFile() {
    report_fname = "/root/logs/luda.hiccup";
    fp_report_file = fopen(report_fname.c_str(), "w+");
    if(!fp_report_file) {
      fprintf(stderr, "ERROR! %s:%d: %s open report file %s failed\n",
        __FILE__, __LINE__, __func__, report_fname.c_str());
      return EXIT_FAILURE;
    }
    fprintf(fp_report_file, "#tID  timestamp      op/s    LatMin    LatMax    LatAvg   Lat99th\n");
    fprintf(stderr, "#tID  timestamp      op/s    LatMin    LatMax    LatAvg   Lat99th\n");
    return 0;
  }

  void IOStats::AppendToReportFile(const char* contents) {
    fprintf(fp_report_file, "%s\n", contents);
  }

  void IOStats::CloseReportFile() {
    fclose(fp_report_file);
  }


}  // namespace leveldb
