// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// Must not be included from any .h files to avoid polluting the namespace
// with macros.

#ifndef STORAGE_LEVELDB_UTIL_MONITOR_IMPL_H_
#define STORAGE_LEVELDB_UTIL_MONITOR_IMPL_H_

#include <stdint.h>
#include <stdio.h>

#include <string>

#include "port/port.h"

#include "leveldb/env.h"
#include <hdr/hdr_histogram.h>
#include "util/monitor.h"
// #include <chrono>

namespace leveldb {

extern IOStats iostats_;

}  // namespace leveldb

#endif  // STORAGE_LEVELDB_UTIL_MONITOR_IMPL_H_
