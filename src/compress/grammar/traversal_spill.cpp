#include "compress/grammar/traversal_spill.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <fcntl.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gfaz {

namespace {

constexpr const char *kSpillErrorPrefix = "Traversal spill error: ";

// Record encodings; see the header. GFAZ_SPILL_CODEC=raw|varint pins one of
// the two simple ones for every record.
constexpr uint32_t kCodecVarint = 0;
constexpr uint32_t kCodecRaw = 1;
constexpr uint32_t kCodecTable = 2; // + table index

enum class ForcedCodec { kNone, kRaw, kVarint };
ForcedCodec forced_codec() {
  static const ForcedCodec forced = [] {
    const char *value = std::getenv("GFAZ_SPILL_CODEC");
    if (value != nullptr && std::strcmp(value, "raw") == 0)
      return ForcedCodec::kRaw;
    if (value != nullptr && std::strcmp(value, "varint") == 0)
      return ForcedCodec::kVarint;
    return ForcedCodec::kNone;
  }();
  return forced;
}

// Table codes: one byte for ranks below kOneByteRanks, two bytes up to
// kTableRanks, and kEscape followed by the int32 for any other symbol.
constexpr uint32_t kOneByteRanks = 128;
constexpr uint32_t kTableRanks = kOneByteRanks + 127 * 256;
constexpr uint8_t kEscape = 0xFF;
constexpr size_t kTableSlots = size_t{1} << 16; // at most half full
constexpr size_t kTrainingSample = size_t{1} << 18;
constexpr size_t kMaxTablesPerPass = 1024;

inline size_t table_slot(NodeId symbol) {
  return (static_cast<uint32_t>(symbol) * 2654435761u) >> 16;
}

// GFAZ_SPILL_DROP_CACHE=1 evicts spill pages from the page cache as soon as
// they are written or read, so the temp file does not occupy RAM either.
bool drop_page_cache() {
  static const bool drop = [] {
    const char *value = std::getenv("GFAZ_SPILL_DROP_CACHE");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return drop;
}

size_t encode_raw(const std::vector<NodeId> &sequence,
                  std::vector<uint8_t> &buffer) {
  const size_t bytes = sequence.size() * sizeof(NodeId);
  if (buffer.size() < bytes)
    buffer.resize(bytes);
  std::memcpy(buffer.data(), sequence.data(), bytes);
  return bytes;
}

// Returns the encoded size; `escapes` counts the symbols not in the table.
size_t encode_with_table(const std::vector<NodeId> &sequence,
                         const std::vector<NodeId> &keys,
                         const std::vector<uint16_t> &ranks,
                         std::vector<uint8_t> &buffer, size_t &escapes) {
  constexpr size_t kBlockSteps = 64 * 1024;
  constexpr size_t kMaxBytes = 5;
  constexpr size_t kMask = kTableSlots - 1;
  size_t used = 0;
  escapes = 0;
  for (size_t begin = 0; begin < sequence.size(); begin += kBlockSteps) {
    const size_t end = std::min(sequence.size(), begin + kBlockSteps);
    const size_t needed = used + (end - begin) * kMaxBytes;
    if (buffer.size() < needed)
      buffer.resize(std::max(needed, buffer.size() + buffer.size() / 2));
    uint8_t *out = buffer.data() + used;
    for (size_t i = begin; i < end; ++i) {
      const NodeId symbol = sequence[i];
      size_t slot = table_slot(symbol);
      uint32_t rank_plus_one;
      while ((rank_plus_one = ranks[slot]) != 0 && keys[slot] != symbol)
        slot = (slot + 1) & kMask;
      if (rank_plus_one == 0) {
        *out++ = kEscape;
        std::memcpy(out, &symbol, sizeof(symbol));
        out += sizeof(symbol);
        ++escapes;
        continue;
      }
      const uint32_t rank = rank_plus_one - 1;
      if (rank < kOneByteRanks) {
        *out++ = static_cast<uint8_t>(rank);
      } else {
        const uint32_t rest = rank - kOneByteRanks;
        *out++ = static_cast<uint8_t>(0x80 + (rest >> 8));
        *out++ = static_cast<uint8_t>(rest);
      }
    }
    used = static_cast<size_t>(out - buffer.data());
  }
  return used;
}

void decode_with_table(const uint8_t *in, size_t steps,
                       const std::vector<NodeId> &symbols,
                       std::vector<NodeId> &out) {
  out.resize(steps);
  NodeId *dst = out.data();
  NodeId *const end = dst + steps;
  const NodeId *by_rank = symbols.data();
  while (dst < end) {
    const uint32_t first = *in++;
    if (first < kOneByteRanks) {
      *dst++ = by_rank[first];
    } else if (first != kEscape) {
      *dst++ = by_rank[kOneByteRanks + ((first - 0x80) << 8) + *in++];
    } else {
      std::memcpy(dst++, in, sizeof(NodeId));
      in += sizeof(NodeId);
    }
  }
}

size_t encode_varint(const std::vector<NodeId> &sequence,
                     std::vector<uint8_t> &buffer) {
  // Grow the buffer block by block instead of reserving the 5-bytes-per-step
  // worst case: typical records need about one byte per step.
  constexpr size_t kBlockSteps = 64 * 1024;
  constexpr size_t kMaxVarintBytes = 5;
  size_t used = 0;
  for (size_t begin = 0; begin < sequence.size(); begin += kBlockSteps) {
    const size_t end = std::min(sequence.size(), begin + kBlockSteps);
    const size_t needed = used + (end - begin) * kMaxVarintBytes;
    if (buffer.size() < needed)
      buffer.resize(std::max(needed, buffer.size() + buffer.size() / 2));
    uint8_t *out = buffer.data() + used;
    for (size_t i = begin; i < end; ++i) {
      const NodeId value = sequence[i];
      uint32_t zigzag = (static_cast<uint32_t>(value) << 1) ^
                        static_cast<uint32_t>(value >> 31);
      while (zigzag >= 0x80) {
        *out++ = static_cast<uint8_t>(zigzag | 0x80);
        zigzag >>= 7;
      }
      *out++ = static_cast<uint8_t>(zigzag);
    }
    used = static_cast<size_t>(out - buffer.data());
  }
  return used;
}

void decode_raw_or_varint(const uint8_t *in, size_t bytes, size_t steps,
                          bool raw, std::vector<NodeId> &out) {
  out.resize(steps);
  if (raw) {
    std::memcpy(out.data(), in, bytes);
    return;
  }

  NodeId *dst = out.data();
  NodeId *const end = dst + steps;
  while (dst < end) {
    uint32_t byte = *in++;
    uint32_t zigzag = byte & 0x7f;
    if (byte & 0x80) {
      unsigned shift = 7;
      do {
        byte = *in++;
        zigzag |= (byte & 0x7f) << shift;
        shift += 7;
      } while (byte & 0x80);
    }
    *dst++ = static_cast<NodeId>((zigzag >> 1) ^ (0u - (zigzag & 1)));
  }
}

// Same transform as Codec::delta_transform_and_max_abs, for one traversal.
uint32_t delta_transform_one(std::vector<NodeId> &sequence) {
  if (sequence.empty())
    return 0;
  uint32_t local_max = static_cast<uint32_t>(std::abs(sequence[0]));
  for (size_t i = sequence.size() - 1; i > 0; --i) {
    sequence[i] = sequence[i] - sequence[i - 1];
    const uint32_t abs_val = static_cast<uint32_t>(std::abs(sequence[i]));
    if (abs_val > local_max)
      local_max = abs_val;
  }
  return local_max;
}

uint32_t max_abs_one(const std::vector<NodeId> &sequence) {
  uint32_t local_max = 0;
  for (const NodeId node : sequence) {
    const uint32_t abs_val = static_cast<uint32_t>(std::abs(node));
    if (abs_val > local_max)
      local_max = abs_val;
  }
  return local_max;
}

} // namespace

TraversalSpill::TraversalSpill(size_t memory_budget_bytes, std::string tmp_dir,
                               int delta_round)
    : memory_budget_bytes_(memory_budget_bytes), tmp_dir_(std::move(tmp_dir)),
      delta_round_(delta_round) {
  if (tmp_dir_.empty()) {
    const char *env_tmp = std::getenv("TMPDIR");
    tmp_dir_ = (env_tmp != nullptr && *env_tmp != '\0') ? env_tmp : "/tmp";
  }
}

TraversalSpill::~TraversalSpill() {
  if (read_fd_ != -1)
    close(read_fd_);
  if (write_fd_ != -1)
    close(write_fd_);
}

int TraversalSpill::open_temp_file() {
  int fd = -1;
#ifdef O_TMPFILE
  fd = open(tmp_dir_.c_str(), O_TMPFILE | O_RDWR | O_CLOEXEC, 0600);
#endif
  if (fd == -1) {
    // Filesystems without O_TMPFILE: create a named file and unlink it at
    // once so it disappears when the process exits, however it exits.
    std::string name = tmp_dir_ + "/gfaz-spill-XXXXXX";
    fd = mkstemp(name.data());
    if (fd != -1)
      unlink(name.c_str());
  }
  if (fd == -1) {
    throw std::runtime_error(std::string(kSpillErrorPrefix) +
                             "cannot create a temp file in '" + tmp_dir_ +
                             "': " + std::strerror(errno));
  }
  return fd;
}

void TraversalSpill::ensure_thread_buffers() {
  size_t threads = 1;
#ifdef _OPENMP
  threads = static_cast<size_t>(std::max(1, omp_get_max_threads()));
#endif
  if (thread_buffers_.size() < threads)
    thread_buffers_.resize(threads);
}

TraversalSpill::ThreadBuffers &TraversalSpill::thread_buffers() {
#ifdef _OPENMP
  return thread_buffers_[static_cast<size_t>(omp_get_thread_num())];
#else
  return thread_buffers_[0];
#endif
}

std::vector<NodeId> &TraversalSpill::scratch() {
  return thread_buffers().sequence;
}

void TraversalSpill::release_thread_buffers() {
  for (auto &buffers : thread_buffers_) {
    std::vector<NodeId>().swap(buffers.sequence);
    std::vector<uint8_t>().swap(buffers.io);
    std::vector<uint8_t>().swap(buffers.io_alternate);
    buffers.table = SymbolTable{};
  }
}

void TraversalSpill::fail(const std::string &message) {
  std::lock_guard<std::mutex> lock(error_mutex_);
  if (!failed_.load()) {
    error_message_ = message;
    failed_.store(true);
  }
}

void TraversalSpill::throw_if_failed() const {
  if (!failed_.load())
    return;
  std::lock_guard<std::mutex> lock(error_mutex_);
  throw std::runtime_error(std::string(kSpillErrorPrefix) + error_message_);
}

void TraversalSpill::prepare(size_t num_paths, size_t num_walks) {
  num_paths_ = num_paths;
  num_walks_ = num_walks;
  records_.assign(num_paths + num_walks, Record{});
  original_lengths_.assign(num_paths + num_walks, 0);
  memory_bytes_.store(0);
  spilled_count_.store(0);
  spilled_steps_.store(0);
  max_abs_.store(0);
  remap_.clear();
  read_tables_.clear();
  write_tables_.clear();
  rewriting_ = false;
  ensure_thread_buffers();

  if (read_fd_ == -1)
    read_fd_ = open_temp_file();
  if (write_fd_ == -1)
    write_fd_ = open_temp_file();
  if (ftruncate(read_fd_, 0) != 0 || ftruncate(write_fd_, 0) != 0)
    fail(std::string("cannot reset temp file: ") + std::strerror(errno));
  read_cursor_.store(0);
  write_cursor_.store(0);
}

bool TraversalSpill::try_reserve_memory(size_t elements) {
  const size_t bytes = elements * sizeof(NodeId);
  size_t current = memory_bytes_.load();
  while (current + bytes <= memory_budget_bytes_) {
    if (memory_bytes_.compare_exchange_weak(current, current + bytes))
      return true;
  }
  return false;
}

void TraversalSpill::train_table(SymbolTable &table,
                                 const std::vector<NodeId> &sequence) {
  // Count symbols in an evenly spaced sample and keep the most frequent.
  const size_t stride = std::max<size_t>(1, sequence.size() / kTrainingSample);
  std::vector<NodeId> sample;
  sample.reserve(sequence.size() / stride + 1);
  for (size_t i = 0; i < sequence.size(); i += stride)
    sample.push_back(sequence[i]);
  std::sort(sample.begin(), sample.end());

  std::vector<std::pair<uint32_t, NodeId>> counts; // (count, symbol)
  for (size_t i = 0; i < sample.size();) {
    size_t j = i + 1;
    while (j < sample.size() && sample[j] == sample[i])
      ++j;
    counts.emplace_back(static_cast<uint32_t>(j - i), sample[i]);
    i = j;
  }
  const size_t kept = std::min<size_t>(counts.size(), kTableRanks);
  std::partial_sort(counts.begin(), counts.begin() + kept, counts.end(),
                    [](const auto &a, const auto &b) {
                      return a.first != b.first ? a.first > b.first
                                                : a.second < b.second;
                    });

  std::vector<NodeId> symbols(kept);
  table.keys.assign(kTableSlots, 0);
  table.ranks.assign(kTableSlots, 0);
  for (size_t rank = 0; rank < kept; ++rank) {
    const NodeId symbol = counts[rank].second;
    symbols[rank] = symbol;
    size_t slot = table_slot(symbol);
    while (table.ranks[slot] != 0)
      slot = (slot + 1) & (kTableSlots - 1);
    table.keys[slot] = symbol;
    table.ranks[slot] = static_cast<uint16_t>(rank + 1);
  }

  std::lock_guard<std::mutex> lock(table_mutex_);
  table.index = static_cast<uint32_t>(write_tables_.size());
  write_tables_.push_back(std::move(symbols));
  table.ready = true;
}

void TraversalSpill::write_record(int fd, std::atomic<uint64_t> &cursor,
                                  size_t i,
                                  const std::vector<NodeId> &sequence,
                                  bool rewritten) {
  auto &buffers = thread_buffers();
  auto &buffer = buffers.io;
  const size_t steps = sequence.size();
  const ForcedCodec forced = forced_codec();
  uint32_t codec = forced == ForcedCodec::kRaw ? kCodecRaw : kCodecVarint;
  size_t bytes = 0;
  bool encoded = false;

  if (rewritten && forced == ForcedCodec::kNone) {
    SymbolTable &table = buffers.table;
    if (!table.useless) {
      size_t escapes = 0;
      bool fresh = false;
      if (!table.ready) {
        train_table(table, sequence);
        fresh = true;
      }
      bytes = encode_with_table(sequence, table.keys, table.ranks, buffer,
                                escapes);
      // A table that misses many symbols was learned from different data:
      // learn a new one from this record, unless too many exist already.
      if (!fresh && escapes > steps / 8) {
        bool room;
        {
          std::lock_guard<std::mutex> lock(table_mutex_);
          room = write_tables_.size() < kMaxTablesPerPass;
        }
        if (room) {
          train_table(table, sequence);
          fresh = true;
          bytes = encode_with_table(sequence, table.keys, table.ranks, buffer,
                                    escapes);
        }
      }
      // Even a table learned from this very record misses half of it: the
      // symbols are too spread out (later rounds). Stop trying.
      if (fresh && steps >= 4096 && escapes > steps / 2)
        table.useless = true;
      codec = kCodecTable + table.index;
      encoded = true;
    }
    // Plain int32 is the bound to beat; small values do better as varints.
    if (!encoded || bytes >= steps * sizeof(NodeId)) {
      uint32_t any_bits = 0;
      for (const NodeId value : sequence)
        any_bits |= value < 0 ? 0u - static_cast<uint32_t>(value)
                              : static_cast<uint32_t>(value);
      const bool small_values = any_bits < (uint32_t{1} << 20);
      if (small_values) {
        std::vector<uint8_t> &other = buffers.io_alternate;
        const size_t varint_bytes = encode_varint(sequence, other);
        if (!encoded || varint_bytes < bytes) {
          buffer.swap(other);
          bytes = varint_bytes;
          codec = kCodecVarint;
          encoded = true;
        }
      } else {
        bytes = encode_raw(sequence, buffer);
        codec = kCodecRaw;
        encoded = true;
      }
    }
  }
  if (!encoded) {
    bytes = codec == kCodecRaw ? encode_raw(sequence, buffer)
                               : encode_varint(sequence, buffer);
  }
  const uint64_t offset = cursor.fetch_add(bytes);

  size_t done = 0;
  while (done < bytes) {
    const ssize_t written =
        pwrite(fd, buffer.data() + done, bytes - done,
               static_cast<off_t>(offset + done));
    if (written < 0) {
      if (errno == EINTR)
        continue;
      fail("write to temp file in '" + tmp_dir_ +
           "' failed: " + std::strerror(errno));
      return;
    }
    done += static_cast<size_t>(written);
  }
  if (drop_page_cache()) {
    sync_file_range(fd, static_cast<off64_t>(offset),
                    static_cast<off64_t>(bytes),
                    SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE |
                        SYNC_FILE_RANGE_WAIT_AFTER);
    posix_fadvise(fd, static_cast<off_t>(offset), static_cast<off_t>(bytes),
                  POSIX_FADV_DONTNEED);
  }

  records_[i] =
      Record{offset, bytes, static_cast<uint32_t>(sequence.size()), codec};
  total_written_bytes_.fetch_add(bytes);
}

void TraversalSpill::consume(bool is_walk, size_t index,
                             std::vector<NodeId> &sequence,
                             std::vector<NodeId> &slot) {
  const size_t i = is_walk ? num_paths_ + index : index;
  original_lengths_[i] = static_cast<uint32_t>(sequence.size());
  slot.clear();
  if (sequence.empty() || failed_.load(std::memory_order_relaxed))
    return;

  uint32_t local_max = delta_round_ == 0 ? max_abs_one(sequence) : 0;
  for (int round = 0; round < delta_round_; ++round)
    local_max = std::max(local_max, delta_transform_one(sequence));
  uint32_t seen_max = max_abs_.load(std::memory_order_relaxed);
  while (local_max > seen_max &&
         !max_abs_.compare_exchange_weak(seen_max, local_max)) {
  }

  if (try_reserve_memory(sequence.size())) {
    slot = std::move(sequence);
    sequence.clear();
    return;
  }
  write_record(read_fd_, read_cursor_, i, sequence, false);
  spilled_count_.fetch_add(1);
  spilled_steps_.fetch_add(sequence.size());
}

void TraversalSpill::load(size_t i, std::vector<NodeId> &out) {
  const Record record = records_[i];
  if (record.bytes == 0 || failed_.load(std::memory_order_relaxed)) {
    out.clear();
    return;
  }

  auto &buffer = thread_buffers().io;
  if (buffer.size() < record.bytes)
    buffer.resize(record.bytes);
  size_t done = 0;
  while (done < record.bytes) {
    const ssize_t got = pread(read_fd_, buffer.data() + done,
                              record.bytes - done,
                              static_cast<off_t>(record.offset + done));
    if (got < 0 && errno == EINTR)
      continue;
    if (got <= 0) {
      fail(std::string("read from temp file failed: ") +
           (got == 0 ? "unexpected end of file" : std::strerror(errno)));
      out.clear();
      return;
    }
    done += static_cast<size_t>(got);
  }
  if (drop_page_cache()) {
    posix_fadvise(read_fd_, static_cast<off_t>(record.offset),
                  static_cast<off_t>(record.bytes), POSIX_FADV_DONTNEED);
  }

  if (record.codec >= kCodecTable) {
    decode_with_table(buffer.data(), record.steps,
                      read_tables_[record.codec - kCodecTable], out);
  } else {
    decode_raw_or_varint(buffer.data(), record.bytes, record.steps,
                         record.codec == kCodecRaw, out);
  }

  if (!remap_.empty()) {
    for (NodeId &node : out) {
      const uint32_t abs_id = static_cast<uint32_t>(std::abs(node));
      if (abs_id < remap_start_id_)
        continue;
      const uint32_t offset = abs_id - remap_start_id_;
      if (offset >= remap_.size() || remap_[offset] == 0)
        continue;
      node = node > 0 ? static_cast<NodeId>(remap_[offset])
                      : -static_cast<NodeId>(remap_[offset]);
    }
  }
}

void TraversalSpill::begin_rewrite() {
  ensure_thread_buffers();
  for (auto &buffers : thread_buffers_)
    buffers.table = SymbolTable{};
  write_tables_.clear();
  write_cursor_.store(0);
  rewriting_ = true;
}

void TraversalSpill::store(size_t i, std::vector<NodeId> &sequence,
                           std::vector<NodeId> &slot) {
  const uint32_t old_steps = records_[i].steps;
  if (sequence.empty() || try_reserve_memory(sequence.size())) {
    slot.assign(sequence.begin(), sequence.end());
    records_[i] = Record{};
    spilled_count_.fetch_sub(1);
    spilled_steps_.fetch_sub(old_steps);
    return;
  }
  write_record(write_fd_, write_cursor_, i, sequence, true);
  spilled_steps_.fetch_sub(old_steps - sequence.size());
}

void TraversalSpill::end_rewrite() {
  peak_file_bytes_ = std::max<size_t>(
      peak_file_bytes_, read_cursor_.load() + write_cursor_.load());
  std::swap(read_fd_, write_fd_);
  read_cursor_.store(write_cursor_.load());
  write_cursor_.store(0);
  if (ftruncate(write_fd_, 0) != 0)
    fail(std::string("cannot reset temp file: ") + std::strerror(errno));
  read_tables_ = std::move(write_tables_);
  write_tables_.clear();
  // Every spilled record was just rewritten with the pending remap applied.
  remap_.clear();
  remap_.shrink_to_fit();
  rewriting_ = false;
  release_thread_buffers();
}

void TraversalSpill::set_pending_remap(uint32_t rules_start_id,
                                       std::vector<uint32_t> id_map) {
  remap_start_id_ = rules_start_id;
  remap_ = std::move(id_map);
}

void TraversalSpill::materialize_all(std::vector<std::vector<NodeId>> &paths,
                                     std::vector<std::vector<NodeId>> &walks) {
  peak_file_bytes_ = std::max<size_t>(peak_file_bytes_, read_cursor_.load());
  ensure_thread_buffers();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t i = 0; i < records_.size(); ++i) {
    if (records_[i].bytes == 0)
      continue;
    auto &slot = i < num_paths_ ? paths[i] : walks[i - num_paths_];
    load(i, slot);
    memory_bytes_.fetch_add(slot.size() * sizeof(NodeId));
    records_[i] = Record{};
  }
  spilled_count_.store(0);
  spilled_steps_.store(0);
  remap_.clear();
  remap_.shrink_to_fit();
  read_tables_.clear();
  read_tables_.shrink_to_fit();
  thread_buffers_.clear();
  thread_buffers_.shrink_to_fit();
  if (ftruncate(read_fd_, 0) != 0)
    fail(std::string("cannot reset temp file: ") + std::strerror(errno));
  read_cursor_.store(0);
}

} // namespace gfaz
