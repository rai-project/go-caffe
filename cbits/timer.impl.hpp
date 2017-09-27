
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>
#include <map>
#include <type_traits>
#include <vector>

#include "json.hpp"
#include "timer.h"

using json = nlohmann::json;

using timestamp_t = std::chrono::time_point<std::chrono::high_resolution_clock>;

static timestamp_t now() { return std::chrono::high_resolution_clock::now(); }

static double elapsed_time(timestamp_t start, timestamp_t end) {
  const auto elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();
  return elapsed;
}

static uint64_t to_nanoseconds(timestamp_t t) {
  const auto duration = t.time_since_epoch();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
}

struct profile_entry {
  profile_entry(std::string name = "", std::string metadata = "")
      : name_(name), metadata_(metadata) {
    start();
  }
  ~profile_entry() {}

  error_t start() {
    start_ = now();
    return success;
  }

  error_t end() {
    end_ = now();
    return success;
  }

  json to_json() {
    const auto start_ns = to_nanoseconds(start_);
    const auto end_ns = to_nanoseconds(end_);
    uint64_t id = std::hash<std::thread::id>()(std::this_thread::get_id());
    return json{
        {"name", name_}, {"metadata", metadata_}, {"start", start_ns},
        {"end", end_ns}, {"thread_id", id},
    };
  }

  void dump() {
    const auto j = this->to_json();
    std::cout << j.dump(2) << "\n";
  }

private:
  std::string name_{""};
  std::string metadata_{""};
  timestamp_t start_{}, end_{};
};

struct profile {
  profile(std::string name = "", std::string metadata = "")
      : name_(name), metadata_(metadata) {
    start();
  }
  ~profile() { this->reset(); }

  error_t start() {
    start_ = now();
    return success;
  }

  error_t end() {
    end_ = now();
    return success;
  }

  error_t reset() {
    std::lock_guard<std::mutex> lock(mut_);
    for (auto e : entries_) {
      delete e.second;
    }
    entries_.clear();
    return success;
  }

  error_t add(int layer, profile_entry *entry) {
    std::lock_guard<std::mutex> lock(mut_);
    entries_.emplace(layer, entry);
    return success;
  }

  profile_entry* get(int layer) {
    std::lock_guard<std::mutex> lock(mut_);
    auto p = entries_.find(layer);
    if (p == entries_.end()) {
      return nullptr;
    }
    return p->second;
  }

  json to_json() {
    std::lock_guard<std::mutex> lock(mut_);
    
    const auto start_ns = to_nanoseconds(start_);
    const auto end_ns = to_nanoseconds(end_);

    json elements = json::array();
    for (const auto e : entries_) {
      elements.emplace_back(e.second->to_json());
    }
    return json{
        {"name", name_}, {"metadata", metadata_}, {"start", start_ns},
        {"end", end_ns}, {"elements", elements},
    };
  }

  void dump() {
    const auto j = this->to_json();
    std::cout << j.dump(2) << "\n";
  }

  std::string read() {
    const auto j = this->to_json();
    return j.dump();
  }

private:
  std::string name_{""};
  std::string metadata_{""};
  std::map<int, profile_entry *> entries_{};
  timestamp_t start_{}, end_{};
  std::mutex mut_;
};
