/*
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef DA_CACHE_HPP
#define DA_CACHE_HPP

#include "aoclda.h"
#include "da_error.hpp"
#include "macros.h"
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

/**
  * LRU (Least Recently Used) Cache implementation
  */
namespace da_cache {
template <typename T> class LRUCache {
  private:
    // Maximum number of items the cache can hold
    da_int capacity_;
    // Length of each item
    da_int len_;

    // List to track usage order (front = most recently used)
    std::list<std::pair<da_int, std::vector<T>>> items_;

    // Map for O(1) access to items in the list
    std::unordered_map<da_int,
                       typename std::list<std::pair<da_int, std::vector<T>>>::iterator>
        cache_;

  public:
    explicit LRUCache() {}

    /**
      * Get value for key and mark it as most recently used
      * @param key The key to look up
      * @return Pointer to the value if found, nullptr otherwise
      */
    T *get(const da_int &key) {
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return nullptr; // Key not found
        }

        // Move the accessed item to the front of the list
        items_.splice(items_.begin(), items_, it->second);
        return it->second->second.data();
    }

    /**
      * Add or update a key-value pair
      * @param key The key to insert or update
      * @param value The value to associate with the key
      */
    void put(const da_int &key, const T *value) {
        auto it = cache_.find(key);
        if (it != cache_.end() || capacity_ <= 0) {
            return; // Key already exists, no need to insert again
        }

        // Check if cache is full
        if ((da_int)items_.size() >= capacity_) {
            // Remove the least recently used item (back of the list)
            da_int lru_key = items_.back().first;
            items_.pop_back();
            cache_.erase(lru_key);
        }

        // Add new item to the front
        items_.emplace_front(key, std::vector<T>(value, value + len_));
        cache_[key] = items_.begin();
    }

    /**
      * Set size of cache
      * @param capacity Size of cache in "number of rows" it can hold
      * @param len Number of columns for a single row that is being held
      */
    void set_size(const da_int &capacity, const da_int &len) {
        capacity_ = capacity;
        len_ = len;
    }

    /**
      * Clear the cache
      */
    void clear() {
        items_.clear();
        cache_.clear();
    }
};
} // namespace da_cache

#endif