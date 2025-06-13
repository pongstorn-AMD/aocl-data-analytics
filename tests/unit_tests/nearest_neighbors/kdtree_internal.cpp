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

// Test code for k-d tree construction (radius neighbors and k neighbors functionality is tested in
// radius_neighbors_internal and k_neighbors_internal)

#include <iostream>
#include <limits>
#include <list>
#include <numeric>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#include "../utest_utils.hpp"
#include "aoclda.h"
#include "kdtree.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

typedef struct params_t {
    std::string test_name;
    std::vector<double> A;
    std::vector<da_int> expected_indices;
    da_int n_samples;
    da_int n_features;
    da_int lda;
    da_int leaf_size;
} params;

const params_t kdtree_params[] = {
    {"kdtree_test1", {0.0, 1.0, 0.0, 2.0}, {0, 1}, 2, 2, 2, 2},
    {"kdtree_test2", {0.0, 1.0, 0.0, 2.0}, {0, 1}, 2, 2, 2, 1},
    {"kdtree_test3",
     {5.0, 2.0, 1.0, 3.0, 7.0, 4.0, 6.0, 0.0},
     {7, 2, 1, 3, 5, 0, 6, 4},
     8,
     1,
     8,
     1},
    {"kdtree_test4",
     {5.0, 8.0, 3.0, 1.0, 2.0, 4.0, 7.0, 0.0, 6.0, 5.0, 1.0, 4.0, 8.0, 3.0, 7.0, 0.0, 2.0,
      6.0},
     {7, 4, 2, 3, 5, 6, 1, 8, 0},
     9,
     2,
     9,
     1},
    {"kdtree_test5",
     {7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
     {5, 6, 7, 4, 0, 1, 2, 3},
     8,
     2,
     8,
     1},
};

class kdtreeSmallTests : public testing::TestWithParam<params> {};

template <class T> void test_kdtree(const params pr) {
    std::vector<T> A = convert_vector<double, T>(pr.A);
    da_int n_samples = pr.n_samples;
    da_int n_features = pr.n_features;
    da_int lda = pr.lda;
    da_int leaf_size = pr.leaf_size;

    auto tree =
        TEST_ARCH::da_kdtree::kdtree<T>(n_samples, n_features, A.data(), lda, leaf_size);

    // Check if the indices in the k-d tree match the expected indices, allowing for consecutive pairs
    // to be swapped because of the way the k-d tree is built and terminates with some leaves of size 2
    // meaning that depending on the nth_element implementation, the order of such pairs may vary
    da_int i = 0;
    while (i < n_samples) {
        if (pr.expected_indices[i] == tree.get_indices()[i]) {
            i += 1;
            continue;
        } else if (pr.expected_indices[i] == tree.get_indices()[i + 1] &&
                   pr.expected_indices[i + 1] == tree.get_indices()[i]) {
            i += 2;
            continue;
        } else {
            FAIL() << "k-d tree indices do not match expected values for large test";
        }
    }
}

void PrintTo(const params &param, ::std::ostream *os) { *os << param.test_name; }

TEST_P(kdtreeSmallTests, double) {
    const params &pr = GetParam();
    test_kdtree<double>(pr);
}

TEST_P(kdtreeSmallTests, float) {
    const params &pr = GetParam();
    test_kdtree<float>(pr);
}

INSTANTIATE_TEST_SUITE_P(kdtreeSuite, kdtreeSmallTests, testing::ValuesIn(kdtree_params));

template <typename T> class kdtreeLargeTest : public testing::Test {
  public:
    using List = std::list<T>;
    static T shared_;
    T value_;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(kdtreeLargeTest, FloatTypes);

TYPED_TEST(kdtreeLargeTest, large) {
    da_int n_samples = 500;
    da_int n_features = 3;
    da_int lda = n_samples;
    da_int leaf_size = 1;

    std::vector<double> A_in(n_samples * n_features);
    std::vector<da_int> expected_indices(n_samples);

    for (da_int i = 0; i < n_samples; i++) {
        for (da_int j = 0; j < n_features; j++) {
            A_in[i + lda * j] = static_cast<double>(n_samples - i - 1);
        }
        expected_indices[i] = n_samples - i - 1;
    }

    std::vector<TypeParam> A = convert_vector<double, TypeParam>(A_in);

    auto tree = TEST_ARCH::da_kdtree::kdtree<TypeParam>(n_samples, n_features, A.data(),
                                                        lda, leaf_size);

    // Check if the indices in the k-d tree match the expected indices, allowing for consecutive pairs
    // to be swapped because of the way the k-d tree is built and terminates with some leaves of size 2
    da_int i = 0;
    while (i < n_samples) {
        if (expected_indices[i] == tree.get_indices()[i]) {
            i += 1;
            continue;
        } else if (expected_indices[i] == tree.get_indices()[i + 1] &&
                   expected_indices[i + 1] == tree.get_indices()[i]) {
            i += 2;
            continue;
        } else {
            FAIL() << "k-d tree indices do not match expected values for large test";
        }
    }
}