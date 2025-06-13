/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "aoclda.h"
#include "basic_handle.hpp"
#include "da_error.hpp"
#include "da_vector.hpp"
#include <algorithm>
#include <memory>
#include <vector>

namespace ARCH {

namespace da_kdtree {

/* Node structure for the k-d tree class */

template <typename T> struct node {

    // Which dimension this node splits on
    da_int dim;

    // The depth of this node
    da_int depth;

    // For a non-leaf node, the index of the point that splits the node
    da_int point;

    // Indices of the points in the dataset that are in this node and its children
    da_int *indices = nullptr;
    da_int n_indices;

    // Is this a leaf node
    bool is_leaf = false;

    // shared_ptr to the child nodes means we don't have to worry about memory management
    std::shared_ptr<node<T>> left_child = nullptr;
    std::shared_ptr<node<T>> right_child = nullptr;

    // bounding box for the node
    std::vector<T> min_bounds;
    std::vector<T> max_bounds;

    //T r2 = 0.0; // squared radius of the bounding box, used for pruning

    // Constructors - only the top level node need have min_bounds and max_bounds supplied
    node(da_int dim, da_int depth, da_int *indices, da_int n_indices,
         std::vector<T> min_bounds, std::vector<T> max_bounds);
    node(da_int dim, da_int depth, da_int *indices, da_int n_indices);
};

// Lightweight partial MaxHeap implementation to keep track of k-NN KDTree searches
template <typename T> class MaxHeap {
  public:
    // Constructor from existing arrays of indices and distances
    MaxHeap(da_int capacity, da_int *indices, T *distances);

    // Get the current maximum distance in the heap
    T GetMaxDist();

    da_int GetSize();

    // Insert a new point to the heap, if it is small enough, maintaining the max-heap property
    void Insert(da_int index, T distance);

  private:
    da_int *indices = nullptr; // Indices of the points in the dataset
    T *distances = nullptr;    // Distances of the points in the dataset
    da_int capacity = 0;       // Maximum capacity of the heap
    da_int size = 0;           // Current size of the heap

    void heapify_up(da_int index);
    void heapify_down(da_int index);
};

/* k-d tree class */
template <typename T> class kdtree {
  public:
    kdtree(da_int n_samples, da_int n_features, const T *A, da_int lda,
           da_int leaf_size = 30);

    da_status radius_neighbors(da_int m_samples_in, da_int m_features_in, const T *X_in,
                               da_int ldx_in, T eps, da_metric metric_in, T p_in,
                               std::vector<da_vector::da_vector<da_int>> &neighbors,
                               da_errors::da_error_t *err);

    da_status k_neighbors(da_int m_samples_in, da_int m_features_in, const T *X_in,
                          da_int ldx_in, da_int k, da_metric metric_in, T p_in,
                          da_int *k_ind, T *k_dist, da_errors::da_error_t *err);

    // Get the indices, for testing purposes
    const std::vector<da_int> &get_indices();

  private:
    // Build the k-d tree from the dataset
    std::shared_ptr<node<T>> build_tree(da_int depth, da_int *indices, da_int n_indices,
                                        std::vector<T> *min_bounds = nullptr,
                                        std::vector<T> *max_bounds = nullptr);

    // Internal functions used in tree construction and tree traversal to find neighbours
    da_status radius_neighbors_recursive(std::shared_ptr<node<T>> current_node, da_int n,
                                         T *X, T eps, T eps_internal, da_metric metric,
                                         T p, T p_inv,
                                         da_vector::da_vector<da_int> &neighbors,
                                         bool X_is_A, da_int index_X, T X_norm);

    da_status k_neighbors_recursive(std::shared_ptr<node<T>> current_node, da_int n, T *X,
                                    da_int k, da_metric metric, T p, T p_inv, bool X_is_A,
                                    da_int index_X, T X_norm, MaxHeap<T> &heap);

    da_status preprocess_data(const T *X_in, da_int m_samples_in, da_int m_features_in,
                              da_int ldx_in, da_metric metric_in, T p_in, const T **X,
                              bool &X_is_A, da_int &m_samples, da_int &m_features,
                              da_int &ldx, da_metric &metric, T &p, T &p_inv,
                              std::vector<T> &X_norms, da_errors::da_error_t *err);

    da_status compute_distance(T &dist, da_int index_A, da_int n, T *X, da_metric metric,
                               T p, T X_norm);

    da_int check_bounding_box(da_int n, T *X, T eps, da_metric metric, T p, T p_inv,
                              std::vector<T> &min_bounds, std::vector<T> &max_bounds);

    da_int leaf_size = 30;

    // Indices of points in the dataset
    std::vector<da_int> indices;

    // Dataset on which to build the tree
    const T *A;
    da_int lda;
    da_int n_samples;
    da_int n_features;

    // Row norms of the dataset - only used for da_(sq)euclidean
    std::vector<T> A_norms;

    // Root node of the tree
    std::shared_ptr<node<T>> root = nullptr;
};

} // namespace da_kdtree

} // namespace ARCH