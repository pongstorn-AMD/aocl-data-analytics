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

#include "kdtree.hpp"
#include "da_omp.hpp"
#include "da_std.hpp"
#include "pairwise_distances.hpp"
#include <limits>
#include <stack>

#define KDTREE_MIN_TASK_SIZE da_int(2048)
#define KDTREE_BLOCK_SIZE da_int(512)

namespace ARCH {

namespace da_kdtree {

// Constructors for a node of the k-d tree
template <typename T>
node<T>::node(da_int dim, da_int depth, da_int *indices, da_int n_indices,
              std::vector<T> min_bounds, std::vector<T> max_bounds)
    : dim(dim), depth(depth), indices(indices), n_indices(n_indices),
      min_bounds(min_bounds), max_bounds(max_bounds) {}

template <typename T>
node<T>::node(da_int dim, da_int depth, da_int *indices, da_int n_indices)
    : dim(dim), depth(depth), indices(indices), n_indices(n_indices),
      min_bounds(std::vector<T>()), max_bounds(std::vector<T>()) {}

template <typename T> const std::vector<da_int> &kdtree<T>::get_indices() {
    return indices;
}

// Lightweight partial MaxHeap implementation to keep track of k-NN k-d tree searches
template <typename T>
MaxHeap<T>::MaxHeap(da_int capacity, da_int *indices, T *distances)
    : indices(indices), distances(distances), capacity(capacity) {
    size = 0;
}

template <typename T> T MaxHeap<T>::GetMaxDist() {
    // Return the maximum distance in the heap, or the maximum possible value if the heap is not full
    return (size < capacity) ? std::numeric_limits<T>::max() : distances[0];
}

// Insert a new point to the heap if the distance is smaller than the max, maintaining the max-heap property
template <typename T> void MaxHeap<T>::Insert(da_int index, T distance) {
    if (size < capacity) {
        indices[size] = index;
        distances[size] = distance;
        heapify_up(size);
        size++;
    } else if (distance < distances[0]) {
        indices[0] = index;
        distances[0] = distance;
        heapify_down(0);
    }
}

template <typename T> da_int MaxHeap<T>::GetSize() {
    // Return the current size of the heap
    return size;
}

template <typename T> void MaxHeap<T>::heapify_up(da_int index) {
    // Move the element at index up the heap until the max-heap property is restored
    while (index > 0) {
        da_int parent = (index - 1) / 2;
        if (distances[index] > distances[parent]) {
            // If the current element is greater than its parent, swap them
            std::swap(indices[index], indices[parent]);
            std::swap(distances[index], distances[parent]);
            index = parent;
        } else {
            break; // The max-heap property is restored
        }
    }
}

template <typename T> void MaxHeap<T>::heapify_down(da_int index) {
    // Move the element at index down the heap until the max-heap property is restored
    while (true) {
        da_int left = 2 * index + 1;
        da_int right = 2 * index + 2;
        da_int largest = index;

        if (left < size && distances[left] > distances[largest]) {
            largest = left;
        }
        if (right < size && distances[right] > distances[largest]) {
            largest = right;
        }
        if (largest != index) {
            std::swap(distances[index], distances[largest]);
            std::swap(indices[index], indices[largest]);
            index = largest;
        } else {
            break;
        }
    }
}

template <typename T>
kdtree<T>::kdtree(da_int n_samples, da_int n_features, const T *A, da_int lda,
                  da_int leaf_size) {
    // Initialize the k-d tree
    this->n_samples = n_samples;
    this->n_features = n_features;
    this->lda = lda;
    this->A = A;
    this->leaf_size = leaf_size;

    // Allocate memory for the indices (which will initially be filled with 0, 1, 2, ..., n_samples - 1)
    // and the initial bounding box
    indices = std::vector<da_int>(n_samples);
    std::vector<T> min_bounds(n_features, std::numeric_limits<T>::max());
    std::vector<T> max_bounds(n_features, std::numeric_limits<T>::lowest());
    // If memory allocation failed an exception will be thrown, so the constructor must be wrapped in a try...catch
    da_std::iota(indices.begin(), indices.end(), 0);

    // Compute the bounding box for the dataset; parallelism optimized for tall, skinny dataset
    da_int n_blocks, block_rem;
    da_int max_block_size = std::min(KDTREE_BLOCK_SIZE, n_samples);
    da_utils::blocking_scheme(n_samples, max_block_size, n_blocks, block_rem);

    da_int block_index;
    da_int block_size = max_block_size;

    bool memory_alloc_failed = false;

#pragma omp parallel firstprivate(block_size) private(block_index) default(none)         \
    shared(n_blocks, block_rem, n_samples, max_block_size, A, lda, min_bounds,           \
               max_bounds, n_features, memory_alloc_failed)
    {
        // Each thread needs its own copy of the min and max bounds; but these are likely to be small
        std::vector<T> min_bounds_private;
        std::vector<T> max_bounds_private;
        try {
            min_bounds_private.resize(n_features, std::numeric_limits<T>::max());
            max_bounds_private.resize(n_features, std::numeric_limits<T>::lowest());
        } catch (std::bad_alloc const &) {
// If memory allocation failed, set a flag so we can throw an exception outside the parallel region
#pragma omp atomic write
            memory_alloc_failed = true;
        }

        if (!memory_alloc_failed) {
#pragma omp for schedule(static)
            for (da_int i = 0; i < n_blocks; i++) {
                if (i == n_blocks - 1 && block_rem > 0) {
                    block_index = n_samples - block_rem;
                    block_size = block_rem;
                } else {
                    block_index = i * max_block_size;
                }
                da_int A_offset = 0;
                for (da_int j = 0; j < n_features; j++) {
                    A_offset = j * lda;
                    for (da_int k = 0; k < block_size; k++) {
                        min_bounds_private[j] = std::min(min_bounds_private[j],
                                                         A[k + block_index + A_offset]);
                        max_bounds_private[j] = std::max(max_bounds_private[j],
                                                         A[k + block_index + A_offset]);
                    }
                }
            }
#pragma omp critical(min_bounds)
            {
                for (da_int j = 0; j < n_features; j++) {
                    min_bounds[j] = std::min(min_bounds[j], min_bounds_private[j]);
                }
            }
#pragma omp critical(max_bounds)
            {
                for (da_int j = 0; j < n_features; j++) {
                    max_bounds[j] = std::max(max_bounds[j], max_bounds_private[j]);
                }
            }
        }

    } // End of parallel region

    if (memory_alloc_failed) {
        throw std::bad_alloc(); // LCOV_EXCL_LINE
    }

// Build the k-d tree
#pragma omp parallel default(none) shared(indices, n_samples, min_bounds, max_bounds)
    {
#pragma omp single
        {
            this->root =
                build_tree(0, indices.data(), n_samples, &min_bounds, &max_bounds);
        }
    }
}

// Recursive function to build the k-d tree
// The k-d tree is built in a top-down manner, starting from the root node and recursively splitting
template <typename T>
std::shared_ptr<node<T>>
kdtree<T>::build_tree(da_int depth, da_int *indices, da_int n_indices,
                      std::vector<T> *min_bounds, std::vector<T> *max_bounds) {

    // If there are no indices, return nullptr
    if (n_indices == 0) {
        return nullptr;
    }

    // Find the dimension to split on
    da_int dim = depth % n_features;

    // Create a new node for this part of the tree, with sensible defaults. Only the root node will have
    // min_bounds and max_bounds supplied; for all other nodes these will be computed later
    auto this_node = (min_bounds == nullptr)
                         ? std::make_shared<node<T>>(dim, depth, indices, n_indices)
                         : std::make_shared<node<T>>(dim, depth, indices, n_indices,
                                                     *min_bounds, *max_bounds);

    da_int A_offset = dim * lda;

    // If needed compute the bounding box for the node
    if (min_bounds == nullptr) {
        this_node->min_bounds.resize(n_features, std::numeric_limits<T>::max());
        this_node->max_bounds.resize(n_features, std::numeric_limits<T>::lowest());

        for (da_int j = 0; j < n_features; j++) {
            da_int A_offset_tmp = j * lda;
            for (da_int i = 0; i < n_indices; i++) {
                this_node->min_bounds[j] =
                    std::min(this_node->min_bounds[j], A[indices[i] + A_offset_tmp]);
                this_node->max_bounds[j] =
                    std::max(this_node->max_bounds[j], A[indices[i] + A_offset_tmp]);
            }
        }
    }

    // If the number of indices is such that further splitting would reduce it to below leaf size,
    // or result in an empty child node, then set the node to be a leaf node then return
    if (n_indices < 2 * leaf_size || n_indices == 2) {
        this_node->is_leaf = true;
        return this_node;
    }

    // Find the median point in the current dimension, accounting for zero-based indexing
    da_int mid = (n_indices - 1) / 2;

    std::nth_element(indices, indices + mid, indices + n_indices,
                     [this, &A_offset](da_int x, da_int y) {
                         return A[x + A_offset] < A[y + A_offset];
                     });

    // Assign the median point as the node's splitting point
    this_node->point = indices[mid];

    // Recursively build the left and right child nodes, but only spawn tasks if the workload is large enough
    if (mid > KDTREE_MIN_TASK_SIZE) {
        // Some older compilers don't like the use of "omp task if" so use an explicit if statement
#pragma omp task firstprivate(depth, mid, indices, this_node)
        { this_node->left_child = build_tree(depth + 1, indices, mid); }
#pragma omp task firstprivate(depth, mid, indices, n_indices, this_node)
        {
            this_node->right_child =
                build_tree(depth + 1, indices + mid + 1, n_indices - mid - 1);
        }
    } else {
        this_node->left_child = build_tree(depth + 1, indices, mid);
        this_node->right_child =
            build_tree(depth + 1, indices + mid + 1, n_indices - mid - 1);
    }

    return this_node;
}

// Find the k nearest neighbors of a point using the k-d tree
template <typename T>
da_status kdtree<T>::k_neighbors(da_int m_samples_in, da_int m_features_in, const T *X_in,
                                 da_int ldx_in, da_int k, da_metric metric_in, T p_in,
                                 da_int *k_ind, T *k_dist, da_errors::da_error_t *err) {
    // We assume here that the k-d tree is already built and that m_samples, m_features, ldx and k
    // are valid and da_metric is valid (i.e. not cosine distance)
    da_status status = da_status_success;

    std::vector<T> X_norms, X_row;
    const T *X = nullptr;
    bool X_is_A = false;
    da_int ldx = 0;
    da_int m_samples = 0;
    da_int m_features = 0;
    T p, p_inv;
    da_metric metric;

    // Call utility function to preprocess the data: check if X_in is null (in which case we should
    // be using the original data matrix, this->A) and update m_samples, m_features, ldx, X, metric, p
    // and compute p_inv and X_norms if needed
    status = preprocess_data(X_in, m_samples_in, m_features_in, ldx_in, metric_in, p_in,
                             &X, X_is_A, m_samples, m_features, ldx, metric, p, p_inv,
                             X_norms, err);
    if (status != da_status_success) {
        return status; // LCOV_EXCL_LINE
    }

    try {
        X_row.resize(m_features * omp_get_max_threads());
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    // Loop over the samples in X and find the neighbors
#pragma omp parallel default(none)                                                       \
    shared(m_samples, m_features, X, ldx, k, metric, p, p_inv, A_norms, X_norms, A, lda, \
               root, X_is_A, status, k_ind, k_dist, X_row)
    {
        da_int X_row_index = m_features * omp_get_thread_num();
#pragma omp for schedule(dynamic, 128)
        for (da_int i = 0; i < m_samples; i++) {

            T X_norm = 0.0;
            if (metric == da_sqeuclidean) {
                X_norm = A_norms[i];
                if (!(X_is_A)) {
                    X_norm = X_norms[i];
                }
            }

            // For better data access later, copy this row of X into a contiguous piece of a vector
            for (da_int j = 0; j < m_features; j++) {
                X_row[j + X_row_index] = X[i + j * ldx];
            }

            auto heap = MaxHeap<T>(k, &k_ind[i * k], &k_dist[i * k]);

            da_status tmp_status =
                k_neighbors_recursive(root, m_features, &X_row[X_row_index], k, metric, p,
                                      p_inv, X_is_A, i, X_norm, heap);
            if (tmp_status != da_status_success) {
// If there was an error, set the status and break out of the loop
#pragma omp atomic write
                status = tmp_status;
            }
        }
    }
    if (status != da_status_success) {
        return da_error(err, status, // LCOV_EXCL_LINE
                        "Failed to compute radius neighbors.");
    }

    return da_status_success;
}

// Utility function called prior to forming radius/k neighbors. Checks whether X_in is null (in which
// case we should be using the original data matrix, this->A) and updates m_samples, m_features, ldx
// and X accordingly. Also precomputes the row norms of A and X (if the metric is da_euclidean) and computes
// p and p_inv for use later.
template <typename T>
da_status
kdtree<T>::preprocess_data(const T *X_in, da_int m_samples_in, da_int m_features_in,
                           da_int ldx_in, da_metric metric_in, T p_in, const T **X,
                           bool &X_is_A, da_int &m_samples, da_int &m_features,
                           da_int &ldx, da_metric &metric, T &p, T &p_inv,
                           std::vector<T> &X_norms, da_errors::da_error_t *err) {

    // Use squared Euclidean distance instead of Euclidean distance to avoid redundant square roots
    metric = (metric_in == da_euclidean || (metric_in == da_minkowski && p == T(2.0)))
                 ? da_sqeuclidean
                 : metric_in;

    // If X_in is null then we use the original dataset which was used to construct the tree
    if (X_in == nullptr) {
        m_samples = n_samples;
        m_features = n_features;
        ldx = lda;
        *X = A;
        X_is_A = true;
    } else {
        m_samples = m_samples_in;
        m_features = m_features_in;
        ldx = ldx_in;
        *X = X_in;
        X_is_A = false;
    }

    // We use p and p_inv when checking bounding boxes
    p = p_in;
    p_inv = (T)1.0 / p;
    if (metric == da_manhattan) {
        p = (T)1.0;
        p_inv = (T)1.0;
    }

    if (metric == da_sqeuclidean) {
        try {
            A_norms.resize(n_samples);
            // Guard against multiple calls
            da_std::fill(A_norms.begin(), A_norms.end(), (T)0.0);
            if (!X_is_A) {
                X_norms.resize(m_samples, (T)0.0);
            }
        } catch (std::bad_alloc const &) {
            return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                            "Memory allocation failed.");
        }
        // Precompute the row norms of A to speed up Euclidean distance computation
        for (da_int j = 0; j < n_features; j++) {
            for (da_int i = 0; i < n_samples; i++) {
                A_norms[i] += A[i + j * lda] * A[i + j * lda];
            }
        }
        if (!X_is_A) {
            // Precompute the row norms of X to speed up Euclidean distance computation
            for (da_int j = 0; j < m_features; j++) {
                for (da_int i = 0; i < m_samples; i++) {
                    X_norms[i] += (*X)[i + j * ldx] * (*X)[i + j * ldx];
                }
            }
        }
    }
    return da_status_success;
}

template <typename T>
da_status
kdtree<T>::radius_neighbors(da_int m_samples_in, da_int m_features_in, const T *X_in,
                            da_int ldx_in, T eps, da_metric metric_in, T p_in,
                            std::vector<da_vector::da_vector<da_int>> &neighbors,
                            da_errors::da_error_t *err) {

    // We assume here that the k-d tree is already built and that m_samples, m_features and ldx are
    // valid and da_metric is valid (not cosine distance)
    da_status status = da_status_success;

    std::vector<T> X_norms, X_row;
    const T *X = nullptr;
    bool X_is_A = false;
    da_int ldx = 0;
    da_int m_samples = 0;
    da_int m_features = 0;
    da_metric metric;
    T p, p_inv;

    // For da_euclidean it is more efficient to use the squared distance for some of the computation
    T eps_internal = (metric_in == da_euclidean) ? eps * eps : eps;

    // Call utility function to preprocess the data: check is X_in is null (in which case we should
    // be using the original data matrix, this->A) and update m_samples, m_features, ldx, X, metric, p
    // and compute p_inv and X_norms if needed
    status = preprocess_data(X_in, m_samples_in, m_features_in, ldx_in, metric_in, p_in,
                             &X, X_is_A, m_samples, m_features, ldx, metric, p, p_inv,
                             X_norms, err);
    if (status != da_status_success) {
        return status; // LCOV_EXCL_LINE
    }

    try {
        X_row.resize(m_features * omp_get_max_threads());
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

// Loop over the samples in X and find the radius neighbors
#pragma omp parallel default(none)                                                       \
    shared(neighbors, m_samples, m_features, X, ldx, eps, eps_internal, metric, p,       \
               p_inv, A_norms, X_norms, A, lda, root, X_is_A, status, X_row)
    {
        da_int X_row_index = m_features * omp_get_thread_num();
#pragma omp for schedule(dynamic, 128)
        for (da_int i = 0; i < m_samples; i++) {

            T X_norm = 0.0;
            if (metric == da_sqeuclidean) {
                X_norm = A_norms[i];
                if (!(X_is_A)) {
                    X_norm = X_norms[i];
                }
            }

            // For better data access later, copy this row of X into a contiguous piece of a vector
            for (da_int j = 0; j < m_features; j++) {
                X_row[j + X_row_index] = X[i + j * ldx];
            }

            // Find the epsilon radius neighbors of the ith point in X by recursively searching the tree
            da_status tmp_status = radius_neighbors_recursive(
                root, m_features, &X_row[X_row_index], eps, eps_internal, metric, p,
                p_inv, neighbors[i], X_is_A, i, X_norm);
            if (tmp_status != da_status_success) {
// If there was an error, set the status and break out of the loop
#pragma omp atomic write
                status = tmp_status;
            }
        }
    }

    if (status != da_status_success) {
        return da_error(err, status, // LCOV_EXCL_LINE
                        "Failed to compute radius neighbors.");
    }

    return da_status_success;
}

// Recursive function to find the radius neighbors of a point (determined by index_X) in X
template <typename T>
da_status kdtree<T>::radius_neighbors_recursive(std::shared_ptr<node<T>> current_node,
                                                da_int n, T *X, T eps, T eps_internal,
                                                da_metric metric, T p, T p_inv,
                                                da_vector::da_vector<da_int> &neighbors,
                                                bool X_is_A, da_int index_X, T X_norm) {

    da_status status = da_status_success;

    // Check the bounding box for quick pruning of the search space
    da_int bounding_box =
        check_bounding_box(n, X, eps_internal, metric, p, p_inv, current_node->min_bounds,
                           current_node->max_bounds);
    if (bounding_box == 0) {
        // The point is too far from the bounding box for this node, we can return and ignore all sub-nodes
        return da_status_success;
    }

    if (bounding_box == 2) {
        // The entire bounding box is inside the search radius, so we can add all points in the node
        for (da_int i = 0; i < current_node->n_indices; i++) {
            da_int index_A = current_node->indices[i];
            if (X_is_A && index_A == index_X) {
                // If we are using the original dataset, skip the point itself so we don't add it to its own neighbors
                continue;
            }
            neighbors.push_back(index_A);
        }
        return da_status_success;
    }

    T dist;

    if (current_node->is_leaf) {
        // Check all the points in the node

        for (da_int i = 0; i < current_node->n_indices; i++) {

            da_int index_A = current_node->indices[i];

            if (X_is_A && index_A == index_X) {
                // If we are using the original dataset, skip the point itself so we don't add it to its own neighbors
                continue;
            }

            status = compute_distance(dist, index_A, n, X, metric, p, X_norm);
            if (status != da_status_success) {
                return status; // LCOV_EXCL_LINE
            }

            if (dist <= eps_internal) {
                // If the distance is less than or equal to eps_internal, add the point to the neighbors list
                neighbors.push_back(index_A);
            }
        }

    } else {
        // This is not a leaf node, so only has a single point, which we need to check
        da_int index_A = current_node->point;

        if (!(X_is_A && index_A == index_X)) {
            // If we are using the original dataset, make sure we don't add the point to its own neighbors

            status = compute_distance(dist, index_A, n, X, metric, p, X_norm);
            if (status != da_status_success) {
                return status; // LCOV_EXCL_LINE
            }

            if (dist <= eps_internal) {
                // If the distance is less than or equal to eps_internal, add the point to the neighbors list
                neighbors.push_back(index_A);
            }
        }

        // Check the splitting dimension
        da_int dim = current_node->dim;
        T diff = X[dim] - A[index_A + lda * dim];

        if (diff <= eps) {
            // Check the left child
            radius_neighbors_recursive(current_node->left_child, n, X, eps, eps_internal,
                                       metric, p, p_inv, neighbors, X_is_A, index_X,
                                       X_norm);
        }
        if (diff >= -eps) {
            // Check the right child
            radius_neighbors_recursive(current_node->right_child, n, X, eps, eps_internal,
                                       metric, p, p_inv, neighbors, X_is_A, index_X,
                                       X_norm);
        }
    }
    return da_status_success;
}

// Recursive function to find the k nearest neighbors of a point (determined by index_X) in X
template <typename T>
da_status kdtree<T>::k_neighbors_recursive(std::shared_ptr<node<T>> current_node,
                                           da_int n, T *X, da_int k, da_metric metric,
                                           T p, T p_inv, bool X_is_A, da_int index_X,
                                           T X_norm, MaxHeap<T> &heap) {

    da_status status = da_status_success;

    // If the heap is full we need to check the bounding box, otherwise we can skip this check
    da_int bounding_box =
        (heap.GetSize() < k)
            ? 1
            : check_bounding_box(n, X, heap.GetMaxDist(), metric, p, p_inv,
                                 current_node->min_bounds, current_node->max_bounds);

    // If the point is too far from the bounding box for this node, we can return and ignore all sub-nodes
    if (bounding_box == 0) {
        return da_status_success;
    }

    if (current_node->is_leaf) {
        // Check all the points in the node
        for (da_int i = 0; i < current_node->n_indices; i++) {

            da_int index_A = current_node->indices[i];

            if (X_is_A && index_A == index_X) {
                // If we are using the original dataset, skip the point itself so we don't add it to its own neighbors
                continue;
            }

            T dist;
            status = compute_distance(dist, index_A, n, X, metric, p, X_norm);
            if (status != da_status_success) {
                return status; // LCOV_EXCL_LINE
            }

            // Add the point to the heap (the heap itself will handle the max distance)
            heap.Insert(index_A, dist);
        }

    } else {
        // This is not a leaf node, so only has a single point, which we should check
        da_int index_A = current_node->point;

        if (!(X_is_A && index_A == index_X)) {
            T dist;
            status = compute_distance(dist, index_A, n, X, metric, p, X_norm);
            if (status != da_status_success) {
                return status; // LCOV_EXCL_LINE
            }

            heap.Insert(index_A, dist);
        }

        // Check the splitting dimension
        da_int dim = current_node->dim;
        T diff = X[dim] - A[index_A + lda * dim];

        // diff_tmp accounts for the square of the distances used in da_sqeuclidean
        T diff_tmp = (metric = da_sqeuclidean) ? diff * std::abs(diff) : diff;

        // Whether we check the left or right child first depends on the sign of diff - this has a significant performance impact

        if (diff < (T)0.0) {
            // Check the left child first
            k_neighbors_recursive(current_node->left_child, n, X, k, metric, p, p_inv,
                                  X_is_A, index_X, X_norm, heap);

            if (diff_tmp >= -heap.GetMaxDist()) {
                // Check the right child
                k_neighbors_recursive(current_node->right_child, n, X, k, metric, p,
                                      p_inv, X_is_A, index_X, X_norm, heap);
            }
        } else {
            // Check the right child first
            k_neighbors_recursive(current_node->right_child, n, X, k, metric, p, p_inv,
                                  X_is_A, index_X, X_norm, heap);

            if (diff_tmp <= heap.GetMaxDist()) {
                // Check the left child
                k_neighbors_recursive(current_node->left_child, n, X, k, metric, p, p_inv,
                                      X_is_A, index_X, X_norm, heap);
            }
        }
    }

    return da_status_success;
}

// Compute the distance between the point at index_A in A and the point at index_X in X
template <typename T>
da_status kdtree<T>::compute_distance(T &dist, da_int index_A, da_int n, T *X,
                                      da_metric metric, T p, T X_norm) {
    // Note that if the user specified metric = da_euclidean, it will have been converted to da_sqeuclidean by now
    if (metric == da_sqeuclidean) {
        // Special case for Euclidean distance using precomputed norms
        dist = 0.0;
        // Typically expect this to be a small number of features so use a simple loop rather than BLAS call
        for (da_int i = 0; i < n; i++) {
            dist += X[i] * A[index_A + i * lda];
        }
        dist = X_norm + A_norms[index_A] - 2 * dist;
    } else {
        // Compute the distance matrix using the specified metric
        da_status status = ARCH::da_metrics::pairwise_distances::pairwise_distance_kernel(
            da_order::column_major, 1, 1, n, X, 1, &A[index_A], lda, &dist, 1, p, metric);
        if (status != da_status_success) {
            return status;
        }
    }
    return da_status_success;
}

/* Check if a point X might be within distance eps of a box defined by the min_bounds and max_bounds coordinates
*  Return: 0 if X is further than eps from the box
*          1 if X is within eps of the box
*          2 if X the entirely of the box is within eps of X
*/
template <typename T>
da_int kdtree<T>::check_bounding_box(da_int n, T *X, T eps, da_metric metric, T p,
                                     T p_inv, std::vector<T> &min_bounds,
                                     std::vector<T> &max_bounds) {

    // Note that if the user specified metric = da_euclidean, it will have been converted to da_sqeuclidean by now and eps will have been squared

    // min_dist will be the minimum distance from X to the bounding box - zero if X is inside
    T min_dist = 0.0;

    // max_dist will be the maximum distance from X to any corner of the bounding box
    T max_dist = 0.0;

    T tmp_min_dist, tmp_max_dist;

    // Special case for (squared)-Euclidean distance as it's a bit faster
    if (metric == da_sqeuclidean) {
        for (da_int i = 0; i < n; i++) {

            if (X[i] < min_bounds[i]) {
                tmp_min_dist = min_bounds[i] - X[i];
                tmp_max_dist = max_bounds[i] - X[i];
            } else if (X[i] > max_bounds[i]) {
                tmp_min_dist = X[i] - max_bounds[i];
                tmp_max_dist = X[i] - min_bounds[i];
            } else {
                tmp_max_dist = std::max(max_bounds[i] - X[i], X[i] - min_bounds[i]);
                tmp_min_dist = (T)0.0;
            }
            tmp_min_dist *= tmp_min_dist;
            tmp_max_dist *= tmp_max_dist;
            if (tmp_min_dist > eps) {
                // Quick return here as we know X is further than eps from the bounding box
                return 0;
            }
            min_dist += tmp_min_dist;
            max_dist += tmp_max_dist;
        }
    } else {

        for (da_int i = 0; i < n; i++) {

            if (X[i] < min_bounds[i]) {
                tmp_min_dist = min_bounds[i] - X[i];
                tmp_max_dist = max_bounds[i] - X[i];
            } else if (X[i] > max_bounds[i]) {
                tmp_min_dist = X[i] - max_bounds[i];
                tmp_max_dist = X[i] - min_bounds[i];
            } else {
                tmp_max_dist = std::max(max_bounds[i] - X[i], X[i] - min_bounds[i]);
                tmp_min_dist = (T)0.0;
            }
            if (tmp_min_dist > eps) {
                // Quick return here as we know X is further than eps from the bounding box
                return 0;
            }
            if (metric == da_manhattan) {
                min_dist += tmp_min_dist;
                max_dist += tmp_max_dist;
            } else {
                min_dist += std::pow(tmp_min_dist, p);
                max_dist += std::pow(tmp_max_dist, p);
            }
        }
        if (metric != da_manhattan) {
            min_dist = std::pow(min_dist, p_inv);
            max_dist = std::pow(max_dist, p_inv);
        }
    }

    if (max_dist <= eps) {
        // If the maximum distance is less than eps, then the entire bounding box is within eps of X
        return 2;
    }
    // If the minimum distance is less than eps, then X is within eps of the bounding box
    if (min_dist <= eps) {
        return 1;
    }
    // Otherwise, the point is outside the bounding box
    return 0;
}

// Explicit instantiation of the k-d tree class for double and float types
template struct node<double>;
template struct node<float>;

template class kdtree<double>;
template class kdtree<float>;

} // namespace da_kdtree
} // namespace ARCH
