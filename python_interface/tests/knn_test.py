# Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

"""
kNN Python test script
"""

import numpy as np
import pytest
from aoclda.nearest_neighbors import knn_classifier
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
@pytest.mark.parametrize("algo", ["auto", "brute", "kd_tree"])
def test_knn_functionality(numpy_precision, numpy_order, algo):
    """
    Test the functionality of the Python wrapper
    """
    x_train = np.array([[-1, -1, 2],
                        [-2, -1, 3],
                        [-3, -2, -1],
                        [1, 3, 1],
                        [2, 5, 1],
                        [3, -1, 2]],
                       dtype=numpy_precision, order=numpy_order)

    y_train = np.array([1, 2, 0, 1, 2, 2],
                       dtype=numpy_precision, order=numpy_order)

    x_test = np.array([[-2 , 2, 3],
                       [-1, -2, -1],
                       [2, 1, -3]],
                       dtype=numpy_precision, order=numpy_order)

    knn = knn_classifier(algorithm=algo)
    knn.fit(x_train, y_train)
    k_dist, k_ind = knn.kneighbors(x_test, n_neighbors=3, return_distance=True)

    assert k_dist.flags.f_contiguous == x_test.flags.f_contiguous
    assert k_ind.flags.f_contiguous == x_test.flags.f_contiguous

    proba = knn.predict_proba(x_test)
    assert proba.flags.f_contiguous == x_test.flags.f_contiguous

    y_test = knn.predict(x_test)

    expected_ind = np.array([[1, 0, 3],
                             [2, 0, 1],
                             [3, 5, 4]])
    expected_dist = np.array([[3.        , 3.31662479, 3.74165739 ],
                              [2.        , 3.16227766, 4.24264069 ],
                              [4.58257569, 5.47722558, 5.65685425]])
    expected_proba = np.array([[0.2, 0.4, 0.4],
                               [0.2, 0.4, 0.4],
                               [0.2, 0.4, 0.4]])
    expected_labels = np.array([[1, 1, 1,]])

    # Check we have the right answer
    tol = np.sqrt(np.finfo(numpy_precision).eps)

    assert k_dist == pytest.approx(expected_dist, tol)
    assert not np.any(k_ind - expected_ind)
    assert proba == pytest.approx(expected_proba, tol)
    assert not np.any(y_test - expected_labels)

@pytest.mark.parametrize("n_samples", [20, 500, 5000])
@pytest.mark.parametrize("n_features", [4, 10, 15])
@pytest.mark.parametrize("n_classes", [3])
@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
@pytest.mark.parametrize("numpy_order", ["C", "F"])
@pytest.mark.parametrize("leaf_size", [1, 2, 10, 41])
def test_knn_kdtree_functionality(n_samples, n_features, n_classes, numpy_precision, numpy_order, leaf_size):
    """
    Test the functionality of the Python wrapper fo different leaf sizes.
    Since we expect the same results for k-d tree and brute force algorithms,
    we can compare against the brute force algorithm.
    """
    x, y = make_classification(n_samples=2*n_samples,
                                n_features=n_features,
                                n_informative=n_features,
                                n_repeated=0,
                                n_redundant=0,
                                n_classes=n_classes,
                                random_state=42)

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.5,
                                                        train_size=0.5,
                                                        random_state=42)

    if numpy_order == "F":
        x_train = np.asfortranarray(x_train, dtype=numpy_precision)
        x_test = np.asfortranarray(x_test, dtype=numpy_precision)
    else:
        # Only ensure numpy_precision is correct
        x_train = np.array(x_train, dtype=numpy_precision)
        x_test = np.array(x_test, dtype=numpy_precision)

    # Compute using k-d tree algorithm
    kdtree_knn = knn_classifier(algorithm="kd_tree", leaf_size=leaf_size)
    kdtree_knn.fit(x_train, y_train)
    kdtree_knn_k_dist, kdtree_knn_k_ind = kdtree_knn.kneighbors(x_test, n_neighbors=7, return_distance=True)

    assert kdtree_knn_k_dist.flags.f_contiguous == x_test.flags.f_contiguous
    assert kdtree_knn_k_ind.flags.f_contiguous == x_test.flags.f_contiguous

    kdtree_knn_proba = kdtree_knn.predict_proba(x_test)
    assert kdtree_knn_proba.flags.f_contiguous == x_test.flags.f_contiguous

    kdtree_knn_y_test = kdtree_knn.predict(x_test)

    # Now compute using brute force algorithm
    brute_knn = knn_classifier(algorithm="brute", leaf_size=leaf_size)
    brute_knn.fit(x_train, y_train)
    brute_knn_k_dist, brute_knn_k_ind = brute_knn.kneighbors(x_test, n_neighbors=7, return_distance=True)

    assert brute_knn_k_dist.flags.f_contiguous == x_test.flags.f_contiguous
    assert brute_knn_k_ind.flags.f_contiguous == x_test.flags.f_contiguous

    brute_knn_proba = brute_knn.predict_proba(x_test)
    assert brute_knn_proba.flags.f_contiguous == x_test.flags.f_contiguous

    brute_knn_y_test = brute_knn.predict(x_test)

    # Check we have the right answer
    tol = np.sqrt(np.finfo(numpy_precision).eps)

    assert kdtree_knn_k_dist == pytest.approx(brute_knn_k_dist, tol)
    assert not np.any(kdtree_knn_k_ind - brute_knn_k_ind)
    assert kdtree_knn_proba == pytest.approx(brute_knn_proba, tol)
    assert not np.any(kdtree_knn_y_test - brute_knn_y_test)

@pytest.mark.parametrize("numpy_precision", [np.float64, np.float32])
def test_knn_error_exits(numpy_precision):
    """
    Test error exits in the Python wrapper
    """
    x_train = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        knn = knn_classifier(n_neighbors = -1)
    with pytest.raises(RuntimeError):
        knn = knn_classifier(weights = "ones")
    with pytest.raises(RuntimeError):
        knn = knn_classifier(metric = "nonexistent")
    with pytest.raises(RuntimeError):
        knn = knn_classifier(algorithm = "nonexistent")
    y_train = np.array([[1,2,3]], dtype=numpy_precision)
    knn = knn_classifier()
    knn.fit(x_train, y_train)
    x_test = np.array([[1, 1], [2, 2], [3, 3]], dtype=numpy_precision)
    with pytest.raises(RuntimeError):
        knn.kneighbors(X=x_test)
    with pytest.raises(RuntimeError):
        knn.predict(X=x_test)
    with pytest.raises(RuntimeError):
        knn.predict_proba(X=x_test)

    x_train = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=numpy_precision, order="F")
    y_train = np.array([[1, 2, 3]], dtype=numpy_precision)
    knn = knn_classifier()
    knn.fit(x_train, y_train)
    x_test = np.array([[1, 1, 3], [2, 2, 3]], dtype=numpy_precision, order="C")
    with pytest.raises(RuntimeError):
        knn.kneighbors(X=x_test, n_neighbors=2)
    with pytest.raises(RuntimeError):
        knn.predict(X=x_test)
    with pytest.raises(RuntimeError):
        knn.predict_proba(X=x_test)

    knn = knn_classifier(check_data=True)
    x_train = np.array([[1, 1, np.nan], [2, 2, 2], [3, 3, 3]], dtype=numpy_precision, order="F")
    with pytest.raises(RuntimeError):
        knn.fit(x_train, y_train)
