# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
Data validation Python test script.
"""

import numpy as np
import pandas as pd
import pytest
from aoclda._internal_utils import check_convert_data


@pytest.mark.parametrize('force_float, dtype', [
    (True, 'float32'), (True, 'float64'),
    (False, 'float32'), (False, 'float64'),
    (False, 'int32'), (False, 'int64')])
def test_no_casting_numpy(force_float, dtype):
    """
        Test data validation when no casting is needed.
    """
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=dtype)
    X = check_convert_data(X, force_float=force_float)

    assert X.dtype == dtype


@pytest.mark.parametrize('ini_dtype, conv_dtype',
                         [('float16', 'float32'),
                          ('int16', 'float32'),
                          ('int32', 'float32'),
                          ('int64', 'float32'),
                          ('object', 'float32'),
                          ('float16', 'float64'),
                          ('int16', 'float64'),
                          ('int32', 'float64'),
                          ('int64', 'float64'),
                          ('object', 'float64')])
def test_valid_numpy_type_casting_force(ini_dtype, conv_dtype):
    """
        Test data validation on NumPy arrays when casting is needed and only floats are accepted.
    """
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=ini_dtype)

    with pytest.warns(UserWarning):
        X = check_convert_data(X, dtype=conv_dtype)

    assert X.dtype == conv_dtype


@pytest.mark.parametrize('ini_dtype', ['float16', 'int16', 'object'])
@pytest.mark.parametrize('conv_dtype', ['float32', 'float64', 'int32', 'int64'])
def test_valid_numpy_type_casting_not_force(ini_dtype, conv_dtype):
    """
        Test data validation on NumPy arrays when casting is needed and integers are accepted.
    """
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=ini_dtype)

    with pytest.warns(UserWarning):
        X = check_convert_data(X, dtype=conv_dtype, force_float=False)

    assert X.dtype == conv_dtype


def test_invalid_numpy_type_casting():
    """
        Test data validation when invalid NumPy data is inputted.
    """
    X = np.array([[1, 1, 1], [2, 'a', 2], [3, 3, 3]])

    with pytest.raises(ValueError):
        X = check_convert_data(X)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_nan_numpy_validation(dtype):
    """
        Test data validation when data has NumPy nan in it.
    """
    X = np.array([[1, 1, 1], [2, np.nan, 2], [3, 3, 3]], dtype=dtype)

    X = check_convert_data(X)
    assert X.dtype == dtype


@pytest.mark.parametrize('ini_dtype, conv_dtype',
                         [('float16', 'float32'),
                          ('object', 'float32'),
                          ('float16', 'float64'),
                          ('object', 'float64')])
def test_nan_numpy_casting(ini_dtype, conv_dtype):
    """
        Test data validation on NumPy arrays when casting is needed and np.nan data is present.
    """
    X = np.array([[1, 1, 1], [2, np.nan, 2], [3, 3, 3]], dtype=ini_dtype)

    with pytest.warns(UserWarning):
        X = check_convert_data(X, dtype=conv_dtype)

    assert np.isnan(X[1][1])
    assert X.dtype == conv_dtype


@pytest.mark.parametrize('force_float, dtype', [
    (True, 'float32'), (True, 'float64'),
    (False, 'float32'), (False, 'float64'),
    (False, 'int32'), (False, 'int64')])
def test_no_casting_pandas(force_float, dtype):
    """
        Test data validation on Pandas when no casting is needed.
    """
    df = pd.DataFrame([[1, 1, 1], [2, 2, 2]], dtype=dtype)
    X = check_convert_data(df, force_float=force_float)

    assert X.dtype == dtype


@pytest.mark.parametrize('ini_dtype, conv_dtype',
                         [('float16', 'float32'),
                          ('int16', 'float32'),
                          ('int32', 'float32'),
                          ('int64', 'float32'),
                          ('object', 'float32'),
                          ('float16', 'float64'),
                          ('int16', 'float64'),
                          ('int32', 'float64'),
                          ('int64', 'float64'),
                          ('object', 'float64')])
def test_valid_pandas_casting_force(ini_dtype, conv_dtype):
    """
        Test data validation on Pandas when casting is needed when only floats are accepted.
    """
    df = pd.DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=ini_dtype)

    with pytest.warns(UserWarning):
        X = check_convert_data(df, dtype=conv_dtype)

    assert X.dtype == conv_dtype


@pytest.mark.parametrize('ini_dtype', ['float16', 'int16', 'object'])
@pytest.mark.parametrize('conv_dtype', ['float32', 'float64', 'int32', 'int64'])
def test_valid_pandas_casting_not_force(ini_dtype, conv_dtype):
    """
        Test data validation on Pandas when casting is needed when integers are also accepted.
    """
    df = pd.DataFrame([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=ini_dtype)

    with pytest.warns(UserWarning):
        X = check_convert_data(df, dtype=conv_dtype, force_float=False)

    assert X.dtype == conv_dtype


def test_invalid_pandas_type_casting():
    """
        Test data validation on Pandas when invalid data is present.
    """
    df = pd.DataFrame([[1, 1, 1], [2, 'a', 2], [3, 3, 3]])

    with pytest.raises(ValueError):
        X = check_convert_data(df)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_nan_pandas_casting(dtype):
    """
        Test data validation on Pandas when None, NA and nan are present.
    """
    # No data conversion as pd.NA cannot be converted to floats, etc.
    df = pd.DataFrame([[1, 1, 1], [2, pd.NA, 2], [3, 3, 3]])

    with pytest.raises(TypeError):
        X = check_convert_data(df, dtype=dtype)

    df = pd.DataFrame([[1, 1, 1], [2, None, 2], [3, 3, 3]], dtype=dtype)
    X = check_convert_data(df, dtype=dtype)

    assert X.dtype == dtype
    assert np.isnan(X[1][1])

    df = pd.DataFrame([[1, 1, 1], [2, np.nan, 2], [3, 3, 3]], dtype=dtype)
    X = check_convert_data(df, dtype=dtype)

    assert X.dtype == dtype
    assert np.isnan(X[1][1])


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
def test_list_casting(dtype):
    """
        Test data validation when input is a Python list.
    """
    x = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]

    with pytest.warns(UserWarning):
        X = check_convert_data(x, dtype=dtype, force_float=False)

    assert X.dtype == dtype

    x = [[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]

    with pytest.warns(UserWarning):
        X = check_convert_data(x, dtype=dtype, force_float=False)
    assert X.dtype == dtype


def test_list_invalid_input():
    """
        Test data validation when input is an invalid Python list.
    """
    x = [[1., 1, 1], [2, 'a', 2], [3, 3, 3]]

    with pytest.raises(ValueError):
        X = check_convert_data(x)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_list_nan_casting(dtype):
    """
        Test data validation when input is a Python list with None.
    """
    x = [[1, 1, 1], [2, None, 2], [3, 3, 3]]

    with pytest.warns(UserWarning):
        X = check_convert_data(x, dtype=dtype)

    assert np.isnan(X[1][1])
    assert X.dtype == dtype


def test_invalid_matrix_dimensions():
    """
        Test data validation when invalid matrix shape is inputted
    """

    x = [[1], [2, 2], [3, 3, 3]]

    with pytest.raises(ValueError):
        X = check_convert_data(x)


@pytest.mark.parametrize('dtype', ['float16', 'int32',
                                   'int64', 'a', '1', 1, True])
def test_incorrect_parameter_force(dtype):
    """
        Test data validation when wrong dtype parameter is given.
    """
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    with pytest.raises(ValueError):
        X = check_convert_data(X, dtype)


@pytest.mark.parametrize('dtype', ['float16', 'a', '1', 1, True])
def test_incorrect_parameter_not_force(dtype):
    """
        Test data validation when wrong dtype parameter is given.
    """
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    with pytest.raises(ValueError):
        X = check_convert_data(X, dtype, force_float=False)
