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
aoclda._internal_utils module
"""

import warnings
import numpy as np


def check_convert_data(X, dtype='float64', force_float=True):
    """
    Checks if input data is a NumPy ndarray of a supported type. 
    If not, a conversion is attempted. The conversion tries to keep the 
    input data type, however, if not possible it is converted 
    to either float32 or float64 when force_float=True; or to 
    float32, float64, int32 or int64 when force_float=False. 

    If the input data contains mixed data type, the first element's/row's/column's 
    (X[0]) type is inferred.

    Args:
        X (array-like): The data matrix to validate.

        dtype (str, optional): Data type to be used if user data is not valid.
            It can take the values 'float32' or 'float64' when force_float=True 
            and 'float32', 'float64', 'int32' or 'int64' when force_float=False. 
            Default is 'float64'.

        force_float (bool, optional): Boolean to force data type converstion to 'float32' or 'float64'
            when original data is 'int32' or 'int64'. Default is True.

    Returns:
        X (numpy.ndarray[float32 | float64 | int32 | int64]): The validated data matrix.
    """    
    if force_float and dtype in ['float32', 'float64']:
        allowed_dtypes = [np.float32, np.float64]
    elif not force_float and dtype in ['float32', 'float64', 'int32', 'int64']:
        allowed_dtypes = [np.float32, np.float64, np.int32, np.int64]
    else:
        raise ValueError(
            "dtype must be 'float32' or 'float64' if force_floa=True or 'float32', 'float64', 'int32' or 'int64' if force_float=False.")
    dtype = np.dtype(dtype)
    
    if isinstance(X, np.ndarray):
        if not X.dtype in allowed_dtypes:
            X = X.astype(dtype=dtype)
            warnings.warn(UserWarning(
                f"Input had an unsupported data type and has been cast to {dtype}."))
        return X

    try:
        if len(X) > 0 and hasattr(X[0], 'dtype') and X[0].dtype in allowed_dtypes:
            X = np.asarray(X, dtype=X[0].dtype)
        else:
            X = np.asarray(X, dtype=dtype)
            warnings.warn(UserWarning(
                f"Input had an unsupported data type and has been cast to {dtype}."))
    except AttributeError:
        raise ValueError(
            f"Input data cannot be converted to a NumPy ndarray of type {dtype}.")

    return X
