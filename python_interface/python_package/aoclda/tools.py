# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and / or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

"""
Tools and utility functions.
"""
# pylint: disable=not-an-iterable,no-self-argument,no-member
# pylint: disable=import-error,invalid-name,too-many-arguments
# pylint: disable=missing-module-docstring,too-many-locals, anomalous-backslash-in-string
from ._aoclda.tools import (
    pybind_debug_set, pybind_debug_get, pybind_debug_print_context_registry)


class _debug():

    def set(dic):
        """ Setter for the context registry """
        if not isinstance(dic, dict):
            raise TypeError("set data ``dic'' must be a dictionary.")
        for key, value in dic.items():
            if value is None:
                value = ""
            pybind_debug_set(key, value)

    def get(key=None):
        """ Getter for the context registry """
        if key is None:
            pybind_debug_print_context_registry()
            return {}

        # check if key is a string or a list, tuple, or set of strings
        ok_list = isinstance(key, (list, tuple, set)) and all(
            isinstance(elem, str) for elem in key)
        ok_str = isinstance(key, str)
        ok = ok_list or ok_str
        if not ok:
            raise TypeError(
                "get data key ``key'' must contain at least one string.")

        keys = [key] if isinstance(key, str) else key

        dic = {}
        for k in keys:
            value = pybind_debug_get(k)
            dic[k] = value

        return dic
