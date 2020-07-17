__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import numpy as np
import numpy.linalg as la  # noqa
import pyopencl.array as cla  # noqa
from pytools.obj_array import flat_obj_array


def compare_states(red_state, blue_state):
    resid = red_state - blue_state
    numfields = len(resid)
    max_errors = [np.max(np.abs(resid[i].get())) for i in range(numfields)]
    return max_errors


def get_field_stats(state):
    numfields = len(state)
    field_mins = [np.min(state[i].get()) for i in range(numfields)]
    field_maxs = [np.max(state[i].get()) for i in range(numfields)]
    return flat_obj_array(field_mins, field_maxs)
