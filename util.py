# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DNC util ops and modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

  
def batch_invert_permutation(prem_idx):
  """Returns batched `tf.invert_permutation` for every row in `permutations`."""
  if len(prem_idx.shape) != 2:
    print('only 2 d tensor prem could be inverted')
  
  batch_size,prem_size = prem_idx.shape
 
  inverted_prem = torch.ones(prem_idx.shape)
  for i in range(batch_size):
    prem_idx_ith_row = prem_idx[i]
    inverted_prem[i][prem_idx_ith_row] = torch.arange(prem_size).float()

  return inverted_prem.long()
    

def batch_gather(values, indices):
  """Returns batched `tf.gather` for every row in the input."""
  if np.all(values.shape == indices.shape) != True:
    print('The shapes of values and indices should be indentical')
  row , col = values.shape
  result  = torch.empty(values.shape)
  for i in range(row):
    result[i] = values[i][indices[i]]
  return result.long()


def one_hot(length, index):
  """Return an nd array of given `length` filled with 0s and a 1 at `index`."""
  result = torch.zeros(length)
  result[index] = 1
  return result

def reduce_prod(x, axis, name=None):
  """Efficient reduce product over axis.

  Uses tf.cumprod and tf.gather_nd as a workaround to the poor performance of calculating tf.reduce_prod's gradient on CPU.
  """

  return torch.prod(x,axis)
  
