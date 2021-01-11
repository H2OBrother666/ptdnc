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
"""Tests for utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

import util




def BatchInvertPermutation_test():
  # Tests that the _batch_invert_permutation function correctly inverts a
  # batch of permutations.
  batch_size = 5
  length = 7

  permutations = torch.zeros([batch_size, length])
  for i in range(batch_size):
    permutations[i] = torch.tensor(np.random.permutation(length))
  inverse = util.batch_invert_permutation(permutations.long())
  

  for i in range(batch_size):
    for j in range(length):
      inv_idx = inverse[i][j]
      if (permutations[i][inv_idx] != j):
        print('BatchInvertPermutation is not working find another way')
        return 
  print('BatchInvertPermutation_test pass')

def batch_gather_test():
  values = torch.tensor([[3, 1, 4, 1], [5, 9, 2, 6], [5, 3, 5, 7]])
  indexs = torch.tensor([[1, 2, 0, 3], [3, 0, 1, 2], [0, 2, 1, 3]])
  target = torch.tensor([[1, 4, 3, 1], [6, 5, 9, 2], [5, 5, 3, 7]])
  result = util.batch_gather(values, indexs)
  if torch.all(target == result) != True:
    print('batch_gather is not working find another way.')
    return
  print('batch_gather_test pass')

if __name__ == '__main__':
  BatchInvertPermutation_test()
  batch_gather_test()
