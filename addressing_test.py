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
"""Tests for memory addressing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
#import sonnet as snt
import tensorflow as tf
import torch

import addressing
import util



class WeightedSoftmaxTest(tf.test.TestCase):
  def WeightedSoftmaxTest(self):
    batch_size = 5
    num_heads = 3
    memory_size = 7

    activations_data = np.random.randn(batch_size, num_heads, memory_size)
    weights_data = np.ones((batch_size, num_heads))
    activations_data_tensor = torch.tensor(activations_data)
    weights_data_tensor = torch.tensor(weights_data)
    activations = tf.placeholder(tf.float32,
                                 [batch_size, num_heads, memory_size])
    weights = tf.placeholder(tf.float32, [batch_size, num_heads])
    # Run weighted softmax with identity placed on weights. Output should be
    # equal to a standalone softmax.
    leave_it  = torch.nn.Identity()
    observed = addressing.weighted_softmax(activations_data_tensor, weights_data_tensor, leave_it)
    expected = snt.BatchApply(
        module_or_op=tf.nn.softmax, name='BatchSoftmax')(activations)
    with self.test_session() as sess:
      expected = sess.run(expected, feed_dict={activations: activations_data})
     

class CosineWeightsTest(tf.test.TestCase):

  def testShape(self):
    batch_size = 5
    num_heads = 3
    memory_size = 7
    word_size = 2

    module = addressing.CosineWeights(num_heads, word_size)
    mem = torch.ones([batch_size, memory_size, word_size])
    keys = torch.ones([batch_size, num_heads, word_size])
    strengths = torch.ones([batch_size, num_heads])
    weights = module(mem, keys, strengths)
    print('expected result:')
    print([batch_size, num_heads, memory_size])
    print('observed shape:')
    print(weights.shape)
  def testValues(self):
    batch_size = 5
    num_heads = 4
    memory_size = 10
    word_size = 2

    mem_data = np.random.randn(batch_size, memory_size, word_size)
    np.copyto(mem_data[0, 0], [1, 2])
    np.copyto(mem_data[0, 1], [3, 4])
    np.copyto(mem_data[0, 2], [5, 6])

    keys_data = np.random.randn(batch_size, num_heads, word_size)
    np.copyto(keys_data[0, 0], [5, 6])
    np.copyto(keys_data[0, 1], [1, 2])
    np.copyto(keys_data[0, 2], [5, 6])
    np.copyto(keys_data[0, 3], [3, 4])
    strengths_data = np.random.randn(batch_size, num_heads)

    mem = torch.tensor(mem_data)
    keys = torch.tensor(keys_data)
    strengths = torch.tensor(strengths_data)
    module = addressing.CosineWeights(num_heads, word_size)
    weights = module(mem, keys, strengths)

    # Manually checks results.
    strengths_softplus = np.log(1 + np.exp(strengths_data))
    similarity = np.zeros((memory_size))

    for b in range(batch_size):
      for h in range(num_heads):
        key = keys_data[b, h]
        key_norm = np.linalg.norm(key)

        for m in range(memory_size):
          row = mem_data[b, m]
          similarity[m] = np.dot(key, row) / (key_norm * np.linalg.norm(row))

        similarity = np.exp(similarity * strengths_softplus[b, h])
        similarity /= similarity.sum()
        similarity = torch.tensor(similarity)
        if  not torch.allclose(weights[b, h], similarity, atol=1e-4, rtol=1e-4):
          print('CosineWeightsTest fail')
          return
    print('test pass')
  def testDivideByZero(self):
    batch_size = 5
    num_heads = 4
    memory_size = 10
    word_size = 2

    module = addressing.CosineWeights(num_heads, word_size)
    keys = torch.randn([batch_size, num_heads, word_size])
    strengths = torch.randn([batch_size, num_heads])

    # First row of memory is non-zero to concentrate attention on this location.
    # Remaining rows are all zero.
    first_row_ones = torch.ones([batch_size, 1, word_size])
    remaining_zeros = torch.zeros([batch_size, memory_size - 1, word_size])
    mem = torch.cat((first_row_ones, remaining_zeros), 1)

    output = module(mem, keys, strengths)
    #gradients = tf.gradients(output, [mem, keys, strengths])

    if torch.any(torch.isnan(output)):
      print('can not caculate gradients')
      


class TemporalLinkageTest(tf.test.TestCase):

  def testModule(self):
    batch_size = 7
    memory_size = 4
    num_reads = 11
    num_writes = 5
    module = addressing.TemporalLinkage(
        memory_size=memory_size, num_writes=num_writes)

    state = addressing.TemporalLinkageState(
        link=torch.zeros([batch_size, num_writes, memory_size, memory_size]).double(),
        precedence_weights=torch.zeros([batch_size, num_writes, memory_size]).double())

    num_steps = 5
    for i in range(num_steps):
      write_weights = np.random.rand(batch_size, num_writes, memory_size)
      write_weights /= write_weights.sum(2,keepdims=True) + 1

      # Simulate (in final steps) link 0-->1 in head 0 and 3-->2 in head 1
      if i == num_steps - 2:
        write_weights[0, 0, :] = util.one_hot(memory_size, 0)
        write_weights[0, 1, :] = util.one_hot(memory_size, 3)
      elif i == num_steps - 1:
        write_weights[0, 0, :] = util.one_hot(memory_size, 1)
        write_weights[0, 1, :] = util.one_hot(memory_size, 2)
      write_weights = torch.tensor(write_weights)
      
      state = module(write_weights.double(),state)

    # link should be bounded in range [0, 1]
    if torch.any(state.link < 0 ):
      print('every elements from state_Temprol matrix should >= 0 ')
      return
      
    #self.assertLessEqual(state.link.max(), 1)

    # link diagonal should be zero
##    self.assertAllEqual(
##        state.link[:, :, range(memory_size), range(memory_size)],
##        np.zeros([batch_size, num_writes, memory_size]))

    # link rows and columns should sum to at most 1
##    self.assertLessEqual(state.link.sum(2).max(), 1)
##    self.assertLessEqual(state.link.sum(3).max(), 1)

    # records our transitions in batch 0: head 0: 0->1, and head 1: 3->2
##    self.assertAllEqual(state.link[0, 0, :, 0], util.one_hot(memory_size, 1))
##    self.assertAllEqual(state.link[0, 1, :, 3], util.one_hot(memory_size, 2))

    # Now test calculation of forward and backward read weights
    prev_read_weights = np.random.rand(batch_size, num_reads, memory_size)
    prev_read_weights[0, 5, :] = util.one_hot(memory_size, 0)  # read 5, posn 0
    prev_read_weights[0, 6, :] = util.one_hot(memory_size, 2)  # read 6, posn 2
    prev_read_weights = torch.tensor(prev_read_weights)
    forward_read_weights = module.directional_read_weights(
        state.link,
        prev_read_weights,
        forward=True)
    backward_read_weights = module.directional_read_weights(
        state.link,
        prev_read_weights,
        forward=False)

    if torch.all(forward_read_weights[0, 5, 0, :].double() == torch.tensor(util.one_hot(memory_size, 1)).double()) != True:
      print('forward directional weights calculated not correctly.')
      return
    if torch.all(backward_read_weights[0, 6, 1, :] == torch.tensor(util.one_hot(memory_size, 3))) != True:
      print('backward directional weights calculated not correctly.')
      return
  def testPrecedenceWeights(self):
    batch_size = 7
    memory_size = 3
    num_writes = 5
    module = addressing.TemporalLinkage(
        memory_size=memory_size, num_writes=num_writes)

    prev_precedence_weights = np.random.rand(batch_size, num_writes,
                                             memory_size)
    write_weights = np.random.rand(batch_size, num_writes, memory_size)

    # These should sum to at most 1 for each write head in each batch.
    write_weights /= write_weights.sum(2, keepdims=True) + 1
    prev_precedence_weights /= prev_precedence_weights.sum(2, keepdims=True) + 1

    write_weights[0, 1, :] = 0  # batch 0 head 1: no writing
    write_weights[1, 2, :] /= write_weights[1, 2, :].sum()  # b1 h2: all writing
    prev_precedence_weights = torch.tensor(prev_precedence_weights)
    write_weights = torch.tensor(write_weights)
    precedence_weights = module._precedence_weights(
        prev_precedence_weights=prev_precedence_weights,
        write_weights=write_weights)
    print(precedence_weights)
    # precedence weights should be bounded in range [0, 1]
    if torch.all(precedence_weights> 0)!=True:
      print('precedence weights should be greater than 0')
    #self.assertLessEqual(precedence_weights.max(), 1)
    if torch.all(precedence_weights < 1)!=True:
      print('precedence weights should be less than 1')
    # no writing in batch 0, head 1
    if torch.allclose(precedence_weights[0, 1, :],prev_precedence_weights[0, 1, :]) !=True :
      print('no writing in batch 0, head 1')
      return

    # all writing in batch 1, head 2
    #self.assertAllClose(precedence_weights[1, 2, :], write_weights[1, 2, :])
    if torch.allclose(precedence_weights[1, 2, :],write_weights[1, 2, :]) !=True :
      print('all writing in batch 1, head 2')
      return

class FreenessTest():

  def testModule(self):
    batch_size = 5
    memory_size = 11
    num_reads = 3
    num_writes = 7
    module = addressing.Freeness(memory_size)

    free_gate = np.random.rand(batch_size, num_reads)

    # Produce read weights that sum to 1 for each batch and head.
    prev_read_weights = np.random.rand(batch_size, num_reads, memory_size)
    prev_read_weights[1, :, 3] = 0  # no read at batch 1, position 3; see below
    prev_read_weights /= prev_read_weights.sum(2, keepdims=True)
    prev_write_weights = np.random.rand(batch_size, num_writes, memory_size)
    prev_write_weights /= prev_write_weights.sum(2, keepdims=True)
    prev_usage = np.random.rand(batch_size, memory_size)

    # Add some special values that allows us to test the behaviour:
    prev_write_weights[1, 2, 3] = 1  # full write in batch 1, head 2, position 3
    prev_read_weights[2, 0, 4] = 1  # full read at batch 2, head 0, position 4
    free_gate[2, 0] = 1  # can free up all locations for batch 2, read head 0

    prev_write_weights = torch.tensor(prev_write_weights)
    prev_read_weights  = torch.tensor(prev_read_weights)
    free_gate          = torch.tensor(free_gate)
    prev_usage         = torch.tensor(prev_usage)
    
    usage = module(prev_write_weights,free_gate,prev_read_weights, prev_usage)
    # Check all usages are between 0 and 1.
    if usage.min() < 0:
      print('usage should greater or equal to 0 ')
      return 
    if usage.max() > 1:
      print('usage should less or equal to 0')
      return 

    # Check that the full write at batch 1, position 3 makes it fully used.
    if usage[1][3] != 1:
      print('full write at batch 1, position 3 makes it fully used')
      return 

    # Check that the full free at batch 2, position 4 makes it fully free.
    if usage[2][4] != 0:
      print('full free at batch 2, position 4 makes it fully free')
      return 

  def testWriteAllocationWeights(self):
    batch_size = 7
    memory_size = 23
    num_writes = 5
    module = addressing.Freeness(memory_size)

    usage = np.random.rand(batch_size, memory_size)
    write_gates = np.random.rand(batch_size, num_writes)

    # Turn off gates for heads 1 and 3 in batch 0. This doesn't scaling down the
    # weighting, but it means that the usage doesn't change, so we should get
    # the same allocation weightings for: (1, 2) and (3, 4) (but all others
    # being different).
    write_gates[0, 1] = 0
    write_gates[0, 3] = 0
    # and turn heads 0 and 2 on for full effect.
    write_gates[0, 0] = 1
    write_gates[0, 2] = 1

    # In batch 1, make one of the usages 0 and another almost 0, so that these
    # entries get most of the allocation weights for the first and second heads.
    usage[1] = usage[1] * 0.9 + 0.1  # make sure all entries are in [0.1, 1]
    usage[1][4] = 0  # write head 0 should get allocated to position 4
    usage[1][3] = 1e-4  # write head 1 should get allocated to position 3
    write_gates[1, 0] = 1  # write head 0 fully on
    write_gates[1, 1] = 1  # write head 1 fully on


    usage = torch.tensor(usage)
    write_gates = torch.tensor(write_gates)
    
    weights = module.write_allocation_weights(usage,write_gates,num_writes)
    print('weights.shape')
    print(weights.shape)
    # Check that all weights are between 0 and 1
    if weights.min() < 0:
      print('all weights should be  >= 0 ')
      return
    if weights.max() >  1:
      print('all weights should be  <= 0 ')
      return
    # Check that weights sum to close to 1
    if torch.allclose(torch.sum(weights, 2).float(), torch.ones([batch_size, num_writes]).float(), atol=1e-3):
      print(' weights sum should be  close to 1 ' )
      return
    # Check the same / different allocation weight pairs as described above.
    #print(weights)
    if np.abs(weights[0, 0, :] - weights[0, 1, :]).max() < 0.1 :
      print('first')
    if torch.all(weights[0, 1, :] ==weights[0, 2, :])!= True:
      print('second')
    if np.abs(weights[0, 2, :] - weights[0, 3, :]).max()< 0.1 :
      print('third')
    if torch.all(weights[0, 3, :] == weights[0, 4, :]) != True:
      print('forth')
      
    if torch.allclose(weights[1][0].float(), util.one_hot(memory_size, 4).float(), atol=1e-3) != True:
      print(weights[1][0].float() - util.one_hot(memory_size, 4).float())
      print('fifth')
    if torch.allclose(weights[1][1].float(), util.one_hot(memory_size, 3).float(), atol=1e-3) != True:
      print('sixth')

  def testWriteAllocationWeightsGradient(self):
    batch_size = 7
    memory_size = 5
    num_writes = 3
    module = addressing.Freeness(memory_size)

    usage = torch.tensor(np.random.rand(batch_size, memory_size))
    write_gates = torch.tensor(np.random.rand(batch_size, num_writes))
    weights = module.write_allocation_weights(usage, write_gates, num_writes)

    
    err = tf.test.compute_gradient_error(
        [usage, write_gates],
        [usage.get_shape().as_list(), write_gates.get_shape().as_list()],
        weights,
        weights.get_shape().as_list(),
        delta=1e-5)
    self.assertLess(err, 0.01)

  def testAllocation(self):
    batch_size = 7
    memory_size = 13
    usage = np.random.rand(batch_size, memory_size)
    module = addressing.Freeness(memory_size)
    usage = torch.tensor(usage)
    allocation = module._allocation(usage)

    # 1. Test that max allocation goes to min usage, and vice versa.
    alloc_usage_relation = np.argmin(usage, axis=1) == np.argmax(allocation, axis=1)
    usage_alloc_relation = np.argmax(usage, axis=1) == np.argmin(allocation, axis=1)
    if torch.all(alloc_usage_relation) != True:
      print('that max allocation should go to min usage, and vice versa')
      print(alloc_usage_relation)
    if torch.all( usage_alloc_relation)!= True:
      print('that max allocation should go to min usage, and vice versa')
      print(usage_alloc_relation)
    # 2. Test that allocations sum to almost 1.
    if torch.allclose(np.sum(allocation, axis=1), np.ones(batch_size), 0.01) != True:
      print('allocations sum to almost 1')

  def testAllocationGradient(self):
    batch_size = 1
    memory_size = 5
    usage = torch.tensor(np.random.rand(batch_size, memory_size))
    module = addressing.Freeness(memory_size)
    allocation = module._allocation(usage)
    
    err = tf.test.compute_gradient_error(
        usage,
        usage.get_shape().as_list(),
        allocation,
        allocation.get_shape().as_list(),
        delta=1e-5)
    self.assertLess(err, 0.01)


if __name__ == '__main__':
##  pass
##  test_1 = WeightedSoftmaxTest()
##  test_1.WeightedSoftmaxTest()


  test_2 = CosineWeightsTest()
##  pass
##  test_2.testShape()

##  pass
##  test_2.testValues()
##  test_2.testDivideByZero()

  test_3 = TemporalLinkageTest()
##ok  
##  test_3.testModule()
##  test_3.testPrecedenceWeights()
  test_4 = FreenessTest()
##  test_4.testModule()
## Not working yet              
##  test_4.testWriteAllocationWeights()
  test_4.testAllocation()
