# Copyright 2013 Pau Haro Negre
# based on C++ code by Carl Staelin Copyright 2009-2011
# 
# See the NOTICE file distributed with this work for additional information
# regarding copyright ownership.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from cognon_extended import Neuron
from cognon_extended import Synapse
from cognon_extended import Word
from cognon_extended import WordSet

from nose.tools import assert_false
from nose.tools import assert_greater_equal
from nose.tools import assert_in
from nose.tools import assert_is_none
from nose.tools import assert_less
from nose.tools import assert_less_equal
from nose.tools import assert_true
from nose.tools import eq_
from nose.tools import ok_
from nose.tools import raises


class TestSynapse:

    @raises(TypeError)
    def test_construct_requires_args(self):
        s = Synapse()

    def test_named_attributes(self):
        s = Synapse(1, 0)
        eq_(s.offset, 1)
        eq_(s.delay, 0)


class TestWord:

    def test_empty(self):
        w = Word()
        eq_(len(w.synapses), 0)

    @raises(ValueError)
    def test_negative_synapse_offset(self):
        w = Word([(-1, 0)])

    def test_fire_1_3_8(self):
        w = Word([(1,0),(3,0),(8,0)])
        eq_(len(w.synapses), 3)
        assert_in((1,0), w.synapses)
        assert_in((3,0), w.synapses)
        assert_in((8,0), w.synapses)
    
    def test_delay_0(self):
        w = Word([(1,0),(3,0),(8,0)])
        for offset, delay in w.synapses:
            eq_(delay, 0)


class TestWordSet:

    def test_small(self):
        num_words = 5
        word_length = 16
        num_delays = 4
        num_active = 4
        
        ws = WordSet(num_words, word_length, num_delays, num_active)
        
        eq_(len(ws.words), num_words)
        eq_(len(ws.delays), num_words)

        for i in range(num_words):
            word = ws.words[i]
            eq_(len(word.synapses), num_active)
            for synapse in word.synapses:
                assert_greater_equal(synapse.offset, 0)
                assert_less(synapse.offset, word_length)
                assert_greater_equal(synapse.delay, 0)
                assert_less(synapse.delay, num_delays)


class TestNeuron:

    def test_defaults(self):
        n = Neuron()
        eq_(n.S0, 200)
        eq_(n.H, 5.0)
        eq_(n.G, 2.0)
        eq_(n.C, 1)
        eq_(n.D1, 4)
        eq_(n.D2, 7)
        assert_false(n.training)
        eq_(len(n.synapses), n.S0)
        assert_true((n.synapses['strength'] == 1.0).all())
        assert_true((n.synapses['delay'] >= 0).all())
        assert_true((n.synapses['delay'] < n.D2).all())
        assert_true((n.synapses['container'] >= 0).all())
        assert_true((n.synapses['container'] < n.C).all())

    def test_attributes_in_range(self):
        n = Neuron()
        assert_greater_equal(n.H, 1.0)
        assert_greater_equal(n.C, 1)
        assert_less_equal(n.D1, n.D2)
        assert_true((n.synapses['strength'] >= 0.0).all())

    def test_expose_not_training(self):
        n = Neuron(S0 = 16, H = 4.0, G = 2.0, C = 1, D1 = 1, D2 = 1)

        w = Word([(1,0), (6,0), (9,0)])
        fired, delay, container = n.expose(w)
        assert_false(fired)
        assert_is_none(delay)
        assert_is_none(container)

        w = Word([(1,0), (3,0), (4,0), (5,0), (6,0), (8,0), (9,0), (14,0)])
        fired, delay, container = n.expose(w)
        assert_true(fired)
        eq_(delay, 0)
        eq_(container, 0)

    @raises(IndexError)
    def test_expose_index_error(self):
        n = Neuron(S0 = 16)
        w = Word([(16,0)])
        n.expose(w)

    def test_expose_multiple_containers(self):
        n = Neuron(S0 = 16, H = 2.0, G = 2.0, C = 3, D1 = 1, D2 = 1)

        # Set container assignment manually to remove randomness
        n.synapses['container'][ 0:10] = 0
        n.synapses['container'][10:14] = 1
        n.synapses['container'][14:16] = 2

        w = Word([(1,0), (2,0), (6,0)])
        fired, delay, container = n.expose(w)
        assert_false(fired)
        assert_is_none(delay)
        assert_is_none(container)

        w = Word([(1,0), (2,0), (3,0), (4,0), (5,0), (6,0)])
        fired, delay, container = n.expose(w)
        assert_true(fired)
        eq_(delay, 0)
        eq_(container, 0)

        w = Word([(10,0), (11,0), (12,0), (13,0)])
        fired, delay, container = n.expose(w)
        assert_true(fired)
        eq_(delay, 0)
        eq_(container, 1)

        w = Word([(14,0), (15,0)])
        fired, delay, container = n.expose(w)
        assert_false(fired)
        assert_is_none(delay)
        assert_is_none(container)

    def test_expose_with_delays(self):
        n = Neuron(S0 = 16, H = 2.0, G = 2.0, C = 1, D1 = 2, D2 = 3)

        # Set delay assignment manually to remove randomness
        n.synapses['delay'][ 0:10] = 0
        n.synapses['delay'][10:14] = 1
        n.synapses['delay'][14:16] = 2

        w = Word([(1,0), (2,0), (6,0)])
        fired, delay, container = n.expose(w)
        assert_false(fired)
        assert_is_none(delay)
        assert_is_none(container)

        w = Word([(1,0), (2,0), (3,0), (4,0), (5,0), (6,0)])
        fired, delay, container = n.expose(w)
        assert_true(fired)
        eq_(delay, 0)
        eq_(container, 0)

        w = Word([(1,1), (2,1), (3,1), (4,1), (5,0), (6,0)])
        fired, delay, container = n.expose(w)
        assert_true(fired)
        eq_(delay, 1)
        eq_(container, 0)

        w = Word([(1,0), (2,0), (3,0), (4,1), (5,1), (6,1)])
        fired, delay, container = n.expose(w)
        assert_false(fired)
        assert_is_none(delay)
        assert_is_none(container)

        w = Word([(10,1), (11,1), (12,1), (13,1)])
        fired, delay, container = n.expose(w)
        assert_true(fired)
        eq_(delay, 2)
        eq_(container, 0)

        w = Word([(12,0), (13,0), (14,0), (15,0)])
        fired, delay, container = n.expose(w)
        assert_false(fired)
        assert_is_none(delay)
        assert_is_none(container)

    def test_train(self):
        n = Neuron(S0 = 16, H = 4.0, G = 2.0, C = 1, D1 = 1, D2 = 1)
        wA = Word([(1,0), (6,0), (9,0), (14,0)])
        wB = Word([(3,0), (4,0), (9,0), (13,0)])

        n.start_training()
        assert_true(n.train(wA))
        assert_true(n.train(wB))
        n.finish_training()

        wD = Word([(2,0), (6,0), (12,0), (14,0)])
        wE = Word([(3,0), (7,0), (9,0), (13,0)])
        fired, delay, container = n.expose(wD)
        assert_false(fired)
        fired, delay, container = n.expose(wE)
        assert_false(fired)

        wF = Word([(1,0), (4,0), (9,0), (14,0)])
        fired, delay, container = n.expose(wF)
        assert_true(fired)

    def test_train_not_training(self):
        n = Neuron()
        w = Word()
        assert_false(n.train(w))

