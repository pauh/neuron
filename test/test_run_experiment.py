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

from run_experiment import Alice
from run_experiment import Bob
from run_experiment import Configuration
from run_experiment import Cognon

from cognon_extended import Neuron
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

from unittest.case import SkipTest


class TestAlice:

    def test_train(self):
        n = Neuron(S0 = 16, H = 4.0, G = 2.0, C = 1, D1 = 1, D2 = 1)

        wA = Word([(1,0), (6,0), (9,0), (14,0)])
        wB = Word([(3,0), (4,0), (9,0), (13,0)])
        wordset = WordSet(num_words = 2, word_length = 16, num_delays = 1,
                          num_active = 4)
        wordset.words = [wA, wB]

        alice = Alice()
        alice.train(n, wordset)

        # Test recognition
        wD = Word([(2,0), (6,0), (12,0), (14,0)])
        fired, delay, container = n.expose(wD)
        assert_false(fired)

        wE = Word([(3,0), (7,0), (9,0), (13,0)])
        fired, delay, container = n.expose(wE)
        assert_false(fired)

        wF = Word([(1,0), (4,0), (9,0), (14,0)])
        fired, delay, container = n.expose(wF)
        assert_true(fired) # False alarm


class TestBob:

    def test_test(self):
        n = Neuron(S0 = 16, H = 4.0, G = 2.0, C = 1, D1 = 1, D2 = 1)

        wA = Word([(1,0), (6,0), (9,0), (14,0)])
        wB = Word([(3,0), (4,0), (9,0), (13,0)])
        train_wordset = WordSet(num_words = 2, word_length = 16,
                                num_delays = 1, num_active = 4)
        train_wordset.words = [wA, wB]

        alice = Alice()
        alice.train(n, train_wordset)

        wD = Word([(2,0), (6,0), (12,0), (14,0)])
        wE = Word([(3,0), (7,0), (9,0), (13,0)])
        wF = Word([(1,0), (4,0), (9,0), (14,0)]) # False alarm
        test_wordset = WordSet(num_words = 3, word_length = 16,
                               num_delays = 1, num_active = 4)
        test_wordset.words = [wD, wE, wF]

        bob = Bob()
        bob.test(n, train_wordset, test_wordset)

        eq_(bob.true_true, 2)
        eq_(bob.true_false, 0)
        eq_(bob.false_true, 1)
        eq_(bob.false_false, 2)


class TestConfiguration:

    def test_config(self):
        raise SkipTest


class TestCognon:

    def test_cognon(self):
        raise SkipTest

