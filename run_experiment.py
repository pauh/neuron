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
from cognon_extended import WordSet

import numpy as np
import random


class Alice(object):

    def train(self, neuron, wordset):
        neuron.start_training()

        for word in wordset.words:
            fired = neuron.train(word)

        neuron.finish_training()



class Bob(object):

    def __init__(self):
        self.true_true = 0   # trained and fired (ok)
        self.true_false = 0  # trained but not fired (false negative)
        self.false_true = 0  # not trained but fired (false positive)
        self.false_false = 0 # not trained and not fired (ok)

    def test(self, neuron, train_wordset, test_wordset):
        # Check the training set
        for word in train_wordset.words:
            fired, delay, container = neuron.expose(word)
            if fired:
                self.true_true += 1
            else:
                self.true_false += 1

        # Check the test set
        #num_active = len(train_wordset.words[0].synapses[0])
        #test_wordset = Wordset(num_test_words, neuron.S0, neuron.D1, num_active)
        for word in test_wordset.words:
            fired, delay, container = neuron.expose(word)
            if fired:
                self.false_true += 1
            else:
                self.false_false += 1



class Cognon(object):

    def run_configuration(self, repetitions):

        # Ensure that at least 10,000 words are learnt
        min_learn_words = 10000
        #N = repetitions
        #if N * w < min_learn_words:
        #    N = min_learn_words/w

        # Ensure that at least 1,000,000 words are tested
        min_test_words = 1000000
        #K = min_test_words/N

        #for i in xrange(N):
        #    run_experiment(i)

        # Config for each experiment:
        # - neuron
        #    S0, H, G, C, D1, D2, (num_active or refractory_period)
        # - training
        #    num_train_words (w)
        # - test
        #    num_test_words


    def run_experiment(self):

        # create a neuron instance with the provided parameters
        neuron = Neuron(S0, H, G, C, D1, D2)

        # create the training and test wordsets
        train_wordset = WordSet(num_train_words, S0, D1, num_active)
        test_wordset = WordSet(num_test_words, S0, D1, num_active)

        # create Alice instance to train the neuron
        alice = Alice()
        alice.train(n, train_wordset)

        # create a Bob instance to test the neuron
        bob = Bob()
        bob.test(neuron, train_wordset, test_wordset)

