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

from math import log
from multiprocessing import Pool
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



class Configuration(object):

    def __init__(self):
        self.neuron_params()
        self.test_params()


    def neuron_params(self, C = 1, D1 = 4, D2 = 7, Q = 40, G = 2, H = 5):
        self.H = H   # Num. of synapses needed to fire a neuron
        self.G = G   # Ratio of strong synapse strength to weak synapse s.
        self.C = C   # Num. of dendrite compartments
        self.D1 = D1 # Num. of posible time slots where spikes can happen
        self.D2 = D2 # Num. of time delays available between two layers
        self.Q = Q   # Q = S0/(H*R*C)


    def test_params(self, num_active = 4, R = None, w = 100, num_test_words = 0):
        self.num_active = num_active # Num. of active synapses per word
        self.R = R   # Avg. num. of patterns per afferent synapse spike
        self.w = w   # Num. of words to train the neuron with
        self.num_test_words = num_test_words # Num. of words to test


    @property
    def S0(self):
        if self.R:
            return int(self.Q * self.H * self.C * self.R)
        else:
            return int(self.Q * self.H * self.C)



class Cognon(object):

    def __call__(self, config):
        return self.run_experiment(config)


    def run_configuration(self, config, repetitions):

        # Ensure that at least 10,000 words are learnt
        MIN_LEARN_WORDS = 10000
        MIN_LEARN_WORDS = 1
        if repetitions * config.w < MIN_LEARN_WORDS:
            N = MIN_LEARN_WORDS/config.w
        else:
            N = repetitions

        # Ensure that at least 1,000,000 words are tested
        MIN_TEST_WORDS = 1000000
        if not config.num_test_words:
            config.num_test_words = MIN_TEST_WORDS/N

        # Run all the experiments
        #values = [self.run_experiment(config) for i in xrange(N)]
        pool = Pool(processes=20)
        values = pool.map(Cognon(), [config,]*N)

        # Store the results in a NumPy structured array
        names = ('pL', 'pF', 'L')
        types = [np.float64,] * len(values)
        r = np.array(values, dtype = zip(names, types))

        return r


    def run_experiment(self, cfg):

        # create a neuron instance with the provided parameters
        neuron = Neuron(cfg.S0, cfg.H, cfg.G, cfg.C, cfg.D1, cfg.D2)

        # create the training and test wordsets
        train_wordset = WordSet(cfg.w, cfg.S0, cfg.D1, cfg.num_active, cfg.R)
        test_wordset = WordSet(cfg.num_test_words, cfg.S0, cfg.D1,
                               cfg.num_active, cfg.R)

        # create Alice instance to train the neuron
        alice = Alice()
        alice.train(neuron, train_wordset)

        # create a Bob instance to test the neuron
        bob = Bob()
        bob.test(neuron, train_wordset, test_wordset)

        # results
        pL = bob.true_true/float(cfg.w)
        pF = bob.false_true/float(cfg.num_test_words)

        # L  = w*((1-pL)*log2((1-pL)/(1-pF)) + pL*log2(pL/pF)) bits
        L = 0
        if pL == 1.0:
            if pF != 0:
                L = -cfg.w*log(pF)/log(2.0)
            else:
                L = cfg.w
        elif pL > pF:
            L = cfg.w/log(2.0) * \
                (log(1.0 - pL) - log(1.0 - pF) +
                 pL * (log(1.0 - pF) - log(1.0 - pL) + log(pL) - log(pF)))

        return pL, pF, L

