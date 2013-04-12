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

import numpy as np


class Word(object):
    """A Word contains a list of those input synapses that fired for the most
    recent given excitation pattern.

    Attributes:
        offset: A set containing the syanpses that fired.
    """
    
    def __init__(self, fired_syn=[]):
        """Inits Word class.

        Args:
            fired_syn: List of input synapses that fired. Can only contain
            positive values.
        """
        if len(fired_syn) > 0 and sorted(fired_syn)[0] < 0:
            raise ValueError('offset values have to be positive')
        self.offset = set(fired_syn)


class WordSet(object):
    """An array of Words.

    Wordset is simply an array of Word instances, which may also store
    information regarding the delay slot learned for the word during training.

    Attributes:
        words: Array of Word instances.
        delays: Delay slots learned for each word during training.
    """

    def __init__(self):
        """Inits WordSet class."""
        self.words = []
        self.delays = []


class Neuron(object):
    """Models a CB neuron.

    Attributes:
        S0: Number of synapses.
        H: Number of synapses needed to fire a neuron.
        G: Ratio of strong synapse strength to weak synapse strength, binary
            approximation.
        training: whether the neuron is in training mode.
    """

    def __init__(self, S0 = 16, H = 4.0, G = 2.0):
        """Inits Neuron class.

        Args:
            S0: Number of synapses.
            H: Number of synapses needed to fire a neuron.
            G: Ratio of strong synapse strength to weak synapse strength,
                binary approximation.
        """
        self.S0 = S0
        self.H = H
        self.G = G
        self.strength = np.ones(S0)
        self.training = False


    def expose(self, w):
        """Models how the neuron reacts to excitation patterns, and how it computes
       whether or not to fire.

       Expose computes the weighted sum of the input word, and the neuron fires if
       that sum meets or exceeds a threshold. The weighted sum is the sum of the
       S0 element-by-element products f the most recent neron vector, the current
       word, and the neuron frozen Boolean vector.

       Args:
           w: A Word to present to the neuron.

       Returns:
           A Boolean indicating whether the neuron will fire or not.
       """
        # Compute the weighted sum of the firing inputs
        s = self.strength[list(w.offset)].sum()
        if self.training:
            return s >= self.H
        else:
            return s >= self.H*self.G
    
    
    def train(self, w):
        """Trains a neuron with an input word.
    
       To train a neuron, "train" is called for each word to be recognized. If the
       neuron fires for that word then all synapses that contributed to that
       firing have their strengths irreversibly increased to G.

       Args:
           w: A Word to train the neuron with.
    
       Returns:
           A Boolean indicating whether the neuron fired or not.
       """
        if not self.training:
            print "[WARN] train(w) was called when not in training mode."
            return False
        
        if not self.expose(w): return False
    
        # Set the srength for participating synapses to G
        self.strength[list(w.offset)] = self.G
    
        return True

    
    def start_training(self):
        """Set the neuron in training mode.
       """
        self.training = True


    def end_training(self):
        """Set the neuron in recognition mode.
    
       Once the training is complete, the neuron's threshold value H is set
       to H*G.
       """
        self.training = False

