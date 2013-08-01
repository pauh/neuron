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

from collections import namedtuple
import numpy as np
import random


class Synapse(namedtuple('Synapse', ['offset', 'delay'])):
    """A Synapse represents a connection between the neuron's input dendrites
    and the output axons of other neurons.

    Attributes:
        offset: Identifies a synapse of the neuron.
        delay: Represents the time the signal takes to traverse the axon to
        reach the synapse. Takes a value in range(D1).
    """
    pass



class Word(object):
    """An input Word represents the input signals to the neuron for a time
    period.

    A Word contains a list of those input synapses that fired for the
    most recent given excitation pattern.

    Attributes:
        synapses: A set of pairs containing the syanpses that fired and the
            associated delay.
    """

    def __init__(self, fired_syn=[]):
        """Inits Word class.

        Args:
            fired_syn: List of pairs of input synapses that fired and
                associated delays. Can only contain positive synapse offset
                values.
        """
        if len(fired_syn) > 0 and sorted(fired_syn)[0][0] < 0:
            raise ValueError('synapse offset values have to be positive')
        self.synapses = [Synapse(*s) for s in fired_syn]



class WordSet(object):
    """An array of Words.

    Wordset is simply an array of Word instances, which may also store
    information regarding the delay slot learned for the word during training.

    Attributes:
        words: Array of Word instances.
        delays: Delay slots learned for each word during training.
    """

    def __init__(self, num_words, word_length, num_delays, num_active):
        """Inits WordSet class.

       Args:
           num_words: Number of Words to initialize the WordSet with.
           word_length: Number of synapses in a Word.
           num_delays: Number of delay slots.
           num_active: Number of active synapses per word.

       TODO: Add the other initialization method
       """
        self.words = []
        self.delays = [0] * num_words
        synapses = range(word_length)

        for i in range(num_words):
            active_syn = random.sample(synapses, num_active)
            active_delays = np.random.randint(num_delays, size=num_active)
            self.words.append(Word(zip(active_syn, active_delays)))



class Neuron(object):
    """Models a CE neuron.

    Attributes:
        S0: Number of synapses.
        H: Number of synapses needed to fire a neuron.
        G: Ratio of strong synapse strength to weak synapse strength, binary
            approximation.
        C: Number of dendrite compartments capable of firing independently.
        D1: Number of possible time slots where neurons can produce spikes.
        D2: Number of different time delays available between two neural
            layers.
        synapses: Represents a connection between the neuron's input dendrites
            and the output axons of other neurons. Each row of the array
            contains 3 fields:
            - strength: Strength of the synapse.
            - delay: Represents the time the signal takes to traverse the axon
              to reach the synapse. Takes a value in range(D2).
            - container: The dendrite compartment of this synapse.
        training: whether the neuron is in training mode.
    """

    def __init__(self, S0 = 200, H = 5.0, G = 2.0, C = 1, D1 = 4, D2 = 7):
        """Inits Neuron class.

        Args:
            S0: Number of synapses.
            H: Number of synapses needed to fire a neuron.
            G: Ratio of strong synapse strength to weak synapse strength,
                binary approximation.
            C: Number of dendrite compartments capable of firing independently.
            D1: Number of possible time slots where neurons can produce spikes.
            D2: Number of different time delays available between two neural
                layers.
        """
        self.S0 = S0
        self.H = H
        self.G = G
        self.C = C
        self.D1 = D1
        self.D2 = D2
        self.training = False
        self.synapses = np.zeros(S0, dtype='float32,uint16,uint16')
        self.synapses.dtype.names = ('strength', 'delay', 'container')
        self.synapses['strength'] = 1.0
        self.synapses['delay'] = np.random.randint(D2, size=S0)
        self.synapses['container'] = np.random.randint(C, size=S0)


    def expose(self, w):
        """Models how the neuron reacts to excitation patterns, and how it
        computes whether or not to fire.

        Expose computes the weighted sum of the input word, and the neuron fires
        if that sum meets or exceeds a threshold. The weighted sum is the sum of
        the S0 element-by-element products of the most recent neuron vector and
        the current word.

        Args:
            w: A Word to present to the neuron.

        Returns:
            A 3-element tuple containing:
                0. A Boolean indicating whether the neuron fired or not.
                1. The delay in which the neuron has fired.
                2. The container where the firing occurred.
        """
        offsets = [syn.offset for syn in w.synapses]
        delays  = [syn.delay  for syn in w.synapses]
        synapses = self.synapses[offsets]

        # Iterate over delays until neuron fires
        for d in range(self.D1 + self.D2):
            delay_indices = (synapses['delay'] + delays) == d

            # Compute the weighted sum of the firing inputs for each container
            for c in range(self.C):
                container_indices = synapses['container'] == c
                indices = delay_indices & container_indices
                s = synapses['strength'][indices].sum()

                # Check if the container has fired
                if (self.training and s >= self.H) or s >= self.H*self.G:
                    return (True, d, c)

        # If no container has fired for any delay
        return (False, None, None)


    def train(self, w):
        """Trains a neuron with an input word.

        To train a neuron, "train" is called for each word to be recognized. If
        the neuron fires for that word then all synapses that contributed to
        that firing have their strengths irreversibly increased to G.

        Args:
            w: A Word to train the neuron with.

        Returns:
            A Boolean indicating whether the neuron fired or not.
        """
        if not self.training:
            print "[WARN] train(w) was called when not in training mode."
            return False

        fired, delay, container = self.expose(w)
        if not fired: return False

        # Update the synapses that contributed to the firing
        offsets = [s.offset for s in w.synapses]
        delays  = [syn.delay  for syn in w.synapses]

        synapses = self.synapses[offsets]

        delay_indices = (synapses['delay'] + delays) == delay
        container_indices = synapses['container'] == container
        active_indices = delay_indices & container_indices

        indices = np.zeros(self.S0, dtype=bool)
        indices[offsets] = active_indices

        self.synapses['strength'][indices] = self.G

        return True


    def start_training(self):
        """Set the neuron in training mode.
       """
        self.training = True


    def finish_training(self):
        """Set the neuron in recognition mode.

       Once the training is complete, the neuron's threshold value H is set
       to H*G.
       """
        self.training = False

