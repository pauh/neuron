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

from run_experiment import Cognon
from run_experiment import Configuration


def run_table_row(w, active, C, D1, D2, Q, R, G, H):
    repetitions = 20

    config = Configuration()
    config.neuron_params(C, D1, D2, Q, R, G, H)
    config.test_params(active, w, 5000)

    cognon = Cognon()
    return cognon.run_configuration(config, repetitions)


def table21():
    print "%%%%%%%%%%%%%%"
    print "% Table 2.1. %"
    print "%%%%%%%%%%%%%%"
    print
              # N | H |  S0  |  w  | G #
    table21_row( 4,  4,    10,   1, 100)
    table21_row( 5,  4,    10,   1, 100)
    table21_row( 4,  4,    10,   2, 100)
    table21_row(10, 10,   100,   4, 100)
    table21_row(11, 10,   100,   4, 100)
    table21_row(11, 10,   100,   5, 100)
    table21_row(11, 10,  1000,  60, 100)
    table21_row(11, 10, 10000, 600, 100)
    table21_row(22, 20, 10000, 450, 100)

    print "\t\\midrule"

    table21_row(10, 10,   100,   6, 1.5)
    table21_row(11, 10,  1000,  15, 1.5)
    table21_row(11, 10, 10000, 160, 1.5)
    table21_row(14, 10, 10000,  10, 1.5)

def table21_row(N, H, S, w, G):
    r = run_table_row(w, N, 1, 1, 1, S/float(H), 1, G, float(H))
    pF_mean = r['pF'].mean()*100
    pF_std  = r['pF'].std()*100
    L_mean  = r['L'].mean()
    L_S0    = L_mean/S
    txt = "\t{:.2f} & {} & {} & {:,} & {} & {} & {:.1f} & {:.2f} \\\\"
    print txt.format(pF_mean, N, H, S, w, G, L_mean, L_S0)


if __name__ == "__main__":
    table21()
