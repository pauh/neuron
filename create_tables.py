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
    config.neuron_params(C, D1, D2, Q, G, H)
    config.test_params(active, R, w, 5000)

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
    r = run_table_row(w, N, 1, 1, 1, S/float(H), None, G, float(H))
    pF_mean = r['pF'].mean()*100
    pF_std  = r['pF'].std()*100
    L_mean  = r['L'].mean()
    L_S0    = L_mean/S
    txt = "\t{:.2f} & {} & {} & {:,} & {} & {} & {:.1f} & {:.2f} \\\\"
    print txt.format(pF_mean, N, H, S, w, G, L_mean, L_S0)


def table23():
    print "%%%%%%%%%%%%%%"
    print "% Table 2.3. %"
    print "%%%%%%%%%%%%%%"
    print
              #  H |  G |  S0  |  R  | w #
    table23_row( 30, 4.0, 10000, 303, 200)
    table23_row(105, 4.0, 10000,  86,  70)
    table23_row( 40, 1.9, 10000, 250, 100)

    print "\t\\midrule"

    table23_row(  5, 3.6,  1000, 333, 300)
    table23_row( 10, 3.6,  1000, 111,  60)
    table23_row(  5, 1.9,  1000, 333, 300)
    table23_row( 15, 4.0,  1000,  66,  30)

    print "\t\\midrule"

    table23_row(  5, 3.6,   200,  57,  40)
    table23_row( 10, 4.0,   200,  20,  10)
    table23_row( 20, 1.9,   200,  12,  10)

def table23_row(H, G, S, R, w):
    r = run_table_row(w, None, 1, 1, 1, S/float(H*R), R, G, float(H))
    L_mean  = r['L'].mean()
    pF_mean = r['pF'].mean()*100
    pF_std  = r['pF'].std()*100
    pL_mean = r['pL'].mean()*100
    pL_std  = r['pL'].std()*100
    txt = "\t{:.1f} & {:.2f} & {:.2f} & {} & {:.1f} & {:,} & {} & {} \\\\"
    print txt.format(L_mean, pF_mean, pL_mean, H, G, S, R, w)



if __name__ == "__main__":
    #table21()
    #print
    table23()

