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


class TestAlice:

    def test_train(self):
        ok_(False)


class TestBob:

    def test_test(self):
        ok_(False)

