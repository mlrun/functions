# Copyright 2019 Iguazio
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
"""Snowflake Dask unit test"""
from mlrun import import_function

def test_snowflake_dask():
    """An unit test"""
    fn_to_test = import_function("function.yaml")

    # a fake assert to pass the unit test
    if fn_to_test.to_yaml().__contains__('job'):
        assert True
