# Copyright 2025 Iguazio
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

class VerifySchema:
    def __init__(self, name: str, schema: list):
        self.name = name
        self.schema = schema

    def do(self, event: dict):
        missing = set(self.schema) - set(event)
        unexpected = set(event) - set(self.schema)
        if missing:
            raise KeyError(f"Schema verification failed: missing keys {missing} in event: {event}")
        if unexpected:
            raise KeyError(f"Schema verification failed: unexpected keys {unexpected} in event: {event}")
        return event