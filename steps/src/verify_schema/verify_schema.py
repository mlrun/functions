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
    """
    This step validates that an event dictionary contains exactly the keys defined in the schema,
    raising a KeyError if any are missing or unexpected.
    """

    def __init__(self, schema: list, allow_unexpected_keys: bool = False):
        self.schema = schema
        self.allow_unexpected_keys = allow_unexpected_keys

    def do(self, event: dict):
        # Check if all keys in the expected schema are present in the event
        missing = set(self.schema) - set(event)
        if missing:
            raise KeyError(
                f"Schema verification failed: missing keys {missing} in event: {event}"
            )

        if self.allow_unexpected_keys:
            return event

        # Check if there are any unexpected keys in the event
        unexpected = set(event) - set(self.schema)
        if unexpected:
            raise KeyError(
                f"Schema verification failed: unexpected keys {unexpected} in event: {event}"
            )

        return event
