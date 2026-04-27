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

from verify_schema import VerifySchema

class TestVerifySchema:
    def test_verify_schema(self):
        schema = ["id", "name", "active"]
        verifier = VerifySchema(schema=schema, allow_unexpected_keys=False)

        # Test with valid event
        event = {
            "id": 1,
            "name": "Test Event",
            "active": True
        }
        result = verifier.do(event)
        assert result == event

        # Test with missing key
        event_missing_key = {
            "id": 1,
            "name": "Test Event"
        }
        try:
            verifier.do(event_missing_key)
        except KeyError as e:
            assert "missing keys {'active'} in event" in str(e)

        # Test with unexpected key
        event_unexpected_key = {
            "id": 1,
            "name": "Test Event",
            "active": True,
            "extra": "unexpected"
        }
        try:
            verifier.do(event_unexpected_key)
        except KeyError as e:
            assert "unexpected keys {'extra'} in event" in str(e)

    def test_verify_schema_allow_unexpected(self):
        schema = ["id", "name", "active"]
        verifier = VerifySchema(schema=schema, allow_unexpected_keys=True)

        # Test with valid event and unexpected key
        event = {
            "id": 1,
            "name": "Test Event",
            "active": True,
            "extra": "unexpected"
        }
        result = verifier.do(event)
        assert result == event