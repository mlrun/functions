from verify_schema import VerifySchema

class TestVerifySchema:
    def test_verify_schema(self):
        schema = ["id", "name", "active"]
        verifier = VerifySchema(name="test_verifier", schema=schema)

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