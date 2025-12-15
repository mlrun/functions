from dummy_step import DummyStep


class TestDummyStep:
    """Test suite for TestOpenAIProxyApp class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.step = DummyStep(name="test_echo_step")

    def test_dummy_step(self):
        res = self.step.do(3)
        assert res == 6


