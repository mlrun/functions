from dummy_step import DummyStep

step = DummyStep(name="test_echo_step")
res = step.do(3)

assert res == 6