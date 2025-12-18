class BaseClass:
    def __init__(self, context, name=None):
        self.context = context
        self.name = name


class DummyStep(BaseClass):
    """ this is a dummy test for testing purpose"""
    def __init__(self, name=None):
        self.name = name

    def do(self, x):
        print(f"My name is {self.name} and my output is: {x+3}")
        return x+3