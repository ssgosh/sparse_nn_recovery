import sys
sys.path.append(".")

import utils.runs_helper as rh

class TestClass:
    def __init__(self):
        self.run_dir = None

config = TestClass()
for i in range(5):
    rh.setup_run_dir(config, 'foobar')
    print(config.run_dir)
