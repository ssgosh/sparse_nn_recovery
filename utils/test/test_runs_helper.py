import sys
sys.path.append(".")

import utils.runs_helper as rh

class SomeClass:
    def __init__(self):
        self.run_dir = None
        self.run_suffix = '_haha'

item = [ 'cat', 'dog', 'hyena', 'leopard', 'zebra' ]
for i in range(5):
    config = SomeClass()
    if i % 2 == 0:
        config.run_suffix = f"_{item[i]}_{i}"

    rh.setup_run_dir(config, 'foobar')
    print(config.run_dir)
