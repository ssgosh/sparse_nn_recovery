import pathlib
import pickle
import time

def setup_run_dir(config, default_run_dir='runs'):
    if config.run_dir is None:
        run_num_file = pathlib.Path(f'{default_run_dir}/.last_run_num')
        run_num_file.parent.mkdir(exist_ok=True)
        if run_num_file.exists():
            try:
                with open(run_num_file, 'rb') as f:
                    last_run_num = pickle.load(f)
                    next_run_num = last_run_num + 1
            except EOFError:
                next_run_num = 0
        else:
            next_run_num = 0

        with open(run_num_file, 'wb') as f:
            pickle.dump(next_run_num, f)
        
        timestamp = time.strftime('%b%d_%H-%M-%S')
        config.run_dir = f'{default_run_dir}/{next_run_num:0>4d}-{timestamp}'
    
    suffix = config.run_suffix
    config.run_dir = config.run_dir + f'{suffix}'
    print("Run directory: ", config.run_dir)
    path = pathlib.Path(config.run_dir)
    path.mkdir(exist_ok=True, parents=True)
    ckpt_path = path / 'ckpt'
    ckpt_path.mkdir(exist_ok=True)
    output_path = path / 'output'
    output_path.mkdir(exist_ok=True)


# Setup run dir from param and return the folder name
def setup_run_dir1(run_dir, default_run_dir='runs', run_suffix=''):
    class Config:
        def __init__(self):
            self.run_dir = run_dir
            self.run_suffix = run_suffix

    config = Config()
    setup_run_dir(config, default_run_dir)
    return config.run_dir
