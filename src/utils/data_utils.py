import random
import numpy as np

import torch
from torch.utils.data import DataLoader

class UnseededDataLoader(DataLoader):
    """
    A DataLoader that does not set the random seed.
    """
    def __init__(self, *args, **kwargs):
        super(UnseededDataLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        # Get a new seed at the beginning of each epoch
        seed = np.random.randint(0, 2**16)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        for batch in super(UnseededDataLoader, self).__iter__():
            yield batch

class RNGStateContext:
    """
    A context manager that saves and restores the state of the random number generators.
    """
    def __enter__(self):
        self.np_state = np.random.get_state()
        self.random_state = random.getstate()
        self.torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            self.torch_cuda_state = torch.cuda.random.get_rng_state_all()
        return self

    def __exit__(self, *args):
        np.random.set_state(self.np_state)
        random.setstate(self.random_state)
        torch.random.set_rng_state(self.torch_state)
        if torch.cuda.is_available():
            torch.cuda.random.set_rng_state_all(self.torch_cuda_state)

class SeededDataLoader(DataLoader):
    """
    A deterministic DataLoader that sets the random seed.
    """
    def __init__(self, seed, *args, **kwargs):
        self.seed = seed
        super(SeededDataLoader, self).__init__(*args, **kwargs)

        print(f"✅ Created SeededDataLoader with seed {self.seed}.")

    def __iter__(self):
        with RNGStateContext():
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)

            for batch in super(SeededDataLoader, self).__iter__():
                yield batch