import sys
import random
from hashids import Hashids

from utils.project_salt import project_salt


class SeedManager:

    @classmethod
    def get_project_seed_mgr(cls):
        return cls(project_salt)

    def __init__(self, salt):
        self.salt = salt
        self.hashids = Hashids(salt=salt)

    def get_random_seed(self):
        return random.randrange(sys.maxsize)

    def get_random_seed_hashid(self):
        return self.hashids.encode(self.get_random_seed())

    def get_seed_from_hashid(self, id):
        lst = self.hashids.decode(id)
        assert len(lst) == 1
        return lst[0]

