from copy import deepcopy


class callback(object):
    def __init__(self, var_names):
        self.store = {v: [] for v in var_names}

    def __call__(self, params):
        for p in self.store.keys():
            if p in params.keys():
                self.store[p].append(deepcopy(params[p]))

    def __getitem__(self, key):
        return self.store[key]

    def get_keys(self):
        return self.store.keys()
