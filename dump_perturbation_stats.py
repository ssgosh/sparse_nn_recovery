import pickle


class Stats:
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.stats = pickle.load(f)
            self.attack_probs = pickle.load(f)

    def dump_key(self, key):
        for d1 in self.stats:
            print(f'class of attack image : {d1}, prob = {self.attack_probs[d1]:.4f}')
            print(f'{key} for different ground truth classes and various alpha')
            print('alpha\t', '\t'.join([f'{d2}' for d2 in self.stats[d1]]))
            # print('||')
            alphas = list(self.stats[d1][0].keys())
            for alpha in alphas:
                vals = [f"{self.stats[d1][d2][alpha][key]:.2f}" for d2 in self.stats[d1]]
                vals.insert(0, str(alpha))
                print('\t'.join(vals))

    def dump_all(self):
        for key in self.stats[0][0][0]:
            if key != 'num':
                self.dump_key(key)


s = Stats('sparse_attack/sparse_attack_stats_n_100_num_100.p')
s.dump_all()
