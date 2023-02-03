
from copy import deepcopy as copy
import numbers
import numpy as np
from tqdm import trange

class Explanation():
    '''a class for order-based explanation'''
    @classmethod
    def empty(cls, L):
        fea2imp = dict()
        unused = set(range(L))
        e = cls(fea2imp, unused, None, 0)
        return e

    def __init__(self, fea2imp, unused, parent, score):
        self.fea2imp = fea2imp
        self.unused = unused
        self.parent = parent
        self.score = score

    def extend(self):
        '''return a list of new explanations'''
        assert len(self.unused) != 0, 'the explanation is already full'
        assert self.score is not None, \
            'the currently expanded explanation has not been scored'
        extensions = []
        cur_imp = len(self.unused) - 1
        for i in self.unused:
            new_fea2imp = copy(self.fea2imp)
            new_fea2imp[i] = cur_imp
            new_unused = copy(self.unused)
            new_unused.remove(i)
            e = Explanation(new_fea2imp, new_unused, self, None)
            extensions.append(e)
        return extensions

    def to_list(self):
        assert len(self.unused) == 0, 'only full explanations can be converted to list'
        L = len(self.fea2imp)
        return [self.fea2imp[i] for i in range(L)]

    def __lt__(self, other):
        assert self.score is not None and other.score is not None, 'invalid comparison'
        return self.score < other.score

class BeamSearchExplainer():
    def __init__(self, masker, f=None, metric='comp-suff', 
                 beam_size=50, batch_size=None):
        self.f = f
        assert metric in ['comp', 'suff', 'comp-suff'], f'unrecognized metric: {metric}'
        self.metric = metric
        self.masker = masker
        self.beam_size = beam_size
        self.batch_size = batch_size

    def explain_instance(self, x, f=None, label=None, logging='progressbar', **kwargs):
        assert (self.f is None) != (f is None), \
            'the function under explanation should be provided exactly once'
        assert logging in ['none', 'progressbar', 'full'], \
            f'unrecognized logging: {logging}'
        if f is None:
            f = self.f
        L = self.masker.get_num_features(x, **kwargs)
        if label is None:
            func_val = f([x])[0]
            label = func_val.argmax()
            func_val = func_val[label]
        else:
            label = int(label)
            func_val = f([x])[0][label]
        beam = [Explanation.empty(L)]
        if logging == 'progressbar':
            iterator = trange(L, ncols=70)
        else:
            iterator = range(L)
        for i in iterator:
            if logging == 'full':
                print(f'Step {i} / {L}: ', end='')
            new_beam = []
            for e in beam:
                new_beam.extend(e.extend())
            self.score(new_beam, x, f, label, func_val, logging=='full', **kwargs)
            new_beam.sort(reverse=True)
            beam = new_beam[:self.beam_size]
        self.shift(beam[0], x, f, label, func_val, **kwargs)
        exp = beam[0].to_list()
        return {'exp': exp, 'label': label, 'func_val': func_val}

    def shift(self, explanation, x, f, label, func_val, **kwargs):
        L = len(explanation.fea2imp)
        single_on_exps = Explanation.empty(L).extend()
        xs_del = [self.masker.delete_assigned(x, e, **kwargs) for e in single_on_exps]
        func_vals = self.run(f, [x] + xs_del, label)
        diff = func_vals[0] - func_vals[1:]
        ref_signs = np.sign(diff)
        exp = np.array(explanation.to_list())
        offsets = (np.arange(0, L + 1) - 0.5)
        offset_exp = exp.reshape(1, -1) - offsets.reshape(-1, 1)
        sign_matrix = np.sign(offset_exp)
        num_agrees = (sign_matrix == ref_signs.reshape(1, -1)).sum(axis=1)
        exp = offset_exp[num_agrees.argmax()]
        for i in range(L):
            explanation.fea2imp[i] = exp[i]

    def score(self, explanations, x, f, label, func_val, logging, **kwargs):
        todo = dict()
        for e in explanations:
            key = tuple(sorted(e.fea2imp.keys()))
            if key not in todo:
                todo[key] = []
            todo[key].append(e)
        if logging:
            print(f'{len(todo)} keys to run for {len(explanations)} explanations')
        if self.metric == 'comp':
            xs_del = [self.masker.delete_assigned(x, v[0], **kwargs) for v in todo.values()]
            func_vals_del = self.run(xs_del)
            scores = func_val - func_vals_del
        elif self.metric == 'suff':
            xs_ins = [self.masker.insert_assigned(x, v[0], **kwargs) for v in todo.values()]
            func_vals_ins = self.run(xs_ins)
            scores = func_vals_ins - func_val
        else:
            xs_del = [self.masker.delete_assigned(x, v[0], **kwargs) for v in todo.values()]
            xs_ins = [self.masker.insert_assigned(x, v[0], **kwargs) for v in todo.values()]
            func_vals = self.run(f, xs_del + xs_ins, label)
            func_vals_del = func_vals[:len(xs_del)]
            func_vals_ins = func_vals[len(xs_ins):]
            scores = func_vals_ins - func_vals_del
        for k, s in zip(todo, scores):
            for e in todo[k]:
                e.score = e.parent.score + s

    def run(self, f, xs, label):
        if self.batch_size is None:
            return f(xs)[:, label]
        else:
            func_vals = []
            num_batches = int(np.ceil(len(xs) / self.batch_size))
            for i in range(num_batches):
                cur_xs = xs[i * self.batch_size : (i + 1) * self.batch_size]
                func_vals.append(f(cur_xs)[:, label])
            return np.concatenate(func_vals)
