
from .explainer import Explanation

from copy import deepcopy as copy
from textwrap import wrap
from functools import lru_cache
import numbers
from tabulate import tabulate, SEPARATING_LINE
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def get_cmap(kwargs):
    pos_color = np.array(kwargs.get('pos_color', (232, 119, 93)))
    neg_color = np.array(kwargs.get('neg_color', (113, 148, 244)))
    zero_color = np.array(kwargs.get('zero_color', (221, 221, 221)))
    cm = LinearSegmentedColormap.from_list('cm', 
        [(0, neg_color / 255), (0.5, zero_color / 255), (1, pos_color / 255)])
    return cm

class Masker():
    def insert_assigned(self, x, e, **kwargs):
        raise NotImplementedError()

    def delete_assigned(self, x, e, **kwargs):
        raise NotImplementedError()

    def get_num_features(self, x, **kwargs):
        raise NotImplementedError()

class TabularMasker():
    def __init__(self, suppression, dataset=None):
        if dataset is not None:
            dataset = np.array(dataset)
        replace_vals = []
        for i, s in enumerate(suppression):
            assert s in ['cat', 'mean', 'median'] or isinstance(s, numbers.Number)
            if s == 'mean':
                replace_vals.append(np.mean(dataset[:, i]))
            elif s == 'median':
                replace_vals.append(np.median(dataset[:, i]))
            else:
                replace_vals.append(s)
        self.suppression = suppression
        self.replace_vals = replace_vals

    def erase(self, x, idxs):
        x = list(x)
        for idx in idxs:
            if self.replace_vals[idx] == 'cat':
                x[idx] = None
            else:
                x[idx] = self.replace_vals[idx]
        return x

    def insert_assigned(self, x, e):
        return self.erase(x, e.unused)

    def delete_assigned(self, x, e):
        return self.erase(x, e.fea2imp.keys())

    def get_num_features(self, x):
        return len(x)

    def render_result(self, x, exp, mode='text', execute=True, flip=False, **kwargs):
        assert mode in ['text', 'plot']
        exp, label, func_val = np.array(exp['exp']), exp['label'], exp['func_val']
        if flip:
            exp = - exp
        if mode == 'text':
            self.render_text(x, exp, label, func_val, execute, **kwargs)
        else:
            self.render_plot(x, exp, label, func_val, execute, **kwargs)

    def render_text(self, x, exp, label, func_val, execute, **kwargs):
        col_names = kwargs.get('col_names', range(len(x)))
        category_mappings = kwargs.get('category_mappings', None)
        msg = (f'Explained label: {label}\n'
               f'Function value for label {label}: {func_val:0.3f}\n'
               f'Feature attribution:\n')
        data = [['Feature', 'Value', 'Attr val']]
        if category_mappings is not None:
            for w, e, c, m in zip(x, exp, col_names, category_mappings):
                data.append([f'{c}', f'{w if m is None else m[int(w)]}', f'{e}'])
        else:
            for w, e, c in zip(x, exp, col_names):
                data.append([f'{c}', f'{w}', f'{e}'])
        msg += tabulate(data, headers='firstrow', tablefmt='psql')
        if execute:
            print(msg)
        else:
            return msg

    def render_plot(self, x, exp, label, func_val, execute, **kwargs):
        if 'fig' not in kwargs:
            figsize = kwargs.get('figsize', (7, 4.5))
            fig = plt.figure(figsize=figsize)
        else:
            fig = kwargs['fig']
            plt.figure(fig.number)
        L = len(exp)
        plt.barh(range(L)[::-1], exp)
        col_names = kwargs.get('col_names', range(len(x)))
        category_mappings = kwargs.get('category_mappings', None)
        labels = []
        if category_mappings is not None:
            for w, c, m in zip(x, col_names, category_mappings):
                labels.append(f'{c} = {w if m is None else m[int(w)]}')
        else:
            for w, c in zip(x, exp, col_names):
                labels.append(f'Feature {c} = {w}')
        plt.yticks(range(L)[::-1], labels)
        plt.title(f'Explained label: {label}. Function value: {func_val:0.3f}')
        plt.tight_layout()
        if execute:
            plt.show()
        else:
            return fig

class TextWordMasker():
    def __init__(self, suppression):
        assert suppression == 'remove' or suppression.startswith('replace-')
        if suppression.startswith('replace-'):
            self.token = suppression[8:]
            suppression = 'replace'
        self.suppression = suppression

    def insert_assigned(self, x, e):
        if self.suppression == 'remove':
            new_x = [w for i, w in enumerate(x) if i in e.fea2imp]
        else:
            new_x = [w if i in e.fea2imp else self.token for (i, w) in enumerate(x)]
        return new_x

    def delete_assigned(self, x, e):
        if self.suppression == 'remove':
            new_x = [w for i, w in enumerate(x) if i not in e.fea2imp]
        else:
            new_x = [w if i not in e.fea2imp else self.token for (i, w) in enumerate(x)]
        return new_x

    def get_num_features(self, x):
        assert not isinstance(x, str)
        return len(x)

    def render_result(self, x, exp, mode='text', execute=True, flip=False, **kwargs):
        assert mode in ['text', 'color', 'plot']
        exp, label, func_val = np.array(exp['exp']), exp['label'], exp['func_val']
        if flip:
            exp = - exp
        if mode == 'text':
            self.render_text(x, exp, label, func_val, execute, **kwargs)
        elif mode == 'color':
            self.render_color(x, exp, label, func_val, execute, **kwargs)
        else:
            self.render_plot(x, exp, label, func_val, execute, **kwargs)

    def render_text(self, x, exp, label, func_val, execute, **kwargs):
        max_len = max([len(w) for w in x])
        msg = (f'Input: {" ".join(x)}\n'
               f'Explained label: {label}\n'
               f'Function value for label {label}: {func_val:0.3f}\n'
               f'Word feature attribution:\n')
        data = [['Word', 'Attr val']]
        for w, e in zip(x, exp):
            data.append([f'{w}', f'{e}'])
        msg += tabulate(data, headers='firstrow', tablefmt='psql')
        if execute:
            print(msg)
        else:
            return msg

    def render_color(self, x, exp, label, func_val, execute, **kwargs):
        cm = get_cmap(kwargs)
        exp = exp / max(abs(exp)) / 2 + 0.5
        html = (f'Explained label: {label}<br>'
                f'Function value for label {label}: {func_val:0.3f}<br>'
                f'<pre style="white-space: pre-wrap; margin: 0px;">')
        for w, e in zip(x, exp):
            color = (np.array(cm(e)[:3]) * 255).astype('uint8')
            html += (f'<span style="background-color: '
                     f'rgb({color[0]}, {color[1]}, {color[2]});">'
                     f'{w}</span> ')
        html += '</pre>'
        if execute:
            with open('explanation.html', 'w') as f:
                f.write(html + '\n')
        else:
            return html

    def render_plot(self, x, exp, label, func_val, execute, **kwargs):
        if 'fig' not in kwargs:
            figsize = kwargs.get('figsize', (7, 3.5))
            fig = plt.figure(figsize=figsize)
        else:
            fig = kwargs['fig']
            plt.figure(fig.number)
        L = len(exp)
        plt.bar(range(L), exp)
        plt.xticks(range(L), x, rotation=90)
        plt.ylabel('Feature Attribution Value')
        plt.title(f'Explained label: {label}. Function value: {func_val:0.3f}')
        plt.tight_layout()
        if execute:
            plt.show()
        else:
            return fig

class TextSentenceMasker():
    def __init__(self):
        import spacy
        spacy.prefer_gpu()
        try:
            self.nlp = spacy.load('en_core_web_trf')
        except OSError:
            print('Downloading spaCy language model...')
            from spacy.cli import download
            download('en_core_web_trf')
            self.nlp = spacy.load('en_core_web_trf')

    @lru_cache(maxsize=1000)
    def sentencize(self, x):
        assert isinstance(x, str)
        doc = self.nlp(x)
        return [str(sent) for sent in doc.sents]

    def insert_assigned(self, x, e):
        sents = self.sentencize(x)
        new_x = [w for i, w in enumerate(sents) if i in e.fea2imp]
        return ' '.join(new_x)

    def delete_assigned(self, x, e):
        sents = self.sentencize(x)
        new_x = [w for i, w in enumerate(sents) if i in e.unused]
        return ' '.join(new_x)

    def get_num_features(self, x):
        return len(self.sentencize(x))

    def render_result(self, x, exp, mode='text', execute=True, flip=False, **kwargs):
        assert mode in ['text', 'color', 'plot']
        exp, label, func_val = np.array(exp['exp']), exp['label'], exp['func_val']
        if flip:
            exp = - exp
        if mode == 'text':
            self.render_text(x, exp, label, func_val, execute, **kwargs)
        elif mode == 'color':
            self.render_color(x, exp, label, func_val, execute, **kwargs)
        else:
            self.render_plot(x, exp, label, func_val, execute, **kwargs)

    def render_text(self, x, exp, label, func_val, execute, **kwargs):
        max_len = max([len(w) for w in x])
        msg = (f'Explained label: {label}\n'
               f'Function value for label {label}: {func_val:0.3f}\n'
               f'Sentence feature attribution:\n')
        data = [['Sentence', 'Attr val']]
        for s, e in zip(self.sentencize(x), exp):
            data.append([f'{s}', f'{e}'])
            data.append(SEPARATING_LINE)
        del data[-1]
        msg += tabulate(data, headers='firstrow', tablefmt='psql', maxcolwidths=[60, None])
        if execute:
            print(msg)
        else:
            return msg

    def render_color(self, x, exp, label, func_val, execute, **kwargs):
        cm = get_cmap(kwargs)
        exp = exp / max(abs(exp)) / 2 + 0.5
        html = (f'Explained label: {label}<br>'
                f'Function value for label {label}: {func_val:0.3f}<br>'
                f'<pre style="white-space: pre-wrap;">')
        for s, e in zip(self.sentencize(x), exp):
            color = (np.array(cm(e)[:3]) * 255).astype('uint8')
            html += (f'<span style="background-color: '
                     f'rgb({color[0]}, {color[1]}, {color[2]});">'
                     f'{s}</span> ')
        html += '</pre>'
        if execute:
            with open('explanation.html', 'w') as f:
                f.write(html + '\n')
        else:
            return html

    def render_plot(self, x, exp, label, func_val, execute, **kwargs):
        if 'fig' not in kwargs:
            figsize = kwargs.get('figsize', (7, 4))
            fig = plt.figure(figsize=figsize)
        else:
            fig = kwargs['fig']
            plt.figure(fig.number)
        L = len(exp)
        plt.barh(range(L)[::-1], exp)
        plt.yticks(range(L)[::-1], ['\n'.join(wrap(s, width=50))
            for s in self.sentencize(x)], fontsize=8)
        plt.title(f'Explained label: {label}. Function value: {func_val:0.3f}')
        plt.tight_layout()
        if execute:
            plt.show()
        else:
            return fig

class ImageMasker():
    def __init__(self, fill_value='global_mean'):
        if fill_value is None:
            fill_value = 'global_mean'
        if isinstance(fill_value, numbers.Number):
            fill_value = [fill_value, fill_value, fill_value]
        assert fill_value in ['global_mean', 'local_mean'] or \
            len(fill_value) == 3
        self.fill_value = fill_value

    def insert_assigned(self, x, e, **kwargs):
        return self.erase(x, e, 'absent', **kwargs)

    def delete_assigned(self, x, e, **kwargs):
        return self.erase(x, e, 'present', **kwargs)

    def render_result(self, x, exp, mode='color', execute=True, flip=False, **kwargs):
        assert mode in ['text', 'color']
        exp, label, func_val = np.array(exp['exp']), exp['label'], exp['func_val']
        if flip:
            exp = - exp
        if mode == 'text':
            self.render_text(x, exp, label, func_val, execute, **kwargs)
        else:
            self.render_color(x, exp, label, func_val, execute, **kwargs)

class ImageGridMasker(ImageMasker):
    def __init__(self, resolution, fill_value=None):
        super().__init__(fill_value)
        if isinstance(resolution, numbers.Number):
            resolution = [resolution, resolution]
        assert len(resolution) == 2
        self.resolution = resolution

    def get_regions(self, h, w, e, which):
        assert which in ['present', 'absent']
        hs = np.linspace(0, h, self.resolution[0] + 1).astype('int32')
        ws = np.linspace(0, w, self.resolution[1] + 1).astype('int32')
        regions = []
        if which == 'present':
            fea_idxs = e.fea2imp.keys()
        else:
            fea_idxs = e.unused
        for fea_idx in fea_idxs:
            h_idx = fea_idx // self.resolution[1]
            w_idx = fea_idx % self.resolution[1]
            regions.append((hs[h_idx], hs[h_idx + 1], ws[w_idx], ws[w_idx + 1]))
        return regions

    def erase(self, x, e, which):
        assert isinstance(x, Image.Image)
        x = np.asarray(x.convert('RGB'))
        assert x.shape[2] == 3
        h, w, _ = x.shape
        regions = self.get_regions(h, w, e, which)
        if self.fill_value == 'local_mean':
            for h1, h2, w1, w2 in regions:
                val = x[h1:h2, w1:w2].mean(axis=(0, 1))
                x[h1:h2, w1:w2] = val
        else:
            if self.fill_value == 'global_mean':
                val = x.mean(axis=(0, 1))
            else:
                val = self.fill_value
            for h1, h2, w1, w2 in regions:
                x[h1:h2, w1:w2] = val
        return Image.fromarray(x)

    def get_num_features(self, x):
        return self.resolution[0] * self.resolution[1]

    def get_saliency_map(self, h, w, exp):
        regions = self.get_regions(h, w, Explanation.empty(len(exp)), 'absent')
        saliency_map = np.zeros((h, w))
        for i, (h1, h2, w1, w2) in enumerate(regions):
            saliency_map[h1:h2, w1:w2] = exp[i]

    def render_text(self, x, exp, label, func_val, execute, **kwargs):
        msg = (f'Explained label: {label}\n'
               f'Function value for label {label}: {func_val:0.3f}\n'
               f'Grid cell feature attribution:\n')
        res = self.resolution[1]
        data = [['Cell idx', 'Row idx', 'Col idx', 'Attr val']]
        for i, e in enumerate(exp):
            data.append([f'Cell {i}', f'row {i // res}', f'col {i % res}', f'{e}'])
        msg += tabulate(data, headers='firstrow', tablefmt='psql')
        if execute:
            print(msg)
        else:
            return msg

    def render_color(self, x, exp, label, func_val, execute, **kwargs):
        w, h = x.size
        if 'fig' not in kwargs:
            figsize = kwargs.get('figsize', (7, 4))
            fig = plt.figure(figsize=figsize)
        else:
            fig = kwargs['fig']
            plt.figure(fig.number)
        lim = abs(exp).max()
        regions = self.get_regions(h, w, Explanation.empty(len(exp)), 'absent')
        saliency_map = np.zeros((h, w))
        for i, (h1, h2, w1, w2) in enumerate(regions):
            saliency_map[h1:h2, w1:w2] = exp[i]
        plt.imshow(x, cmap='gray')
        cm = get_cmap(kwargs)
        plt.imshow(saliency_map, vmin=-lim, vmax=lim, cmap=cm, alpha=0.8)
        plt.title(f'Explained label: {label}. Function value: {func_val:0.3f}')
        plt.axis('off')
        plt.colorbar()
        plt.tight_layout()
        if execute:
            plt.show()
        else:
            return fig

class ImageSegmentationMasker(ImageMasker):
    def __init__(self, fill_value=None):
        super().__init__(fill_value)

    def erase(self, x, e, which, seg_mask):
        assert which in ['present', 'absent']
        assert isinstance(x, Image.Image)
        x = np.asarray(x.convert('RGB'))
        assert x.shape[2] == 3
        h, w, _ = x.shape
        if which == 'present':
            fea_idxs = e.fea2imp.keys()
        else:
            fea_idxs = e.unused
        if self.fill_value == 'local_mean':
            for fea_idx in fea_idxs:
                val = x[seg_mask == fill_value].mean(axis=0)
                x[seg_mask == fill_value] = val
        else:
            if self.fill_value == 'global_mean':
                val = x.mean(axis=(0, 1))
            else:
                val = self.fill_value
            for fea_idx in fea_idxs:
                x[seg_mask == fea_idx] = val
        return Image.fromarray(x)

    def get_num_features(self, x, seg_mask):
        return seg_mask.max() + 1

    def render_text(self, x, exp, label, func_val, execute, **kwargs):
        msg = (f'Explained label: {label}\n'
               f'Function value for label {label}: {func_val:0.3f}\n'
               f'Superpixel feature attribution:\n')
        data = [['Superpixel idx', 'Attr val']]
        for i, e in enumerate(exp):
            data.append([f'Superpixel {i}', f'{e}'])
        msg += tabulate(data, headers='firstrow', tablefmt='psql')
        if execute:
            print(msg)
        else:
            return msg

    def render_color(self, x, exp, label, func_val, execute, **kwargs):
        assert 'seg_mask' in kwargs
        seg_mask = kwargs['seg_mask']
        w, h = x.size
        if 'fig' not in kwargs:
            figsize = kwargs.get('figsize', (7, 4))
            fig = plt.figure(figsize=figsize)
        else:
            fig = kwargs['fig']
            plt.figure(fig.number)
        lim = abs(exp).max()
        saliency_map = np.zeros((h, w))
        for i, e in enumerate(exp):
            saliency_map[seg_mask == i] = e
        plt.imshow(x, cmap='gray')
        cm = get_cmap(kwargs)
        plt.imshow(saliency_map, vmin=-lim, vmax=lim, cmap=cm, alpha=0.8)
        plt.title(f'Explained label: {label}. Function value: {func_val:0.3f}')
        plt.axis('off')
        plt.colorbar()
        plt.tight_layout()
        if execute:
            plt.show()
        else:
            return fig
