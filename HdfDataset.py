import h5py, re
import numpy as np

sortKey = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', str(s))]

class Real:
    def __init__(self, X, channels):
        self.X = X
        self.channels = channels

class Categorical(Real):
    def __init__(self, X, channels=None):
        if channels is None:
            channels = list(set(X))
            channels.sort(key=sortKey)
            channels = np.array(channels, dtype=object)
            X_cat = np.zeros(len(X), dtype=int)
            for ci, ch in enumerate(channels):
                X_cat[X==ch] = ci
            X = X_cat
        super(Categorical, self).__init__(X, channels)

    def captioned(self, suffix=None):
        if type(self.channels) is list:
            v = []
            for i in self.X:
                val = self.channels[int(i)]
                if type(val) is h5py.Group:
                    val = str(val.name)
                v.append(val)
        else:
            v = self.channels[self.X]
        if suffix is not None:
            v = [x + suffix for x in v]
        return np.array(v)

class Sample:
    def __init__(self, name=None):
        self.name = name
        self.series = {}
        self.nodes = None
        self.parent = None
        self.unnormed = {}

    def __getitem__(self, item):
        return self.series[item]

    def __len__(self):
        if len(self.series) == 0:
            return 0
        else:
            return len(list(self.series.values())[0].X)

    def __repr__(self):
        s = str(self.name)
        if len(self.series):
            group = list(self.series.values())[0]
            s += ': {} events, {} channels'.format(len(group.X), len(group.channels))
        if self.nodes is not None:
            s += ', {} nodes'.format(len(self.nodes.channels))
        return s

    def add(self, series, name, cat=True):
        if name not in self.series:
            self.series[name] = series
        else:
            if cat == False:
                print('Series {} already exists.'.format(name))
            else:
                if self[name].channels != series.channels:
                    print('Incompatible channels:')
                    print(self[name].channels)
                    print(series.channels)
                else:
                    X = self[name].X
                    X = np.concatenate((X, series.X))
                    self[name].X = X

    def addNodeMapping(self, name, mapping:dict, unkToken='-'):
        node_comps = [n.name.split('/')[1:] for n in self.nodes.channels]
        vals = np.array(np.repeat(unkToken, len(self.nodes.X)), dtype=object)
        for ni, comps in enumerate(node_comps):
            for c in comps:
                if c in mapping:
                    node_mask = self.nodes.X == int(ni)
                    vals[node_mask] = mapping[c]
                    break
        self.series[name] = Categorical(vals)
        return self.series[name]

    def getAverage(self, grouping:Categorical, averageOf, exclude:dict=None, channels:list=None) -> {object: dict}:
        mask = np.ones(len(grouping.X), dtype=bool)
        if exclude is not None:
            for ex_ser, ex_types in exclude.items():
                s_v = self[ex_ser].captioned()
                for ex_t in ex_types:
                    mask[s_v == ex_t] = 0
        avgs = {}
        if channels is None:
            channels = averageOf.channels
        for ci, cl in enumerate(grouping.channels):
            mask = grouping.X == ci
            vals = averageOf.X[mask]
            if len(vals):
                if type(averageOf) is Categorical:
                    cts = np.zeros(len(channels), dtype=int)
                    for ch_i, ch in enumerate(channels):
                        if ch in averageOf.channels:
                            ci = list(averageOf.channels).index(ch)
                            cts[ch_i] = np.sum(vals == ci)
                    props = cts / np.sum(cts)
                    avgs[cl] = {channels[ch_i]: props[ch_i] for ch_i in range(len(channels))}
                elif type(averageOf) is Real:
                    median = np.median(vals, axis=0)
                    avgs[cl] = {channels[ch_i]: median[ch_i] for ch_i in range(len(channels))}
        return avgs

    def nodeAvgList(self, series:str= 'Z', exclude:dict=None):
        d_avgs = self.getAverage(self.nodes, averageOf=self[series], exclude=exclude)
        avgs = []
        for ni, node in enumerate(self.nodes.channels):
            if node in d_avgs:
                avgs.append([node, d_avgs[node]])
        return avgs


    def standardize(self, what, by, stats=None, backup=True):
        if stats is None:
            stats = {}
        if what not in self.unnormed and backup:
            self.unnormed[what] = np.copy(self.series[what].X)
        if what in self.unnormed:
            X = np.copy(self.unnormed[what])
        else:
            X = self.series[what].X
        B = self.series[by].captioned()
        for b in set(B):
            mask = B == b
            if b in stats:
                mean = stats[b]['mean']
                sd = stats[b]['sd']
            else:
                stats[b] = {}
                stats[b]['mean'] = mean = np.mean(X[mask], axis=0)
                stats[b]['sd'] = sd = np.std(X[mask], axis=0)
            X[mask] -= mean
            X[mask] /= sd
        self.series[what].X = X
        return stats

    def restore(self):
        for series in self.unnormed:
            self[series].X = np.copy(self.unnormed[series])


class HdfDataset:
    def __init__(self, path, mode='r'):
        self.hdf = h5py.File(path, mode)
        self.channelAttr = 'captions'
        self.renamedChs = {}
        self.hiddenNodes = []

    def getPath(self, path:list):
        leafs = [self.hdf]
        for level, keys in enumerate(path):
            new_leafs = []
            for d in leafs:
                for k in d:
                    if keys is None or (str(k) in keys):
                        new_leafs.append(d[k])
            leafs = new_leafs
        return leafs

    def datasetChannels(self, ds):
        if ds not in self.renamedChs:
            return [c if type(c) is str else c.decode('UTF-8') for c in ds.attrs[self.channelAttr]]
        else:
            return self.renamedChs[ds]

    def renameChannels(self, path, dataset, rename:dict):
        leafs = self.getPath(path)
        for l in leafs:
            if dataset in l:
                chs = self.datasetChannels(l[dataset])
                self.renamedChs[l[dataset]] = [c if c not in rename else rename[c] for c in chs]

    def hideNodes(self, names):
        self.hiddenNodes.extend(names)

    def getPanels(self, path:list, groupBy:str, labels:list=[], embeddings:list=[], channelList=None, nullLabel='-'):
        all_nodes = self.getPath(path)
        channel_sets = {}
        for n in all_nodes:
            exclude = False
            for excl in self.hiddenNodes:
                if excl in str(n):
                    exclude = True
                    break
            if groupBy in n and not exclude:
                data = n[groupBy]
                chs = frozenset(self.datasetChannels(data))
                if channelList is not None:
                    chs = frozenset([c for c in chs if c in channelList])
                if len(chs):
                    if chs not in channel_sets:
                        channel_sets[chs] = [n]
                    else:
                        channel_sets[chs].append(n)
        order = np.argsort([len(vs) for vs in channel_sets.values()])[::-1]
        channel_sets = {list(channel_sets.keys())[k]: list(channel_sets.values())[k] for k in order}
        panels = []
        for i, (chs, nodes) in enumerate(channel_sets.items()):
            ord_chs = sorted(list(chs))
            Xacc = np.zeros((0, len(ord_chs)), dtype=float)
            Yacc = {l: np.repeat('', 0) for l in labels}
            Yacc_real = {r: [] for r in embeddings}
            Yacc_real_chs = {}
            nodeAcc = np.zeros((0), dtype=int)
            for ni, node in enumerate(nodes):
                X = node[groupBy]
                chs = self.datasetChannels(X)
                ch_perm = np.array([chs.index(c) for c in ord_chs], dtype=int)
                ord_X = np.array(X)[:, ch_perm]
                Xacc = np.concatenate((Xacc, ord_X), axis=0)
                for l in labels:
                    if l in node:
                        Y = node[l]
                        Ychs = np.array(self.datasetChannels(Y))
                        Y = Ychs[np.array(Y, dtype=int)]
                    else:
                        Y = np.repeat(nullLabel, len(ord_X))
                    Yacc[l] = np.concatenate((Yacc[l], Y))
                for e in embeddings:
                    if e in node:
                        Y = node[e]
                        Yacc_real_chs[e] = np.array(self.datasetChannels(Y))
                    else:
                        Y = np.zeros((len(ord_X), 1))
                    Yacc_real[e].append(Y)
                nodeAcc = np.concatenate((nodeAcc, np.repeat(ni, len(ord_X))))
            samp = Sample(name=str(i))
            samp.parent = self
            samp.series['X'] = Real(Xacc, ord_chs)
            for l in Yacc:
                samp.series[l] = Categorical(Yacc[l])
            for e in Yacc_real:
                vals = Yacc_real[e]
                max_chs = np.max([v.shape[-1] for v in vals])
                cat = np.zeros((0, max_chs), dtype=float)
                for v in vals:
                    if v.shape[-1] != max_chs:
                        cat = np.concatenate((cat, np.zeros((len(v), max_chs))), axis=0)
                    else:
                        cat = np.concatenate((cat, v), axis=0)
                samp.series[e] = Real(cat, Yacc_real_chs[e])
            samp.nodes = Categorical(nodeAcc, channels=nodes)
            panels.append(samp)
        return panels
