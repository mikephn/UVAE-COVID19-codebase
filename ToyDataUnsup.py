from sklearn.mixture import GaussianMixture
from src.UVAE_diag import savePlot
import umap
import flowkit as fk
from src.tools import *
import copy

# input file
fpath = 'data/WB_main 20200504_YA HC WB_main_018.fcs'

rootFolder = ensureFolder('ToyData/')

outFolder = ensureFolder(rootFolder + 'WB/')
dspath = outFolder + 'ToyDataWB-3x3.pkl'
keepCtypes = ["B cells", "Basophils", "DC cells", "Eosinophils", "Monocytes", "Neutrophils", "NK cells", "T cells", "Doublets"]

n_clusters = 20

class FlowFile():
    def __init__(self, path, labelColumnsAtEnd=1, labtype='merge', logicle=False, compensation=False):
        self.path = path
        self.f = None
        try:
            self.f = fk.Sample(path, ignore_offset_error=True)
        except:
            e = sys.exc_info()[0]
            print('Unable to load file.', e)
        if self.f is not None:
            if compensation:
                self.f.apply_compensation(self.f.compensation)
            if labtype == 'pnn':
                self.markers = self.f.pnn_labels
            elif labtype == 'pns':
                self.markers = self.f.pns_labels
            else:
                self.markers = self.f.pns_labels
                for n in range(len(self.markers)):
                    if len(self.markers[n]) == 0:
                        self.markers[n] = self.f.pnn_labels[n]
            self.nCells = self.f.event_count
            self.means = np.zeros(len(self.markers))
            self.sds = np.ones(len(self.markers))
            if labelColumnsAtEnd > 0:
                self.useMarkers(self.markers[:-labelColumnsAtEnd])
            else:
                self.useMarkers(self.markers)
            self.nLabelCols = labelColumnsAtEnd
            self.logicle = None
            if logicle:
                self.logicle = fk.transforms.LogicleTransform('logicle', 262144, 0.5, 4.5, 0)


    def useMarkers(self, mk_labels):
        self.usedMarkers = mk_labels
        self._useMarkers = np.array([self.markers.index(m) for m in mk_labels], dtype=int)

    def standardise(self, reset=False):
        if reset:
            self.means = np.zeros(len(self.markers))
            self.sds = np.ones(len(self.markers))
        else:
            eve = self.f.get_events(source='raw')
            if self.logicle is not None:
                eve = self.logicle.apply(eve)
            self.means = np.mean(eve, axis=0)
            self.sds = np.std(eve, axis=0)

    def rawEvents(self):
        e = np.copy(self.f.get_events(source='raw'))
        if self.logicle is not None:
            e = self.logicle.apply(e)
        return e

    def sample(self, n=None, replacement=True, label=None, returnLabel=False, labelInd=-1):
        eve = np.copy(self.f.get_events(source='raw'))
        if label is not None:
            eve = eve[eve[:, labelInd] == label, :]
        if len(eve) > 0:
            if n is not None:
                if n <= len(eve):
                    perm = np.random.permutation(len(eve))[:n]
                    eve = eve[perm]
                elif replacement:
                    perm = np.random.randint(0, len(eve), n)
                    eve = eve[perm]
            labs = copy.copy(eve[:, -(self.nLabelCols):])
            if self.logicle is not None:
                eve = self.logicle.apply(eve)
            eve -= self.means
            eve /= self.sds
        else:
            labs = None
        if not returnLabel:
            return eve[:, self._useMarkers]
        else:
            return eve[:, self._useMarkers], labs


if not fileExists(dspath):

    n_panels = 3
    n_batches = 3
    uneven_prop = 0.6
    cap = 100000
    n_anchors = 1000
    nn_per_anchor = 10
    toDropChsPerPanel = 2

    # load file, separate labels from flow markers
    f = FlowFile(fpath, labelColumnsAtEnd=1)
    Xorg, ctypes = f.sample(returnLabel=True)
    ctype_caps = ["unk", "B cells", "Basophils", "DC cells", "Eosinophils", "Monocytes", "Neutrophils", "NK cells", "T cells", "Doublets", "Debris"]
    ctypes = np.array([ctype_caps[int(i)] for i in np.squeeze(ctypes)])
    valid = np.isin(ctypes, keepCtypes)
    valid_inds = np.arange(len(Xorg))[valid]
    subsample_inds = np.random.permutation(valid_inds)[:cap]
    Xorg = Xorg[subsample_inds]
    ctypes = ctypes[subsample_inds]

    Xorg -= np.mean(Xorg, axis=0)
    Xorg /= np.std(Xorg, axis=0)

    channels = f.usedMarkers

    channelDropOrder = [channels.index(ch) for ch in ['CD45', 'CD11c', 'CD16', 'CD11b', 'CD10', 'CD14']]
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
    Y = gmm.fit_predict(Xorg)
    Y_caps = list(set(Y))

    anchor_inds = np.random.permutation(cap)[:n_anchors]
    nn_inds = []
    anchor_vars = []
    for a_i in anchor_inds:
        a_x = Xorg[a_i]
        expand = np.tile(a_x, [cap, 1])
        dist = np.sqrt(np.sum(np.square(expand-Xorg), axis=1))
        ord = np.argsort(dist)
        a_inds = ord[:nn_per_anchor]
        nn_inds.append(a_inds)
        var_mean = np.mean(np.var(Xorg[a_inds], axis=0))
        anchor_vars.append(var_mean)
    print('Average anchor variance:', np.mean(anchor_vars))

    # generate a random disproportional split
    def unevenProportions(n_classes, n_parts, uneven=0.3):
        class_props = []
        for cn in range(n_classes):
            per_part = (1.0 - uneven) / n_parts
            props = np.random.exponential(1, n_parts)
            per_part_extra = uneven * (props / np.sum(props))
            per_part_sum = per_part_extra + per_part
            class_props.append(per_part_sum)
        return np.transpose(class_props)

    # split label vector Y according to specified proportions, return batch designation
    def labelSplit(Y, props):
        cl, ct = np.unique(Y, return_counts=True)
        B = np.zeros(len(Y), dtype=int)
        props = np.transpose(props)
        for ci, c_prop in enumerate(props):
            count = ct[ci]
            per_part = c_prop * count
            c_inds = np.arange(len(Y))[Y == ci]
            c_inds = c_inds[np.random.permutation(len(c_inds))]
            for pi, c_ct in enumerate(per_part):
                start = np.sum(per_part[:pi])
                end = start + per_part[pi]
                B[c_inds[int(start):int(end)]] = pi
        return B

    # define proportions for random panel and batch split
    effs = {}
    effs['p_split'] = unevenProportions(len(Y_caps), n_panels, uneven_prop)
    for pn in range(n_panels):
        p_effs = {}
        p_effs['b_split'] = unevenProportions(len(Y_caps), n_batches, uneven_prop)
        effs[pn] = p_effs

    # add batch effects to the data and save as dataset
    ds = {}
    ds['GT_X'] = []
    ds['GT_markers'] = channels
    ds['X'] = []
    ds['markers'] = []
    ds['batch'] = []
    ds['celltype'] = []
    ds['cluster'] = []

    # split into panels
    panel_split = labelSplit(Y, effs['p_split'])
    dropped = {}
    pan_inds = []
    for pn in range(n_panels):
        p_inds = np.arange(len(Xorg))[panel_split == pn]
        pan_inds.append(p_inds)
        panX_gt = np.copy(Xorg[p_inds])
        panY_gt = np.copy(Y[p_inds])
        panCtype_gt = np.copy(ctypes[p_inds])
        panX = np.copy(panX_gt)
        # split panel into batches
        B = labelSplit(panY_gt, effs[pn]['b_split'])
        b_caps = []
        for bi in B:
            b_caps.append('p{}b{}'.format(int(pn), int(bi)))
        b_caps = np.array(b_caps)
        batches = list(set(b_caps))
        # re-normalise each batch
        for batch in batches:
            b_mask = b_caps == batch
            b_x = np.copy(panX[b_mask])
            panX[b_mask] -= np.mean(b_x, axis=0)
            panX[b_mask] /= np.std(b_x, axis=0)
        panelChannels = channels
        if toDropChsPerPanel > 0:
            # drop channels from the end in each panel
            drop_ch_start = pn*toDropChsPerPanel
            drop_ch_inds = channelDropOrder[drop_ch_start:drop_ch_start+toDropChsPerPanel]
            keep_ch_inds = np.array([n for n in range(len(channels)) if (n not in drop_ch_inds)])
            panX = panX[:, keep_ch_inds]
            panelChannels = [channels[i] for i in keep_ch_inds]
            dropped[pn] = [channels[i] for i in drop_ch_inds]

        ds['GT_X'].append(panX_gt)
        ds['X'].append(panX)
        ds['markers'].append(panelChannels)
        ds['cluster'].append(panY_gt)
        ds['celltype'].append(panCtype_gt)
        ds['batch'].append(b_caps)

    split_anchors = []
    for a_nns in nn_inds:
        p_ref_inds = [[] for p in range(n_panels)]
        for ind in a_nns:
            p_n = panel_split[ind]
            p_ind = np.where(pan_inds[p_n] == ind)
            p_ref_inds[p_n].append(int(p_ind[0]))
        split_anchors.append(p_ref_inds)

    effs['dropped'] = dropped
    ds['effects'] = effs
    ds['anchors'] = split_anchors
    doPickle(ds, dspath)

ds = unpickle(dspath)
X_GT = ds['GT_X']
X_GT_cat = np.vstack(X_GT)
Y_cat = np.concatenate(ds['cluster'])
ctype_cat = np.concatenate(ds['celltype'])
anchors = ds['anchors']

anchor_vars = []
anchor_labs = [np.zeros(len(X_GT[pi]), dtype=int) for pi in range(len(X_GT))]
for ai, anch in enumerate(anchors):
    points = []
    for pi, p_inds in enumerate(anch):
        points.extend(X_GT[pi][p_inds])
        anchor_labs[pi][p_inds] = ai + 1
    var_mean = np.mean(np.var(points, axis=0))
    anchor_vars.append(var_mean)
print('Average anchor variance:', np.mean(anchor_vars))
print(ds['effects']['dropped'])

for pi in range(len(X_GT)):
    um = umap.UMAP()
    um_emb = um.fit_transform(ds['X'][pi])
    savePlot(um_emb, ds['celltype'][pi], outFolder + 'p{}_celltype.png'.format(pi), title='Panel {}, cell type'.format(pi))
    savePlot(um_emb, ds['cluster'][pi], outFolder + 'p{}_cluster.png'.format(pi), title='Panel {}, clustering'.format(pi))
    savePlot(um_emb, ds['batch'][pi], outFolder + 'p{}_batch.png'.format(pi), title='Panel {}, batch'.format(pi))
    savePlot(um_emb, anchor_labs[pi], outFolder + 'p{}_anchors.png'.format(pi), title='Panel {}, anchors'.format(pi), firstBlack=True)