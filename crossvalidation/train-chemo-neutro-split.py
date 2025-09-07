# Script to train UVAE on several chemokine panels normalised by batch
from src.UVAE import *
from src.UVAE_diag import *
from HdfDataset import *
from src.tools import *
import pandas as pd
import sys, os

#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

fold_number = 0#task_id - 1

outFolder = ensureFolder('chemokine-split/{}/'.format(fold_number))

# load HDF dataset
ds = HdfDataset('data/chemo.h5')
# path to save normalisation statistics so they can be kept consistent between runs
statsFile = outFolder + 'stats.pkl'

# Read clinical variable file and create maps of samples
cv = pd.read_csv('data/meta.csv')
cv_names = list(cv['sample_name'].values)
cv_pids = list(cv['patient_id'].values)

# Path for reading HDF tree, None means read all nodes at a given level.
# Levels are: <workspace>/<filename>/...
# Third level has data stored in 'random' and 'extra'. Extra contains the additional up-sampled data.
path = [None, None, 'random'] # only random (proportional) sample

# Mapping of flow marker names to a common space. Allows creating fewer panels by merging synonymous channels.
synonyms = csvFile('data/channels-rename.tsv', delimiter='\t', remNewline=True)
synonyms = {synonyms[i][0]: synonyms[i][1] for i in range(len(synonyms)) if len(synonyms[i][1])}
allowed = list(set(synonyms.values()))

ds.renameChannels(path, 'X', synonyms)

# Determine which files to hold out from initial optimization
fold_split = unpickle('severity-reg/pid_split_4.pkl')
pid_held = fold_split[str(fold_number)]
fnames_held = [cv_names[i] for i in range(len(cv_names)) if cv_pids[i] in pid_held]
print('Held out PIDs: {}'.format(pid_held))

# Get the panels from a given tree path, concatenated by shared channels in X. Also get specified labellings.
allPanels = ds.getPanels(path, 'X', ['label', 'live', 'doublets'], channelList=allowed)
chs = [p['X'].channels for p in allPanels]
allChs = sorted(list(set(np.concatenate(chs))))

# Panels are sorted by size. Only use the largest for initial training.
n_primary_panels = 4

pans = []
heldOutPans = []
for i, p in enumerate(allPanels):
    nodes = p.nodes.captioned()
    fnames = np.array([fn.split('/')[2] for fn in nodes])
    held_mask = np.isin(fnames, fnames_held)
    p_hold = p.extract(held_mask, name=f'{i}_held')
    if len(p['X'].X) > 0:
        pans.append(p)
        print('{} files: {}'.format(len(p.nodes.channels), p['X'].channels))
    if len(p_hold['X'].X) > 0:
        heldOutPans.append(p_hold)
        print('Held out files: {}'.format(len(p_hold.nodes.channels)))

# Get sample names from HDF nodes.
allSamples = [[str(n).split('/')[2] for n in p.nodes.channels] for p in pans]
allSamplesCat = np.concatenate(allSamples)

# batch identifier consists of combined acquisition and processing dates
cv_batch = list(cv['batch'].values)
batchMap = dict(zip(allSamplesCat, [cv_batch[cv_names.index(sn)] for sn in allSamplesCat]))

# Batch standardise the data and save the statistics for consistency.
if fileExists(statsFile):
    stats = unpickle(statsFile)
    for pi, p in enumerate(pans+heldOutPans):
        p.addNodeMapping('batch', batchMap)
        st = p.standardize('X', 'batch', stats=stats[pi])
else:
    stats = []
    for p in pans+heldOutPans:
        p.addNodeMapping('batch', batchMap)
        st = p.standardize('X', 'batch')
        stats.append(st)
    doPickle(stats, statsFile)

followingPans = pans[n_primary_panels:]
pans = pans[:n_primary_panels]  # Use only the largest panels for initial training
subsampleDataBy = 0.1

modelName = 'model_chemo'
uv = UVAE(outFolder + modelName + '.uv')

p_label = {}
file_panel = {}
p_filenames = {}

for i, P in enumerate(pans):
    # Add each panel
    X = P['X'].X
    subsample = np.random.permutation(len(X))[:int(len(X) * subsampleDataBy)]
    X = X[subsample]
    chs = P['X'].channels
    d = uv + Data(X, name='Panel_{}'.format(P.name), channels=chs, valProp=0.3)
    print("Added panel: {} samples: {}".format(P.name, len(X)))
    # Add batch labeling
    B = P['batch'].captioned()[subsample]
    b_p = uv + Labeling(Y={d: B}, name='Batches {}'.format(i))
    b_all = uv + Labeling(Y={d: B}, name='Batches all')

    batch_clust = gmmClustering(X, B=B, path=ensureFolder(outFolder + 'gmm/') + 'GMM-{}.pkl'.format(d.name), comps=[10])[0]
    batch_clust_global_ids = np.array(['{}-{}'.format(d.name, cl) for cl in batch_clust], dtype=object)
    b_clust = uv + Labeling(Y={d: batch_clust_global_ids}, name='Batch clustering')

    # Store sample origin nodes of each cell for further use
    nodes = P.nodes.captioned()[subsample]
    fnames = np.array([fn.split('/')[2] for fn in nodes])  # filename is the second component of node path
    p_filenames[P.name] = set(fnames)
    f_n = uv + Labeling(Y={d: fnames}, name='Files {}'.format(i))
    f_n_all = uv + Labeling(Y={d: fnames}, name='Files')

    p_label = uv + Labeling(Y={d: np.repeat(P.name, len(X))}, name='Panel label')

    # Get cell-type annotation
    Y = P['label'].captioned()[subsample]
    # Check if 'live' annotation exists, if it does then add Debris classification
    L = P['live'].captioned()[subsample]
    if not all(L == '-'):
        Y[(Y == '-') & (L == '-')] = 'Debris'

    clsf = uv + Classification(Y={d: Y}, name='Cell type', nullLabel='-', trainEmbedding=True, equalizeLabels=False)
    ctype_label = uv + Labeling(Y={d: Y}, name='Cell type label')

    norm = uv + Normalization(Y=b_p.Y, name='Latent norm', balanceBatchAvg=False)
    clsf.resample(norm)
    b_clust.resample(norm)

    doublets = P['doublets'].captioned()[subsample]
    doublet_clsf = uv + Classification(Y={d: doublets}, name='Doublet clsf', nullLabel='-', trainEmbedding=False)
    doublet_label = uv + Labeling(Y={d: doublets}, name='Doublet label')

    ae = uv + Autoencoder(name=d.name, masks=d, conditions=[b_p])

sh = uv + Subspace(name='Shared', pull=5, frequency=2, masks=uv.data[:len(pans)], conditions=[b_all])
mmd = uv + MMD(Y=p_label.Y, name='MMD', pull=5, frequency=2, balanceBatchAvg=False)
clsf.resample(mmd)
b_clust.resample(mmd)

def umapData(uv):
    dm = uv.allDataMap(subsample=200000)
    return uv.predictMap(dm, mean=True, stacked=True)

uv.hyper = {'ease_epochs': 5, 'lr_merge': 2}

uv.train(50, samplesPerEpoch=100000, valSamplesPerEpoch=0)

# Add consecutive panels on top of the initial model:
for _, P in enumerate(followingPans+heldOutPans):
    # Add each panel
    X = P['X'].X
    subsample = np.random.permutation(len(X))[:int(len(X) * subsampleDataBy)]
    X = X[subsample]
    chs = P['X'].channels
    B = P['batch'].captioned()[subsample]
    Y = P['label'].captioned()[subsample]
    doublets = P['doublets'].captioned()[subsample]

    nodes = P.nodes.captioned()[subsample]
    fnames = np.array([fn.split('/')[2] for fn in nodes])  # filename is the second component of node path
    p_filenames[P.name] = set(fnames)

    d = uv + Data(X, name='Panel_{}'.format(P.name), channels=chs, valProp=0.0)
    print("Added panel: {} samples: {}".format(P.name, len(X)))

    batch_clust = gmmClustering(X, B=B, path=outFolder + 'gmm/GMM-{}.pkl'.format(d.name), comps=[10])[0]
    batch_clust_global_ids = np.array(['{}-{}'.format(d.name, cl) for cl in batch_clust], dtype=object)
    b_clust = uv + Labeling(Y={d: batch_clust_global_ids}, name='Batch clustering')

    doublet_label = uv + Labeling(Y={d: doublets}, name='Doublet label')
    f_n_all = uv + Labeling(Y={d: fnames}, name='Files')
    p_label = uv + Labeling(Y={d: np.repeat(P.name, len(X))}, name='Panel label')
    ctype_label = uv + Labeling(Y={d: Y}, name='Cell type label')
    b_p = uv + Labeling(Y={d: B}, name='Batches {}'.format(P.name))

    norm = uv + Normalization(Y=b_p.Y, name='Latent norm', balanceBatchAvg=False)
    clsf.resample(norm)
    b_clust.resample(norm)

    # Determine most similar panel to merge shared channels with
    n_shared = []
    for pi in range(len(pans)):
        shared_chs = [ch for ch in chs if ch in uv.data[pi].channels]
        n_shared.append(len(shared_chs))
    most_similar_p_ind = int(np.argmax(n_shared))
    merge_with = uv.data[most_similar_p_ind]
    print('Shared channels: {}, most similar panel: {}'.format(n_shared, merge_with))

    merge_with_P_batches = uv['Batches {}'.format(pans[most_similar_p_ind].name)]
    sub_mask = {merge_with: np.ones(len(merge_with.X)), d: np.ones(len(d.X))}
    b_join = uv + Labeling(Y={d: B, merge_with: merge_with_P_batches.Y[merge_with]}, name='Batches join {}'.format(P.name))
    sub_join = uv + Subspace(masks=sub_mask, name='Subspace_{}'.format(P.name), conditions=[b_join], pull=sh.pull, frequency=sh.frequency)

    mmd_mask = {ex_d: np.zeros(len(ex_d.X)) for ex_d in uv.data[0:len(pans)]}
    mmd_mask[d] = np.ones(len(d.X))
    mmd_join = uv + MMD(Y=mmd_mask, name='MMD {}'.format(P.name), frequency=mmd.frequency, pull=mmd.pull, balanceBatchAvg=False)
    clsf.resample(mmd_join)
    b_clust.resample(mmd_join)

    ae = uv + Autoencoder(name=d.name, masks=d, conditions=[b_p])

    t0 = time.time()
    uv.train(30, samplesPerEpoch=100000, valSamplesPerEpoch=0)
    print("Added panel {}: {:.2f} seconds".format(P.name, time.time() - t0))


# Function to save corrected data samples for use for longitudinal models training:
def saveSample(vals:dict, ctypes:{str:dict}, files:dict, path, channels:[str] = None):
    values_by_file = {}
    classifications_by_file = {ct: {} for ct in ctypes}
    channels_by_file = {}
    for d, vec_files in files.items():
        panel_files = list(set(vec_files))
        for f in panel_files:
            file_mask = vec_files == f
            values_by_file[f] = vals[d][file_mask]
            if channels is None:
                channels_by_file[f] = d.channels
            else:
                channels_by_file[f] = channels
            for ctype in ctypes:
                vec_clsf = ctypes[ctype][d][file_mask]
                classifications_by_file[ctype][f] = vec_clsf

    saveDict = {'vals': values_by_file, 'channels': channels_by_file,
                'ctypes': classifications_by_file,
                'batch': batchMap, 'panels': p_filenames}
    doPickle(saveDict, path)

# Generate combined embeddings and predictions
dm = uv.allDataMap(subsample=500000)
emb = uv.predictMap(dm, mean=True, stacked=False)
ctype_pred = clsf.predictMap(dm, stacked=False, mean=True)
ctype_lab = ctype_label.predictMap(dm, stacked=False)
panel_lab = p_label.predictMap(dm, stacked=False)
batch_lab = b_all.predictMap(dm, stacked=False)
file_lab = uv['Files'].predictMap(dm, stacked=False)
doublet_pred = doublet_clsf.predictMap(dm, stacked=False, mean=True)
doublet_lab = doublet_label.predictMap(dm, stacked=False)

# Reconstruct markers in the style of the top panels for all included panels:
gen_pans = uv.data[0:len(pans)]
allMarkers = sorted(list(set(np.concatenate([d.channels for d in gen_pans]))))
conds = {uv.autoencoders[d].conditions[0]: uv.autoencoders[d].conditions[0].enum[0] for d in gen_pans}
rec_full = uv.reconstruct(dm, channels=allMarkers, decoderPanels=gen_pans, conditions=conds, stacked=False)
# Filter neutrophils
ctype_pred_stacked = stack(ctype_pred)
nodeb_mask = ctype_pred_stacked != 'Debris'
neutro_mask = ctype_pred_stacked == 'Neutrophils'
rec_full_stack = stack(rec_full)
panel_stack = stack(panel_lab)
heldout_mask = np.isin(panel_stack, [p.name for p in heldOutPans])
train_mask = heldout_mask == False
neutro_rec = rec_full_stack[neutro_mask & train_mask]
neutro_held = rec_full_stack[neutro_mask & heldout_mask]

from sklearn.neighbors import KNeighborsClassifier

# Apply GMM clustering to the corrected data:
n_gmm_clusters = 5
neutro_clust_gmm = gmmClustering(neutro_rec, outFolder+'gmm/neutro_gmm.pkl', comps=[n_gmm_clusters])[0]

cluster_assg_gmm = np.array(np.repeat('Debris', len(rec_full_stack)), dtype=object)
cluster_assg_gmm[nodeb_mask] = 'Other'
cluster_assg_gmm[neutro_mask & train_mask] = np.array(neutro_clust_gmm, dtype=str)

# Train KNN to classify based on GMM labels
knn_gmm = KNeighborsClassifier(n_neighbors=15)
print("  Fitting KNN for GMM labels...")
knn_gmm.fit(neutro_rec, neutro_clust_gmm)
# Predict labels on the held-out test data
print("  Predicting GMM labels for test data...")
test_labels_gmm = knn_gmm.predict(neutro_held)
cluster_assg_gmm[neutro_mask & heldout_mask] = np.array(test_labels_gmm, dtype=str)

# Apply Leiden clustering to the corrected data:
neutro_clust_leiden = leidenClustering(neutro_rec)
cluster_assg_leiden = np.array(np.repeat('Debris', len(rec_full_stack)), dtype=object)
cluster_assg_leiden[nodeb_mask] = 'Other'
cluster_assg_leiden[neutro_mask & train_mask] = np.array(neutro_clust_leiden, dtype=str)

# Train KNN to classify based on Leiden labels
knn_leiden = KNeighborsClassifier(n_neighbors=15)
print("  Fitting KNN for Leiden labels...")
knn_leiden.fit(neutro_rec, neutro_clust_leiden)
# Predict labels on the held-out test data
print("  Predicting Leiden labels for test data...")
test_labels_leiden = knn_leiden.predict(neutro_held)
cluster_assg_leiden[neutro_mask & heldout_mask] = np.array(test_labels_leiden, dtype=str)

# Save corrected samples for further analysis
saveSample(vals=rec_full, channels=allMarkers,
               ctypes={'label': ctype_lab,
                       'prediction': ctype_pred,
                       'db_label': doublet_lab,
                       'db_pred': doublet_pred,
                       'gmm': unstack(cluster_assg_gmm, rec_full),
                       'leiden': unstack(cluster_assg_leiden, rec_full)
                       },
               files=file_lab,
               path=ensureFolder(outFolder + 'embs-split/') + 'chemo-{}.pkl'.format(fold_number))

