from src.UVAE import *
from src.UVAE_diag import *
from HdfDataset import *
from src.tools import *
import pandas as pd
import sys, os

#task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

fold_number = 0#task_id - 1

outFolder = ensureFolder('lineage-split/{}/'.format(fold_number))

# load HDF dataset
ds = HdfDataset('data/lineage.h5')
# path to save normalisation statistics so they can be kept consistent between runs
statsFile = outFolder + 'stats.pkl'
# this script is designed to be run twice:
# test = False will use upsampled data to train the model
# test = True will use the trained model to predict the cell-type proportions without including extra samples

test = False

# Read clinical variable file and create maps of samples
cv = pd.read_csv('data/meta.csv')
cv_names = list(cv['sample_name'].values)
cv_pids = list(cv['patient_id'].values)
# Paths for reading HDF tree, None means read all nodes at a given level.
# Levels are: <workspace>/<filename>/...
# Third level has data stored in 'random' and 'extra'. Extra contains the additional up-sampled data.
if test:
    path = [None, None, 'random'] # only random (proportional) sample
else:
    path = [None, None, None] # both random and extra samples

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
chs = [p['X'].channels for p in allPanels] # read channels for reference
allChs = sorted(list(set(np.concatenate(chs))))

# Panels are sorted by size. Only use the largest for initial training.
n_primary_panels = 3

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

# Batch standardise the data and save the statistics. In test, the stats from training are loaded and used.
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

# Create the model
modelFile = outFolder + 'model_lineage'
uv = UVAE(modelFile + '.uv')
p_nodes = {}
p_filenames = {}
b_labs = []

for i, P in enumerate(pans):
    # Add each panel
    X = P['X'].X
    chs = P['X'].channels
    subsample = np.random.permutation(len(X))[:int(len(X)*subsampleDataBy)]
    X = X[subsample]

    d = uv + Data(X, name='Panel_{}'.format(P.name), channels=chs, valProp=0.3)

    # Add batch labeling
    B = P['batch'].captioned()[subsample]
    b_p = uv + Labeling(Y={d: B}, name='Batches {}'.format(i))
    b_all = uv + Labeling(Y={d: B}, name='Batches all')

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

    norm = uv + Normalization(Y=b_p.Y, name='Latent norm')
    clsf.resample(norm)

    doublets = P['doublets'].captioned()[subsample]
    doublet_clsf = uv + Classification(Y={d: doublets}, name='Doublet clsf', nullLabel='-', trainEmbedding=False)
    doublet_label = uv + Labeling(Y={d: doublets}, name='Doublet label')

    # Create conditional VAE for each panel
    ae = uv + Autoencoder(name=d.name, masks=d, conditions=[b_p])


sh = uv + Subspace(name='Shared', pull=5, frequency=2, masks=uv.data[:len(pans)], conditions=[b_all])
mmd = uv + MMD(Y=p_label.Y, name='MMD', pull=5, frequency=2)
clsf.resample(mmd)

def umapData(uv):
    dm = uv.allDataMap(subsample=200000)
    return uv.predictMap(dm, mean=True, stacked=True)

uv.hyper = {'ease_epochs': 5, 'lr_merge': 2}

uv.train(50, samplesPerEpoch=100000, valSamplesPerEpoch=0)

# Add consecutive panels on top of the initial model:
for _, P in enumerate(followingPans+heldOutPans):
    # Add each panel
    X = P['X'].X
    chs = P['X'].channels
    subsample = np.random.permutation(len(X))[:int(len(X) * subsampleDataBy)]
    X = X[subsample]
    B = P['batch'].captioned()[subsample]
    Y = P['label'].captioned()[subsample]
    doublets = P['doublets'].captioned()[subsample]

    nodes = P.nodes.captioned()[subsample]
    fnames = np.array([fn.split('/')[2] for fn in nodes])  # filename is the second component of node path
    p_filenames[P.name] = set(fnames)

    d = uv + Data(X, name='Panel_{}'.format(P.name), channels=chs, valProp=0.0)

    doublet_label = uv + Labeling(Y={d: doublets}, name='Doublet label')
    f_n_all = uv + Labeling(Y={d: fnames}, name='Files')
    p_label = uv + Labeling(Y={d: np.repeat(P.name, len(X))}, name='Panel label')
    ctype_label = uv + Labeling(Y={d: Y}, name='Cell type label')
    b_p = uv + Labeling(Y={d: B}, name='Batches {}'.format(P.name))

    norm = uv + Normalization(Y=b_p.Y, name='Latent norm')
    clsf.resample(norm)

    ae = uv + Autoencoder(name=d.name, masks=d, conditions=[b_p])

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
    mmd_join = uv + MMD(Y=mmd_mask, name='MMD {}'.format(P.name), pull=mmd.pull, frequency=mmd.frequency)
    clsf.resample(mmd_join)

    uv.train(30, samplesPerEpoch=100000, valSamplesPerEpoch=0)


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

# Reconstruct markers in the style of the top panels
gen_pans = uv.data[0:len(pans)]
allMarkers = sorted(list(set(np.concatenate([d.channels for d in gen_pans]))))
conds = {uv.autoencoders[d].conditions[0]: uv.autoencoders[d].conditions[0].enum[0] for d in gen_pans}
rec_full = uv.reconstruct(dm, channels=allMarkers, decoderPanels=gen_pans, conditions=conds, stacked=False)

if test:
    # Save corrected samples for further analysis
    # Only do it in test runs, since training runs may contain up-sampled cell-types

    saveSample(vals=rec_full, channels=allMarkers,
               ctypes={'label': ctype_lab,
                       'prediction': ctype_pred,
                       'db_label': doublet_lab,
                       'db_pred': doublet_pred},
               files=file_lab,
               path=ensureFolder(outFolder + 'embs-split/') + 'lineage-{}.pkl'.format(fold_number))
