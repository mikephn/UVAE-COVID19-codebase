from src.UVAE import *
from src.UVAE_diag import *
from HdfDataset import *
from src.tools import *
import pandas as pd

outFolder = ensureFolder('lineage/')

# load HDF dataset
ds = HdfDataset('data/lineage.h5')
# path to save normalisation statistics so they can be kept consistent between runs
statsFile = outFolder + 'stats.pkl'
# this script is designed to be run twice:
# test = False will use upsampled data to train the model
# test = True will use the trained model to predict the cell-type proportions without including extra samples

test = True

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

# Get the panels from a given tree path, concatenated by shared channels in X. Also get specified labellings.
pans = ds.getPanels(path, 'X', ['label', 'live', 'doublets'], channelList=allowed)
chs = [p['X'].channels for p in pans] # read channels for reference
allChs = sorted(list(set(np.concatenate(chs))))

ch_file_count = np.zeros(len(allChs), dtype=int)

for p in pans:
    print('{} files: {}'.format(len(p.nodes.channels), p['X'].channels))
    for ch in p['X'].channels:
        ch_file_count[int(allChs.index(ch))] += len(p.nodes.channels)

# Get sample names from HDF nodes.
allSamples = [[str(n).split('/')[2] for n in p.nodes.channels] for p in pans]
allSamplesCat = np.concatenate(allSamples)

# Read clinical variable file and create maps of samples
cv = pd.read_csv('data/meta.csv')
cv_names = list(cv['sample_name'].values)
# batch identifier consists of combined acquisition and processing dates
cv_batch = list(cv['batch'].values)
batchMap = dict(zip(allSamplesCat, [cv_batch[cv_names.index(sn)] for sn in allSamplesCat]))

# Batch standardise the data and save the statistics. In test, the stats from training are loaded and used.
if fileExists(statsFile):
    stats = unpickle(statsFile)
    for pi, p in enumerate(pans):
        p.addNodeMapping('batch', batchMap)
        st = p.standardize('X', 'batch', stats=stats[pi])
else:
    stats = []
    for p in pans:
        p.addNodeMapping('batch', batchMap)
        st = p.standardize('X', 'batch')
        stats.append(st)
    doPickle(stats, statsFile)

# Panels are sorted by size. Only use the largest for initial training.
followingPans = pans[3:]
pans = pans[0:3]
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


sh = uv + Subspace(name='Shared', pull=1.5, masks=uv.data[:len(pans)], conditions=[b_all])
mmd = uv + MMD(Y=p_label.Y, name='MMD', pull=1.5, frequency=3.0)
clsf.resample(mmd)

def umapData(uv):
    dm = uv.allDataMap(subsample=200000)
    return uv.predictMap(dm, mean=True, stacked=True)

# Optimise the base model hyper-parameters:

lisiSet = LisiValidationSet(uv.allDataMap(subsample=100000), labelConstraint=clsf, batchConstraint=b_all,
                                    normClasses=True, perplexity=50, labelRange=(1.0, 2.0), labelWeight=1.0,
                                    batchRange=(10.9, 12.9), batchWeight=1.0)

uv.optimize(iterations=20, maxEpochs=50, samplesPerEpoch=100000, valSamplesPerEpoch=100000,
                lossWeight=1.0,
                subset=['frequency-MMD',
                        'frequency-Shared',
                        'pull-MMD',
                        'pull-Shared',
                        'lr_unsupervised',
                        'lr_merge',
                        'ease_epochs'],
                lisiValidationSet=lisiSet, callback=None)

# Train with the best found configuration
uv.train(50, samplesPerEpoch=100000, valSamplesPerEpoch=0)

# Create visualisations:

# # Visualise the latent space:
# um = cachedUmap(outFolder + 'um.pkl', umapData, uv=uv)
# plotsCallback(uv, True, dataMap=uv.allDataMap(subsample=100000), outFolder=outFolder + 'umap/', um=um)
#
# # Visualise the reconstructed data space:
# dm = uv.allDataMap(subsample=200000)
# emb = uv.predictMap(dm, mean=True, stacked=True)
# ctype_pred = clsf.predictMap(dm, stacked=True, mean=True)
# ctype_lab = ctype_label.predictMap(dm, stacked=True, mean=True)
# panel_lab = p_label.predictMap(dm, stacked=True, mean=True)
# batch_lab = b_all.predictMap(dm, stacked=True)
# file_lab = uv['Files'].predictMap(dm, stacked=True)
#
# gen_pans = uv.data[0:len(pans)]
# allMarkers = sorted(list(set(np.concatenate([d.channels for d in gen_pans]))))
# conds = {uv.autoencoders[d].conditions[0]: uv.autoencoders[d].conditions[0].enum[0] for d in gen_pans}
# rec_org = uv.reconstruct(dm, channels=allMarkers, decoderPanels=gen_pans, conditions=conds, stacked=True)
#
# um_rec = cachedUmap(outFolder + 'um_rec.pkl', lambda: rec_org)
# um_rec_emb = um_rec.transform(rec_org)
#
# um_rec_folder = ensureFolder(outFolder + 'umap_rec/')
# savePlot(um_rec_emb, ctype_pred, path=um_rec_folder + 'ctype.png')
# savePlot(um_rec_emb, ctype_lab, path=um_rec_folder + 'ctype_lab.png')
# savePlot(um_rec_emb, panel_lab, path=um_rec_folder + 'panel.png')
# savePlot(um_rec_emb, batch_lab, path=um_rec_folder + 'batch.png')

# Add consecutive panels on top of the initial model:

for i, P in enumerate(followingPans):
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

## Visualise the combined latent space:
# um = cachedUmap(outFolder + 'um_full.pkl', umapData, uv=uv)
# plotsCallback(uv, True, dataMap=uv.allDataMap(subsample=100000), outFolder=outFolder + 'umap_full/', um=um)

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

# Generate combined embeddings and predictions:
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

if test:
    # Save corrected samples for further analysis
    # Only do it in test runs, since training runs may contain up-sampled cell-types

    saveSample(vals=rec_full, channels=allMarkers,
               ctypes={'label': ctype_lab,
                       'prediction': ctype_pred,
                       'db_label': doublet_lab,
                       'db_pred': doublet_pred},
               files=file_lab,
               path=ensureFolder(outFolder + 'embs/') + 'lineage.pkl')
