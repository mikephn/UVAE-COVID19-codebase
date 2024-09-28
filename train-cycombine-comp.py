from src.UVAE import *
from src.UVAE_diag import *
import os
import pandas as pd
import pyreadr

sourceDir = 'FlowRepository_FR-FCM-Z52G_files/'

# train on panel 1:
filename = 'cll_p1_ds.RDS'
model_name = 'P1_ds_nct_wbeta_mad3'

# train or panel 2:
# filename = 'cll_p2_ds.RDS'
# model_name = 'P2_ds_nct_wbeta_mad3'

sharedB3 = False # set True to train on shared batch
if sharedB3:
    filename = 'cll_shared_ds.RDS'
    model_name = 'b3_ds_nct_wbeta_mad3'

epochs = 20
cond_batch = True
cond_cond = True
norm_batch = True
norm_cond = False
mmd_batch = True
res_together = True
res_batch = False
n_clusters = 6
large = False
small = True
optimise = True
prob_out = True

model_name += '_e{}'.format(epochs)
if cond_batch:
    model_name += '-cb'
if cond_cond:
    model_name += '-cc'
if norm_batch:
    model_name += '-nb'
if norm_cond:
    model_name += '-nc'
if mmd_batch:
    model_name += '-bmmd'
if res_together:
    model_name += '-res{}'.format(n_clusters)
if res_batch:
    model_name += '-resb{}'.format(n_clusters)
if large:
    model_name += '-large'
if small:
    model_name += '-small'
if optimise:
    model_name += '-opt'

outFolder = ensureFolder('cycombine/{}/'.format(model_name))
resFolder = ensureFolder('cycombine/results/')
clustFolder = ensureFolder('cycombine/clustering/')
plotFolder = ensureFolder('cycombine/plots/')

modelFile = outFolder + model_name
uv = UVAE(modelFile)

if optimise:
    hyperRanges = {'latent_dim': randint(20, 201),
                   'hidden': randint(1, 3),
                   'width': randint(16, 1025),
                   'relu_slope': uniform(0, 0.3),
                   'dropout': uniform(0, 0.3),
                   'pull': uniform(0.1, 10),
                   'cond_dim': randint(10, 21),
                   'cond_hidden': randint(0, 2),
                   'cond_width': randint(16, 513),
                   'lr_unsupervised': uniform(0.5, 3.0),
                   'lr_supervised': uniform(0.5, 3.0),
                   'lr_merge': uniform(0.5, 3.0),
                   'grad_clip': uniform(0.0001, 1.0),
                   'ease_epochs': randint(1, 10),
                   'frequency': uniform(0.1, 3.0),
                   'batch_size': randint(128, 1025),
                   'beta': uniform(0.0, 1.0)
                   }
if large:
    hyperRanges = {'latent_dim': randint(20, 201),
                   'hidden': randint(1, 4),
                   'width': randint(16, 2049),
                   'relu_slope': uniform(0, 0.3),
                   'dropout': uniform(0, 0.3),
                   'pull': uniform(0.1, 10),
                   'cond_dim': randint(10, 21),
                   'cond_hidden': randint(0, 2),
                   'cond_width': randint(16, 513),
                   'lr_unsupervised': uniform(0.5, 3.0),
                   'lr_supervised': uniform(0.5, 3.0),
                   'lr_merge': uniform(0.5, 3.0),
                   'grad_clip': uniform(0.0001, 1.0),
                   'ease_epochs': randint(1, 10),
                   'frequency': uniform(0.1, 3.0),
                   'batch_size': randint(128, 1025),
                   'beta': uniform(0.0, 1.0)
                   }

channel_sets = []
channel_sets.append(['LAG3', 'CD56', 'CD137', 'CD161', 'CD80', 'CD270', 'CD275', 'CD278', 'CD355', 'CD69', 'CD4', 'CD337', 'CD8', 'FoxP3', 'CD20', 'CD3', 'TBet', 'CD45RA', 'CD279', 'CD5', 'CD19', 'CD14', 'GranzymeA', 'FCRL6', 'CD27', 'CD45RO', 'GranzymeK', 'CD152', 'CD33', 'CD197', 'CD134', 'CD127', 'KLRG1', 'CD25', 'HLADR', 'XCL1'])
channel_sets.append(['IL1RA', 'CD74', 'CD56', 'DR3', 'CD161', 'CD34', 'IL23A', 'SMAD2', 'CD123', 'JAK1', 'CD4', 'CD16', 'CD8', 'IFNG', 'FoxP3', 'CD20', 'CD3', 'CD54', 'CD45RA', 'CD1c', 'CD5', 'CD19', 'CD14', 'CD11c', 'HLADR', 'CD1d', 'CD33', 'CD197', 'CD11b', 'CD184', 'TGFBR2', 'FCeR1a', 'TGFB1', 'XCL1'])
channel_sets.append(['CD56', 'CD161', 'CD4', 'CD8', 'FoxP3', 'CD20', 'CD3', 'CD45RA', 'CD5', 'CD19', 'CD14', 'CD33', 'CD197', 'HLADR', 'XCL1'])

meta = pd.read_csv(sourceDir + 'attachments/Metadata.txt', delimiter='\t')

data = pd.DataFrame(pyreadr.read_r(sourceDir+filename)[None])
allSamples = list(set(data['sample']))

batchMap = {}
condMap = {}
for i, fname in enumerate(allSamples):
    fname_split = fname.split('_')
    s_id = '{}_{}'.format(fname_split[1], fname_split[2])
    meta_row = meta[meta['Patient.id'] == s_id]
    batchMap[fname] = str(meta_row['Batch'].values[0])
    condMap[fname] = str(meta_row['Condition'].values[0])

d_chs = list(data.columns)
use_chs = None
for chs in channel_sets:
    if all([ch in d_chs for ch in chs]):
        use_chs = chs
        break

X = np.array(data[use_chs])
if not sharedB3:
    batch = np.array([batchMap[fn] for fn in data['sample']])
else:
    batch = np.array(data['batch'])

cond = np.array([condMap[fn] for fn in data['sample']])
clust = None
if res_together:
    clust = gmmClustering(X, clustFolder+'GMM-{}-{}'.format(filename, n_clusters), B=None, comps=[n_clusters])[0]
elif res_batch:
    clust = gmmClustering(X, clustFolder + 'GMMB-{}-{}'.format(filename, n_clusters), B=batch, comps=[n_clusters])[0]

d = uv + Data(X, name='Data', channels=use_chs, valProp=0.2)
b = uv + Labeling(Y={d: batch}, name='Batch')
c = uv + Labeling(Y={d: cond}, name='Cond')
if clust is not None:
    cl = uv + Labeling(Y={d: clust}, name='Clustering')

if mmd_batch:
    mmd = uv + MMD(Y=b.Y, name='MMD', balanceBatchAvg=res_together)
    if clust is not None:
        cl.resample(mmd)

conds = []
targets = {}
if cond_batch:
    conds.append(b)
    if '5' in batch:
        targets[b] = '5'
    else:
        targets[b] = 'p1b3'
if cond_cond:
    conds.append(c)

ae = uv + Autoencoder(name=d.name, masks=d, conditions=conds)
ae.encoder.weight = 0

if norm_batch:
    norm = uv + Normalization(Y=b.Y, name='Norm', balanceBatchAvg=res_together)
    if '5' in batch:
        targets[norm] = '5'
    else:
        targets[norm] = 'p1b3'
    if clust is not None:
        cl.resample(norm)
if norm_cond:
    norm_c = uv + Normalization(Y=c.Y, name='Norm cond', balanceBatchAvg=res_together)
    if clust is not None:
        cl.resample(norm_c)

def umapData(uv):
    dm = uv.allDataMap(subsample=100000)
    return uv.predictMap(dm, mean=True, stacked=True)

def saveReconstruction(uv, path):
    rec = uv.reconstruct(uv.allDataMap(), conditions=targets, mean=(not prob_out))
    vals = np.array(rec[d], dtype=float)
    rec = pd.DataFrame(vals, columns=use_chs)
    rec['id'] = np.array(data['id'])
    rec['sample'] = np.array(data['sample'])
    rec['batch'] = batch
    pyreadr.write_rds(path, rec)

def emdMadLoss(model):
    rec_opt_file = outFolder + 'rec-opt.RDS'
    saveReconstruction(model, rec_opt_file)
    emd, mad = calculateEmdMad(sourceDir+filename, rec_opt_file, outFolder + 'emdMadOpt.csv')
    return (-1*emd+3*mad)

if optimise:
    uv.optimize(50, maxEpochs=epochs, samplesPerEpoch=100000, valSamplesPerEpoch=100000,
                customLoss=emdMadLoss)

uv.train(epochs, samplesPerEpoch=100000, valSamplesPerEpoch=0)

# Make plots:
# um_base = cachedUmap(outFolder + 'um.pkl', umapData, uv=uv)
# plotsCallback(uv, True, dataMap=uv.allDataMap(subsample=100000), outFolder=outFolder + 'umap/', um=um_base)

if not prob_out:
    rec_file = outFolder + "rec.RDS"
else:
    rec_file = outFolder + "rec-prob.RDS"
    model_name += '-prob'

saveReconstruction(uv, rec_file)
cmd = 'Rscript src/calculateEmdMad.R {} {} {} {}'.format(sourceDir+filename, rec_file,
                                                         resFolder + model_name + '.csv',
                                                         plotFolder + model_name + '.pdf')
print(cmd)
os.system(cmd)