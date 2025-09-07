from src.UVAE import *
from src.UVAE_diag import *
import os
import pandas as pd
import pyreadr

isDFCI = True # else is van Gassen
p_n = 'p1'  # 'p1', 'p2', or 'b3'

if isDFCI:
    sourceDir = 'DFCI/'
    if p_n == 'p1':
        filename = 'cll_p1_ds.RDS'
        model_name = 'P1'
        use_chs = ['LAG3', 'CD56', 'CD137', 'CD161', 'CD80', 'CD270', 'CD275', 'CD278', 'CD355', 'CD69', 'CD4', 'CD337', 'CD8', 'FoxP3', 'CD20', 'CD3', 'TBet', 'CD45RA', 'CD279', 'CD5', 'CD19', 'CD14', 'GranzymeA', 'FCRL6', 'CD27', 'CD45RO', 'GranzymeK', 'CD152', 'CD33', 'CD197', 'CD134', 'CD127', 'KLRG1', 'CD25', 'HLADR', 'XCL1']
    elif p_n == 'p2':
        filename = 'cll_p2_ds.RDS'
        model_name = 'P2'
        use_chs = ['IL1RA', 'CD74', 'CD56', 'DR3', 'CD161', 'CD34', 'IL23A', 'SMAD2', 'CD123', 'JAK1', 'CD4', 'CD16', 'CD8', 'IFNG', 'FoxP3', 'CD20', 'CD3', 'CD54', 'CD45RA', 'CD1c', 'CD5', 'CD19', 'CD14', 'CD11c', 'HLADR', 'CD1d', 'CD33', 'CD197', 'CD11b', 'CD184', 'TGFBR2', 'FCeR1a', 'TGFB1', 'XCL1']
    elif p_n == 'b3':
        filename = 'cll_p1p2_b3_ds.RDS'
        model_name = 'B3'
        use_chs = ['CD56', 'CD161', 'CD4', 'CD8', 'FoxP3', 'CD20', 'CD3', 'CD45RA', 'CD5', 'CD19', 'CD14', 'CD33', 'CD197', 'HLADR', 'XCL1']
else:
    sourceDir = 'van Gassen/'
    filename = 'vanGassen_ds.RDS'
    model_name = 'VG'
    use_chs = ["CCR2", "CCR7", "CCR9", "CD3", "CD4", "CD7", "CD8a", "CD11b", "CD11c", "CD14", "CD15", "CD16", "CD25", "CD33", "CD19", "CD45", "CD45RA", "CD56", "CD66", "CD123", "CD161", "CD235abCD61", "HLADR", "TCRgd", "CREB", "CXCR3", "ERK", "FoxP3", "IkB", "MAPKAPK2", "NFkB", "p38",  "S6", "STAT1", "STAT3", "STAT5", "Tbet"]


epochs = 20
cond_batch = True
cond_cond = True
norm_batch = True
norm_cond = False
mmd_batch = True
res_together = True
res_batch = False
n_clusters = 6

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

if optimise:
    model_name += '-opt'

outFolder = ensureFolder('cycombine/{}/'.format(model_name))
resFolder = ensureFolder('cycombine/results/')
clustFolder = ensureFolder('cycombine/clustering/')
plotFolder = ensureFolder('cycombine/plots/')

modelFile = outFolder + model_name
uv = UVAE(modelFile)

if optimise:
    hyperRanges = {
                'pull': uniform(0.1, 10),
                'cond_dim': randint(10, 21),
                'lr_unsupervised': uniform(0.5, 3.0),
                'lr_merge': uniform(0.5, 3.0),
                'frequency': uniform(0.1, 3.0),
                'batch_size': randint(128, 1025),
                'beta': uniform(0.0, 1.0)
                }


data = pd.DataFrame(pyreadr.read_r(sourceDir+filename)[None])
allSamples = list(set(data['sample']))
d_chs = list(data.columns)
X = np.array(data[use_chs])

if isDFCI:
    meta = pd.read_csv(sourceDir + 'Metadata.txt', delimiter='\t')
    batchMap = {}
    condMap = {}
    for i, fname in enumerate(allSamples):
        fname_split = fname.split('_')
        s_id = '{}_{}'.format(fname_split[1], fname_split[2])
        meta_row = meta[meta['Patient.id'] == s_id]
        batchMap[fname] = str(meta_row['Batch'].values[0])
        condMap[fname] = str(meta_row['Condition'].values[0])
    if filename == 'cll_p1p2_b3_ds.RDS':
        batch = np.array(data['batch'])
    else:
        batch = np.array([batchMap[fn] for fn in data['sample']])        
    cond = np.array([condMap[fn] for fn in data['sample']])
else:
    batch = np.array(data['batch'], dtype=str)
    cond = np.array(data['condition'], dtype=str)

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
    if isDFCI:
        if '5' in batch:
            targets[b] = '5'
        else:
            targets[b] = 'p1b3'
    else:
        targets[b] = batch[0]
if cond_cond:
    conds.append(c)

ae = uv + Autoencoder(name=d.name, masks=d, conditions=conds)
ae.encoder.weight = 0

if norm_batch:
    norm = uv + Normalization(Y=b.Y, name='Norm', balanceBatchAvg=res_together)
    if isDFCI:
        if '5' in batch:
            targets[norm] = '5'
        else:
            targets[norm] = 'p1b3'
    else:
        targets[norm] = batch[0]
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
    if cond_cond:
        # set the condition target to preserve original condition
        vals = np.zeros((len(data), len(use_chs)))
        p_conds = c.predictMap(uv.allDataMap(), stacked=True)
        for c_id in c.enum:
            mask = np.arange(len(data))[p_conds == c_id]
            targets[c] = c_id
            vals[mask] = uv.reconstruct({d: mask}, conditions=targets, mean=(not prob_out))[d]
    else:
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
    return {'emd': -emd, 'mad': mad}

if optimise:
    uv.optimize(50, maxEpochs=epochs, samplesPerEpoch=100000, valSamplesPerEpoch=100000,
                customLoss=emdMadLoss)

uv.train(epochs, samplesPerEpoch=100000, valSamplesPerEpoch=0)

# Make plots:
um_base = cachedUmap(outFolder + 'um.pkl', umapData, uv=uv)
plotsCallback(uv, True, dataMap=uv.allDataMap(subsample=100000), outFolder=outFolder + 'umap/', um=um_base)

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