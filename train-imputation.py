from src.UVAE import *
from src.UVAE_diag import *
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description="UVAE Training")
parser.add_argument("--source_dir", type=str, default="DFCI/cll_p1_impute/", help="Path to the data directory")
parser.add_argument("--out_dir", type=str, default="imputation_results/p1/", help="Path to the output directory")
parser.add_argument("--m_name", type=str, default="uvae", help="Model name")
# add parser params:
parser.add_argument("--cond_batch", action='store_true', help="Use batch conditioning")
parser.add_argument("--norm_batch", action='store_true', help="Use batch normalization")
parser.add_argument("--shared_all", action='store_true', help="Use shared all")
parser.add_argument("--shared_split", action='store_true', help="Use shared split")
parser.add_argument("--mmd_batch", action='store_true', help="Use MMD batch")
parser.add_argument("--mmd_panel", action='store_true', help="Use MMD panel")
parser.add_argument("--resample", action='store_true', help="Use resampling")
parser.add_argument("--gmm_clusters", type=int, default=6, help="Number of GMM clusters")
args = parser.parse_args()

ensureFolder(args.out_dir)
modelsFolder = ensureFolder(args.out_dir + '/models/')
recFolder = ensureFolder(args.out_dir + '/rec/' + f'{args.m_name}/')
recTargetedFolder = ensureFolder(args.out_dir + '/rec_targeted/' + f'{args.m_name}/')

uv = UVAE(modelsFolder + args.m_name)

Xs = []
channels = []
Bs = []

for n in range(3):
    data = pd.read_csv(args.source_dir + f'panel{n+1}_subset.csv')
    d_chs = list(data.columns)
    use_chs = [c for c in d_chs if c not in ['id', 'sample', 'batch', 'condition']]
    X = np.array(data[use_chs])
    batch = np.array(data['batch'], dtype=str)
    Xs.append(X)
    channels.append(use_chs)
    Bs.append(batch)
    print("Batches in panel {}: {}".format(n+1, sorted(list(set(batch)))))

all_chs = list(set(channels[0] + channels[1] + channels[2]))
missing_p1 = [ch for ch in all_chs if ch not in channels[0]]
missing_p2 = [ch for ch in all_chs if ch not in channels[1]]
missing_p3 = [ch for ch in all_chs if ch not in channels[2]]

def addPanel(n):
    d = uv + Data(Xs[n], name='Panel {}'.format(n+1), channels=channels[n], valProp=0.2)
    b = uv + Labeling(Y={d: Bs[n]}, name='Batch {}'.format(n+1))
    if args.cond_batch:
        conds = [b]
    else:
        conds = None

    ae = uv + Autoencoder(name=d.name, masks=d, conditions=conds)

    return (d, b, ae)

d1, b1, ae1 = addPanel(0)
d2, b2, ae2 = addPanel(1)
d3, b3, ae3 = addPanel(2)
datas = [d1, d2, d3]
batch_assg = [b1, b2, b3]

if args.resample:
    Xcat = np.concatenate(Xs, axis=0)
    clust_all = gmmClustering(Xcat, args.out_dir+'/GMM{}'.format(args.gmm_clusters), B=None, comps=[args.gmm_clusters])[0]
    clustering = uv + Labeling(Y=unstack(clust_all, d={d: d.X for d in datas}), name='Clustering')

pan_l = uv + Labeling(Y={datas[n]: np.repeat(n, len(datas[n].X)) for n in range(3)}, name='Panel')
batch = uv + Labeling(Y={d: Bs[n] for n, d in enumerate(datas)}, name='All batches')
if args.cond_batch:
    conds = [batch]
else:
    conds = None
if args.shared_split:
    sub12 = uv + Subspace(conditions=conds, name='Subspace12', pull=1, frequency=1, masks=[d1, d2])
    sub13 = uv + Subspace(conditions=conds, name='Subspace13', pull=1, frequency=1, masks=[d1, d3])
    sub23 = uv + Subspace(conditions=conds, name='Subspace23', pull=1, frequency=1, masks=[d2, d3])
if args.shared_all:
    sub = uv + Subspace(conditions=conds, name='Subspace', pull=1, frequency=1)
if args.mmd_batch:
    mmd = uv + MMD(Y=batch.Y, name='MMD batch', balanceBatchAvg=True, frequency=1, pull=1)
    if args.resample:
        clustering.resample(mmd)
if args.mmd_panel:
    mmd_p = uv + MMD(Y=pan_l.Y, name='MMD panel', balanceBatchAvg=True, frequency=1, pull=1)
    if args.resample:
        clustering.resample(mmd_p)
if args.norm_batch:
    norm = uv + Normalization(Y=batch.Y, name='Norm', balanceBatchAvg=True)
    if args.resample:
        clustering.resample(norm)

uv.train(100, samplesPerEpoch=20000, valSamplesPerEpoch=20000, earlyStopEpochs=5, saveBest=True)

use_mean = False
targets = [b1, b2, b3, batch]
if args.norm_batch:
    targets.append(norm)

# reconstruct averaging all common batches:
rec_p1_avg = np.zeros((len(d1.X), len(missing_p1)))
rec_p2_avg = np.zeros((len(d2.X), len(missing_p2)))
rec_p3_avg = np.zeros((len(d3.X), len(missing_p3)))

common_batches = list(set(b1.enum) & set(b2.enum) & set(b3.enum))
# we use common batches here for consistency, to avoid model auto-switching to first available batch if the chosen batch is not present across different panels
for b_id in common_batches:
    target_dict = {t: b_id for t in targets}
    rec_p1_avg += uv.reconstruct({d1: np.arange(len(d1.X))}, conditions=target_dict, mean=use_mean, channels=missing_p1, stacked=True) / len(common_batches)
    rec_p2_avg += uv.reconstruct({d2: np.arange(len(d2.X))}, conditions=target_dict, mean=use_mean, channels=missing_p2, stacked=True) / len(common_batches)
    rec_p3_avg += uv.reconstruct({d3: np.arange(len(d3.X))}, conditions=target_dict, mean=use_mean, channels=missing_p3, stacked=True) / len(common_batches)

# Save reconstructed outputs to TSV files
pd.DataFrame(rec_p1_avg, columns=missing_p1).to_csv(recFolder + "p1.tsv", sep='\t', index=False)
pd.DataFrame(rec_p2_avg, columns=missing_p2).to_csv(recFolder + "p2.tsv", sep='\t', index=False)
pd.DataFrame(rec_p3_avg, columns=missing_p3).to_csv(recFolder + "p3.tsv", sep='\t', index=False)

print(f"Reconstructed averaged outputs saved to {recFolder}")

# reconstruct targeting correct batch
rec_p1 = np.zeros((len(d1.X), len(missing_p1)))
rec_p2 = np.zeros((len(d2.X), len(missing_p2)))
rec_p3 = np.zeros((len(d3.X), len(missing_p3)))
recs = [rec_p1, rec_p2, rec_p3]

for b_id in batch.enum: # for each existing batch
    target_dict = {t: b_id for t in targets} # set batch as reconstruction target
    to_impute = [(d1, missing_p1), (d2, missing_p2), (d3, missing_p3)]
    for i, (d, missing_chs) in enumerate(to_impute): # in each panel to impute
        mask = np.arange(len(d.X))[batch.Y[d] == b_id] # find samples of this batch
        if len(mask):
            decoders = [datas[pi] for pi in range(len(datas)) if b_id in batch_assg[pi].enum] # find decoders which contain this batch
            recs[i][mask] = uv.reconstruct({d: mask}, conditions=target_dict, mean=use_mean, channels=missing_chs, decoderPanels=decoders, stacked=True)

# Save reconstructed outputs to TSV files
pd.DataFrame(rec_p1, columns=missing_p1).to_csv(recTargetedFolder + "p1.tsv", sep='\t', index=False)
pd.DataFrame(rec_p2, columns=missing_p2).to_csv(recTargetedFolder + "p2.tsv", sep='\t', index=False)
pd.DataFrame(rec_p3, columns=missing_p3).to_csv(recTargetedFolder + "p3.tsv", sep='\t', index=False)

print(f"Reconstructed outputs saved to {recTargetedFolder}")

# # Import and use the scoring module
from imputation_scoring import score_imputation

ground_truth_files = [
    f"{args.source_dir}/panel1_gt.csv",
    f"{args.source_dir}/panel2_gt.csv",
    f"{args.source_dir}/panel3_gt.csv"
]

missing_channels_list = [missing_p1, missing_p2, missing_p3]

reconstructed_files = [
    f"{recFolder}/p1.tsv",
    f"{recFolder}/p2.tsv",
    f"{recFolder}/p3.tsv"
]

results_file = f"{args.out_dir}/imputation_averaged.tsv"
verbose = True

# Run evaluation
score_imputation(args.m_name, reconstructed_files, ground_truth_files, missing_channels_list, 
                results_file, verbose)


reconstructed_targeted_files = [
    f"{recTargetedFolder}/p1.tsv",
    f"{recTargetedFolder}/p2.tsv",
    f"{recTargetedFolder}/p3.tsv"
]

results_file = f"{args.out_dir}/imputation_targeted.tsv"

# Run evaluation
score_imputation(args.m_name, reconstructed_targeted_files, ground_truth_files, missing_channels_list, 
                results_file, verbose)