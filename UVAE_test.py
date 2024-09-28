#!/usr/bin/python
import os

os.environ["OMP_NUM_THREADS"] = f"1"
os.environ['TF_NUM_INTEROP_THREADS'] = f"1"
os.environ['TF_NUM_INTRAOP_THREADS'] = f"1"

import faulthandler
faulthandler.enable()

from src.UVAE import *
from src.UVAE_diag import *

if len(sys.argv) > 1:
    ds_name = sys.argv[1]
    config_ind = int(os.getenv('SGE_TASK_ID'))
    config_params = []
    if len(sys.argv) > 2:
        config_params = sys.argv[2:]

    print(ds_name, config_ind)
else:
    ds_name = 'ToyDataWB-3x3'
    config_ind = np.arange(48, 50)
    config_params = []

# use sample data from:
if ds_name == 'ToyDataWB-3x3':
    toyDs = unpickle('data/ToyDataWB-3x3.pkl')
    lisiRanges = {'supervised': {'labelRange': (1.0, 3.2), 'batchRange': (2.45, 7.6)},
                  'unsupervised': {'labelRange': (3.0, 23.3), 'batchRange': (2.2, 7.3)}}
else:
    exit()

outFolder = ensureFolder('test/{}/'.format(ds_name))

# load test data
X0, X1, X2 = toyDs['X']
chs0, chs1, chs2 = toyDs['markers']
Y0, Y1, Y2 = toyDs['celltype']
B0, B1, B2 = toyDs['batch']

ctype_labels = np.concatenate(toyDs['celltype'])
batches = np.concatenate(toyDs['batch'])

# ground truth values
GT0, GT1, GT2 = toyDs['GT_X']
gt_chs = toyDs['GT_markers']
gt_cat = np.vstack(toyDs['GT_X'])
anchors = toyDs['anchors']
clusters_gt = np.concatenate(toyDs['cluster'])

lisiMasksFile = outFolder + 'lisi_masks.pkl'
if fileExists(lisiMasksFile):
    lisiMasks = unpickle(lisiMasksFile)
else:
    mask_ctype = classNormalizationMask(batches=batches, labels=ctype_labels)
    mask_gt = classNormalizationMask(batches=batches, labels=clusters_gt)
    lisiMasks = {'gt': mask_gt, 'ctype': mask_ctype}
    doPickle(lisiMasks, lisiMasksFile)


n_clusters_per_batch = [20, 15, 10]
n_clusters_per_panel = [20]
batch_clusterings = []
batch_clustering_unique = []
panel_clusterings = []
for pi in range(3):
    clusts = gmmClustering(toyDs['X'][pi],
                          path=outFolder+ds_name+'p{}'.format(pi),
                          B=toyDs['batch'][pi],
                          comps=n_clusters_per_batch)
    batch_clusterings.append(clusts)
    batch_clustering_unique.append(clusts[0] + (pi * n_clusters_per_batch[0]))
    p_clusts = gmmClustering(toyDs['X'][pi],
                           path=outFolder + ds_name + 'p{}-nb'.format(pi),
                           comps=n_clusters_per_panel)
    panel_clusterings.append(p_clusts)

# number of repeats and epochs for each configuration
repeats = 10
n_epochs = 50

# test configurations
from UVAE_test_configs import testConfigs, configName

hyperRanges.update({
    'lr_unsupervised': uniform(0.2, 3.0),
    'lr_supervised': uniform(0.2, 3.0),
    'lr_merge': uniform(0.2, 3.0),
    'pull': uniform(1, 20),
    'frequency': uniform(1, 10),
})

if config_ind is not None:
    if type(config_ind) is int:
        testConfigs = [testConfigs[config_ind]]
    else:
        testConfigs = [testConfigs[n] for n in config_ind]

def groundTruthModel(path, verbose=True, rep=0):
    repPath = path + 'r{}'.format(rep)
    uv = UVAE(repPath)
    p = uv + Data(gt_cat, channels=gt_chs, name='All data', valProp=0.0)
    uv.train(n_epochs, samplesPerEpoch=100000, valSamplesPerEpoch=0, callback=None, verbose=verbose,
             skipTrained=True)
    rec = uv.reconstruct(dataMap=uv.allDataMap(), stacked=True)
    pred_norm = rec - np.mean(rec, axis=0)
    pred_norm /= np.std(pred_norm, axis=0)
    mse_rec = np.mean(np.square(gt_cat - pred_norm))
    return uv, {'reconstructed': mse_rec}

def worstCaseModel(path, verbose=True, rep=0):
    repPath = path + 'r{}'.format(rep)
    uv = UVAE(repPath)
    p0 = uv + Data(X0, channels=chs0, name='Panel 0', valProp=0.0)
    p1 = uv + Data(X1, channels=chs1, name='Panel 1', valProp=0.0)
    p2 = uv + Data(X2, channels=chs2, name='Panel 2', valProp=0.0)
    uv.train(n_epochs, samplesPerEpoch=100000, valSamplesPerEpoch=0, callback=None, verbose=verbose,
             skipTrained=True)
    return uv


# train a model given a config
def trainConfig(config, path, verbose=True, rep=0):
    repPath = path + 'r{}'.format(rep)
    if fileExists(repPath):
        uv = UVAE(repPath)
    else:
        uv = UVAE(path)

    p0 = uv + Data(X0, channels=chs0, name='Panel 0', valProp=0.2)
    p1 = uv + Data(X1, channels=chs1, name='Panel 1', valProp=0.2)
    p2 = uv + Data(X2, channels=chs2, name='Panel 2', valProp=0.2)
    ps = [p0, p1, p2]

    panel_lab = uv + Labeling(Y={p0: np.repeat(0, len(p0.X)),
                                 p1: np.repeat(1, len(p1.X)),
                                 p2: np.repeat(2, len(p2.X))},
                              name='Panel')

    batch_lab = uv + Labeling(Y={p0: B0, p1: B1, p2: B2}, name='Batch')
    batch_lab0 = uv + Labeling(Y={p0: B0}, name='Batch 0')
    batch_lab1 = uv + Labeling(Y={p1: B1}, name='Batch 1')
    batch_lab2 = uv + Labeling(Y={p2: B2}, name='Batch 2')

    ctype_lab = uv + Labeling(Y={p0: Y0, p1: Y1, p2: Y2}, name='CT lab')

    clustering_lab = uv + Labeling(Y={p0: batch_clustering_unique[0],
                                      p1: batch_clustering_unique[1],
                                      p2: batch_clustering_unique[2]}, name='Clustering')

    res_clusterings = []
    if 'res-uns' in config:
        n_clusterings = int(config['res-uns'])
        for n_cl in range(n_clusterings):
            cl_lab = uv + Labeling(Y={p0: batch_clusterings[0][n_cl],
                                      p1: batch_clusterings[1][n_cl],
                                      p2: batch_clusterings[2][n_cl]}, name='Clustering {}'.format(n_cl))
            res_clusterings.append(cl_lab)
    if 'res-uns-pan' in config:
        cl_lab = uv + Labeling(Y={p0: panel_clusterings[0][0],
                                  p1: panel_clusterings[1][0],
                                  p2: panel_clusterings[2][0]}, name='Clustering panels')
        res_clusterings.append(cl_lab)

    c_cond = config['cond']
    if type(c_cond) is bool and c_cond == False:
        cond = cond0 = cond1 = cond2 = None
    else:
        cond = [batch_lab]
        cond0 = [batch_lab0]
        cond1 = [batch_lab1]
        cond2 = [batch_lab2]
        # default is linear projection of conditioning vector (1 layer), to this dimensionality:
        uv.hyper = {'cond_dim': 2}
        # else:
        if type(c_cond) is int:
            if c_cond == 0:
                uv.hyper = {'cond_dim': 0} # direct concatenation of one-hot vector

    c_enc = True
    if 'no-ce' in config:
        c_enc = False

    variational = True
    if 'no-vae' in config:
        variational = False

    ae0 = uv + Autoencoder(name=p0.name, masks=p0, conditions=cond0, condEncoder=c_enc, variational=variational)
    ae1 = uv + Autoencoder(name=p1.name, masks=p1, conditions=cond1, condEncoder=c_enc, variational=variational)
    ae2 = uv + Autoencoder(name=p2.name, masks=p2, conditions=cond2, condEncoder=c_enc, variational=variational)

    bal_batch = True
    if 'no-bal' in config:
        bal_batch = False

    if config['norm']:
        norm = uv + Normalization(Y=batch_lab.Y, name='Z norm', balanceBatchAvg=bal_batch)
        if config['resample']:
            ctype_lab.resample(norm)
        for cl_res in res_clusterings:
            cl_res.resample(norm)

    if config['mmd']:
        mmd = uv + MMD(Y=panel_lab.Y, name='MMD', pull=1, frequency=1, balanceBatchAvg=bal_batch)
        if config['resample']:
            ctype_lab.resample(mmd)
        for cl_res in res_clusterings:
            cl_res.resample(mmd)

    if 'mmdb' in config:
        if not config['mmd']:
            mmdb = uv + MMD(Y=batch_lab.Y, name='MMD', pull=1, frequency=1, balanceBatchAvg=bal_batch)
            if config['resample']:
                ctype_lab.resample(mmdb)
        else:
            mmdb0 = uv + MMD(Y=batch_lab0.Y, name='MMD-B0', pull=1, frequency=1, balanceBatchAvg=bal_batch)
            mmdb1 = uv + MMD(Y=batch_lab1.Y, name='MMD-B1', pull=1, frequency=1, balanceBatchAvg=bal_batch)
            mmdb2 = uv + MMD(Y=batch_lab2.Y, name='MMD-B2', pull=1, frequency=1, balanceBatchAvg=bal_batch)
            if config['resample']:
                ctype_lab.resample(mmdb0)
                ctype_lab.resample(mmdb1)
                ctype_lab.resample(mmdb2)

    if config['sub']:
        sub = uv + Subspace(name='Sub', pull=1, conditions=cond, condEncoder=c_enc, variational=variational)

    if 'reg-sup' in config:
        ctype = uv + Classification(Y={p0: Y0, p1: Y1, p2: Y2}, name='Cell type')

    if 'reg-uns' in config:
        for pi, cl in enumerate(batch_clustering_unique):
            bts = list(set(toyDs['batch'][pi]))
            for b in bts:
                b_mask = toyDs['batch'][pi] == b
                masked = repeatMasked(mask=b_mask, value=cl[b_mask], nullValue='-')
                cl_clsf = uv + Classification(Y={ps[pi]: masked}, nullLabel='-', name='Cl_p{}_b{}'.format(pi, b),
                                              trainEmbedding=True, equalizeLabels=False)

    if 'lisi-sup' in config:
        lisiSet = LisiValidationSet(uv.allDataMap(), labelConstraint=ctype_lab, batchConstraint=batch_lab,
                                    normClasses=True, labelRange=lisiRanges['supervised']['labelRange'], labelWeight=1.0,
                                    batchRange=lisiRanges['supervised']['batchRange'], batchWeight=1.0, perplexity=100)
    else:
        lisiSet = LisiValidationSet(uv.allDataMap(), labelConstraint=clustering_lab, batchConstraint=batch_lab,
                                    normClasses=False, labelRange=lisiRanges['unsupervised']['labelRange'], labelWeight=1.0,
                                    batchRange=lisiRanges['unsupervised']['batchRange'], batchWeight=1.0, perplexity=100)

    uv.optimize(iterations=30, maxEpochs=n_epochs, samplesPerEpoch=100000, valSamplesPerEpoch=100000,
                lossWeight=3.0,
                subset=['frequency-MMD',
                        'frequency-Sub',
                        'pull-MMD',
                        'pull-Sub',
                        'lr_unsupervised',
                        'lr_supervised',
                        'lr_merge',
                        'ease_epochs'],
                lisiValidationSet=lisiSet, callback=None)

    if uv.path != repPath:
        uv.train(n_epochs, samplesPerEpoch=100000, valSamplesPerEpoch=0, callback=None, verbose=verbose,
                 skipTrained=False)
        copyFile(uv.path, repPath)
    return uv

# test reconstruction from a model against the ground truth values
def reconstructionError(uv, shuffle=False, medians=False):
    if uv is not None and not medians:
        dm = uv.allDataMap()
        if shuffle:
            # shuffle points for worst case reconstruction reference
            for k in dm:
                dm[k] = np.random.permutation(dm[k])

        # reconstruct own markers for each panel
        rec = uv.reconstruct(dataMap=dm)
        # imputed and merged outputs (all markers, first valid target batch)
        rec_merged = uv.reconstruct(dataMap=dm, channels=gt_chs)
    else:
        # impute values based on medians
        classes = list(set(ctype_labels))
        class_dict = {cl: {ch: [] for ch in gt_chs} for cl in classes}
        for cl in classes:
            for chi, ch in enumerate(gt_chs):
                for xi, x_vals in enumerate(toyDs['X']):
                    if ch in toyDs['markers'][xi]:
                        ctype_mask = toyDs['celltype'][xi] == cl
                        vals = x_vals[ctype_mask, int(toyDs['markers'][xi].index(ch))]
                        class_dict[cl][ch].extend(vals)

    reconstructed = {ch: {'gt': [], 'rec': []} for ch in gt_chs}
    imputed = {ch: {'gt': [], 'rec': []} for ch in gt_chs}
    merged = {ch: {'gt': [], 'rec': []} for ch in gt_chs}

    # sort reconstructions into the three types
    for ch in gt_chs:
        gt_xs = [GT0, GT1, GT2]
        ch_ind = list(gt_chs).index(ch)
        for pi, p_chs in enumerate([chs0, chs1, chs2]):
            gt_x = gt_xs[pi][:, ch_ind]
            rec_merged_x = rec_merged[uv.data[pi]][:, ch_ind]
            if ch in p_chs:
                merged[ch]['gt'].append(gt_x)
                merged[ch]['rec'].append(rec_merged_x)
                p_ch_ind = list(p_chs).index(ch)
                rec_x = rec[uv.data[pi]][:, p_ch_ind]
                reconstructed[ch]['gt'].append(gt_x)
                reconstructed[ch]['rec'].append(rec_x)
            else:
                imputed[ch]['gt'].append(gt_x)
                imputed[ch]['rec'].append(rec_merged_x)

    def calculateError(vals):
        ch_mse = {}
        for ch in vals:
            if len(vals[ch]['gt']):
                cat_gt = np.concatenate(vals[ch]['gt'])
                cat_pred = np.concatenate(vals[ch]['rec'])
                cat_pred_norm = cat_pred - np.mean(cat_pred, axis=0)
                cat_pred_norm /= np.std(cat_pred, axis=0)
                mse = np.mean(np.square(cat_gt - cat_pred_norm))
                ch_mse[ch] = {'mse': mse}
        return ch_mse

    def channelMean(d):
        return np.mean([d[ch]['mse'] for ch in d])

    results = {'reconstructed': channelMean(calculateError(reconstructed)),
            'merged': channelMean(calculateError(merged)),
            'imputed': channelMean(calculateError(imputed))}

    return results


def printChannelError(rec_ch_err, merged_ch_err, imp_ch_err):
    print('\nReconstructed:')
    for ch in rec_ch_err:
        print('{}: MSE: {}'.format(ch, round(rec_ch_err[ch]['mse'], 4)))
    print('\nMerged:')
    for ch in merged_ch_err:
        print('{}: MSE: {}'.format(ch, round(merged_ch_err[ch]['mse'], 4)))
    print('\nImputed:')
    for ch in imp_ch_err:
        print('{}: MSE: {}'.format(ch, round(imp_ch_err[ch]['mse'], 4)))


def testAnchors(stackedEmb):
    vars = []
    panelOffsets = [np.sum([len(X) for X in toyDs['X'][:pn]]) for pn in range(len(toyDs['X']))]
    for ai, anch in enumerate(anchors):
        points = []
        for pi, p_inds in enumerate(anch):
            inds_adjusted = np.array(p_inds, dtype=int) + int(panelOffsets[pi])
            points.extend(stackedEmb[inds_adjusted])
        mean_var = np.mean(np.var(points, axis=0))
        vars.append(mean_var)
    meanvar = np.mean(vars)
    global_meanvar = np.mean(np.var(stackedEmb, axis=0))
    return {'anchor_var': meanvar,
            'global_var': global_meanvar,
            'anchor_var_ratio': meanvar / global_meanvar}


def makePlots(emb, name, folder, emb2d=None):
    import umap
    um = umap.UMAP()
    emb_um = um.fit_transform(emb)
    ensureFolder(folder)
    savePlot(emb_um, ctype_labels, folder + '{}-ctype.png'.format(name))
    savePlot(emb_um, batches, folder + '{}-batches.png'.format(name))
    savePlot(emb_um, clusters_gt, folder + '{}-gt.png'.format(name))
    if emb2d is not None:
        savePlot(emb2d, ctype_labels, folder + '{}-2d-ctype.png'.format(name))
        savePlot(emb2d, batches, folder + '{}-2d-batches.png'.format(name))
        savePlot(emb2d, clusters_gt, folder + '{}-2d-gt.png'.format(name))


def lisiMetrics(embs_stack, name, rep, outFolder, resDict):
    if 'LISI' not in resDict:
        resDict['LISI'] = calculateLISI(emb=embs_stack,
                                      batches=batches,
                                      classes={'gt': clusters_gt,
                                               'ctype': ctype_labels,
                                               'bclust': np.concatenate(batch_clustering_unique)},
                                      name='{}_rep{}'.format(name, rep), outFolder=outFolder,
                                      scoreFilename='LISI.csv',
                                      perplexity=100)
    if 'LISI_gt_normed' not in resDict:
        gt_norm_mask = lisiMasks['gt']
        resDict['LISI_gt_normed'] = calculateLISI(emb=embs_stack[gt_norm_mask],
                                    batches=batches[gt_norm_mask],
                                    classes=clusters_gt[gt_norm_mask],
                                    name='{}_rep{}'.format(name, rep), outFolder=outFolder,
                                    scoreFilename='LISI_gt_normed.csv',
                                    perplexity=100)
    if 'LISI_ctype_normed' not in resDict:
        ctype_norm_mask = lisiMasks['ctype']
        resDict['LISI_ctype_normed'] = calculateLISI(emb=embs_stack[ctype_norm_mask],
                                          batches=batches[ctype_norm_mask],
                                          classes=ctype_labels[ctype_norm_mask],
                                          name='{}_rep{}'.format(name, rep), outFolder=outFolder,
                                          scoreFilename='LISI_ctype_normed.csv',
                                          perplexity=100)


for rep in range(repeats):

    for ii, config in enumerate(testConfigs):
        # add extra parameters from command line
        config = dict(config)
        for par in config_params:
            config[par] = True

        # generate a string from config to use as unique hashable key
        cf_key = configName(config, ds_name)
        print('\nTraining config ({}/{} rep. {}/{}):'.format(ii+1, len(testConfigs), rep+1, repeats))
        print(cf_key)

        cf_folder = ensureFolder(outFolder + '{}/'.format(cf_key))
        cf_results_file = cf_folder + 'results-{}.pkl'.format(cf_key)
        res_dict = {}

        if fileExists(cf_results_file):
            cf_results = unpickle(cf_results_file)
            if len(cf_results['scores']) > rep:
                res_dict = cf_results['scores'][rep]
        else:
            cf_results = {'dataset': ds_name, 'config': config, 'cf_name': cf_key, 'scores': []}

        if 'GT-VAE' in config:
            # ground truth config
            if 'GT-data' not in cf_results:
                # LISI metrics on the ground truth in data space
                lisiMetrics(gt_cat, 'GT-data', 0, cf_folder, res_dict)
                doPickle(cf_results, cf_results_file)

            # VAE trained on ground truth without batch effects or panel split
            uv, rec_errors = groundTruthModel(cf_folder + 'gt-vae.uv', rep=rep)

        else:
            if 'worst-VAE' in config:
                # worst case configuration, no panel merging or batch effect correction
                uv = worstCaseModel(cf_folder + '{}.uv'.format(cf_key), rep=rep)
                rec_errors = reconstructionError(uv, shuffle=config['rand'])
            else:

                # normal configuration
                uv = trainConfig(config, path=cf_folder + '{}.uv'.format(cf_key), rep=rep)

                viewHyper(uv, cf_folder + 'hyper/')

                rec_errors = reconstructionError(uv)
        res_dict.update(rec_errors)

        embs_stack = uv.predictMap(uv.allDataMap(), mean=True, stacked=True)

        if 'worst-VAE' in config and config['rand']:
            # permute embedding for worst case of maximum mixing
            embs_stack = np.random.permutation(embs_stack)

        # calculate anchor metrics:
        anchor_metrics = testAnchors(embs_stack)
        res_dict.update(anchor_metrics)

        # calculate LISI metrics:
        lisiMetrics(embs_stack, cf_key, rep, cf_folder, res_dict)

        # save results

        print(config, res_dict)
        if len(cf_results['scores']) > rep:
            cf_results['scores'][rep] = res_dict
        else:
            cf_results['scores'].append(res_dict)
        doPickle(cf_results, cf_results_file)

        makePlots(embs_stack, name='rep_{}'.format(rep), folder=cf_folder)


