import os
import pickle
import pandas as pd
import numpy as np
from src.UVAE_diag import calculateLISI, calculateEmdMad
from src.tools import unpickle
import pyreadr

# Load the toy dataset
toyDs = pickle.load(open('data/ToyDataWB-3x3.pkl', "rb"), encoding='latin1')
ds_name = 'ToyDataWB-3x3'
outFolder = '.'

# Ground truth values
GT0, GT1, GT2 = toyDs['GT_X']
chs0, chs1, chs2 = toyDs['markers']
gt_chs = toyDs['GT_markers']
shared_chs = np.array([ch for ch in gt_chs if (ch in chs0) and (ch in chs1) and (ch in chs2)], dtype=str)

ctype_labels = np.concatenate(toyDs['celltype'])
batches = np.concatenate(toyDs['batch'])
clusters_gt = np.concatenate(toyDs['cluster'])

# Create DataFrames for each panel
df_gt0 = pd.DataFrame(GT0, columns=gt_chs)
df_gt1 = pd.DataFrame(GT1, columns=gt_chs)
df_gt2 = pd.DataFrame(GT2, columns=gt_chs)

# Load imputed values from TSV files
df_imp0 = pd.read_csv('cc_imp_0.tsv', sep='\t')
df_imp1 = pd.read_csv('cc_imp_1.tsv', sep='\t')
df_imp2 = pd.read_csv('cc_imp_2.tsv', sep='\t')

cat_imputed = pd.concat([df_imp0, df_imp1, df_imp2], axis=0)
cat_imputed.drop('batch', axis=1, inplace=True)
cat_imputed = cat_imputed.to_numpy()
nan_mask = np.isnan(cat_imputed).any(axis=1).astype(bool)
print("NaNs: {} out of {}, {}%".format(np.sum(nan_mask), len(nan_mask), 100*np.sum(nan_mask)/len(nan_mask)))

# Function to calculate channel-wise MSE for imputed channels
def calculateErrorImputed(GT, imputed, original_chs):
    ch_mse = {}
    imputed_chs = set(GT.columns) - set(original_chs)
    
    for ch in imputed_chs:
        gt_values = GT[ch].values
        pred_values = imputed[ch].values

        # Identify NaN values
        nan_mask = np.isnan(pred_values)
        num_nans = np.sum(nan_mask)
        
        # Print the number of NaNs
        print(f"Channel {ch}: {num_nans} NaNs found ({float(num_nans) / len(gt_values)})")
        
        # Filter out NaN values
        valid_gt_values = gt_values[~nan_mask]
        valid_pred_values = pred_values[~nan_mask]
        
        # Normalize the valid predicted values
        if len(valid_pred_values) > 0:
            valid_pred_values_norm = (valid_pred_values - np.mean(valid_pred_values)) / np.std(valid_pred_values)
            # Calculate MSE
            mse = np.mean(np.square(valid_gt_values - valid_pred_values_norm))
        else:
            mse = np.nan
        
        ch_mse[ch] = {'mse': mse}
    return ch_mse

def channelMean(d):
    return np.nanmean([d[ch]['mse'] for ch in d if not np.isnan(d[ch]['mse'])])

# Calculate MSE for each panel
channel_mse0 = calculateErrorImputed(df_gt0, df_imp0, chs0)
channel_mse1 = calculateErrorImputed(df_gt1, df_imp1, chs1)
channel_mse2 = calculateErrorImputed(df_gt2, df_imp2, chs2)

# Calculate the average channel MSE for each panel
average_channel_mse0 = channelMean(channel_mse0)
average_channel_mse1 = channelMean(channel_mse1)
average_channel_mse2 = channelMean(channel_mse2)

print("Average Channel MSE for Panel 0:", average_channel_mse0)
print("Average Channel MSE for Panel 1:", average_channel_mse1)
print("Average Channel MSE for Panel 2:", average_channel_mse2)

# Calculate the combined channel MSE
imputed_channels_mse = dict()
imputed_channels_mse.update(channel_mse0)
imputed_channels_mse.update(channel_mse1)
imputed_channels_mse.update(channel_mse2)
average_channel_mse = channelMean(imputed_channels_mse)

print("Average Channel MSE:", average_channel_mse)

def calcLISI(rec_stack, batches, clusters_gt, ctype_labels, no_nan_mask, name="cyCombine", rep=0):

    lisiMasksFile = outFolder + 'lisi_masks.pkl'
    lisiMasks = unpickle(lisiMasksFile)

    resDict = {}
    resDict['recLISI'] = calculateLISI(emb=rec_stack[no_nan_mask],
                                       batches=batches[no_nan_mask],
                                       classes={'gt': clusters_gt[no_nan_mask],
                                                'ctype': ctype_labels[no_nan_mask]},
                                       name='{}_rep{}'.format(name, rep), outFolder=outFolder,
                                       scoreFilename='cyComb_recLISI.csv',
                                       perplexity=100)
    gt_norm_mask = lisiMasks['gt'][no_nan_mask]
    resDict['recLISI_gt_normed'] = calculateLISI(emb=rec_stack[no_nan_mask][gt_norm_mask],
                                                 batches=batches[no_nan_mask][gt_norm_mask],
                                                 classes=clusters_gt[no_nan_mask][gt_norm_mask],
                                                 name='{}_rep{}'.format(name, rep), outFolder=outFolder,
                                                 scoreFilename='cyComb_recLISI_gt_normed.csv',
                                                 perplexity=100)
    ctype_norm_mask = lisiMasks['ctype'][no_nan_mask]
    resDict['recLISI_ctype_normed'] = calculateLISI(emb=rec_stack[no_nan_mask][ctype_norm_mask],
                                                    batches=batches[no_nan_mask][ctype_norm_mask],
                                                    classes=ctype_labels[no_nan_mask][ctype_norm_mask],
                                                    name='{}_rep{}'.format(name, rep), outFolder=outFolder,
                                                    scoreFilename='cyComb_recLISI_ctype_normed.csv',
                                                    perplexity=100)

calcLISI(cat_imputed, batches=batches, clusters_gt=clusters_gt, ctype_labels=ctype_labels, no_nan_mask=nan_mask==False)

def calcEmdMad():
    col_inds = np.array([gt_chs.index(ch) for ch in shared_chs], dtype=int)
    rec_subset = cat_imputed[nan_mask==False][:, col_inds]

    vals = pd.DataFrame(rec_subset, columns=shared_chs)
    vals['id'] = np.arange(len(vals))[nan_mask==False]
    vals['batch'] = batches[nan_mask==False]
    emd_mad_path = outFolder + 'emd_mad_cycomb.RDS'
    pyreadr.write_rds(emd_mad_path, vals)
    emd, mad = calculateEmdMad("emd_mad/emd_mad_unc.RDS", emd_mad_path,
                               outFolder + 'emd_mad_cyComb.csv')
    os.remove(emd_mad_path)

calcEmdMad()