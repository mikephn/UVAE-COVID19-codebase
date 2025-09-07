"""
Imputation scoring script for evaluating reconstruction performance.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import os

def evaluate_imputation(reconstructed, ground_truth_df, missing_channels, panel_name):
    """
    Evaluate imputation performance for a single panel
    
    Parameters:
    - reconstructed: numpy array of reconstructed values (samples x channels)
    - ground_truth_df: pandas DataFrame with ground truth
    - missing_channels: list of channel names that were imputed
    - panel_name: string identifier for the panel
    
    Returns:
    - dict with evaluation metrics per channel
    """
    results = {}
    
    for i, channel in enumerate(missing_channels):
        if channel in ground_truth_df.columns:
            # Get ground truth values for this channel
            gt_values = ground_truth_df[channel].values
            
            # Get predicted values
            pred_values = reconstructed[:, i]
            
            # Ensure same length (in case of sample mismatch)
            min_len = min(len(gt_values), len(pred_values))
            gt_values = gt_values[:min_len]
            pred_values = pred_values[:min_len]
            
            # Convert to numeric and handle any remaining non-numeric values
            try:
                gt_values = pd.to_numeric(gt_values, errors='coerce')
                pred_values = pd.to_numeric(pred_values, errors='coerce')
                
                # Remove NaN values
                valid_mask = ~(np.isnan(gt_values) | np.isnan(pred_values))
                gt_values = gt_values[valid_mask]
                pred_values = pred_values[valid_mask]
                
                if len(gt_values) == 0:
                    print(f"Warning: No valid numeric data for {channel} in {panel_name}")
                    continue
                
                # Calculate metrics
                mse = mean_squared_error(gt_values, pred_values)
                pearson_r, pearson_p = pearsonr(gt_values, pred_values)
                spearman_rho, spearman_p = spearmanr(gt_values, pred_values)
                
            except Exception as e:
                print(f"Error calculating metrics for {channel} in {panel_name}: {e}")
                continue
            
            results[channel] = {
                'MSE': mse,
                'Pearson_R': pearson_r,
                'Pearson_p': pearson_p,
                'Spearman_Rho': spearman_rho,
                'Spearman_p': spearman_p
            }
            
            print(f"{panel_name} - {channel}:")
            print(f"  MSE: {mse:.6f}")
            print(f"  Pearson R: {pearson_r:.4f} (p={pearson_p:.4f})")
            print(f"  Spearman Rho: {spearman_rho:.4f} (p={spearman_p:.4f})")
        else:
            print(f"Warning: {channel} not found in {panel_name} ground truth")
    
    return results


def score_imputation(model_name, reconstructed_files, ground_truth_files, missing_channels_list, 
                    results_file="imputation_results.tsv", verbose=True):
    """
    Load reconstructed outputs from TSV files and calculate scores
    
    Parameters:
    - model_name: string identifier for the model
    - reconstructed_files: list of paths to reconstructed TSV files
    - ground_truth_files: list of paths to ground truth CSV files
    - missing_channels_list: list of lists containing missing channels for each panel
                           e.g., [['HLADR', 'CD19', ...], ['CD134', 'CD45RO', ...], ...]
    - results_file: path to the results TSV file
    - verbose: whether to print detailed results
    
    Returns:
    - dict with all evaluation results
    """
    
    all_results = {}
    
    # Set random seed for reproducible sampling
    np.random.seed(42)

    # Evaluate each panel
    for i, (rec_file, gt_file, missing_channels) in enumerate(zip(reconstructed_files, ground_truth_files, missing_channels_list)):
        
        if not os.path.exists(rec_file):
            print(f"Warning: Reconstructed file not found: {rec_file}")
            continue
            
        if not os.path.exists(gt_file):
            print(f"Warning: Ground truth file not found: {gt_file}")
            continue
        
        # Load data
        rec_data = pd.read_csv(rec_file, sep='\t')
        gt_data = pd.read_csv(gt_file)
        
        # First, check which missing channels are actually available in the reconstructed data
        available_channels = [ch for ch in missing_channels if ch in rec_data.columns]
        if not available_channels:
            print(f"Warning: None of the missing channels {missing_channels} found in {rec_file}")
            continue
        
        # Select only the channels we need from reconstructed data
        rec_data_filtered = rec_data[available_channels]
        
        # Convert to numeric
        rec_data_numeric = rec_data_filtered.apply(pd.to_numeric, errors='coerce')
        
        # Convert to numpy array
        rec_data_array = rec_data_numeric.values
        
        # Evaluate using only the available channels
        panel_name = f"Panel {i+1}"
        eval_results = evaluate_imputation(rec_data_array, gt_data, available_channels, panel_name)
        
        # Add to all results with panel prefix
        all_results.update({f"P{i+1}_{k}": v for k, v in eval_results.items()})
    
    # Calculate average metrics across all channels
    all_mse = [metrics['MSE'] for metrics in all_results.values()]
    all_pearson_r = [metrics['Pearson_R'] for metrics in all_results.values() if not np.isnan(metrics['Pearson_R'])]
    all_spearman_rho = [metrics['Spearman_Rho'] for metrics in all_results.values() if not np.isnan(metrics['Spearman_Rho'])]

    avg_mse = np.mean(all_mse)
    # For correlations, convert to Fisher Z-transform, average, then convert back
    fisher_z_pearson = np.arctanh(np.clip(all_pearson_r, -0.999, 0.999))
    avg_pearson_r = np.tanh(np.mean(fisher_z_pearson))

    fisher_z_spearman = np.arctanh(np.clip(all_spearman_rho, -0.999, 0.999))
    avg_spearman_rho = np.tanh(np.mean(fisher_z_spearman))

    if verbose:
        print("SUMMARY STATISTICS")
        print(f"Average MSE across all channels: {avg_mse:.6f}")
        print(f"Average Pearson R (Fisher Z-averaged): {avg_pearson_r:.4f}")
        print(f"Average Spearman Rho (Fisher Z-averaged): {avg_spearman_rho:.4f}")

    # Save results to TSV file
    # Prepare the row data
    row_data = [model_name, avg_mse, avg_pearson_r, avg_spearman_rho]

    # Add individual channel results
    for channel_key in sorted(all_results.keys()):
        metrics = all_results[channel_key]
        row_data.extend([metrics['MSE'], metrics['Pearson_R'], metrics['Spearman_Rho']])

    # Create header if file doesn't exist
    file_exists = os.path.exists(results_file)

    if not file_exists:
        header = ['Model', 'Avg_MSE', 'Avg_Pearson_R', 'Avg_Spearman_Rho']
        for channel_key in sorted(all_results.keys()):
            header.extend([f'{channel_key}_MSE', f'{channel_key}_Pearson_R', f'{channel_key}_Spearman_Rho'])
        
        with open(results_file, 'w') as f:
            f.write('\t'.join(header) + '\n')

    # Append the results
    with open(results_file, 'a') as f:
        f.write('\t'.join(map(str, row_data)) + '\n')

    if verbose:
        print(f"\nResults saved to {results_file}")
        print(f"Row added for model: {model_name}")
    
    return all_results

if __name__ == "__main__":
    # Example run for cyCombine-generated data
    model_name = "cycomb_g16"


    p_n_gt = "p1"
    p_n_out = "p1_16"
    # File paths as provided
    reconstructed_files = [
        f"imputation/{p_n_out}/cc_imp_0.tsv",
        f"imputation/{p_n_out}/cc_imp_1.tsv",
        f"imputation/{p_n_out}/cc_imp_2.tsv"
    ]

    ground_truth_files = [
        f"DFCI/cll_{p_n_gt}_impute/panel1_gt.csv",
        f"DFCI/cll_{p_n_gt}_impute/panel2_gt.csv",
        f"DFCI/cll_{p_n_gt}_impute/panel3_gt.csv"
    ]

    # p_n_gt = "vg"
    # p_n_out = "vg_16"
    # ground_truth_files = [
    #     f"van Gassen/impute/panel1_gt.csv",
    #     f"van Gassen/impute/panel2_gt.csv",
    #     f"van Gassen/impute/panel3_gt.csv"
    # ]

    # Missing channels for each panel
    if p_n_gt == "p1":
        missing_channels_list = [
            ["HLADR", "CD19", "GranzymeK", "CD27", "FoxP3"],
            ["CD134", "CD45RO", "CD355", "CD20", "CD197"],
            ["CD279", "CD4", "GranzymeA", "CD278", "CD127"]
        ]
    elif p_n_gt == "p2":
        missing_channels_list = [
            ["CD5", "CD33", "HLADR", "IFNG", "CD1d"],
            ["CD123", "FoxP3", "CD184", "CD45RA", "CD4"],
            ["CD14", "SMAD2", "TGFBR2", "CD8", "CD161"]
        ]
    elif p_n_gt == "vg":
        missing_channels_list = [
            ["CD56", "HLADR", "CD33", "NFkB", "IkB"], 
            ["p38", "CD3", "CD16", "CD8a", "TCRgd"], 
            ["CD25", "CXCR3", "CD123", "Tbet", "CD4"]
        ]

    results_file = f"imputation/{p_n_gt}/evaluation_results.tsv"
    verbose = True

    # Run evaluation
    score_imputation(model_name, reconstructed_files, ground_truth_files, missing_channels_list, results_file, verbose)