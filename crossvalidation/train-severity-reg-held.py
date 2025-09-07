import re
import tensorflow as tf
import tensorflow.keras as keras
from src.tools import *
import pandas as pd

outFolder = ensureFolder('severity-reg-held/')

valid_sevs = ['Healthy', 'Mild', 'Moderate', 'Severe']
severityEmbedding = {'Healthy': 0.5, 'Mild': 2, 'Moderate': 3, 'Severe': 4.5}

valid_types = ['Control', 'Acute']

# Read clinical variable file and create maps of samples
cv = pd.read_csv('data/meta.csv')
cv_names = list(cv['sample_name'].values)
pidMap = dict(zip(cv_names, cv['patient_id']))
typeMap = dict(zip(cv_names, cv['sample_type']))
severityMap = dict(zip(cv_names, cv['peak_severity']))
timeMap = dict(zip(cv_names, cv['timepoint']))

params_default = {'modelName': 'severity-reg',
          'useLineage': True,
          'useChemo': True,
          'chemoType': 'gmm',
          'sampleInput': None,
          'poolTime': 'max',
          'nFilters': 50,
          'nHidden': 0,
          'embDepth': 0,
          'maxEpochs': 2000,
          'patience': 50,
          'upsample': 10,
          'batchSize': 256,
          'nFolds': 4,
          }

configs = [

# trained on proportions
# linear models
    {'useLineage': True, 'useChemo': False},
    {'useLineage': False, 'useChemo': True},
    {'useLineage': False, 'useChemo': True, 'chemoType': 'leiden'},
    {'useLineage': True, 'useChemo': True},
    {'useLineage': True, 'useChemo': True, 'chemoType': 'leiden'},
# 1-hidden layer, max pooling
    {'useLineage': True, 'useChemo': False, 'nHidden': 1},
    {'useLineage': False, 'useChemo': True, 'nHidden': 1},
    {'useLineage': False, 'useChemo': True, 'chemoType': 'leiden', 'nHidden': 1},
    {'useLineage': True, 'useChemo': True, 'nHidden': 1},
    {'useLineage': True, 'useChemo': True, 'chemoType': 'leiden', 'nHidden': 1},
# 1-hidden layer, rnn pooling
    {'useLineage': True, 'useChemo': False, 'nHidden': 1, 'poolTime': 'rnn'},
    {'useLineage': False, 'useChemo': True, 'nHidden': 1, 'poolTime': 'rnn'},
    {'useLineage': False, 'useChemo': True, 'chemoType': 'leiden', 'nHidden': 1, 'poolTime': 'rnn'},
    {'useLineage': True, 'useChemo': True, 'nHidden': 1, 'poolTime': 'rnn'},
    {'useLineage': True, 'useChemo': True, 'chemoType': 'leiden', 'nHidden': 1, 'poolTime': 'rnn'},
# trained on set input
# 1-layer embedding
# max pooling
    {'useLineage': True, 'useChemo': False, 'sampleInput': 20, 'batchSize': 64, 'embDepth': 1, 'nHidden': 1},
    {'useLineage': False, 'useChemo': True, 'sampleInput': 20, 'batchSize': 64, 'embDepth': 1, 'nHidden': 1},
    {'useLineage': True, 'useChemo': True, 'sampleInput': 20, 'batchSize': 64, 'embDepth': 1, 'nHidden': 1},
# rnn pooling
    {'useLineage': True, 'useChemo': False, 'sampleInput': 20, 'batchSize': 64, 'embDepth': 1, 'poolTime': 'rnn'},
    {'useLineage': False, 'useChemo': True, 'sampleInput': 20, 'batchSize': 64, 'embDepth': 1, 'poolTime': 'rnn'},
    {'useLineage': True, 'useChemo': True, 'sampleInput': 20, 'batchSize': 64, 'embDepth': 1, 'poolTime': 'rnn'},
# max pooling
    {'useLineage': True, 'useChemo': False, 'sampleInput': 50, 'batchSize': 64, 'embDepth': 1, 'nHidden': 1},
    {'useLineage': False, 'useChemo': True, 'sampleInput': 50, 'batchSize': 64, 'embDepth': 1, 'nHidden': 1},
    {'useLineage': True, 'useChemo': True, 'sampleInput': 50, 'batchSize': 64, 'embDepth': 1, 'nHidden': 1},
# rnn pooling
    {'useLineage': True, 'useChemo': False, 'sampleInput': 50, 'batchSize': 64, 'embDepth': 1, 'poolTime': 'rnn'},
    {'useLineage': False, 'useChemo': True, 'sampleInput': 50, 'batchSize': 64, 'embDepth': 1, 'poolTime': 'rnn'},
    {'useLineage': True, 'useChemo': True, 'sampleInput': 50, 'batchSize': 64, 'embDepth': 1, 'poolTime': 'rnn'},
# max pooling
    {'useLineage': True, 'useChemo': False, 'sampleInput': 100, 'batchSize': 64, 'embDepth': 1, 'nHidden': 1},
    {'useLineage': False, 'useChemo': True, 'sampleInput': 100, 'batchSize': 64, 'embDepth': 1, 'nHidden': 1},
    {'useLineage': True, 'useChemo': True, 'sampleInput': 100, 'batchSize': 64, 'embDepth': 1, 'nHidden': 1},
# rnn pooling
    {'useLineage': True, 'useChemo': False, 'sampleInput': 100, 'batchSize': 64, 'embDepth': 1, 'poolTime': 'rnn'},
    {'useLineage': False, 'useChemo': True, 'sampleInput': 100, 'batchSize': 64, 'embDepth': 1, 'poolTime': 'rnn'},
    {'useLineage': True, 'useChemo': True, 'sampleInput': 100, 'batchSize': 64, 'embDepth': 1, 'poolTime': 'rnn'},
]

print('n. configs:', len(configs))

#config_ind = int(os.getenv('SLURM_ARRAY_TASK_ID')) - 1 # different IDs are provided by external batch script
config_ind = 0
resultsFolder = ensureFolder(outFolder+'scores/')

def configParams(config):
    params = dict(params_default)
    for k, v in config.items():
        params[k] = v
    modelName = params['modelName']
    for k, v in list(params.items())[1:]:
        modelName += '-{}'.format(str(v))
    return params, modelName

if config_ind is None:
    # aggregate results into a single file
    lines = [[k for k in params_default] + list(valid_sevs) + ['f1avg', 'accuracy', 'test_loss', 'val_loss']]
    for config in configs:
        params, modelName = configParams(config)
        resultFile = resultsFolder + modelName + '.csv'
        if fileExists(resultFile):
            results = csvFile(resultFile, remNewline=True)[-1]
            lines.append(list(params.values()) + results)
    saveAsCsv(lines, resultsFolder + 'combined.csv')
    exit()

params, modelName = configParams(configs[config_ind])

modelFolder = ensureFolder(outFolder+'models/{}/'.format(modelName))

# group files by patient ID and determine associated sample files, times and severities
def groupPidData(data):
    sample_pids = {fn: pidMap[fn] for fn in data['vals']}
    valid_pids = []
    valid_samples = []
    for fname, pid in sample_pids.items():
        if (type(pid) is str and pid != 'unk') and (typeMap[fname] in valid_types) and (severityMap[fname] in valid_sevs):
            valid_pids.append(pid)
            valid_samples.append(fname)
    valid_pids = list(set(valid_pids))
    pid_samples = {pid: [] for pid in valid_pids}
    pid_times = {pid: [] for pid in valid_pids}
    pid_sevs = {pid: [] for pid in valid_pids}
    for k, v in sample_pids.items():
        if v in valid_pids and k in valid_samples:
            pid_samples[v].append(k)
            pid_times[v].append(timeMap[k] if type(timeMap[k]) is str else 'nan')
            pid_sevs[v].append(severityMap[k] if type(severityMap[k]) is str else 'nan')

    # assert that all samples for pid have the same severity (designating maximum severity)
    for pid in pid_sevs:
        assert len(set(pid_sevs[pid])) == 1

    # order samples for each PID by time the sample was taken (used in RNN model)
    sortKey = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', str(s))]
    all_times_sorted = sorted(list(set(np.concatenate(list(pid_times.values())))), key=sortKey) + ['nan']
    for pid in valid_pids:
        time_ord = np.argsort([all_times_sorted.index(i) for i in pid_times[pid]])
        pid_samples[pid] = [pid_samples[pid][i] for i in time_ord]
        pid_times[pid] = [pid_times[pid][i] for i in time_ord]
        pid_sevs[pid] = pid_sevs[pid][0] # severities for each time point are the same

    return pid_samples, pid_times, pid_sevs

# The `pid_split_file` is the source of truth for which patients exist and which fold they belong to.
# This must be pre-generated before training the UVAE models, so correct PIDs can be held out.
pid_split_file = outFolder + 'pid_split_{}.pkl'.format(params['nFolds'])
if not fileExists(pid_split_file):
    all_pids_from_meta = np.array(list(set(cv['patient_id'].values)))
    valid_pids_from_meta = []
    for i, row in cv.iterrows():
        pid = row['patient_id']
        if (type(pid) is str and pid != 'unk') and (row['sample_type'] in valid_types) and (row['peak_severity'] in valid_sevs):
            valid_pids_from_meta.append(pid)
    all_pids = np.array(list(set(valid_pids_from_meta)))

    pid_fold_split = {}
    pid_per = np.random.permutation(len(all_pids))
    pid_per_fold = int(np.ceil(float(len(all_pids)) / params['nFolds']))
    for i in range(params['nFolds']):
        start = i * pid_per_fold
        end = (i+1) * pid_per_fold
        pid_fold_split[str(i)] = all_pids[pid_per][start:end]
    doPickle(pid_fold_split, pid_split_file)
else:
    pid_fold_split = unpickle(pid_split_file)


# function reading in valid classes and filtering doublets
def filteredValues(fn, embedding:dict, clsf:dict, valid:list=None, doublets:dict=None):
    emb = embedding[fn]
    mask = np.ones(len(emb), dtype=bool)
    if doublets is not None:
        mask = np.logical_and(mask, doublets[fn] == 'Singlet')
    if np.sum(mask) == 0:
        print(f"Warning: No values found in embedding for {fn}")
    if clsf is None:
        return emb[mask]
    else:
        c = clsf[fn]
        if valid is not None:
            mask = np.logical_and(mask, np.isin(c, valid))
        if np.sum(mask) == 0:
            print(f"Warning: No values found in classification for {fn}")
        return emb[mask], c[mask]

# Helper function to extract data for a given list of PIDs.
def extract_pid_data(pids, pid_samples_common, pid_sevs_all, dataLineage, ctype_lin, valid_classes_lin, doublets_lin, dataChemo, ctype_chemo, valid_classes_chemo, doublets_chemo):
    inputs_props = []
    inputs_emb = []
    class_labels = []
    targets = []
    valid_pids_found = []
    for pid in pids:
        if pid in pid_samples_common:
            valid_pids_found.append(pid)
            pid_files_lin = pid_samples_common[pid]['lineage']
            pid_files_chemo = pid_samples_common[pid]['chemo']
            pid_input_props = []
            pid_input_emb = []
            pid_class_labels = []
            for ti in range(len(pid_files_lin)):
                emb_lin, ct_lin = filteredValues(pid_files_lin[ti],
                                                    embedding=dataLineage['vals'],
                                                    clsf=ctype_lin, valid=valid_classes_lin,
                                                    doublets=doublets_lin)
                if len(ct_lin) == 0:
                    props_lin = [0.0] * len(valid_classes_lin)
                else:
                    props_lin = [np.sum(ct_lin == ct)/len(ct_lin) for ct in valid_classes_lin]
                emb_chemo, ct_chemo = filteredValues(pid_files_chemo[ti],
                                                     embedding=dataChemo['vals'],
                                                     clsf=ctype_chemo, valid=valid_classes_chemo,
                                                     doublets=doublets_chemo)
                if len(ct_chemo) == 0:
                    props_chemo = [0.0] * len(valid_classes_chemo)
                else:
                    props_chemo = [np.sum(ct_chemo == ct)/len(ct_chemo) for ct in valid_classes_chemo]
                pid_input_props.append([props_lin, props_chemo])
                pid_input_emb.append([emb_lin, emb_chemo])
                pid_class_labels.append([ct_lin, ct_chemo])
            inputs_props.append(pid_input_props)
            inputs_emb.append(pid_input_emb)
            class_labels.append(pid_class_labels)
            targets.append(severityEmbedding[pid_sevs_all[pid]])
    return {'props': inputs_props, 'vals': inputs_emb, 'labels': class_labels, 'targets': targets, 'pids': valid_pids_found}


# `max_timesteps` must be redefined for each fold's dataset.
# It is declared here to make it accessible to the helper functions below.
max_timesteps = 0

# define architecture which takes sets of cell embeddings as input
def setInputModel(input_shapes, n_filters=50, n_embed=1, n_hidden=0, rnn=False):
    # sub-model for processing sets of cells
    def set_conv_pool(input_shape, n_filters, n_embed):
        inp = out = keras.layers.Input(input_shape)
        for _ in range(n_embed):
            out = keras.layers.Conv1D(filters=n_filters, kernel_size=1, activation='relu')(out)
        out_avg = keras.layers.GlobalAveragePooling1D(data_format="channels_last")(out)
        out_max = keras.layers.GlobalMaxPooling1D(data_format="channels_last")(out)
        # concatenate and max- and average-pool embeddings per sample
        out = keras.layers.Concatenate(axis=-1)([out_avg, out_max])
        return keras.Model(inp, out)

    inputs = []
    outs = []
    # first each input type (lineage or chemo) is passed through a separate embedding
    for i_inp, inp_shape in enumerate(input_shapes):
        inp = out = keras.layers.Input(inp_shape)
        inputs.append(inp)
        # zero padding is used for variable number of samples in time
        out = keras.layers.Masking(mask_value=0.0, input_shape=inp_shape)(out)
        inner_model = set_conv_pool(inp_shape[1:], n_filters, n_embed)
        out = keras.layers.TimeDistributed(inner_model)(out)
        outs.append(out)
    # concatenate inputs if more than one
    if len(outs) > 1:
        out = keras.layers.Concatenate(axis=-1)(outs)
    else:
        out = outs[0]
    for n_h in range(n_hidden):
        out = keras.layers.TimeDistributed(keras.layers.Dense(units=n_filters, activation='relu'))(out)

    if rnn:
        # pool timesteps with RNN
        out = keras.layers.SimpleRNN(units=n_filters, return_sequences=False)(out)
        out = keras.layers.Dense(units=1, activation='linear')(out)
    else:
        out = keras.layers.TimeDistributed(keras.layers.Dense(units=1, activation='linear'))(out)
        # pool timesteps with max (predicting maximum severity)
        out = keras.layers.GlobalMaxPooling1D()(out)
    model = keras.Model(inputs, out)
    return model

# define architecture which takes vectors of class proportions as input
def proportionsModel(input_shape, n_filters=50, n_hidden=0, rnn=False):
    inp = out = keras.layers.Input(input_shape)
    # zero padding is used for variable number of samples in time
    out = keras.layers.Masking(mask_value=0., input_shape=input_shape)(out)
    for n_h in range(n_hidden):
        out = keras.layers.TimeDistributed(keras.layers.Dense(units=n_filters, activation='relu'))(out)
    if rnn:
        # pool timesteps with RNN
        out = keras.layers.SimpleRNN(units=n_filters, return_sequences=False)(out)
        out = keras.layers.Dense(units=1, activation='linear')(out)
    else:
        out = keras.layers.TimeDistributed(keras.layers.Dense(units=1, activation='linear'))(out)
        # pool timesteps with max (predicting maximum severity)
        out = keras.layers.GlobalMaxPool1D()(out)
    model = keras.Model(inp, out)
    return model

# upsample training data to equalise samples per severity type
def upsampleSeverity(Xs:list, Y, upsample):
    existing_counts = [np.sum(Y == se) for se in severityEmbedding.values()]
    upsample_to = np.max(existing_counts) * upsample
    up_inds = []
    for se in severityEmbedding.values():
        inds = np.arange(len(Y))[Y == se]
        if len(inds):
            upsampled_inds = np.random.randint(0, len(inds), upsample_to)
            up_inds.extend(list(inds[upsampled_inds]))
    permuted = np.random.permutation(np.array(up_inds, dtype=int))
    return [np.array(x)[permuted] for x in Xs], np.array(Y)[permuted]

# function to get a sample of cell embeddings from data
def getEmbeddings(inputs, targets, useLineage, useChemo, n_samples, poolTime, upsample=0, classLabels=None):
    X_lin = []
    X_chemo = []
    labels_lin = []
    labels_chemo = []
    Y = []
    for i_pid, t_ins in enumerate(inputs): # for each PID
        pid_ins_lin = []
        pid_ins_chemo = []
        pid_labels_lin = []
        pid_labels_chemo = []
        for i_t, inp in enumerate(t_ins): # for each timepoint in PID
            # Handle cases where a sample has 0 valid cells to prevent sampling errors.
            # Create a zero-array placeholder which will be handled by the model's Masking layer.
            if len(inp[0]) > 0:
                lin_rand = np.random.randint(0, len(inp[0]), n_samples)
                sampled_lin_emb = inp[0][lin_rand]
                if classLabels is not None:
                    sampled_lin_labels = classLabels[i_pid][i_t][0][lin_rand]
            else:
                # Get the number of markers from the shape (0, n_markers)
                n_markers_lin = inp[0].shape[1]
                sampled_lin_emb = np.zeros((n_samples, n_markers_lin))
                if classLabels is not None:
                    sampled_lin_labels = np.repeat('empty', n_samples) # Placeholder labels
            pid_ins_lin.append(sampled_lin_emb)

            if len(inp[1]) > 0:
                chemo_rand = np.random.randint(0, len(inp[1]), n_samples)
                sampled_chemo_emb = inp[1][chemo_rand]
                if classLabels is not None:
                    sampled_chemo_labels = classLabels[i_pid][i_t][1][chemo_rand]
            else:
                n_markers_chemo = inp[1].shape[1]
                sampled_chemo_emb = np.zeros((n_samples, n_markers_chemo))
                if classLabels is not None:
                    sampled_chemo_labels = np.repeat('empty', n_samples) # Placeholder labels
            pid_ins_chemo.append(sampled_chemo_emb)
            
            if classLabels is not None:
                pid_labels_lin.append(sampled_lin_labels)
                pid_labels_chemo.append(sampled_chemo_labels)

        if poolTime is not None: # model will pool timepoints of each PID
            while len(pid_ins_lin) < max_timesteps: # pad time with zeros
                pid_ins_lin.append(np.zeros(pid_ins_lin[0].shape))
                pid_ins_chemo.append(np.zeros(pid_ins_chemo[0].shape))
                if classLabels is not None:
                    pid_labels_lin.append(np.zeros(len(pid_labels_lin[0])))
                    pid_labels_chemo.append(np.zeros(len(pid_labels_chemo[0])))
            X_lin.append(pid_ins_lin)
            X_chemo.append(pid_ins_chemo)
            labels_lin.append(pid_labels_lin)
            labels_chemo.append(pid_labels_chemo)
            Y.append(targets[i_pid])
        else: # use each timepoint separately
            for ti in range(len(pid_ins_lin)):
                X_lin.append([pid_ins_lin[ti]])
                X_chemo.append([pid_ins_chemo[ti]])
                labels_lin.append([pid_labels_lin[ti]])
                labels_chemo.append([pid_labels_chemo[ti]])
                Y.append(targets[i_pid])
    Y = np.array(Y)
    if useLineage and useChemo:
        Xs = [np.array(X_lin), np.array(X_chemo)]
    else:
        if useLineage:
            Xs = [np.array(X_lin)]
        if useChemo:
            Xs = [np.array(X_chemo)]
    if classLabels is not None:
        if useLineage:
            Xs.append(np.array(labels_lin))
        if useChemo:
            Xs.append(np.array(labels_chemo))
    if upsample == 0:
        return Xs, Y
    else:
        return upsampleSeverity(Xs, Y, upsample)

# repeat prediction multiple times with different random samples of cells and return average
def averagePrediction(model, inputs, targets, reps=100):
    preds = []
    for rep in range(reps):
        X, Y = getEmbeddings(inputs=inputs,
                               targets=targets,
                               useLineage=params['useLineage'], useChemo=params['useChemo'],
                               poolTime=params['poolTime'],
                               n_samples=params['sampleInput'])
        y = model.predict(X, verbose=0)
        preds.append(y)
    return np.mean(preds, axis=0)

# function to get concatenated class proportions as inputs from data
def getProportions(inputs, targets, useLineage, useChemo, poolTime, upsample=0):
    X = []
    Y = []
    for i_pid, t_ins in enumerate(inputs): # for each PID
        pid_ins = []
        for inp in t_ins: # for each timepoint in PID
            if useLineage and useChemo: # concatenate vectors from lineage and chemo
                pid_ins.append(np.concatenate([inp[0], inp[1]]))
            else: # use only one proportion vector
                if useLineage:
                    pid_ins.append(inp[0])
                if useChemo:
                    pid_ins.append(inp[1])
        if poolTime is not None: # model will pool timepoints of each PID
            while len(pid_ins) < max_timesteps: # pad time with zeros
                pid_ins.append(np.zeros(len(pid_ins[0])))
            X.append(pid_ins)
            Y.append(targets[i_pid])
        else: # use each timepoint separately
            for x in pid_ins:
                X.append([x])
                Y.append(targets[i_pid])
    X = np.array(X)
    Y = np.array(Y)
    if upsample == 0:
        return X, Y
    else:
        Xs_up, Y_up = upsampleSeverity([X], Y, upsample)
        return Xs_up[0], Y_up

def mse(y, Y):
    return np.mean(np.square(np.squeeze(y) - np.squeeze(Y)))

def f1Metrics(gt, pred):
    nclasses = int(np.max(gt) + 1)
    matrix = np.zeros((nclasses, nclasses), dtype=int)
    for n in range(len(pred)):
        matrix[int(pred[n]), int(gt[n])] += 1
    totalTrue = np.sum(matrix, axis=0)
    totalPredicted = np.sum(matrix, axis=1)
    tp = np.array(np.diag(matrix), dtype=float)
    precision = np.divide(tp, totalPredicted)
    recall = np.divide(tp, totalTrue)
    f1s = (2 * precision * recall) / (precision + recall)
    f1s = np.nan_to_num(f1s)
    totalCorrect = np.sum(tp)
    totalEntries = np.sum(matrix)
    accuracy = totalCorrect / totalEntries

    return f1s, accuracy

# function to score the regression output as classification of severity types
def metricsFromRegression(gt, pred, val_loss, saveAsCsvPath=None):
    gt = np.squeeze(gt) - 1.0
    pred = np.squeeze(pred) - 1.0
    loss = mse(pred, gt)
    gt_rounded = np.clip(np.round(gt), 0, len(valid_sevs) - 1)
    pred_rounded = np.clip(np.round(pred), 0, len(valid_sevs) - 1)
    f1s, accuracy = f1Metrics(gt_rounded, pred_rounded)
    if saveAsCsvPath is not None:
        if fileExists(saveAsCsvPath):
            rows = csvFile(saveAsCsvPath, remNewline=True)
        else:
            rows = []
        rows.append(list(f1s) + [np.mean(f1s), accuracy, loss, val_loss])
        saveAsCsv(rows, saveAsCsvPath)

# train model of a specified type, or load if already trained
def trainedModel(X, Y, vX, vY, modelPath,
                 useLineage=True, useChemo=True,
                 sampleInput:int=None,  # if using raw embedding specify number of cells to sample per timepoint
                 poolTime:str=None):
    if sampleInput is None:
        X_val, Y_val = getProportions(vX, vY, useLineage=useLineage, useChemo=useChemo,
                                      poolTime=poolTime, upsample=0)
    else:
        X_val, Y_val = getEmbeddings(vX, vY, useLineage=useLineage, useChemo=useChemo,
                                     n_samples=sampleInput, poolTime=poolTime, upsample=0)

    if fileExists(modelPath):
        model = keras.models.load_model(modelPath)
    else:
        if sampleInput is None:
            X_train, Y_train = getProportions(X, Y, useLineage=useLineage, useChemo=useChemo,
                                        poolTime=poolTime, upsample=params['upsample'])
            model = proportionsModel(X_train[0].shape, rnn=(poolTime=='rnn'),
                                     n_filters=params['nFilters'], n_hidden=params['nHidden'])
        else:
            n_inputs = len(X_val)
            in_shapes = []
            for n_in in range(n_inputs):
                in_shapes.append(X_val[n_in][0].shape)
            model = setInputModel(in_shapes, rnn=(poolTime == 'rnn'),
                                  n_embed=params['embDepth'], n_hidden=params['nHidden'], n_filters=params['nFilters'])

        model.compile(optimizer='rmsprop', loss='mse')
        model.summary()

        best_val = None
        model_weights = model.get_weights()
        p_count = params['patience']
        for epoch in range(params['maxEpochs']):
            if sampleInput is None:
                h = model.fit(X_train, Y_train, batch_size=params['batchSize'], epochs=1, verbose=0)
                y_val = model.predict(X_val, verbose=0)
            else:
                X_train, Y_train = getEmbeddings(X, Y, useLineage=useLineage, useChemo=useChemo,
                                           n_samples=sampleInput, poolTime=poolTime, upsample=params['upsample'])
                h = model.fit(X_train, Y_train, batch_size=params['batchSize'], epochs=1, verbose=0)
                y_val = averagePrediction(model, vX, vY, reps=10)
            val_loss = mse(y_val, Y_val)
            if epoch % 50 == 0:
                print('Epoch {}, loss: {}, valLoss: {}'.format(epoch, h.history['loss'][0], val_loss))
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                model_weights = model.get_weights()
                p_count = params['patience']
            else:
                p_count -= 1
                if p_count == 0:
                    break
        model.set_weights(model_weights)
        model.save(modelPath)

    if sampleInput is None:
        y_val = model.predict(X_val, verbose=0)
    else:
        y_val = averagePrediction(model, vX, vY)
    val_loss = mse(y_val, Y_val)
    print('Val loss:', val_loss)
    return model, val_loss


# load first fold to get the sample information (it's the same across all folds)
dataLineage = unpickle(f'lineage-split/0/embs-split/lineage-0.pkl')
dataChemo = unpickle(f'chemokine-split/0/embs-split/chemo-0.pkl')

markers = {'lineage': dataLineage['channels'],
            'chemo': dataChemo['channels']}

if type(markers['lineage']) is dict:
    markers['lineage'] = list(markers['lineage'].values())[0]
if type(markers['chemo']) is dict:
    markers['chemo'] = list(markers['chemo'].values())[0]

pid_samples_lin, pid_times_lin, pid_sevs_lin = groupPidData(dataLineage)
pid_samples_chemo, pid_times_chemo, pid_sevs_chemo = groupPidData(dataChemo)
pid_sevs_all = dict(pid_sevs_lin)
pid_sevs_all.update(pid_sevs_chemo)

# Find samples that have both types of panels for each timepoint
common_pids = [pid for pid in pid_samples_lin if pid in pid_samples_chemo]
pid_samples_common = {}
pid_times_missing_chemo = {}
for pid in pid_samples_lin:
    times_lin = pid_times_lin[pid]
    missing_times_chemo = []
    if pid not in pid_samples_chemo:
        missing_times_chemo.extend(times_lin)
    else:
        times_chemo = pid_times_chemo[pid]
        matching_times = []
        matching_files_lin = []
        matching_files_chemo = []
        for t in times_lin:
            if t in times_chemo:
                matching_times.append(t)
                matching_files_lin.append(pid_samples_lin[pid][int(times_lin.index(t))])
                matching_files_chemo.append(pid_samples_chemo[pid][int(times_chemo.index(t))])
            else:
                missing_times_chemo.append(t)
        if len(matching_times):
            pid_samples_common[pid] = {'lineage': matching_files_lin, 'chemo': matching_files_chemo, 'times': matching_times}
    if len(missing_times_chemo):
        pid_times_missing_chemo[pid] = missing_times_chemo

max_timesteps = 0
if pid_samples_common:
    max_timesteps = np.max([len(v['times']) for v in pid_samples_common.values()])

valid_pid_fold_split = {fold_id: [pid for pid in pid_samples_common if pid in pid_fold_split[fold_id]] for fold_id in pid_fold_split}

# store predictions grouped by severity and class
models = []
targets = []
predictions = []
prediction_pids = []
val_losses = []

# The main cross-validation loop
for fold_id in sorted(list(pid_fold_split.keys())):
    print('\n' + '='*20 + f' Processing Test Fold {fold_id} ' + '='*20)
    # Load the specific dataset for this fold. This dataset was created by holding out
    # the test patients defined in pid_fold_split[fold_id].
    dataLineage = unpickle(f'lineage-split/{fold_id}/embs-split/lineage-{fold_id}.pkl')
    dataChemo = unpickle(f'chemokine-split/{fold_id}/embs-split/chemo-{fold_id}.pkl')

    ctype_lin = dataLineage['ctypes']['prediction']
    ctype_chemo = dataChemo['ctypes'][params['chemoType']]
    doublets_lin = dataLineage['ctypes']['db_pred']
    doublets_chemo = dataChemo['ctypes']['db_pred']

    valid_classes_lin = ['B cells', 'Basophils', 'DC cells', 'Eosinophils', 'Monocytes', 'NK cells', 'Neutrophils', 'T cells']
    valid_classes_chemo = sorted(list(set(np.concatenate(list(ctype_chemo.values())))))
    valid_classes_chemo = [c for c in valid_classes_chemo if c not in ['Debris', 'Other']]

    # Prepare test set for this fold
    test_pids = valid_pid_fold_split[fold_id]
    test_values = extract_pid_data(test_pids, pid_samples_common, pid_sevs_all, dataLineage, ctype_lin,
                                   valid_classes_lin, doublets_lin, dataChemo, ctype_chemo, valid_classes_chemo, doublets_chemo)

    # Train an ensemble of models using the remaining folds for train/validation splits
    validation_fold_keys = [f_ii for f_ii in pid_fold_split if f_ii != fold_id]
    fold_predictions = []
    fold_models = []
    fold_val_losses = []

    for f_val_key in validation_fold_keys:
        print('Test fold {} (validation fold {})'.format(fold_id, f_val_key))
        # Get PIDs for validation set
        val_pids = pid_fold_split[f_val_key]
        # Get PIDs for training set
        train_keys = [k for k in validation_fold_keys if k != f_val_key]
        training_pids = np.concatenate([pid_fold_split[k] for k in train_keys]) if train_keys else []

        # Extract data for train/validation sets
        training_data = extract_pid_data(training_pids, pid_samples_common, pid_sevs_all, dataLineage, ctype_lin,
                                   valid_classes_lin, doublets_lin, dataChemo, ctype_chemo, valid_classes_chemo, doublets_chemo)
        validation_data = extract_pid_data(val_pids, pid_samples_common, pid_sevs_all, dataLineage, ctype_lin,
                                   valid_classes_lin, doublets_lin, dataChemo, ctype_chemo, valid_classes_chemo, doublets_chemo)

        Y = training_data['targets']
        vY = validation_data['targets']
        if params['sampleInput'] is None: # train model on proportions
            X = training_data['props']
            vX = validation_data['props']
        else: # train model on raw inputs (sets of marker values)
            X = training_data['vals']
            vX = validation_data['vals']

        # train model with early stopping
        model, val_loss = trainedModel(X=X, Y=Y, vX=vX, vY=vY,
                                     modelPath=modelFolder+'fold{}-{}'.format(fold_id, f_val_key),
                                     useLineage=params['useLineage'],
                                     useChemo=params['useChemo'],
                                     sampleInput=params['sampleInput'],
                                     poolTime=params['poolTime'])
        fold_val_losses.append(val_loss)

        # predict on test data and accumulate ensemble predictions
        if params['sampleInput'] is None: # proportions model
            X_test, Y_test = getProportions(inputs=test_values['props'],
                                            targets=test_values['targets'],
                                            useLineage=params['useLineage'], useChemo=params['useChemo'], poolTime=params['poolTime'])
            y_test = model.predict(X_test)
            fold_predictions.append(y_test)
        else: # set model
            y_test = averagePrediction(model, test_values['vals'], test_values['targets'])
            fold_predictions.append(y_test)

        fold_models.append(model)
    models.append(fold_models)
    mean_fold_pred = np.mean(fold_predictions, axis=0)
    predictions.append(mean_fold_pred)
    targets.append(test_values['targets'])
    val_losses.extend(fold_val_losses)
    prediction_pids.append(test_values['pids'])
    # show and save scores for each fold
    print(f'--- Results for Test Fold {fold_id} ---')
    metricsFromRegression(test_values['targets'],
                          mean_fold_pred,
                          np.mean(fold_val_losses),
                          saveAsCsvPath=None)

# show and save combined scores
print('\n' + '='*20 + ' Final Combined Results ' + '='*20)
print('Val losses:', val_losses)
print('Avg val loss:', np.mean(val_losses))
print('Test error:')
metricsFromRegression(np.concatenate(targets, axis=0),
                      np.concatenate(predictions, axis=0),
                      np.mean(val_losses),
                      saveAsCsvPath=resultsFolder+modelName+'.csv')