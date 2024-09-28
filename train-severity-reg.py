import re
import tensorflow as tf
import tensorflow.keras as keras
from src.tools import *

dataLineage = unpickle('lineage/embs/lineage.pkl')
dataChemo = unpickle('chemokine/embs/chemo.pkl')
outFolder = ensureFolder('severity-reg/')

all_fnames_lineage = sorted(list(dataLineage['embs'].keys()))
all_fnames_chemo = sorted(list(dataChemo['embs'].keys()))

valid_sevs = ['Healthy', 'Mild', 'Moderate', 'Severe']
severityEmbedding = {'Healthy': 0.5, 'Mild': 2, 'Moderate': 3, 'Severe': 4.5}

valid_types = ['Control', 'Acute']

gradientAttribution = False
nonzero_grads = False
group_grads_by_labels = False

markers = {'lineage': dataLineage['channels'],
           'chemo': dataChemo['channels']}

if type(markers['lineage']) is dict:
    markers['lineage'] = list(markers['lineage'].values())[0]
if type(markers['chemo']) is dict:
    markers['chemo'] = list(markers['chemo'].values())[0]

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

]

print('n. configs:', len(configs))

#config_ind = int(os.getenv('SGE_TASK_ID')) - 1 # different IDs are provided by external batch script
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

# define valid classes and plot colors for each class
ctype_lin = dataLineage['ctypes']['prediction']
ctype_chemo = dataChemo['ctypes'][params['chemoType']]
doublets_lin = dataLineage['ctypes']['db_pred']
doublets_chemo = dataChemo['ctypes']['db_pred']

valid_classes_lin = ['B cells', 'Basophils', 'DC cells', 'Eosinophils', 'Monocytes', 'NK cells', 'Neutrophils', 'T cells']
valid_classes_chemo = sorted(list(set(np.concatenate(list(ctype_chemo.values())))))
valid_classes_chemo = [c for c in valid_classes_chemo if c not in ['Debris', 'Other']]

# group files by patient ID and determine associated sample files, times and severities
def groupPidData(data):
    sample_pids = {fn: pidMap[fn] for fn in data['embs']}
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

# get pid samples from lineage and chemo panel
pid_samples_lin, pid_times_lin, pid_sevs_lin = groupPidData(dataLineage)
pid_samples_chemo, pid_times_chemo, pid_sevs_chemo = groupPidData(dataChemo)
pid_sevs_all = dict(pid_sevs_lin)
pid_sevs_all.update(pid_sevs_chemo)

all_pids = np.array(list(set(list(pid_samples_lin.keys()) + list(pid_samples_chemo.keys()))))

# find samples that have both lineage and chemo files for the same time point
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

lin_s, lin_sc = np.unique(list(pid_sevs_lin.values()), return_counts=True)
print('PID severities lineage: ')
print(lin_s, lin_sc)

ch_s, ch_sc = np.unique(list(pid_sevs_chemo.values()), return_counts=True)
print('PID severities chemo: ')
print(ch_s, ch_sc)

pid_common_sevs = [pid_sevs_all[pid] for pid in pid_samples_common]
s, c = np.unique(pid_common_sevs, return_counts=True)
print('PID severities common: ')
print(s, c)

total_lin_samples = np.sum([len(v) for v in pid_samples_lin])
total_chemo_samples = np.sum([len(v) for v in pid_samples_chemo])
total_common_samples = np.sum([len(v['times']) for v in pid_samples_common.values()])

# split PIDs randomly into folds
pid_split_file = outFolder + 'pid_split_{}.pkl'.format(params['nFolds'])
if not fileExists(pid_split_file):
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
    if clsf is None:
        return emb[mask]
    else:
        c = clsf[fn]
        if valid is not None:
            mask = np.logical_and(mask, np.isin(c, valid))
        return emb[mask], c[mask]

# get the input proportions and marker values for each fold
split_values = {}
for fold_id, pids in pid_fold_split.items():
    inputs_props = []
    inputs_emb = []
    class_labels = []
    targets = []
    valid_pids = []
    for pid in pids:
        if pid in pid_samples_common:
            valid_pids.append(pid)
            pid_files_lin = pid_samples_common[pid]['lineage']
            pid_files_chemo = pid_samples_common[pid]['chemo']
            pid_input_props = []
            pid_input_emb = []
            pid_class_labels = []
            for ti in range(len(pid_files_lin)):
                emb_lin, ct_lin = filteredValues(pid_files_lin[ti],
                                                    embedding=dataLineage['embs'],
                                                    clsf=ctype_lin, valid=valid_classes_lin,
                                                    doublets=doublets_lin)
                props_lin = [np.sum(ct_lin == ct)/len(ct_lin) for ct in valid_classes_lin]
                emb_chemo, ct_chemo = filteredValues(pid_files_chemo[ti],
                                                     embedding=dataChemo['embs'],
                                                     clsf=ctype_chemo, valid=valid_classes_chemo,
                                                     doublets=doublets_chemo)
                props_chemo = [np.sum(ct_chemo == ct)/len(ct_chemo) for ct in valid_classes_chemo]
                pid_input_props.append([props_lin, props_chemo])
                pid_input_emb.append([emb_lin, emb_chemo])
                pid_class_labels.append([ct_lin, ct_chemo])
            inputs_props.append(pid_input_props)
            inputs_emb.append(pid_input_emb)
            class_labels.append(pid_class_labels)
            targets.append(severityEmbedding[pid_sevs_all[pid]])
    split_values[fold_id] = {'props': inputs_props, 'embs': inputs_emb, 'labels': class_labels, 'targets': targets, 'pids': valid_pids}

# check the maximum number of timepoints in the data
max_timesteps = np.max([np.max([len(pid_ins) for pid_ins in fold_ins['props']]) for fold_ins in split_values.values()])

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
            lin_rand = np.random.randint(0, len(inp[0]), n_samples)
            pid_ins_lin.append(inp[0][lin_rand]) # sample random set of lineage cells
            chemo_rand = np.random.randint(0, len(inp[1]), n_samples)
            pid_ins_chemo.append(inp[1][chemo_rand]) # sample random set of chemo cells
            # get cell-type labels for attribution:
            if classLabels is not None:
                pid_labels_lin.append(classLabels[i_pid][i_t][0][lin_rand])
                pid_labels_chemo.append(classLabels[i_pid][i_t][1][chemo_rand])
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

# function to obtain gradients split by severity type and cell-type
def attributePrediction(model, inputs, targets, labels, reps=10):
    grad_acc_sev = {}
    n_inp = 0
    if params['useLineage']:
        grad_acc_sev['lineage'] = {sev: {cl: [] for cl in valid_classes_lin} for sev in valid_sevs}
        n_inp += 1
    if params['useChemo']:
        grad_acc_sev['chemo'] = {sev: {cl: [] for cl in valid_classes_chemo} for sev in valid_sevs}
        n_inp += 1

    for rep in range(reps):
        X, Y = getEmbeddings(inputs=inputs,
                             targets=targets,
                             useLineage=params['useLineage'], useChemo=params['useChemo'],
                             poolTime=params['poolTime'],
                             n_samples=params['sampleInput'],
                             classLabels=labels)

        inps = X[:n_inp]
        labs = X[n_inp:]

        inps = [tf.convert_to_tensor(inp) for inp in inps]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inps)
            out = model(inps)
        # get predicted categories from outputs
        out_rounded = np.clip(np.round(out), 0, len(valid_sevs) - 1)
        out_rounded = np.array([o[0] for o in out_rounded], dtype=int)
        for n_i, inp in enumerate(inps):
            # get gradients
            grads = np.array(tape.gradient(out, inp))
            for i_sev, sev in enumerate(valid_sevs):
                # get gradients for each severity type
                if group_grads_by_labels:
                    pid_mask = Y == severityEmbedding[sev]
                else:
                    pid_mask = out_rounded == i_sev
                if np.any(pid_mask):
                    grads_sev = grads[pid_mask]
                    total_cell_dim = np.product(grads_sev.shape[0:3])
                    # flatten valid samples
                    grads_rs = np.reshape(grads_sev, (total_cell_dim, inp.shape[-1]))
                    grads_valid = grads_rs
                    classes_valid = np.reshape(labs[n_i][pid_mask], (total_cell_dim,))
                    if nonzero_grads:
                        # get non-zero gradients
                        mask = np.any(grads_rs != 0, axis=-1)
                        grads_valid = grads_valid[mask]
                        classes_valid = classes_valid[mask]
                    # split gradients per cell-type
                    for cl in list(grad_acc_sev.values())[n_i][sev]:
                        list(grad_acc_sev.values())[n_i][sev][cl].extend(grads_valid[classes_valid == cl])
    return grad_acc_sev

# function to obtain gradients for the entire sample
def sampleGradients(model, inputs, targets, reps=10):
    grad_acc = {}
    n_inp = 0
    if params['useLineage']:
        grad_acc['lineage'] = [[] for _ in inputs]
        n_inp += 1
    if params['useChemo']:
        grad_acc['chemo'] = [[] for _ in inputs]
        n_inp += 1

    for rep in range(reps):
        X, Y = getEmbeddings(inputs=inputs,
                             targets=targets,
                             useLineage=params['useLineage'], useChemo=params['useChemo'],
                             poolTime=params['poolTime'],
                             n_samples=params['sampleInput'],
                             classLabels=None)

        inps = X[:n_inp]
        inps = [tf.convert_to_tensor(inp) for inp in inps]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inps)
            out = model(inps)
        for n_i, inp in enumerate(inps):
            grads = np.array(tape.gradient(out, inp))
            for i_samp, g_samp in enumerate(grads):
                total_cell_dim = np.product(g_samp.shape[0:2])
                grads_rs = np.reshape(g_samp, (total_cell_dim, inp.shape[-1]))
                if nonzero_grads:
                    mask = np.any(grads_rs != 0, axis=-1)
                    grads_valid = grads_rs[mask]
                else:
                    grads_valid = grads_rs
                list(grad_acc.values())[n_i][i_samp].extend(grads_valid)
    return grad_acc


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

# store predictions and gradients grouped by severity and class
models = []
targets = []
predictions = []
val_losses = []
gradients_sev = []

# filter only specified samples for plotting
def pidsFilter(severity, minTimes=0):
    matching = [pid for pid in pid_samples_common if pid_sevs_all[pid] == severity]
    matching = [pid for pid in matching if len(pid_samples_common[pid]['lineage']) > minTimes]
    return matching

if gradientAttribution:
    plot_min_timesteps = 3
    pids_to_plot = pidsFilter('Healthy')
    pids_to_plot.extend(pidsFilter('Mild', plot_min_timesteps))
    pids_to_plot.extend(pidsFilter('Moderate', plot_min_timesteps))
    pids_to_plot.extend(pidsFilter('Severe', plot_min_timesteps))
else:
    pids_to_plot = []

# prepare data structure for holding plot information
pid_plot_vals = {}
for pid in pids_to_plot:
    pid_plot_vals[pid] = {}
    for t in pid_samples_common[pid]['times']:
        pid_plot_vals[pid][t] = {'pred': [], 'props': {}, 'embs': {}, 'grads': {'lineage': [], 'chemo': []}}

# go over data folds, train models on remaining data, predict and attribute
for fold_id, test_values in split_values.items():
    validation_folds = [f_ii for f_ii in split_values if f_ii != fold_id]
    fold_predictions = []
    fold_models = []
    fold_val_losses = []
    for f_val in validation_folds:
        print('Fold {} (validation {})'.format(fold_id, f_val))
        # concatenate training folds
        training_props = []
        training_embs = []
        training_targets = []
        for f_train in validation_folds:
            if f_train != f_val:
                training_props.extend(split_values[f_train]['props'])
                training_embs.extend(split_values[f_train]['embs'])
                training_targets.extend(split_values[f_train]['targets'])
        Y = training_targets
        vY = split_values[f_val]['targets']
        if params['sampleInput'] is None: # train model on proportions
            X = training_props
            vX = split_values[f_val]['props']
        else: # train model on raw inputs (sets of marker values)
            X = training_embs
            vX = split_values[f_val]['embs']
        # train model with early stopping
        model, val_loss = trainedModel(X=X, Y=Y, vX=vX, vY=vY,
                                     modelPath=modelFolder+'fold{}-{}'.format(fold_id, f_val),
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
            y_test = averagePrediction(model, test_values['embs'], test_values['targets'])
            fold_predictions.append(y_test)

            if gradientAttribution:
                # perform gradient attribution
                # get gradients split by input, severity and class
                grads_sev = attributePrediction(model, test_values['embs'], test_values['targets'], test_values['labels'])
                gradients_sev.append(grads_sev)

            # get predictions and gradients for each timepoint by making single-timestep samples (to prevent temporal maxpooling)
            for i_pid, pid in enumerate(test_values['pids']):
                if pid not in pids_to_plot:
                    continue
                pid_embs = test_values['embs'][i_pid]
                pid_props = test_values['props'][i_pid]
                # expand timepoints into separate samples
                pid_embs_expand = [[e] for e in pid_embs]
                pid_target_expand = [test_values['targets'][i_pid]] * len(pid_embs)
                # predict each timepoint as if it were separate sample
                exp_pred = averagePrediction(model, pid_embs_expand, pid_target_expand)
                # get input attribution for each timepoint
                exp_grad = sampleGradients(model, pid_embs_expand, pid_target_expand)
                # save values for plots
                for t_i, pt in enumerate(pid_plot_vals[pid]):
                    pid_plot_vals[pid][pt]['props']['lineage'] = pid_props[t_i][0]
                    pid_plot_vals[pid][pt]['props']['chemo'] = pid_props[t_i][1]
                    pid_plot_vals[pid][pt]['embs']['lineage'] = pid_embs[t_i][0]
                    pid_plot_vals[pid][pt]['embs']['chemo'] = pid_embs[t_i][1]
                    pid_plot_vals[pid][pt]['pred'].append(exp_pred[t_i][0])
                    if 'lineage' in exp_grad:
                        pid_plot_vals[pid][pt]['grads']['lineage'].extend(list(exp_grad['lineage'][t_i]))
                    if 'chemo' in exp_grad:
                        pid_plot_vals[pid][pt]['grads']['chemo'].extend(list(exp_grad['chemo'][t_i]))

        fold_models.append(model)
    mean_fold_pred = np.mean(fold_predictions, axis=0)
    predictions.append(mean_fold_pred)
    targets.append(test_values['targets'])
    val_losses.extend(fold_val_losses)
    # show and save scores for each fold
    metricsFromRegression(test_values['targets'],
                          mean_fold_pred,
                          np.mean(fold_val_losses),
                          saveAsCsvPath=resultsFolder + modelName + '.csv')

# show and save combined scores
print('Val losses:', val_losses)
print('Avg val loss:', np.mean(val_losses))
print('Test error:')
metricsFromRegression(np.concatenate(targets, axis=0),
                      np.concatenate(predictions, axis=0),
                      np.mean(val_losses),
                      saveAsCsvPath=resultsFolder+modelName+'.csv')
