testConfigs = [{'GT-VAE': True},
               {'worst-VAE': True, 'rand': False},
               {'worst-VAE': True, 'rand': True},

# no resampling
# sub, mmd, sub+mmd
{'cond': False, 'norm': False, 'mmd': False, 'sub': True, 'resample': False}, #3
{'cond': False, 'norm': False, 'mmd': True, 'sub': False, 'resample': False},
{'cond': False, 'norm': False, 'mmd': True, 'sub': True, 'resample': False},
# sub, mmd, sub+mmd | with norm
{'cond': False, 'norm': True, 'mmd': False, 'sub': True, 'resample': False},
{'cond': False, 'norm': True, 'mmd': True, 'sub': False, 'resample': False},
{'cond': False, 'norm': True, 'mmd': True, 'sub': True, 'resample': False},
# sub, mmd, sub+mmd | with cond
{'cond': True, 'norm': False, 'mmd': False, 'sub': True, 'resample': False},
{'cond': True, 'norm': False, 'mmd': True, 'sub': False, 'resample': False},
{'cond': True, 'norm': False, 'mmd': True, 'sub': True, 'resample': False},
# sub, mmd, sub+mmd | with cond and norm
{'cond': True, 'norm': True, 'mmd': False, 'sub': True, 'resample': False},
{'cond': True, 'norm': True, 'mmd': True, 'sub': False, 'resample': False},
{'cond': True, 'norm': True, 'mmd': True, 'sub': True, 'resample': False},

# resampling
# mmd, sub+mmd
{'cond': False, 'norm': False, 'mmd': True, 'sub': False, 'resample': True}, #15
{'cond': False, 'norm': False, 'mmd': True, 'sub': True, 'resample': True},
# sub, mmd, sub+mmd | with norm
{'cond': False, 'norm': True, 'mmd': False, 'sub': True, 'resample': True},
{'cond': False, 'norm': True, 'mmd': True, 'sub': False, 'resample': True},
{'cond': False, 'norm': True, 'mmd': True, 'sub': True, 'resample': True},
# mmd, sub+mmd | with cond
{'cond': True, 'norm': False, 'mmd': True, 'sub': False, 'resample': True},
{'cond': True, 'norm': False, 'mmd': True, 'sub': True, 'resample': True},
# sub, mmd, sub+mmd | with cond and norm
{'cond': True, 'norm': True, 'mmd': False, 'sub': True, 'resample': True},
{'cond': True, 'norm': True, 'mmd': True, 'sub': False, 'resample': True},
{'cond': True, 'norm': True, 'mmd': True, 'sub': True, 'resample': True},

# conditioning experiments
# linear conditioning channel
{'cond': True, 'norm': False, 'mmd': False, 'sub': True, 'resample': False}, #25
{'cond': True, 'norm': False, 'mmd': True, 'sub': False, 'resample': True},
{'cond': True, 'norm': False, 'mmd': True, 'sub': True, 'resample': True},
{'cond': True, 'norm': True, 'mmd': False, 'sub': True, 'resample': True},
{'cond': True, 'norm': True, 'mmd': True, 'sub': False, 'resample': True},
{'cond': True, 'norm': True, 'mmd': True, 'sub': True, 'resample': True},
# direct concatenation of one-hot conditioning vector
{'cond': 0, 'norm': False, 'mmd': False, 'sub': True, 'resample': False}, #31
{'cond': 0, 'norm': False, 'mmd': True, 'sub': False, 'resample': True},
{'cond': 0, 'norm': False, 'mmd': True, 'sub': True, 'resample': True},
{'cond': 0, 'norm': True, 'mmd': False, 'sub': True, 'resample': True},
{'cond': 0, 'norm': True, 'mmd': True, 'sub': False, 'resample': True},
{'cond': 0, 'norm': True, 'mmd': True, 'sub': True, 'resample': True},

# conditioning with mmd experiments
# no variational prior
{'cond': True, 'norm': False, 'mmd': True, 'sub': False, 'resample': True, 'no-vae': True}, #37
{'cond': True, 'norm': False, 'mmd': False, 'sub': True, 'resample': False, 'no-vae': True},
{'cond': True, 'norm': False, 'mmd': True, 'sub': True, 'resample': True, 'no-vae': True},
# mmd on batches, resampling, subspace
{'cond': True, 'norm': False, 'mmd': False, 'sub': True, 'resample': True, 'mmdb': True},
{'cond': True, 'norm': True, 'mmd': False, 'sub': True, 'resample': True, 'mmdb': True},
# mmd on batches, no resampling, subspace
{'cond': True, 'norm': False, 'mmd': False, 'sub': True, 'resample': False, 'mmdb': True},
{'cond': True, 'norm': True, 'mmd': False, 'sub': True, 'resample': False, 'mmdb': True},
# mmd on batches, resampling, no subspace
{'cond': True, 'norm': False, 'mmd': False, 'sub': False, 'resample': True, 'mmdb': True},
{'cond': True, 'norm': True, 'mmd': False, 'sub': False, 'resample': True, 'mmdb': True},
# mmd on batches, no resampling, no subspace
{'cond': True, 'norm': False, 'mmd': False, 'sub': False, 'resample': False, 'mmdb': True},
{'cond': True, 'norm': True, 'mmd': False, 'sub': False, 'resample': False, 'mmdb': True},

# unsupervised resampling
{'cond': False, 'norm': False, 'mmd': True, 'sub': True, 'resample': False, 'res-uns': 1},# 48
{'cond': False, 'norm': True, 'mmd': False, 'sub': True, 'resample': False, 'res-uns': 1},
{'cond': False, 'norm': True, 'mmd': True, 'sub': True, 'resample': False, 'res-uns': 1},
{'cond': True, 'norm': False, 'mmd': True, 'sub': True, 'resample': False, 'res-uns': 1},
{'cond': True, 'norm': True, 'mmd': False, 'sub': True, 'resample': False, 'res-uns': 1},
{'cond': True, 'norm': True, 'mmd': True, 'sub': True, 'resample': False, 'res-uns': 1},

{'cond': False, 'norm': False, 'mmd': True, 'sub': True, 'resample': False, 'res-uns': 2},
{'cond': False, 'norm': True, 'mmd': False, 'sub': True, 'resample': False, 'res-uns': 2},
{'cond': False, 'norm': True, 'mmd': True, 'sub': True, 'resample': False, 'res-uns': 2},
{'cond': True, 'norm': False, 'mmd': True, 'sub': True, 'resample': False, 'res-uns': 2},
{'cond': True, 'norm': True, 'mmd': False, 'sub': True, 'resample': False, 'res-uns': 2},
{'cond': True, 'norm': True, 'mmd': True, 'sub': True, 'resample': False, 'res-uns': 2},

{'cond': False, 'norm': False, 'mmd': True, 'sub': True, 'resample': False, 'res-uns': 3},
{'cond': False, 'norm': True, 'mmd': False, 'sub': True, 'resample': False, 'res-uns': 3},
{'cond': False, 'norm': True, 'mmd': True, 'sub': True, 'resample': False, 'res-uns': 3},
{'cond': True, 'norm': False, 'mmd': True, 'sub': True, 'resample': False, 'res-uns': 3},
{'cond': True, 'norm': True, 'mmd': False, 'sub': True, 'resample': False, 'res-uns': 3},
{'cond': True, 'norm': True, 'mmd': True, 'sub': True, 'resample': False, 'res-uns': 3},

{'cond': False, 'norm': False, 'mmd': True, 'sub': True, 'resample': False, 'res-uns-pan': True},# 66
{'cond': False, 'norm': True, 'mmd': False, 'sub': True, 'resample': False, 'res-uns-pan': True},
{'cond': False, 'norm': True, 'mmd': True, 'sub': True, 'resample': False, 'res-uns-pan': True},
{'cond': True, 'norm': False, 'mmd': True, 'sub': True, 'resample': False, 'res-uns-pan': True},
{'cond': True, 'norm': True, 'mmd': False, 'sub': True, 'resample': False, 'res-uns-pan': True},
{'cond': True, 'norm': True, 'mmd': True, 'sub': True, 'resample': False, 'res-uns-pan': True},
]#72

# names are constructed from parameters by adding them in this order:

param_order = ['GT-VAE', 'worst-VAE', 'rand', 'cond', 'norm', 'mmd', 'mmdb', 'sub', 'resample', 'res-uns', 'res-uns-pan', 'reg-sup', 'reg-uns', 'lisi-sup', 'no-ce', 'no-bal', 'no-vae']
def configName(config, ds_name):
    cf_key = '' + ds_name
    for k in param_order:
        if k in config:
            v = config[k]
            if type(v) is bool:
                if v:
                    cf_key += '_{}'.format(k)
            else:
                cf_key += '_{}-{}'.format(k, v)
    return cf_key