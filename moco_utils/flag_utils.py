"""

"""


# Built-in
import collections.abc

# Libs

# Own modules


def recursive_update(d, u):
    """
    Recursively update nested dictionary d with u
    :param d: the dictionary to be updated
    :param u: the new dictionary
    :return:
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def update_flags(flags, cf_dict):
    """
    Overwrite the configs in flags if it is given by cf_dict
    :param flags: dictionary of configurations, this is from the config.json file
    :param cf_dict: dictionary of configurations, this is from command line
    :return:
    """
    recursive_update(flags, cf_dict)
    return historical_update_flag(flags, cf_dict)


def historical_update_flag(flags, cf_dict):
    """
    This function updates flag to make it backward compatible with old versions
    :param flags: dictionary of configurations, this is from the config.json file
    :param cf_dict: dictionary of configurations, this is from command line
    :return:
    """
    flags['config'] = cf_dict['config']

    return historical_process_flag(flags)


def historical_process_flag(flags):
    """
    This function updates flag to make it backward compatible with old versions
    :param flags: dictionary of configurations, this is from the config.json file
    """
    if 'imagenet' not in flags:
        flags['imagenet'] = 'True'
    if 'name' not in flags['optimizer']:
        flags['optimizer']['name'] = 'sgd'
    if 'aux_loss' not in flags['optimizer']:
        flags['optimizer']['aux_loss'] = 0
    if 'aux_loss' in flags['optimizer']:
        if 'aux_loss_weight' not in flags['optimizer']:
            flags['optimizer']['aux_loss_weight'] = 0.4
    if 'class_weight' not in flags['trainer']:
        flags['trainer']['class_weight'] = '({})'.format(','.join(['1' for _ in range(flags['dataset']['class_num'])]))
    if 'loss_weights' not in flags['trainer']:
        flags['trainer']['loss_weights'] = 'None'
    if isinstance(flags['trainer']['bp_loss_idx'], str) and len(flags['trainer']['bp_loss_idx']) == 1:
        flags['trainer']['bp_loss_idx'] = '({},)'.format(flags['trainer']['bp_loss_idx'])
    if isinstance(flags['trainer']['bp_loss_idx'], int):
        flags['trainer']['bp_loss_idx'] = '({},)'.format(flags['trainer']['bp_loss_idx'])
    if isinstance(flags['trainer']['loss_weights'], int):
        flags['trainer']['loss_weights'] = (flags['trainer']['loss_weights'],)
    if 'further_train' not in flags['trainer']:
        flags['trainer']['further_train'] = False
    elif isinstance(flags['trainer']['further_train'], str):
        flags['trainer']['further_train'] = eval(flags['trainer']['further_train'])

    flags['ds_cfgs'] = [a for a in sorted(flags.keys()) if 'dataset' in a]
    assert flags['ds_cfgs'][0] == 'dataset'
    if 'gamma' not in flags['trainer']:
        flags['trainer']['gamma'] = 2
    if 'alpha' not in flags['trainer']:
        flags['trainer']['alpha'] = 0.25
    if 'load_func' not in flags['dataset']:
        flags['dataset']['load_func'] = 'default'
    else:
        assert flags['dataset']['load_func'] == 'default' or flags['dataset']['load_func'] == 'None'

    return flags
