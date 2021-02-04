import time

import torch


def getLossMask(outputs, node_first, seq_list, using_cuda=False):
    '''
    Get a mask to denote whether both of current and previous data exsist.
    Note: It is not supposed to calculate loss for a person at time t if his data at t-1 does not exsist.
    '''

    if outputs.dim() == 3:
        seq_length = outputs.shape[0]
    else:
        seq_length = outputs.shape[1]

    node_pre = node_first
    lossmask = torch.zeros(seq_length, seq_list.shape[1])

    if using_cuda:
        lossmask = lossmask.cuda()

    # For loss mask, only generate for those exist through the whole window
    for framenum in range(seq_length):
        if framenum == 0:
            lossmask[framenum] = seq_list[framenum] * node_pre
        else:
            lossmask[framenum] = seq_list[framenum] * lossmask[framenum - 1]

    return lossmask, sum(sum(lossmask))


def L2forTest(outputs, targets, obs_length, lossMask):
    '''
    Evaluation.
    '''
    seq_length = outputs.shape[0]
    error = torch.norm(outputs - targets, p=2, dim=2)
    # only calculate the pedestrian presents fully presented in the time window
    pedi_full = torch.sum(lossMask, dim=0) == seq_length
    error_full = error[obs_length - 1:, pedi_full]
    error = torch.sum(error_full)
    error_cnt = error_full.numel()
    final_error = torch.sum(error_full[-1])
    final_error_cnt = error_full[-1].numel()

    return error.item(), error_cnt, final_error.item(), final_error_cnt, error_full


def L2forTestS(outputs, targets, loss_mask, obs_length):
    '''
    Evaluation, stochastic version
    '''
    seq_length = outputs.shape[1]
    error = torch.norm(outputs - targets, p=2, dim=3)
    # only calculate the pedestrian presents fully presented in the time window
    pedi_full = torch.sum(loss_mask, dim=0) == seq_length
    error_full = error[:, obs_length - 1:, pedi_full]

    error_full_sum = torch.sum(error_full, dim=1)
    error_full_sum_min, min_index = torch.min(error_full_sum, dim=0)

    best_error = []
    for index, value in enumerate(min_index):
        best_error.append(error_full[value, :, index])
    best_error = torch.stack(best_error)
    best_error = best_error.permute(1, 0)

    error = torch.sum(best_error)  # ADE
    error_cnt = best_error.numel()  # ADE denominator count

    final_error = torch.sum(best_error[-1])  # FDE
    final_error_cnt = best_error.shape[-1]  # FDE denominator count

    return error.item(), error_cnt, final_error.item(), final_error_cnt


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('Function', method.__name__, 'time:', round((te - ts) * 1000, 1), 'ms')
        print()
        return result

    return timed


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod