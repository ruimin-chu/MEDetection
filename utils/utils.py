import numpy as np

def generate_gt_margin(L, timestep, e, length):
    gt = np.where(L == 1)[0][0]
    if (gt - timestep < 0) or (gt + timestep > length):
        return None
    margin = (max(0, gt - e), min(gt + e, length))
    return margin

def generate_results(results, n, Y, error, thresholds, margin, max_idx, with_error):
    THRESHOLDS1 = [np.log(n) * 1 / i for i in thresholds]
    if max_idx == -1:
        for k in range(len(thresholds)):
            if with_error:
                results[thresholds[k]]['FN'] += 1
            else:
                results[thresholds[k]]['TN'] += 1
    else:
        for k in range(len(THRESHOLDS1)):
            threshold = THRESHOLDS1[k]
            gain = error - (Y[0:max_idx].var(axis=0) * max_idx + Y[max_idx:len(Y)].var(axis=0) * (len(Y) - max_idx))
            if gain > threshold:
                if with_error:
                    if max_idx >= margin[0] and max_idx <= margin[1]:
                        results[thresholds[k]]['TP'] += 1
                    else:
                        results[thresholds[k]]['FN'] += 1
                else:
                    results[thresholds[k]]['FP'] += 1
            else:
                if with_error:
                    results[thresholds[k]]['FN'] += 1
                else:
                    results[thresholds[k]]['TN'] += 1
    return results

def update_roc(results, thresholds):
    for threshold in thresholds:
        try:
            results[threshold]['Precision'] = results[threshold]['TP'] / (results[threshold]['TP'] + results[threshold]['FP'])
        except:
            results[threshold]['Precision'] = 0

        results[threshold]['TPR'] = results[threshold]['TP'] / (results[threshold]['TP'] + results[threshold]['FN'])
        results[threshold]['FPR'] = results[threshold]['FP'] / (results[threshold]['FP'] + results[threshold]['TN'])
    return results

def sliding_window(seq, ws):
    i = 0
    tl = len(seq)-ws
    windows = np.empty((tl, ws, 1))
    while i < tl:
        windows[i] = seq[i:i+ws].reshape(-1,1)
        i += 1
    return windows

