import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import medfilt
from scipy.signal import argrelmax
from sklearn import metrics
from models.LSTMVAE import LSTM_VAE
from tensorflow.keras import backend as K
from utils.utils import update_roc, generate_results, sliding_window

def main():
    Data = np.load('data/real/real.npz', allow_pickle=True)
    data_info = pd.read_csv('data/real/label.csv')
    WINDOW_SIZE = 100
    ERROR_MARGIN = 50
    timestep = WINDOW_SIZE
    THRESHOLDS = np.array(
        [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10,
         11, 15, 18, 50, 100, 400, 800, 1000, 10000])

    model = LSTM_VAE(WINDOW_SIZE, 1, 4, 'elu', 1, 0.2, 0, 0.01)
    # model = VAE(WINDOW_SIZE, 1, 4, 'elu', 1)
    # model.load_weights('experiment_log/vae')
    model.load_weights('experiment_log/lstmvae')
    encoder = model.encoder
    results = {k: {'TPR': 0, 'FPR': 0, 'Precision': 0, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for k in THRESHOLDS}

    test = Data['Test'].item()
    for fn in test:
        inputs = test.get(fn)
        margin = None
        error_rate = data_info.loc[data_info['ID'] == int(fn)]['Meter_error_rate'].values[0]
        if error_rate != 0:
            with_error = True
        else:
            with_error = False
        gt_dt = -1
        data = inputs['var_sales']
        dates = inputs['dates']
        # global filtering
        indices = np.where(abs(data) > 0.5)[0]
        data = np.delete(data, indices)
        dates = np.delete(dates, indices)
        # IQR filtering
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        outliers = np.where((data > upper_bound) | (data < lower_bound))[0]
        data = np.delete(data, outliers)
        dates = np.delete(dates, outliers)
        # Normalisation
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        # Smoothing
        data = medfilt(data, 11)
        # exclude the test sample if its length is < 210 or the drift occurs within the first or last window
        if len(data) < 210:
            continue
        if with_error:
            gt_dt = next(x[0] for x in enumerate(dates) if
                         x[1] > datetime.strptime(data_info.loc[data_info['ID'] == int(fn)]['Report_date'].values[0],
                                                  '%d/%m/%Y %H:%M'))
            if (gt_dt - WINDOW_SIZE < 0) or (gt_dt + WINDOW_SIZE > data.shape[0]):
                continue
            margin = (max(0, gt_dt - ERROR_MARGIN), min(gt_dt + ERROR_MARGIN, data.shape[0]))

        n = len(data)
        segments = sliding_window(data, WINDOW_SIZE)
        z_mean, z_log_sigma, enc_out = encoder.predict(segments)
        length = enc_out.shape[0] - WINDOW_SIZE
        Y_pred_zmean = []
        Y_pred_zsample = []
        s_idx = 0
        while (s_idx < length):
            enc_out_i = z_mean[s_idx]
            enc_out_in = z_mean[s_idx + WINDOW_SIZE]
            dis = K.sqrt(K.sum(K.square(enc_out_i - enc_out_in), axis=-1))
            Y_pred_zmean.append(dis)
            s_idx += 1

        Y_pred_zmean = np.array(Y_pred_zmean)
        error = data[0:n].var(axis=0) * n
        peak_inds_shifted = argrelmax(Y_pred_zmean, order=timestep // 2, mode="wrap")[0]
        gains = np.take(Y_pred_zmean, peak_inds_shifted)
        try:
            peaks_zmean, peak_inds = zip(*sorted(zip(gains, peak_inds_shifted)))
            peak_inds = list(peak_inds)
        except:
            peak_inds = []

        if len(peak_inds) == 0:
            max_idx = -1
        else:
            try:
                max_idx = peak_inds.pop() + timestep
            except IndexError:  # peak_inds is empty
                max_idx = -1
        results = generate_results(results, n, data, error, THRESHOLDS, margin, max_idx, with_error)

    results = update_roc(results, THRESHOLDS)

    def print_res(results):
        fpr, tpr = [], []
        for key in results:
            fpr.append(results[key]['FPR'])
            tpr.append(results[key]['TPR'])
        fpr.append(1)
        tpr.append(tpr[-1])
        return fpr, tpr

    fpr, tpr = print_res(results)
    auc = metrics.auc(np.array(fpr), np.array(tpr))
    print('auc: ', auc)

if __name__ == '__main__':
    main()



