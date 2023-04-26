import os
import argparse
from scipy.signal import argrelmax
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
from models.LSTMVAE import LSTM_VAE
from utils.utils import update_roc, generate_gt_margin, generate_results
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'

parser = argparse.ArgumentParser(description='Meter error detection')
parser.add_argument('--noise', type=float, default=0, help='Gaussian noise for LSTM-VAE training')
parser.add_argument('--wnd_dim', type=int, default=100, help='window size')
parser.add_argument('--sub_dim', type=int, default=1, help='input data dimension')
parser.add_argument('--lstm_dim', type=int, default=4, help='number of nodes in LSTM')
parser.add_argument('--activation', type=str, default='elu', help='activation function for LSTM')
parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
parser.add_argument('--max_iter', type=int, default=50, help='max iteration for pretraining LSTM_VAE')
parser.add_argument('--latent_dim', type=int, default=1, help='dimension of subspace embedding')
parser.add_argument('--optimiser', type=str, default='rmsprop', help='optimiser')
parser.add_argument('--kl_weight', type=float, default=0.01, help='kl_weight')
parser.add_argument('--save_path', type=str,  default='lstmvae3', help='path to save the pretrained model')
parser.add_argument('--last_act', type=str,  default='linear', help='activation function for last layer')
parser.add_argument('--test', type=bool, default=False, help='test mode')
parser.add_argument('--result_path', type=str, default='15IQRMED11WND100', help='result.pkl')
args = parser.parse_args()

if __name__ == '__main__':
    RATES = ['05', '10', '15', '20', '25']
    ERROR_MARGIN = 50
    THRESHOLDS = np.array(
        [0.01, 0.05, 0.1, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 9, 10,
         11, 15, 18, 20, 25, 35, 45, 50, 80, 100, 200, 300, 400, 800,
         1000, 3000, 6000, 10000, 50000, 100000, 1000000])
    timestep = args.wnd_dim
    trained = False

    for error_rate in RATES:
        info_path = 'data/simulated/' + error_rate + '/' + error_rate + '.csv'
        data_info = pd.read_csv(info_path)
        data_path = 'data/simulated/' + error_rate + '/' + error_rate + 'Simulated.npz'
        Data = np.load(data_path, allow_pickle=True)

        model = LSTM_VAE(timestep, args.sub_dim, args.lstm_dim, args.activation, args.latent_dim, 0.2, 0,
                         args.kl_weight)
        # model = VAE(timestep, args.sub_dim, args.lstm_dim, args.activation, args.latent_dim, args.kl_weight)

        # training phase
        if not args.test and not trained:
            X_p, X_f = Data['Train_L'], Data['Train_R']
            X_p = np.array(X_p)
            X_p_noisy = X_p
            noise = np.random.normal(0, args.noise, size=(X_p.shape[0], X_p.shape[1], X_p.shape[2]))
            X_p_noisy = X_p_noisy + noise

            es = EarlyStopping(patience=5, verbose=1, min_delta=0.00001, monitor='val_loss', mode='auto',
                               restore_best_weights=True)
            optimis = RMSprop(learning_rate=0.001, momentum=0.9)
            model.compile(loss=None, optimizer=optimis)
            model.fit(X_p_noisy, batch_size=args.batch_size, epochs=args.max_iter, validation_split=0.2, shuffle=True,
                      callbacks=[es])
            model.summary()
            model.save_weights('experiment_log/' + args.save_path)
            trained = True

        # testing phase
        model.load_weights('experiment_log/' + args.save_path)
        encoder = model.encoder
        results = {k: {'TPR': 0, 'FPR': 0, 'Precision': 0, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for k in THRESHOLDS}
        test = Data['Test'].item()
        for fn in test:
            margin = None
            with_error = False
            if data_info.loc[data_info['ID'] == int(fn)]['error'].values[0] == 1:
                with_error = True

            inputs = test.get(fn)
            X_p, X_f, Y, L, DATE = inputs['X_p'], inputs['X_f'], inputs['Y'], inputs['L'], inputs['D']
            n = Y.shape[0]

            # exclude the test sample if its length is < 300 or the drift occurs within the first or last window
            if n < 300:
                continue
            if with_error:
                margin = generate_gt_margin(L, timestep, ERROR_MARGIN, n)
                if margin is None:
                    continue

            # generate dissimilarity scores
            z_mean_p, z_log_sigma_p, enc_out_p = encoder.predict(X_p)
            z_mean_f, z_log_sigma_f, enc_out_f = encoder.predict(X_f)
            length = enc_out_p.shape[0]
            Y_pred_zmean = []
            s_idx = 0
            while (s_idx < length):
                enc_out_p = z_mean_p[s_idx]
                enc_out_f = z_mean_f[s_idx]
                dis = K.sqrt(K.sum(K.square(enc_out_p - enc_out_f), axis=-1))
                Y_pred_zmean.append(dis)
                s_idx += 1

            # me identification partially adapt from https://github.com/deepcharles/ruptures
            Y_pred_zmean = np.array(Y_pred_zmean)
            error = Y[0:n].var(axis=0) * (n - 0)
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

            results = generate_results(results, n, Y, error, THRESHOLDS, margin, max_idx, with_error)

        results = update_roc(results, THRESHOLDS)
        isExist = os.path.exists(args.save_path)
        if not isExist:
            os.makedirs(args.save_path)
        result_path = args.save_path + '/' + error_rate + '_' + args.result_path + '.pkl'
        pred_result = open(result_path, 'wb')
        pickle.dump(results, pred_result)
        pred_result.close()
