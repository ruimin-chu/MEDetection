# MEDetection

MEDetection is a meter error detection framrwork based on a LSTM-VAE model. More details can be found in the paper **Automatic meter error detection with a data-driven approach**

## Abstract
Meter error is one of the main contributing factors to unexpected fuel losses or gains in storage tanks at service stations. Although fuel dispensers are expected to be calibrated to standard accuracy periodically to ensure fair and reliable trade in the fuel market, some fuel retailers are unable to keep up with the standards. Current industry practice relies on onsite inspection to identify the issue, which leads to cost burden if inspections are scheduled too frequently. To the best of our knowledge, there is no previously reported research that is tailored to the remote meter error detection problem. In this paper, we propose a novel framework for remote and automatic meter error detection via a data-driven approach based on inventory data and fuel transaction data. Specifically, we propose to use mean shift change point detection methods, including statistical-based as well as deep learning-based methods (LSTM-VAE, VAE, Kernel learning), to approach the problem. We present results on our data sets containing both real-world and simulated meter error data, and further evaluate these methods on several widely-used benchmark datasets, to assess their validity, advantages and limitations. The obtained results show that LSTM-VAE outperforms other models in most of the settings for the meter error dataset and the benchmark datasets. 

## Environment
- Python 3.8.10
- Tensorflow 2.5.0

## Dataset
```
├── data
│   ├── real
│   │    ├── label.csv     # store the pump calibration date (change point) and meter error rate
│   │    ├── real.npz      # the relative error sequence of isoldated transactions occurred in the pump
│   ├── Simulated
│   │    ├── 05
│   │    │    ├── 05.csv             # contain the simulated error rate (avg error rate is 0.5%) and the date of meter error occurrance 
│   │    │    ├── 05_simulated.npz    # contain the simulated meter data and no meter error data, contain training data in the format of windows of 100, and test data
│   │    ├── 10
│   │    │    ├── 10.csv 
│   │    │    ├── 10_simulated.npz    
│   │    ├── 15
│   │    │    ├── 15.csv 
│   │    │    ├── 15_simulated.npz    
│   │    ├── 20
│   │    │    ├── 20.csv 
│   │    │    ├── 20_simulated.npz    
│   │    ├── 25
│   │    │    ├── 25.csv 
│   │    │    ├── 25_simulated.npz    
│   │    ├── train.npz              # contain train data (not divided into windows)
│   │    ├── wnds_train.npz         # contain train data (that has been divided into windows)
```
The full simulated and real-world meter error data can be downloaded here.
- [Google Drive](https://drive.google.com/drive/folders/1vO9BUl8RYKkQcZGYXeZM3_lnDR-DRMqa?usp=share_link)

## Main Usage
For running experiments using the simulated data and train a LSTM-VAE model for meter error data, you can  run `main.py`. For evaluating the performance of the framework with the trained LSTM-VAE model on real-world meter error data, you can run `real_ME.py`.
## Contact
If you have any questions, please contact the author at ruimin.chu@rmit.edu.au
