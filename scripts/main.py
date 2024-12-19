import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat

from scipy.signal import butter, filtfilt
from scipy.signal import iirnotch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def notch_filter(data, notch_freq, fs, quality_factor):
    b, a = iirnotch(notch_freq, quality_factor, fs)
    return filtfilt(b, a, data)


def bandpass_filter(data, low_cut, high_cut, fs):
    low = low_cut / fs
    high = high_cut / fs
    order = 4
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def filter_the_data(df, fs):
    notch_freq = 60  # Powerline frequency in Hz
    quality_factor = 30 # Quality factor

    FilteredDf = {}
    for ii in range(1, 13):
        temp = bandpass_filter(df['EMG' + str(ii)], 20, 1000, fs)
        temp = bandpass_filter(temp, 20, 1000, fs)
        FilteredDf['EMG' + str(ii)] = notch_filter(temp, notch_freq, fs, quality_factor)

    FilteredDf = pd.DataFrame(FilteredDf)

    return FilteredDf

def extract_features(signal, ii):
    features = {}
    features['MAV' + str(ii)] = np.mean(np.abs(signal))
    features['std' + str(ii)] = np.std(signal)
    features['var' + str(ii)] = np.var(signal)
    features['max' + str(ii)] = np.max(signal)
    features['min' + str(ii)] = np.min(signal)
    features['range' + str(ii)] = np.ptp(signal)
    features['zero_crossings' + str(ii)] = ((signal[:-1] * signal[1:]) < 0).sum()
    features['waveform_length' + str(ii)] = np.sum(np.abs(np.diff(signal)))
    return features

def extract_features_window(ms, fs, FilteredEMG, df):
    print('Extracting features...')
    window_size = int(ms * fs / 1000)
    features_list = [[] for _ in range(13)]
    for ii in range(1, 13):
        signal = FilteredEMG['EMG' + str(ii)]
        for start in range(0, len(signal), window_size):
            end = start + window_size
            if end <= len(signal):
                window = signal[start:end]
                window_labels = df['restimulus'][start:end]
                features = extract_features(window, ii)
                features_list[ii].append(features)

        print(f'Status: {ii + 1} out of 13', end='\r')

    # Flatten the list of lists and create a DataFrame
    features_df = pd.concat([pd.DataFrame(f) for f in features_list], axis=1)
    print()
    return features_df

def extract_labels_window(ms, fs, lenSignal, df):
    print('Extracting labels...')
    window_size = int(ms * fs / 1000)    
    labels = []
    for start in range(0, lenSignal, window_size):
        end = start + window_size
        if end <= lenSignal:
            window_labels = df['restimulus'][start:end]
            # Majority voting per determinare l'etichetta della finestra
            label = window_labels.value_counts().idxmax()
            labels.append(label)
        print('Status: ', start, end, lenSignal, end='\r')
    return labels


def train_and_evaluate_model(features_df):
    print('Training and evaluating model...')
    top4Features = ['MAV', 'std', 'zero_crossings', 'waveform_length']
    featuresTraining = []
    for ii in range(1, 13):
        for ff in top4Features:
            featuresTraining.append(ff + str(ii))

    X = features_df.drop(columns=featuresTraining)
    Y = features_df['restimulus']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Generate a classification report
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)

    return rf_model, accuracy, report

import os

def load_data():
    # data_path = str(os.getcwd()) + "/data/s1/S1_E1_A1.mat" 
    data_path = "./data/s1/S1_E1_A1.mat"
    print(f'Loading data from {data_path}')
    
    if not os.path.exists(data_path):
        print(f'File at path {data_path} does not exist')
    data = loadmat(data_path)
    rows = data

    # Remove all the useless data that were not supposed to be in the dataset
    drop = ['__header__', '__version__', '__globals__']
    for dd in drop:
        if dd in rows.keys():
            rows.pop(dd)

    return rows


if __name__ == "__main__":
    data = load_data()
    dataLabels = ['EMG' + str(x) for x in range(1, 13)]
    emgDf = pd.DataFrame(data=data['emg'], columns=dataLabels)
    df = emgDf
    df['restimulus'] = data['restimulus']

    ## Pre processing part
    # Normalize the data

    #scalerEmg = StandardScaler()
    #emgDf_norm = scalerEmg.fit_transform(emgDf)
    #emgDf.head()

    fs = data['frequency'][0][0]  # Frequenza di Nyquist
    FilteredEMG = filter_the_data(emgDf, fs)

    ms = 200 # Durata della finestra in ms
    features_df = extract_features_window(ms, fs, FilteredEMG, df) 
    labels = extract_labels_window(ms, fs, len(FilteredEMG), df)
    features_df['restimulus'] = labels

    rf_model, accuracy, report = train_and_evaluate_model(features_df)

