import numpy as np
import time
import os
from scipy.signal import welch
import pandas as pd
from PyEMD import EMD
from scipy.signal import hilbert

from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# Function to calculate mobility and complexity (Hjorth parameters)
def calculate_hjorth_parameters(signal):
    first_derivative = np.diff(signal)
    second_derivative = np.diff(first_derivative)
    variance = np.var(signal)
    mobility = np.sqrt(np.var(first_derivative) / variance)
    complexity = np.sqrt(np.var(second_derivative) / np.var(first_derivative)) / mobility
    return mobility, complexity

# Function to calculate bandpowers (alpha and beta)
def calculate_bandpowers(signal, fs=250):
    freqs, psd = welch(signal, fs=fs, nperseg=fs)
    alpha_band = np.logical_and(freqs >= 8, freqs <= 13)
    beta_band = np.logical_and(freqs >= 13, freqs <= 30)
    alpha_power = np.sum(psd[alpha_band])
    beta_power = np.sum(psd[beta_band])
    return alpha_power, beta_power

all_output_data = []

def sample_entropy(signal, m=2, r=None):
    if r is None:
        r = 0.2 * np.std(signal)
    N = len(signal)
    def _phi(m):
        x = np.array([signal[i:i + m] for i in range(N - m + 1)])
        C = np.sum([np.sum(np.max(np.abs(x - xi), axis=1) <= r) - 1 for xi in x])
        return C / (N - m + 1)
    return -np.log(_phi(m + 1) / _phi(m))


def process_attention_windows(attention_indices, df, window_size=500, num_windows=4):
    processed_data = []
    channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
    for start_idx in attention_indices:
        actual_class = df.iloc[start_idx, 4]
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end]
            features = []
            for channel in channel_names:
                signal = window[channel].values
                # EMD decomposition
                emd = EMD()
                IMFs = emd(signal)
                # IMF2 and IMF5 (check if they exist)
                for imf_idx in [1, 4]:
                    if len(IMFs) > imf_idx:
                        imf = IMFs[imf_idx]
                        analytic_signal = hilbert(imf)
                        amplitude_envelope = np.abs(analytic_signal)
                        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * 250  # fs=250
                        # Bandpass filter via frequency mask
                        alpha_mask = (instantaneous_frequency >= 8) & (instantaneous_frequency <= 13)
                        beta_mask = (instantaneous_frequency >= 13) & (instantaneous_frequency <= 30)
                        alpha_power = np.mean(amplitude_envelope[1:][alpha_mask]) if np.any(alpha_mask) else 0
                        beta_power = np.mean(amplitude_envelope[1:][beta_mask]) if np.any(beta_mask) else 0
                    else:
                        alpha_power = 0
                        beta_power = 0
                    features.extend([alpha_power, beta_power])
                # Sample entropy from original signal
                sampen = sample_entropy(signal)
                features.append(sampen)
            # Append class label
            features.append(actual_class)
            features.append(1)
            processed_data.append(features)
    return processed_data

# Load your CSV file (replace with your actual CSV path)
csv_path = os.path.join(os.path.dirname(__file__), 'calibration.csv')
df = pd.read_csv(csv_path)

attention_indices = df.index[(df['Class'] == 2) | (df['Class'] == 1)].tolist()

print(attention_indices)

all_output_data.extend(process_attention_windows(attention_indices, df))

all_output_data = np.array(all_output_data)

# Prepare features and labels for SVM
# Features: all columns except the last two (actual_class, active_state)
# Label: actual_class (second last column)
X = all_output_data[:, :-2]
y = all_output_data[:, -2]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap=plt.cm.Blues)
plt.title('SVM Confusion Matrix (Hjorth + Bandpower Features)')
plt.show()
