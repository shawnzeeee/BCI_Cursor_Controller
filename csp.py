import numpy as np
import time
import os
from scipy.signal import welch
import pandas as pd

from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import mne

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

def extract_csp_attention_windows(attention_indices, df, window_size=500, num_windows=4):
    channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]  # Use only the columns for TP9 and TP10
    windows = []
    labels = []
    for start_idx in attention_indices:
        actual_class = df.iloc[start_idx, 4]
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end][channel_names].values.T  # shape: (channels, samples)
            windows.append(window)
            labels.append(actual_class)
    return windows, labels

# Load your CSV file (replace with your actual CSV path)
csv_path = os.path.join(os.path.dirname(__file__), 'calibration.csv')
df = pd.read_csv(csv_path)

# Get indices where class is 2 (attention) and 1 (idle)
attention_indices = df.index[(df['Class'] == 2) | (df['Class'] == 1)].tolist()

# Extract windows and labels using the provided function
windows, labels = extract_csp_attention_windows(attention_indices, df)

# Convert to numpy arrays for sklearn
X = np.array(windows)  # shape: (n_samples, n_channels, n_times)
y = np.array(labels)

# Prepare MNE EpochsArray from windows
sfreq = 250  # Set your actual sampling frequency here if different
ch_names = ["TP9","AF7","AF8", "TP10"]  # Only use TP9 and TP10 for MNE
ch_types = ["eeg"] * 4
info = mne.create_info(ch_names, sfreq, ch_types)

# Set a standard montage for plotting (required for CSP pattern plots)
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# X shape: (n_epochs, n_channels, n_times)
epochs_data = X  # already in (n_epochs, n_channels, n_times)
epochs = mne.EpochsArray(epochs_data, info, events=None, verbose=False)

# Use epochs.get_data() for CSP
X_epochs = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_epochs, y, test_size=0.3, shuffle=True, random_state=42, stratify=y)

print(y_train, len(y_train))

# CSP + SVM pipeline
csp = CSP(n_components=4)
svm = SVC(kernel='linear', random_state=42)
pipeline = Pipeline([('csp', csp), ('svm', svm)])

# Fit pipeline
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap=plt.cm.Blues)
plt.title('CSP + SVM Confusion Matrix (MNE Epochs)')
plt.show()

# Fit CSP on training data only
csp.fit(X_train, y_train)

# Transform data with CSP
X_train_csp = csp.transform(X_train)
X_test_csp = csp.transform(X_test)



# Plot CSP patterns as topomaps (spatial patterns)
csp.plot_patterns(info, ch_type='eeg', units='Patterns (a.u.)', size=1.5)
plt.suptitle('CSP Patterns (Topomap)')
plt.show()

# After CSP transform, extract additional features and concatenate with CSP features
# Calculate bandpowers, mobility, and complexity for each window (trial)
additional_features = []
for window in X_train:
    feats = []
    for ch in range(window.shape[0]):
        signal = window[ch]
        mobility, complexity = calculate_hjorth_parameters(signal)
        alpha_power, beta_power = calculate_bandpowers(signal, fs=sfreq)
        feats.extend([alpha_power, beta_power, mobility, complexity])
    additional_features.append(feats)
additional_features = np.array(additional_features)

# Repeat for test set
additional_features_test = []
for window in X_test:
    feats = []
    for ch in range(window.shape[0]):
        signal = window[ch]
        mobility, complexity = calculate_hjorth_parameters(signal)
        alpha_power, beta_power = calculate_bandpowers(signal, fs=sfreq)
        feats.extend([alpha_power, beta_power, mobility, complexity])
    additional_features_test.append(feats)
additional_features_test = np.array(additional_features_test)

# Get CSP features
X_train_csp = csp.transform(X_train)
X_test_csp = csp.transform(X_test)

# Concatenate CSP features with additional features
X_train_combined = np.concatenate([X_train_csp, additional_features], axis=1)
X_test_combined = np.concatenate([X_test_csp, additional_features_test], axis=1)

# Train SVM on combined features
svm_combined = SVC(kernel='linear', random_state=42)
svm_combined.fit(X_train_combined, y_train)

# Predict and plot confusion matrix
y_pred_combined = svm_combined.predict(X_test_combined)
cm_combined = confusion_matrix(y_test, y_pred_combined)
disp_combined = ConfusionMatrixDisplay(confusion_matrix=cm_combined, display_labels=np.unique(y))
disp_combined.plot(cmap=plt.cm.Blues)
plt.title('SVM Confusion Matrix (CSP + Bandpower + Hjorth)')
plt.show()