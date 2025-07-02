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

def process_idle_windows(idle_indices, df, window_size=500, num_windows=4):
    processed_data = []
    channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
    for start_idx in idle_indices:
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end]
            features = []
            for channel in channel_names:
                signal = window[channel].values
                mobility, complexity = calculate_hjorth_parameters(signal)
                alpha_power, beta_power = calculate_bandpowers(signal)
                features.extend([mobility, complexity, alpha_power, beta_power])
            #Appending the actual class (right hand or left hand)
            features.append(0)
            #Appending the active state (idle or attentive)
            features.append(0)
            processed_data.append(features)
    return processed_data

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
                mobility, complexity = calculate_hjorth_parameters(signal)
                alpha_power, beta_power = calculate_bandpowers(signal)
                features.extend([mobility, complexity, alpha_power, beta_power])
            #Appending the actual class (right hand or left hand)
            features.append(actual_class)
            #Appending the active state (idle or attentive)
            features.append(1)
            processed_data.append(features)
    return processed_data


def extract_csp_idle_windows(idle_indices, df, window_size=500, num_windows=4):
    """
    Extracts raw EEG windows for CSP from idle indices.
    Returns: list of windows (channels x samples), list of labels (all 0)
    """
    channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
    windows = []
    labels = []
    for start_idx in idle_indices:
        # actual_class = df.iloc[start_idx, 4]  # Not needed for idle
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end][channel_names].values.T  # shape: (channels, samples)
            windows.append(window)
            labels.append(0)  # Only append one label per window
    return windows, labels

def extract_csp_attention_windows(attention_indices, df, window_size=500, num_windows=4):
    channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
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
            labels.append(actual_class)  # Only append actual_class ONCE per window
    return windows, labels

# Load your CSV file (replace with your actual CSV path)
csv_path = os.path.join(os.path.dirname(__file__), 'calibration.csv')
df = pd.read_csv(csv_path)

# Get indices where class is 2 (attention) and 1 (idle)
attention_indices = df.index[(df['Class'] == 2) | (df['Class'] == 1)].tolist()
idle_indices = df.index[df['Class'] == 3].tolist()

all_output_data.extend(process_attention_windows(attention_indices, df))
all_output_data.extend(process_idle_windows(idle_indices, df))


all_output_data = np.array(all_output_data)

# Save processed features to CSV
feature_columns = []
for ch in range(1, 5):
    feature_columns.extend([
        f"Ch{ch}_mobility", f"Ch{ch}_complexity", f"Ch{ch}_alpha", f"Ch{ch}_beta"
    ])
feature_columns += ["ActualClass", "ActiveState"]

df_features = pd.DataFrame(all_output_data, columns=feature_columns)
df_features.to_csv("processed_features.csv", index=False)

# Prepare features and labels for double SVM pipeline
X = all_output_data[:, :-2]
y_hand = all_output_data[:, -2]   # 1=left, 2=right, 0=close
y_openclose = all_output_data[:, -1]  # 0=close, 1=open

# 70/30 train/test split for both labels (keep indices aligned)
from sklearn.model_selection import train_test_split
X_train, X_test, y_openclose_train, y_openclose_test, y_hand_train, y_hand_test = train_test_split(
    X, y_openclose, y_hand, test_size=0.3, random_state=42, stratify=y_openclose)

# First SVM: open (0) vs close (1)
from sklearn.svm import SVC
svm_openclose = SVC(kernel='linear', random_state=42)
svm_openclose.fit(X_train, y_openclose_train)
y_openclose_pred = svm_openclose.predict(X_test)

# Second SVM: left (1) vs right (2), only for open samples
close_indices_train = np.where(y_openclose_train == 1)[0]
close_indices_test = np.where(y_openclose_pred == 1)[0]

X_train_close = X_train[close_indices_train]
y_hand_train_close = y_hand_train[close_indices_train]

svm_leftright = SVC(kernel='linear', random_state=42)
svm_leftright.fit(X_train_close, y_hand_train_close)

# Prepare final predictions: 0=close, 1=left open, 2=right open
final_pred = np.zeros_like(y_hand_test)
for i, pred in enumerate(y_openclose_pred):
    if pred == 0:
        final_pred[i] = 0  # close
    else:
        # Predict left/right for open samples
        final_pred[i] = svm_leftright.predict(X_test[i].reshape(1, -1))[0]

# Prepare true labels for evaluation: 0=close, 1=left open, 2=right open
final_true = np.zeros_like(y_hand_test)
for i, val in enumerate(y_openclose_test):
    if val == 0:
        final_true[i] = 0
    else:
        final_true[i] = y_hand_test[i]

# Plot confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
cm = confusion_matrix(final_true, final_pred, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Close", "Left Open", "Right Open"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Double SVM Pipeline Confusion Matrix")
plt.show()

# Compute F1 scores for each class
f1_scores = f1_score(final_true, final_pred, labels=[0, 1, 2], average=None)

# Plot F1 scores
plt.figure()
plt.bar(["Close", "Left Open", "Right Open"], f1_scores, color='skyblue')
plt.ylim(0, 1)
plt.ylabel("F1 Score")
plt.title("Double SVM Pipeline F1 Scores per Class")
plt.show()



# Build pipeline: CSP -> SVM
from sklearn.pipeline import Pipeline



# Each extraction returns (windows, labels), so collect them separately
csp_windows_all = []
csp_labels_all = []

windows, labels = extract_csp_attention_windows(attention_indices, df)
csp_windows_all.extend(windows)
csp_labels_all.extend(labels)

windows, labels = extract_csp_idle_windows(idle_indices, df)
csp_windows_all.extend(windows)
csp_labels_all.extend(labels)

# Convert to numpy arrays
csp_windows_all = np.array(csp_windows_all)
csp_labels_all = np.array(csp_labels_all)

# Split into train and test sets (before filtering)
csp_windows_train_all, csp_windows_test_all, csp_labels_train_all, csp_labels_test_all = train_test_split(
    csp_windows_all, csp_labels_all, test_size=0.3, random_state=42, stratify=csp_labels_all
)

# Now filter out label 0 from both train and test sets
csp_windows_train = csp_windows_train_all[csp_labels_train_all != 0]
csp_labels_train = csp_labels_train_all[csp_labels_train_all != 0]

csp = CSP(n_components=4)
svm = SVC(kernel='linear', random_state=42)
svm_leftright_pipeline = Pipeline([
    ('csp', csp),
    ('svm', svm)
])

# After fitting the pipeline
X_csp_train = csp.fit_transform(csp_windows_train, csp_labels_train)

print(csp_labels_train)

svm_leftright_pipeline.fit(csp_windows_train, csp_labels_train)

final_pred = np.zeros_like(y_hand_test)
csp_idx = 0  # Counter for csp_windows_test

print(len(y_openclose_pred), csp_windows_test_all.shape)
for i, pred in enumerate(y_openclose_pred):
    if pred == 0:
        final_pred[i] = 0  # close
    else:
        # Predict left/right for open samples using the next CSP window
        window = np.array(csp_windows_test_all[i])  # shape (n_channels, n_times)
        final_pred[i] = svm_leftright_pipeline.predict(window[np.newaxis, ...])[0]


# Prepare true labels for evaluation: 0=close, 1=left open, 2=right open
final_true = np.zeros_like(y_hand_test)
for i, val in enumerate(y_openclose_test):
    if val == 0:
        final_true[i] = 0
    else:
        final_true[i] = y_hand_test[i]

# Plot confusion matrix
cm = confusion_matrix(final_true, final_pred, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Close", "Left Open", "Right Open"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Final SVM Pipeline Confusion Matrix")
plt.show()

# F1 scores
f1_scores = f1_score(final_true, final_pred, labels=[0, 1, 2], average=None)
plt.figure()
plt.bar(["Close", "Left Open", "Right Open"], f1_scores, color='skyblue')
plt.ylim(0, 1)
plt.ylabel("F1 Score")
plt.title("Final SVM Pipeline F1 Scores per Class")
plt.show()