import os
from scipy.io import loadmat, savemat
import numpy as np
import mne
from sklearn.decomposition import PCA
from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import Covariances
from scipy.signal import find_peaks
from mne.viz import ClickableImage  # noqa
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score, cross_validate

# Informative start of the script
print('Processing started...')

# Load positional data for electrodes
mat_pos = loadmat(os.getcwd() + '/Ecog_data/pos.mat')
pos = mat_pos['pos']

# Load ECoG hand pose data
mat = loadmat(os.getcwd() + '/Ecog_data/ECoG_Handpose.mat')
y = mat['y'][:, 25:]

# Extract channels data (first 60 channels)
ch_channels = y[1:61]

# Extract hand pose labels, cropping the first 2 seconds (1200*2 samples)
hand_pose = y[61, 1200*2 + 25:]

# Find the first non-zero hand pose event (e.g., gesture event)
first_pose = np.argwhere(hand_pose != 0)[0]

# Compute the overall mean of the first pose event across channels
all_mean = np.mean(np.mean(ch_channels[:, :int(first_pose)], axis=1), axis=0)

# Add the mean back to the signal
ch_channels += all_mean

# Generate channel names for ECoG electrodes
ch_names = ['{}'.format(i) for i in range(len(ch_channels))]

# Sampling frequency for ECoG data (1200 Hz)
Fs = 1200

# Create a montage and raw MNE object using the electrode positions
montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, pos)), coord_frame='head')
info = mne.create_info(ch_names=ch_names, sfreq=Fs, ch_types='ecog').set_montage(montage)
new_raw = mne.io.RawArray(ch_channels, info, verbose=0)

print('Raw data created successfully.')

# Crop out the first 2 seconds to avoid artifacts
new_raw.crop(tmin=2)

# Apply a band-pass filter in the high gamma range (50-300 Hz)
new_raw.filter(50., 300., fir_design='firwin')

# Apply a notch filter to remove powerline noise (50 Hz and its harmonics)
new_raw.notch_filter(np.arange(50, 351, 50), mt_bandwidth=5)

# Get data after filtering for further processing
all_ch = new_raw.get_data()

# Create a new MNE raw object after potential peak trimming
new_raw = mne.io.RawArray(all_ch, info, verbose=0)

# Print summary of the raw data info
print('Data successfully filtered and prepared. Here is a summary:')
print(new_raw.info)

# Plot the Power Spectral Density (PSD) of the raw data between 30-350 Hz
new_raw.plot_psd(fmin=30, fmax=350, picks=['0'])

# Create events for different hand poses (gestures)
events = np.argwhere(np.diff(hand_pose) != 0) + 1
events = np.append(events, np.zeros([len(events), 1]), axis=1)
events = np.append(events, hand_pose[np.argwhere(np.diff(hand_pose) != 0) + 1], axis=1)
events = np.append(np.array([[0, 0, 0]]), events, axis=0)
events = events.astype(int)

# Define event IDs corresponding to hand gestures (relax, fist, peace, open)
event_id = dict(relax=0, fist=1, peace=2, open=3)

# Create epochs (segments of data) around the events for CSP and further processing
epochs = mne.Epochs(new_raw, events, event_id, picks='ecog', tmin=0., tmax=2., baseline=None, preload=True)

# Extract labels (gestures) from the epochs
labels = epochs.events[:, -1]

# Initialize lists for storing scores
scores = []
epochs_data = epochs.get_data()  # Get the epochs data for the gestures
class_accur = []  # List to store classification accuracy per class

print('Starting gesture classification...')

# Iterate over each gesture class (1, 2, 3 corresponds to fist, peace, open gestures)
for j in range(1, 4):
    labels1 = np.copy(labels)  # Copy the labels
    scores = []

    # Set labels not equal to the current gesture class to 0
    labels1[labels != j] = 0

    # Create a ShuffleSplit cross-validator (10 splits, 30% test data)
    cv = ShuffleSplit(10, test_size=0.3, random_state=30)
    cv_split = cv.split(epochs_data)  # Split the data into training and test sets

    # Assemble a classifier pipeline
    lda = LinearDiscriminantAnalysis()  # LDA classifier
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)  # CSP for feature extraction

    # Compute covariance matrices for each epoch (used in Riemannian classifiers)
    cov_data_train = Covariances().transform(epochs_data)

    # Using Riemannian geometry classifier
    clf = TSclassifier()

    # Perform cross-validation and compute classification accuracy
    scores = cross_val_score(clf, cov_data_train, labels1, cv=cv, n_jobs=-1)

    # Store the mean accuracy score for the current gesture class
    class_accur.append(np.mean(scores))

    # Print the classification accuracy for the current gesture
    print(f"Classification accuracy for gesture {j}: {np.mean(scores) * 100:.2f}%")

# Print overall classification accuracy
print(f"Overall classification accuracy: {np.mean(class_accur) * 100:.2f}%")

# End of the script
print('Processing completed.')