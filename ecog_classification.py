import os
from scipy.io import loadmat,savemat
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
from sklearn.model_selection import ShuffleSplit, cross_val_score,cross_validate
print('start')
mat_pos = loadmat(os.getcwd()+ '/Ecog_data/pos.mat')
pos = mat_pos['pos']
mat = loadmat(os.getcwd()+ '/Ecog_data/ECoG_Handpose.mat')
y = mat['y'][:, 25:]
# only the channels data
ch_channels = y[1:61]
# labels per sample with 2.00025 sec cropped out
hand_pose = y[61,1200*2+25:]
first_pose=np.argwhere(hand_pose!=0)[0]
all_mean = np.mean(np.mean(ch_channels[:,:int(first_pose)],axis=1),axis=0)
ch_channels += all_mean
# normlaize the signals
# for i in range(len(ch_channels)):
#     ch_channels[i] -= np.mean(ch_channels[i])
#     ch_channels[i] /= np.max(np.abs(ch_channels[i]))
#set names to the ch_channels
ch_names = ['{}'.format(i) for i in range(len(ch_channels))]
#sample rate
Fs = 1200 #hz
# print(ch_names)
#create montage and raw file for mni
montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, pos)),
                                        coord_frame='head')
info = mne.create_info(ch_names=ch_names, sfreq=Fs, ch_types='ecog').set_montage(montage)
new_raw = mne.io.RawArray(ch_channels, info, verbose=0)
print('raw was created')
#crop first 2s
new_raw.crop(tmin=2)
# new_raw.filter(8., 30., fir_design='firwin')
#band pass filter only for high gamma
new_raw.filter(50., 300., fir_design='firwin')
# notch-filter power line frequency
new_raw.notch_filter(np.arange(50, 351, 50),mt_bandwidth=5)#,method='iir',iir_params=dict(order=6, ftype='butter'))
#find high amplitude and trim it from all electrodes
all_ch = new_raw.get_data()
# for ds in range(0, len(new_raw.get_data())):
#     a,_ = find_peaks(new_raw.get_data()[ds, :], threshold=8e-05)
    # if len(a)!=0:
    #     for remove in range(0,len(a)):
    #         all_ch = np.delete(all_ch,np.append(a[remove], [a[remove]+1, a[remove]-1]), axis=1)
new_raw = mne.io.RawArray(all_ch, info, verbose=0)
new_raw.plot_psd(fmin=30,fmax=350,picks=['0'])
print(new_raw.info)
# create epochs with labels for CSP
events = np.argwhere(np.diff(hand_pose)!=0)+1
events = np.append(events,np.zeros([len(events),1]),axis=1)
events = np.append(events,hand_pose[np.argwhere(np.diff(hand_pose)!=0)+1],axis=1)
events = np.append(np.array([[0,0,0]]),events,axis=0)
# events = events[events[:,2]!=0]
events = events.astype(int)
event_id = dict(relax=0,fist=1, peace=2,open=3)
epochs = mne.Epochs(new_raw, events, event_id,picks='ecog', tmin=0., tmax=2., baseline=None, preload=True)
labels = epochs.events[:, -1]
scores = []
epochs_data = epochs.get_data()
class_accur = []
for j in range(1,4):
    labels1 = np.copy(labels)
    scores = []
    labels1[labels != j] = 0
    cv = ShuffleSplit(10, test_size=0.3,random_state=30)
    cv_split = cv.split(epochs_data)
    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    # pca = mne.decoding.UnsupervisedSpatialFilter(PCA(n_components=2,whiten=True),average=False)
    cov_data_train = Covariances().transform(epochs_data)
    # X = pca.fit_transform(epochs_data)
    # X = (X ** 2).var(axis=2)
    # mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
    clf = TSclassifier()
    # X = np.log(X)
    # Use scikit-learn Pipeline with cross_val_score function
    # clf = Pipeline([('CSP', csp), ('LDA', lda)])
    # clf = Pipeline([('LDA', lda)])
    scores = cross_val_score(clf, cov_data_train, labels1, cv=cv, n_jobs=-1)
    class_accur.append(np.mean(scores))
    print("Classification accuracy: %f" % (np.mean(scores)))
    print(class_accur)
    print(np.mean(class_accur))

print('Eidan')