import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')

dataset_path = "Speech DataStet"

file_paths = []
labels = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            file_paths.append(os.path.join(root, file))
            
            # Extract emotion from filename
            emotion = file.split("_")[-1].replace(".wav", "")
            labels.append(emotion)

print("Total files:", len(file_paths))
print("Sample labels:", labels[:5])
print(files)
print(file_paths[:5])
print(labels[:5])

df = pd.DataFrame()
df['speech'] = file_paths
df['label'] = labels
print(df.head())
print(df['label'].value_counts())


#Explatory Data Analysis
sns.countplot(df['label'])
plt.title("Emotion Distribution")
plt.show()

def waveplot(data, sr, emotion):
    plt.figure(figsize = (10, 4))
    plt.title(emotion, size = 20)
    librosa.display.waveshow(data, sr = sr)
    plt.show()

def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize = (10, 4))
    plt.title(emotion, size = 20)
    librosa.display.specshow(xdb, sr = sr, x_axis= 'time', y_axis= 'hz')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

emotion = 'fear'
path = df[df['label'] == emotion]['speech'].values[2]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
#print(Audio(path))

emotion = 'angry'
path = df[df['label'] == emotion]['speech'].values[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
#print(Audio(path))

emotion = 'disgust'
path = df[df['label'] == emotion]['speech'].values[1]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
#print(Audio(path))

emotion = 'happy'
path = df[df['label'] == emotion]['speech'].values[3]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
#print(Audio(path))

emotion = 'neutral'
path = df[df['label'] == emotion]['speech'].values[4]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
#print(Audio(path))

emotion = 'ps'
path = df[df['label'] == emotion]['speech'].values[5]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
#print(Audio(path))

emotion = 'sad'
path = df[df['label'] == emotion]['speech'].values[6]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
#print(Audio(path))


#Feature Extraction
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 40).T, axis = 0)
    return mfcc

print(extract_mfcc(df['speech'][0]))

X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
print(X_mfcc)

X = [x for x in X_mfcc]
X = np.array(X)
print(X.shape)

# X  = np.expand_dims(X_mfcc, -1)
X = np.array(X_mfcc.tolist())
X = np.expand_dims(X, -1)
print(X.shape)

#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(df[['label']])
print(y)
#y = y.toarray()
print(y.shape)

# Now Creating LSTM Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential([
    LSTM(123, return_sequences = False, input_shape = (40, 1)),
    Dense(64, activation = 'relu'),
    Dropout(0.2),
    Dense(32, activation = 'relu'),
    Dropout(0.2),
    Dense(7, activation = 'softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())

history = model.fit(X, y, validation_split = 0.2, epochs = 100,  batch_size = 512, shuffle = True)

#Plot the results
epochs = list(range(100))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, label = 'train accuracy')
plt.plot(epochs, val_acc, label = 'val_accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, acc, label = 'train loss')
plt.plot(epochs, val_acc, label = 'val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()