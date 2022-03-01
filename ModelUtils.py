import os
import math
import json
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPool2D
'''
This module is specifically made to create a deep learning model
with the "GTZAN Dataset - Music Genre Classification" Dataset.

Nothing special here, just classic tensorflow methods.
'''
class ModelUtils:  
    @staticmethod
    def prepare_data(
        data_path, 
        json_path = 'data.json', 
        sr = 22050, 
        duration = 3, 
        n_segments = 5, 
        n_mfcc = 13, 
        n_fft = 2048, 
        hop_len = 512):
        '''
        Like I said above, this method is specifically made to collect audio data
        from the "GTZAN Dataset - Music Genre Classification" Dataset.
        Other datasets can be collected too as long as the dataset structure is similar.

        The method will walk into all the genre folders in the dataset. Each genre name will
        be stored in the 'mapping' key of {data}. Then, the mfcc data for each audio file will
        be collected and stored into the 'mfcc' key of {data}. The corresponding labels will 
        also be stored in the 'labels' key of {data}.
        '''
        data ={'mapping' : [],'mfcc' : [],'labels' : []}
        samp_p_track = sr * duration
        samp_p_seg = int(samp_p_track / n_segments)
        mfcc_p_seg = math.ceil(samp_p_seg / hop_len)
        for i, (dirpath, dirname, filenames) in enumerate(os.walk(data_path)):
            if dirpath is not data_path:
                label = dirpath.split('\\')[-1]
                data['mapping'].append(label) #Add genre name
                print(f'Processing {label} - {len(filenames)}x...')
                for file in filenames: #Collect mfcc data from each audio file on genre
                    file_path = os.path.join(dirpath, file)
                    signal, sr = librosa.load(file_path, sr = sr)
                    print(signal.shape)
                    for s in range(n_segments):
                        s_samp = samp_p_seg * s
                        f_samp = s_samp + samp_p_seg

                        mfcc = librosa.feature.mfcc(
                            y = signal[s_samp:f_samp], sr = sr, n_mfcc = n_mfcc, 
                            n_fft = n_fft, hop_length = hop_len)
                        mfcc = mfcc.T
                        if len(mfcc) == mfcc_p_seg:
                            data['mfcc'].append(mfcc.tolist())
                            data['labels'].append(i-1)
        #Saving data into json
        with open(json_path, 'w') as fp:
            json.dump(data, fp, indent=4)
            print(f'Data saved to {json_path}')
        return data
    
    @staticmethod      
    def load_JSON(json_path):
        with open(json_path, 'r') as fp:
            data = json.load(fp)
        return data
    
    @staticmethod   
    def split_data(x_raw, y_raw, test_size = 0.25, val_size = 0.2, val_set = False):
        inputs = {}
        x = np.array(x_raw)
        y = to_categorical(np.array(y_raw))
        x_train, x_test, y_train, y_test = train_test_split(x, 
                                                            y, 
                                                            test_size = test_size)
        
        if val_set:
            x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                                y_train, 
                                                                test_size = val_size)
            x_val = np.expand_dims(x_val, axis=3)
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        inputs['x_train'] = x_train
        inputs['x_test'] = x_test
        inputs['x_val'] = x_val
        inputs['y_train'] = y_train
        inputs['y_test'] = y_test
        inputs['y_val'] = y_val
        return inputs
    
    @staticmethod
    def create_model(shape, n_class):
        model = Sequential()
        model.add(Conv2D(32, (3,3),
                        activation='relu',
                        input_shape=shape))
        model.add(MaxPool2D((3, 3), strides=(2,2,), padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3,3),
                        activation='relu',
                        input_shape=shape))
        model.add(MaxPool2D((3, 3), strides=(2,2,), padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (2,2),
                        activation='relu',
                        input_shape=shape))
        model.add(MaxPool2D((2, 2), strides=(2,2,), padding='same'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(64,activation='relu'))
        model.add(Dense(n_class, activation='softmax'))
        model.compile(optimizer='Adam', 
                      loss='categorical_crossentropy', 
                      metrics=['categorical_accuracy'])
        return model
    
    @staticmethod
    def train(
        model, 
        x, 
        y, 
        epochs, 
        val_data = ([], []), 
        model_name='model', 
        plot = False, 
        earlyStop = True):
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
            print('checkpoints folder made')
        check_model_path = 'checkpoints/mdl-{epoch:02d}-{val_categorical_accuracy:2f}.hdf5'
        model_check = ModelCheckpoint(check_model_path, 
                                      monitor='val_categorical_accuracy', 
                                      verbose = 1, 
                                      save_best_only = True, 
                                      mode='max')
        log_csv = CSVLogger('test_log.csv', separator=',', append=False)
        cb = [model_check, log_csv]
        if earlyStop:
            cb.append(EarlyStopping(monitor='val_loss', patience=5, verbose=3))
        if val_data != ([], []):
            history = model.fit(x, y, epochs = epochs, callbacks = cb, validation_data = val_data)
            
            if plot:
                plt.figure(figsize=(16,10))
                val = plt.plot(history.epoch, 
                        history.history['val_categorical_accuracy'],
                        '--', 
                            label='Val')
                plt.plot(history.epoch, 
                        history.history['categorical_accuracy'], 
                        color=val[0].get_color(), 
                        label='Train')
                plt.plot(history.epoch, 
                        history.history['loss'], 
                        label='loss')
                plt.plot(history.epoch, 
                        history.history['val_loss'], 
                        label='val_loss')
                plt.xlabel('Epochs')
                plt.ylabel('categorical_accuracy')
                plt.legend()
                plt.xlim([0,max(history.epoch)])
        model.save(model_name + '.h5')
        return model

    @staticmethod
    def save_model(model, name):
        path = './{}_bert'.format(name.replace('/', '_'))
        model.save(path, include_optimizer=False)
    