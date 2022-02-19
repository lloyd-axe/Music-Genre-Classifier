import math
import librosa
import numpy as np

class MusicGenreClassifier:
    @staticmethod
    def predict_genre(model, data):
        data = np.expand_dims(np.array(data), axis=3)
        label_map = [
            'blues',
            'classical',
            'country',
            'disco',
            'hiphop',
            'jazz',
            'metal',
            'pop',
            'reggae',
            'rock'
            ]
        pred_idx = []  
        pred = model.predict(data)
        for p in pred:
            pred_idx.append(np.argmax(p))
        return label_map[max(pred_idx, key = pred_idx.count)]

    @staticmethod
    def get_sample(
        file_path, 
        sr = 22050, 
        duration = 3, 
        n_segments = 5, 
        n_mfcc = 13, 
        n_fft = 2048, 
        hop_len = 512):

        #Check if wav file
        if file_path.split('\\')[-1].split('.')[-1] != 'wav':
            print('Only .wav files are allowed for now.')
            return None

        samp_p_track = sr * duration
        samp_p_seg = int(samp_p_track / n_segments)
        mfcc_p_seg = math.ceil(samp_p_seg / hop_len)
        
        mfcc_data = []
        
        print(f'Loading audio file...')
        signal, sr = librosa.load(file_path, sr = sr)
        for part in range((signal.shape[0]//samp_p_track)):
            sig = signal[samp_p_track*part : (samp_p_track*part) + samp_p_track]
            for seg in range(n_segments):
                mfcc = librosa.feature.mfcc(
                    y = sig[samp_p_seg * seg:(samp_p_seg * seg) + samp_p_seg], 
                    sr = sr, n_mfcc = n_mfcc, 
                    n_fft = n_fft, hop_length = hop_len)
                mfcc = mfcc.T
                if len(mfcc) == mfcc_p_seg:
                    mfcc_data.append(mfcc.tolist())
        return mfcc_data