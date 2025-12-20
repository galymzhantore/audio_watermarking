import numpy as np
from scipy.io import wavfile
from scipy import signal
from pydub import AudioSegment
import soundfile as sf
import librosa

import io
# import soundfile as sf

class AudioData:
    def __init__(self, fname=None, sr=None, values=None):
        '''
        Note: sound is converted to mono format

        To make AudioData from file provide fname
        To make AudioData from raw data provide values and sr

        fname: file-name or IO object

        sr: sample-rate 

        values: raw sound values
        '''
        if values is not None:
            self.y = values.copy()
            self.power = np.mean(self.y**2)
            if sr is None:
                raise Exception("missing sr for data initialization")
            self.sr = sr
            return

        if fname is not None:
            self.y, self.sr = librosa.load(fname, sr=sr)
            self.power = np.mean(self.y**2)
            return
        
        raise Exception("Invalid inputs: provide either fname or values")

    def save_to(self, file, format=None, quantization=None, sr=None):
        ''' 
        file: file name of IO object

        format: audio format (.mp3, .flac, .wav, etc.)

        quantization is applied only to lossless formats such as wav or flac

        sr: sample rate of saved file. If None than self.sr is used
        '''

        subtype=None

        if quantization is not None:
                if quantization not in [8, 16, 24]:
                    raise Exception("Quantization should be in [8, 16, 24]")

                if quantization == 8:
                    subtype="PCM_U" + str(quantization)
                else:
                    subtype="PCM_" + str(quantization)

        if format is not None:
            format = format.upper()

        if sr is None:
            sr = self.sr

        sf.write(file, self.y, self.sr, format=format, subtype=subtype)

    
    def add_noise(self, snr, mode="renorm", inplace=False):
        '''
        snr: signal-to-noise ratio, the higher - the worse

        mode: is either "renorm" of "clip"
        if mode is "renorm" and noise overflows maximum loudness (1.0),
        sound will be renormalized to keep maximum loudness <= 1.0
        if mode if "clip" and noise overflows maximum loudness (1.0),
        it is clipped to 1.0

        inplace: either apply noise to self or make new sound
        '''

        noise_power = self.power / (10 ** (snr / 10))        
        noise_sigma = np.sqrt(noise_power)
        noise = np.random.normal(loc=0, scale=noise_sigma, size=self.y.shape)

        noised_sound = self.y + noise

        if mode=="renorm":
            norm = np.max(noised_sound)
            noised_sound /= norm
        elif mode=="clip":
            noised_sound = np.clip(noised_sound, -1, 1)
        else:
            raise Exception(f"Invalid mode value: should be 'renorm' or 'clip', got {mode}")

        if inplace:
            self.y = noised_sound
            return
        
        return AudioData(values=noised_sound, sr=self.sr)

    def mp3_compression(self, inplace=False):
        '''
        Applies mp3 compression

        if inplace is True, applies to self
        ohterwise creates new compressed AudioData
        '''
        buf_io = io.BytesIO()
        self.save_to(buf_io, format="mp3")
        buf_io.seek(0)
        res = AudioData(fname=buf_io)

        if not inplace:
            return res
        
        self.y=res.y
        self.sr = res.sr

    def resample(self, sr, inplace=False):
        '''
        Resamples audio

        if inplace is True, applies to self
        ohterwise creates new resampled AudioData
        '''

        y_resampled = librosa.resample(self.y, orig_sr=self.sr, target_sr=sr)

        if inplace:
            self.y = y_resampled
            self.sr = sr
            return
        
        return AudioData(values=y_resampled, sr=sr)
        
    def requantize(self, quantization, inplace=False):
        '''
        Re-quantize audio

        if inplace is True, applies to self
        ohterwise creates new re-quantized AudioData
        '''
        buf_io = io.BytesIO()
        self.save_to(buf_io, format="wav", quantization=quantization)
        buf_io.seek(0)
        res = AudioData(fname=buf_io)

        if not inplace:
            return res
        
        self.y=res.y
        self.sr = res.sr

    def crop(self, begin=None, end=None, inplace=False):
        ''' 
        Crops audo

        begin: time in seconds. Remove audio before begin

        end: time in seconds. Remove audio after end

        if inplace is True, applies to self
        ohterwise creates new re-quantized AudioData
        '''
        if begin is None:
            begin = 0
        if end is None:
            # just a big enough number
            end = self.y.shape[0]

        i_begin = np.round(begin * self.sr)
        i_end = np.round(end * self.sr)

        y_res = self.y[i_begin:i_end]

        if inplace:
            self.y = y_res
            return

        return AudioData(values=y_res, sr=self.sr)
        