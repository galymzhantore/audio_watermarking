from typing import Literal, Union, Optional
import torch
import numpy as np
import warnings

from decompositions import embed_SVD, embed_QR, embed_Schur, extract_SVD, extract_QR, extract_Schur, embed_Cholesky, extract_Cholesky, embed_LU, extract_LU
from LWT_FWHT import BatchLWT, BatchFWHT
from preprocessing import encrypt_watermark, decrypt_watermark

class Watermark():
    def __init__(self, encryption_type: Literal['bernoulli', 'gaussian', 'tent', 'none'] = 'bernoulli',
                 encryption_params: dict = {},
                 need_LWT: bool = True, LWT_params: dict = {},
                 need_FWHT: bool = True, FWHT_params: dict = {},
                 decomposition_type: Literal['SVD', 'QR', 'Schur'] = 'SVD',
                 decomposition_params: dict = {},
                 frame_size: int = 256, # has to be a power of 4
                 ):

        self.encryption_type = encryption_type
        self.encryption_params = encryption_params

        self.need_LWT = need_LWT
        if need_LWT:
            self.LWT = BatchLWT(**LWT_params)

        self.need_FWHT = need_FWHT
        if need_FWHT:
            self.FWHT = BatchFWHT(**FWHT_params)

        self.decomposition_type = decomposition_type
        
        if decomposition_type == 'SVD':
            self.embedder = embed_SVD
            self.extractor = extract_SVD
        elif decomposition_type == 'QR':
            self.embedder = embed_QR
            self.extractor = extract_QR
        elif decomposition_type == 'Schur':
            self.embedder = embed_Schur
            self.extractor = extract_Schur
        elif decomposition_type == 'Cholesky':
            self.embedder = embed_Cholesky
            self.extractor = extract_Cholesky
        elif decomposition_type == 'LU':
            self.embedder = embed_LU
            self.extractor = extract_LU
        else:
            raise ValueError(f"Unsupported decomposition type: {decomposition_type}")
            
        self.decomposition_params = decomposition_params
        self.frame_size = frame_size
        self.bits_original_shape = (-1,)

    def encode(self, audio: Union[torch.Tensor, np.ndarray], bits: Union[torch.Tensor, np.ndarray]):
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio).float()
        audio = audio.squeeze()

        self.bits_original_shape = bits.shape
        if not torch.is_tensor(bits):
            bits = torch.from_numpy(bits).float()

        if self.encryption_type != 'none':
            bits = torch.from_numpy(
                encrypt_watermark(
                                    np.array(bits), method = self.encryption_type,
                                    encryption_params=self.encryption_params
                )
            )

        num_bits = bits.shape[0]

        audio_length = audio.shape[0]
        assert num_bits * self.frame_size <= audio_length, "Too many bits for the given audio sample and frame size"

        num_frames = audio_length // self.frame_size

        bits_with_repetitions = bits[torch.arange(num_frames) % num_bits]
        frames = audio[:num_frames * self.frame_size].reshape(num_frames, -1)
        leftover = audio[num_frames * self.frame_size:]

        if self.need_LWT:
            frames = torch.from_numpy(self.LWT.forward_transform(np.array(frames)))

        if self.need_FWHT:
            frames = torch.from_numpy(self.FWHT.forward_transform(np.array(frames)))

        n = int(np.sqrt(frames.shape[1]))
        frames_reshaped = frames.reshape(frames.shape[0], n, n)

        x = self.embedder(frames_reshaped, bits_with_repetitions, **self.decomposition_params)
        
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()

        frames = x.reshape(frames.shape)

        if self.need_FWHT:
            frames = torch.from_numpy(self.FWHT.inverse_transform(np.array(frames)))

        if self.need_LWT:
            frames = torch.from_numpy(self.LWT.inverse_transform(np.array(frames)))

        watermarked_audio = torch.cat((frames.reshape(-1), leftover), dim=0)
        return watermarked_audio

    def _decode_with_repetitions(self, audio: Union[torch.Tensor, np.ndarray]):
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio).float()
        audio = audio.squeeze()

        audio_length = audio.shape[0]
        num_frames = audio_length // self.frame_size

        frames = audio[:num_frames * self.frame_size].reshape(num_frames, -1)
        leftover = audio[num_frames * self.frame_size:]

        if self.need_LWT:
            frames = torch.from_numpy(self.LWT.forward_transform(np.array(frames)))

        if self.need_FWHT:
            frames = torch.from_numpy(self.FWHT.forward_transform(np.array(frames)))

        n = int(np.sqrt(frames.shape[1]))
        frames_reshaped = frames.reshape(frames.shape[0], n, n)

        bits_with_repetitions = self.extractor(frames_reshaped, **self.decomposition_params)

        return bits_with_repetitions

    @staticmethod
    def _collapse_repetitions(bits_with_repetitions: Union[torch.Tensor, np.ndarray],
                             num_bits: int, include_leftover_in_mean: bool = True):
        audio_length = bits_with_repetitions.shape[0]
        num_repetitions = audio_length//num_bits
        no_leftover = bits_with_repetitions[:num_repetitions*num_bits].reshape((num_repetitions, num_bits))
        leftover = bits_with_repetitions[num_repetitions*num_bits:]
        mean_bit_values = no_leftover.type(torch.float64).mean(axis = 0)
        if include_leftover_in_mean:
            leftover_length = audio_length - num_repetitions*num_bits
            mean_bit_values[:leftover_length]  = mean_bit_values[:leftover_length]*num_repetitions + leftover
            mean_bit_values[:leftover_length] /= num_repetitions + 1
        return mean_bit_values, no_leftover, leftover

    def decode(self, audio: Union[torch.Tensor, np.ndarray],
                            num_bits: int, include_leftover_in_mean: bool = True):
        
        bits_with_repetitions = self._decode_with_repetitions(audio)
        if num_bits != -1:
            mean_bits = self._collapse_repetitions(bits_with_repetitions, num_bits,
                                                include_leftover_in_mean = include_leftover_in_mean
                                                )[0]
            bits = torch.round(mean_bits)
        else:
            bits = bits_with_repetitions
            if self.encryption_type != 'none':
                warnings.warn("Decryption is likely to fail when num_bits = -1")

        if self.encryption_type != 'none':
            bits = torch.from_numpy(
                decrypt_watermark(
                                    np.array(bits), original_shape = self.bits_original_shape,
                                    method = self.encryption_type,
                                    encryption_params=self.encryption_params
                )
            )
        return bits