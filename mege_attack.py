from typing import Literal, Union, Optional
import torch
import numpy as np
import warnings

from LWT_FWHT import BatchLWT, BatchFWHT


def mege_attack(audio: Union[torch.Tensor, np.ndarray], frame_sizes, offsets=None, need_LWT=False, need_FWHT=True,):
    if offsets is None:
        offsets = [0]
    offset = offsets[0]
    res = audio

    # for offset in offsets:
    #     print(res.shape)
    #     res[offset:] = self._apply(res[offset:])
    for frame in frame_sizes:
        res[offset:] = _apply_mege_attack(res[offset:], this_frame_size=frame, need_LWT=need_LWT, need_FWHT=need_FWHT)
    return res
        

def _apply_mege_attack(audio: Union[torch.Tensor, np.ndarray], this_frame_size, need_LWT, need_FWHT,):
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio).float()
    audio = audio.squeeze()

    audio_length = audio.shape[0]

    num_frames = audio_length // this_frame_size

    frames = audio[:num_frames * this_frame_size].reshape(num_frames, -1)
    leftover = audio[num_frames * this_frame_size:]

    if need_LWT:
        frames = torch.from_numpy(BatchLWT().forward_transform(np.array(frames)))

    if need_FWHT:
        frames = torch.from_numpy(BatchFWHT().forward_transform(np.array(frames)))

    n = int(np.sqrt(frames.shape[1]))
    frames_reshaped = frames.reshape(frames.shape[0], n, n)

    x = _svd_attack(frames_reshaped)
    
    if not torch.is_tensor(x):
        x = torch.from_numpy(x).float()

    frames = x.reshape(frames.shape)

    if need_FWHT:
        frames = torch.from_numpy(BatchFWHT().inverse_transform(np.array(frames)))

    if need_LWT:
        frames = torch.from_numpy(BatchLWT().inverse_transform(np.array(frames)))

    n = int(np.sqrt(frames.shape[1]))
    frames_reshaped = frames.reshape(frames.shape[0], n, n)
    x = _svd_attack(frames_reshaped)

    watermarked_audio = torch.cat((frames.reshape(-1), leftover), dim=0)


    return watermarked_audio

def _svd_attack(M):
    if not torch.is_tensor(M):
        M = torch.from_numpy(M).float()
    
    if M.dim() == 2:
        M = M.unsqueeze(0)
    
    batch_size = M.shape[0]
    results = []
    
    for i in range(batch_size):
        U, S, Vh = torch.linalg.svd(M[i], full_matrices=False)
        
        S_prime = S.clone()
        blend = 0.5

        S_prime[0] = blend * S[0] + (1 - blend) * S[1]
        
        # S_prime[1]=S_prime[0]
        # S_prime[1] = 0.9 * S[1] + (1 - 0.9) * S[2]
        # S_prime[1] = 1.1 * S[1]
        # S_prime[2] = S[1]
        # S_prime *= np.random.normal(loc=1, scale=0.01, size=len(S))
        # S_prime *= np.exp(-np.linspace(0, 10, len(S_prime))**2)

        S_diag = torch.diag(S_prime)

        tmp = U @ S_diag @ Vh
        tmp = tmp.numpy()
        tmp = tmp.flatten()
        # numels = int(np.round(len(S_prime)*0.2))
        # numels = 2
        # inds = np.argpartition(tmp, numels)[-numels:]
        # print(tmp.shape, inds)
        # print(tmp[inds])
        # print()
        # tmp[inds] = np.mean(np.abs(tmp[inds]))
        # tmp = tmp.reshape(tmp_shp)

        results.append(torch.from_numpy(tmp))
    
    result = torch.stack(results)
    return result.squeeze() if result.shape[0] == 1 else result