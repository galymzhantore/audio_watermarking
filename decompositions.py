import torch
import numpy as np
from scipy.linalg import schur

def embed_SVD(M, bits, beta=10):
    if not torch.is_tensor(M):
        M = torch.from_numpy(M).float()
    
    if M.dim() == 2:
        M = M.unsqueeze(0)
        bits = torch.tensor([bits]) if not torch.is_tensor(bits) else bits.unsqueeze(0)
    
    batch_size = M.shape[0]
    results = []
    
    for i in range(batch_size):
        U, S, Vh = torch.linalg.svd(M[i], full_matrices=False)
        S1 = S[0]
        Y_i = torch.round(beta * S1)
        B_i = Y_i % 2
        
        if bits[i].item() == 1:
            if B_i == 0:
                Y_i_prime = Y_i + 1
            else:
                Y_i_prime = Y_i
        else:
            if B_i == 0:
                Y_i_prime = Y_i
            else:
                Y_i_prime = Y_i + 1
        
        S_prime = S.clone()
        S_prime[0] = Y_i_prime / beta
        S_diag = torch.diag(S_prime)
        results.append(U @ S_diag @ Vh)
    
    result = torch.stack(results)
    return result.squeeze() if result.shape[0] == 1 else result

def extract_SVD(M, beta=10):
    if not torch.is_tensor(M):
        M = torch.from_numpy(M).float()
    
    if M.dim() == 2:
        M = M.unsqueeze(0)
    
    batch_size = M.shape[0]
    extracted_bits = []
    
    for i in range(batch_size):
        _, S, _ = torch.linalg.svd(M[i], full_matrices=False)
        S1 = S[0]
        Y_i_star = torch.round(beta * S1)
        B_i_star = Y_i_star % 2
        extracted_bit = 1 if B_i_star == 1 else 0
        extracted_bits.append(extracted_bit)
    
    result = torch.tensor(extracted_bits, dtype=torch.long)
    return result.squeeze() if len(result) == 1 else result

def embed_QR(M, bits, D=0.01, C=1):
    if not torch.is_tensor(M):
        M = torch.from_numpy(M).float()
    
    if M.dim() == 2:
        M = M.unsqueeze(0)
        bits = torch.tensor([bits]) if not torch.is_tensor(bits) else bits.unsqueeze(0)
    
    batch_size = M.shape[0]
    results = []
    
    for i in range(batch_size):
        Q, R = torch.linalg.qr(M[i])
        R_flat = R.flatten()
        idx = torch.argmax(torch.abs(R_flat))
        row = idx // R.shape[1]
        col = idx % R.shape[1]
        
        R_max = R[row, col]
        Y_i = torch.round(R_max / D)
        M_param = 2 * C
        
        if bits[i].item() == 1:
            Y_i_prime = Y_i + C - (Y_i % M_param)
        else:
            Y_i_prime = Y_i + C - ((Y_i + C) % M_param)
        
        R_prime = R.clone()
        R_prime[row, col] = Y_i_prime * D
        results.append(Q @ R_prime)
    
    result = torch.stack(results)
    return result.squeeze() if result.shape[0] == 1 else result

def extract_QR(M, D=0.01, C=1):
    if not torch.is_tensor(M):
        M = torch.from_numpy(M).float()
    
    if M.dim() == 2:
        M = M.unsqueeze(0)
    
    batch_size = M.shape[0]
    extracted_bits = []
    
    for i in range(batch_size):
        _, R = torch.linalg.qr(M[i])
        R_flat = R.flatten()
        idx = torch.argmax(torch.abs(R_flat))
        row = idx // R.shape[1]
        col = idx % R.shape[1]
        
        R_max = R[row, col]
        Y_i_star = torch.round(R_max / D)
        M_param = 2 * C
        extracted_bit = 1 if (Y_i_star % M_param).item() == 1 else 0
        extracted_bits.append(extracted_bit)
    
    result = torch.tensor(extracted_bits, dtype=torch.long)
    return result.squeeze() if len(result) == 1 else result

def embed_Schur(M, bits, D=0.01, C=1):
    if torch.is_tensor(M):
        M_np = M.numpy()
    else:
        M_np = M
    
    batch_size = M_np.shape[0]
    results = []
    
    for i in range(batch_size):
        T, Z = schur(M_np[i])
        idx = np.unravel_index(np.argmax(np.abs(T)), T.shape)
        Y_i = np.round(T[idx] / D)
        M_param = 2 * C
        
        if bits[i] == 1:
            Y_i_prime = Y_i + C - (Y_i % M_param)
        else:
            Y_i_prime = Y_i + C - ((Y_i + C) % M_param)
        
        T_prime = T.copy()
        T_prime[idx] = Y_i_prime * D
        results.append(Z @ T_prime @ Z.conj().T)
    
    return torch.from_numpy(np.stack(results)).float()

def extract_Schur(M, D=0.01, C=1):
    if torch.is_tensor(M):
        M_np = M.numpy()
    else:
        M_np = M
    
    batch_size = M_np.shape[0]
    results = []
    
    for i in range(batch_size):
        T, _ = schur(M_np[i])
        idx = np.unravel_index(np.argmax(np.abs(T)), T.shape)
        Y_i_star = np.round(T[idx] / D)
        M_param = 2 * C
        extracted_bit = 1 if (Y_i_star % M_param) == 1 else 0
        results.append(extracted_bit)
    
    return torch.tensor(results).long()