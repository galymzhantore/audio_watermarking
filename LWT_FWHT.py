import pywt
import numpy as np
import torch
from hadamard_transform import hadamard_transform

class BatchLWT:
    """Класс для пакетного LWT как в главе 2 книги"""

    def __init__(self, wavelet='haar', level=2, mode='symmetric'):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        # Для сохранения коэффициентов
        self.ca2 = None
        self.cd2 = None
        self.cd1 = None
        self.original_shape = None

    def forward_transform(self, frames):
        """Применяет LWT к батчу фреймов и сохраняет коэффициенты"""
        #frames.shape = (num_frames, frame_len)
        self.original_shape = frames.shape

        # Применяем LWT ко всем фреймам по последней оси
        coeffs = pywt.wavedec(frames, self.wavelet, level=self.level,
                             axis=1, mode=self.mode)

        # Сохраняем коэффициенты
        self.ca2 = coeffs[0]  # shape: (num_frames, frames_len/4)
        self.cd2 = coeffs[1]  # shape: (num_frames, frames_len/4)
        self.cd1 = coeffs[2]  # shape: (num_frames, frames_len/2)

        #type is np.array
        return self.ca2.copy()

    def inverse_transform(self, modified_ca2):
        """Восстанавливает исходные фреймы из модифицированных A2 коэффициентов"""
        if self.ca2 is None:
            raise ValueError("Сначала вызовите forward_transform()")

        # Проверяем размеры
        if modified_ca2.shape != self.ca2.shape:
            raise ValueError(f"Ожидалась форма {self.ca2.shape}, получена {modified_ca2.shape}")

        # Собираем коэффициенты с модифицированными A2
        modified_coeffs = [modified_ca2, self.cd2, self.cd1]

        # Восстанавливаем фреймы
        reconstructed = pywt.waverec(modified_coeffs, self.wavelet,
                                    axis=1, mode=self.mode)

        #type is np.array
        return reconstructed
    

class BatchFWHT:
    """Класс для пакетного FWHT как в книге"""

    def __init__(self):
        self.original_shape = None

    def forward_transform(self, coefficients):
        """
        coefficients: numpy array или torch tensor формы (num_frames, coeff_length)
        coeff_length должно быть степенью 2
        """
        self.original_shape = coefficients.shape

        # Конвертируем в torch tensor если нужно
        if not isinstance(coefficients, torch.Tensor):
            coefficients = torch.tensor(coefficients, dtype=torch.float64)


        # Применяем FWHT ко всем фреймам
        fwht_result = hadamard_transform(coefficients)

        return fwht_result.numpy()

    def inverse_transform(self, fwht_coefficients):
        """Обратное FWHT для восстановления исходных коэффициентов"""
        if not isinstance(fwht_coefficients, torch.Tensor):
            fwht_coefficients = torch.tensor(fwht_coefficients, dtype=torch.float64)


        # Обратное преобразование = прямое для нормализованного FWHT
        original = hadamard_transform(fwht_coefficients)

        return original.numpy()