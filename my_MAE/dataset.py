import hdf5storage
import numpy as np
import torch


def read_data(dataset_path, device):
    raw_data = hdf5storage.loadmat(dataset_path)['H_dg_all_norm']
    raw_data = np.expand_dims(raw_data, axis=0)
    raw_data = np.vstack((np.real(raw_data), np.imag(raw_data)))
    raw_data = raw_data.astype(np.float32)
    raw_data = np.transpose(raw_data, (2, 1, 0, 3, 4))
    raw_data = np.reshape(raw_data, (-1, 2, 32, 32))
    train_dataset = raw_data[:int(0.8 * len(raw_data))]
    test_dataset = raw_data[int(0.8 * len(raw_data)):]
    return torch.tensor(train_dataset, dtype=torch.float32, device=device), torch.tensor(test_dataset, dtype=torch.float32, device=device)
