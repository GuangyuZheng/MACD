import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image


def save_h5(h5f, data, target, dtype=None):
    with h5py.File(h5f) as dfile:
        shape_list = list(data.shape)
        if not dfile.__contains__(target):
            shape_list[0] = None
            dataset = dfile.create_dataset(target, data=data, dtype=dtype, maxshape=tuple(shape_list), chunks=True)
            return
        else:
            dataset = dfile[target]
        len_old = dataset.shape[0]
        len_new = len_old + data.shape[0]
        shape_list[0] = len_new
        dataset.resize(tuple(shape_list))
        dataset[len_old:len_new] = data
        return


class HDF5Dataset(Dataset):

    def __init__(self, file_path, attributes):
        super().__init__()
        self.file_path = file_path
        self.attributes = attributes
        with h5py.File(self.file_path, 'r') as dfile:
            self.len = dfile['input_ids'].shape[0]

    def __getitem__(self, index):
        vals = []
        for attr in self.attributes:
            val = torch.tensor(self.get_data(attr, index))
            vals.append(val)
        return tuple(vals)

    def __len__(self):
        return self.len

    def get_data(self, attribute, idx):
        with h5py.File(self.file_path, 'r') as dfile:
            return dfile[attribute][idx]


class HDF5DatasetWithImage(Dataset):

    def __init__(self, file_path, attributes, transform):
        super().__init__()
        self.file_path = file_path
        self.attributes = attributes
        self.transform = transform
        with h5py.File(self.file_path, 'r') as dfile:
            self.len = dfile['input_ids'].shape[0]

    def __getitem__(self, index):
        vals = []
        for attr in self.attributes:
            if attr == 'images':
                image = Image.open(self.get_data(attr, index)).convert('RGB')
                val = self.transform(image)
            else:
                val = torch.tensor(self.get_data(attr, index))
            vals.append(val)
        return tuple(vals)

    def __len__(self):
        return self.len

    def get_data(self, attribute, idx):
        with h5py.File(self.file_path, 'r') as dfile:
            return dfile[attribute][idx]
