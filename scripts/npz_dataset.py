import torch
import numpy as np
import torch.utils.data as data
# import h5py
import pdb

class NpzDataset(data.Dataset):
    def __init__(self, data_dir, normalize=True, permute=True, rank=0, world_size=1):
        # data original ranges from 0 to 255, data type are unit8
        # if normalize, we range it from -1 to 1 and make it float type
        
        data = np.load(data_dir)
        # print(data)
        self.input_data = data['arr_0'] # B,H,W,3
        if 'arr_1' in data.files:
            self.labels = data['arr_1'].astype(int)
        else:
            self.labels = np.zeros(self.input_data.shape[0], dtype=int)
        data.close()

        if permute: # input_data is of shape B,H,W,3
            self.input_data = self.input_data.transpose(0,3,1,2)
            # now it is of shape B,3,H,W
        # else: input_data is of shape B,3,H,W
        
        if world_size > 1:
            num_samples_per_rank = int(np.ceil(self.input_data.shape[0] / world_size))
            start = rank * num_samples_per_rank
            end = (rank+1) * num_samples_per_rank
            self.input_data = self.input_data[start:end]
            self.labels = self.labels[start:end]
            self.num_samples_per_rank = num_samples_per_rank
        else:
            self.num_samples_per_rank = self.input_data.shape[0]

        if normalize:
            self.input_data = (self.input_data.astype(np.float32)/255) * 2 - 1 # from -1 to 1
        
        print('dataset %s:' % data_dir)
        print('input_data:', self.input_data.shape)
        print('labels:', self.labels.shape)
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = self.input_data[index]
        label = self.labels[index]
        # label = 0
        return x, label

class DummyDataset(data.Dataset):
    def __init__(self, num_samples, rank=0, world_size=1):
        self.input_data = np.arange(num_samples)
        if world_size > 1:
            num_samples_per_rank = int(np.ceil(self.input_data.shape[0] / world_size))
            start = rank * num_samples_per_rank
            end = (rank+1) * num_samples_per_rank
            self.input_data = self.input_data[start:end]
            self.num_samples_per_rank = num_samples_per_rank
        else:
            self.num_samples_per_rank = self.input_data.shape[0]

        print('dummy dataset:')
        print('input_data:', self.input_data.shape)
        self.len = self.input_data.shape[0]
        # print(self.input_data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = self.input_data[index]
        return x, 0

if __name__ == '__main__':
    import pdb
    path='../evaluations/precomputed/biggan_deep_trunc1_imagenet256.npz'
    dataset = NpzDataset(path, rank=0, world_size=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    for i, (image, label) in enumerate(dataloader):
        print(image.shape, label.shape)
        pdb.set_trace()