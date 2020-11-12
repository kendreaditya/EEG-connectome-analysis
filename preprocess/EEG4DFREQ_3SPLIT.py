import sys
import numpy as np
import torch
from tqdm import tqdm
from segmenter import Segmenter

class Prerocessor(Segmenter):
    def __init__(self, tensor_path=None):
        super().__init__()
        self.data = np.array(self.data)

        if tensor_path is None:
            fn = sys.argv[0]
            fn = fn[fn.find('/')+1:fn.find('.')]
            tensor_path = f"../data/tensor-data/{fn}.pt"
            del fn

        # dataset
        dataset = {"split_1": {"train":None, "test": None},
                   "split_2": {"trian":None, "test": None},
                   "split_3": {"train":None, "test": None}}

        dataums = np.array(list(zip(self.data, self.labels)))
        indexs = [self.indexs_range([66,77],[77,88],[88,99]),
                  self.indexs_range([0,11],[22,33],[44,55]),
                  self.indexs_range([11,22],[33,44],[55,66])]

        for i, test_idx in zip(range(1,4), indexs):
            train_idx = [x for x in range(len(dataums)) if x not in test_idx]

            print(f"Averging test set {i} of 3.")
            test = np.array([dataums[x] for x in test_idx])
            test_data = self.split_bands(test)

            print(f"Averging train/val set {i} of 3.")
            train = np.array([dataums[x] for x in train_idx])
            train_data = self.split_bands(train)

            del train
            del test

            dataset[f"split_{i}"] = {"train": train_data,
                                     "test": test_data}

        del dataums
        print("Saving datasets.")
        torch.save(dataset, tensor_path)
        print(f"Data saved at {tensor_path}")

    def indexs_range(self, *args):
        indexs = []
        for arg in args:
            indexs += [i for i in range(arg[0], arg[1])]
        return indexs

    def transform_data(self, data):
        data = np.array(data)
        shape = data.shape
        t_data = np.zeros([shape[-1]]+list(shape[:-1]))
        for i in range(shape[-1]):
            t_data[i] = data[:,:,:,i]
        del data
        return t_data

    def split_bands(self, dataset):
        band_data = {"delta":[],
                 "theta":[],
                 "alpha":[],
                 "beta":[],
                 "all":[],
                 "labels":[]}

        for datum, label in tqdm(dataset):
            datum = [datum[:,:,:,0:3], datum[:,:,:,3:7], datum[:,:,:,7:12], datum[:,:,:,12:30], datum[:,:,:,0:30]]
            for i in range(len(datum)):
                time_data = []
                for n in range(len(datum[i][0][0][0][0])):
                    # n = 0-130
                    # appended.shape = [34, 34]
                    # time_data.shape = [130,34,34,4]
                    time_data.append(datum[i][:,:,:,:,n][0])
                time_data = self.transform_data(time_data)
                band_data[list(band_data.keys())[i]].append(time_data)
            band_data["labels"].append(label-1.)

        # converts data in list to torch.tensor
        for key in band_data.keys():
            band_data[key] = torch.Tensor(band_data[key])
            if key == "labels_test":
                band_data[key] = band_data[key].long()
        return band_data

preprocessor = Prerocessor()
