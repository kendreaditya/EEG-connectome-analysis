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

        dataums = np.array(list(zip(self.data, self.labels)))

        testset = dataums[66:]
        data_test = {"delta_test":[],
                     "theta_test":[],
                     "alpha_test":[],
                     "beta_test":[],
                     "all_test":[],
                     "labels_test":[]}

        print("averging test set.")
        for datum, label in tqdm(testset):
            datum = [datum[:,:,:,0:3].tolist(), datum[:,:,:,3:7].tolist(), datum[:,:,:,7:12].tolist(), datum[:,:,:,12:30].tolist(), dataums[:,:,:0:30].tolist()]
            for i in range(len(datum)):
                data_test[data_test.keys()[i]].append(self.average(datum[i], [1, 34, 34, len(datum[i][0][0][0])]))
            data_test["labels_test"].append(label-1.)

        # converts data in list to torch.tensor
        for key in data_test.keys():
            data_test[key] = torch.Tensor(data_test[key])
            if key == "labels_test":
                data_test[key] = data_test[key].long()
        del testset

        train_val = dataums[:66]
        del dataums
        data_train = {"delta_train":[],
                     "theta_train":[],
                     "alpha_train":[],
                     "beta_train":[],
                     "all_train":[],
                     "labels_train":[]}

        print("Averging training/validation set.")
        for datum, label in tqdm(train_val):
            datum = [datum[:,:,:,0:3].tolist(), datum[:,:,:,3:7].tolist(), datum[:,:,:,7:12].tolist(), datum[:,:,:,12:30].tolist(), dataums[:,:,:0:30].tolist()]
            for i in range(len(datum)):
                data_train[data_train.keys()[i]].append(self.average(datum[i], [1, 34, 34, len(datum[i][0][0][0])]))
            data_train["labels_train"].append(label-1.)

        # converts data in list to torch.tensor
        for key in data_train.keys():
            data_train[key] = torch.tensor(data_train[key])
            if key == "labels_train":
                data_train[key] = data_train[key].long()
        del train_val

        torch.save([data_test, data_train], tensor_path)
        print(f"Data saved at {tensor_path}")

    def split_bands(self, dataset):
        band_data = {"delta":[],
                 "theta":[],
                 "alpha":[],
                 "beta":[],
                 "all":[],
                 "labels":[]}

        print("Averging dataset.")
        for datum, label in tqdm(dataset):
            datum = [datum[:,:,:,0:3].tolist(), datum[:,:,:,3:7].tolist(), datum[:,:,:,7:12].tolist(), datum[:,:,:,12:30].tolist(), dataums[:,:,:0:30].tolist()]
            for i in range(len(datum)):
                band_data[band_data.keys()[i]].append(self.average(datum[i], [1, 34, 34, len(datum[i][0][0][0])]))
            band_data["labels"].append(label-1.)

        # converts data in list to torch.tensor
        for key in band_data.keys():
            band_data[key] = torch.Tensor(band_data[key])
            if key == "labels_test":
                band_data[key] = band_data[key].long()
        return band_data


preprocessor = Prerocessor()
