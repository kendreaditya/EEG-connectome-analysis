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
        dataums = np.array(list(zip(self.data, self.labels)))

        # Preprocessing test set
        print("Averging test set.")
        testset = dataums[66:]
        test_data = self.split_bands(testset)
        del testset

        # Preprocessing train/validation set
        print("Averging train/val set.")
        train_val = dataums[:66]
        train_data = self.split_bands(train_val)
        del train_val

        del dataums
        print("Saving datasets.")
        torch.save({"test":test_data, "train":train_data}, tensor_path)
        print(f"Data saved at {tensor_path}")

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
                    # time_data.shape = [99,1,130,34,34]
                    time_data.append(self.average(datum[i][:,:,:,:,n], [34, 34, -1]))
                band_data[list(band_data.keys())[i]].append([time_data])
            band_data["labels"].append(label-1.)

        # converts data in list to torch.tensor
        for key in band_data.keys():
            band_data[key] = torch.Tensor(band_data[key])
            if key == "labels_test":
                band_data[key] = band_data[key].long()
        return band_data

preprocessor = Prerocessor()
