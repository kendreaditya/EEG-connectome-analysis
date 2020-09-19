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

        # Shuffle data and labels
        dataums = np.array(list(zip(self.data, self.labels)))

        testset = dataums[66:]
        test_X, test_y = [], []
        print("Averging test set.")
        for datum, label in tqdm(testset):
            datum = self.average(datum, [1, 34, 34, 50, -1])
            datum = datum[:,:,:,7:11]
            datum = self.average(datum, [1, 34, 34, -1])
            test_X.append(datum)
            test_y.append(label)
        test_X, test_y = torch.Tensor(test_X), torch.Tensor(test_y).long()-1
        del testset

        train_val = dataums[:66]
        del dataums
        print("Averging training/validation set.")
        X, y = [], []
        for i in tqdm(range(2,self.data.shape[-1])):
            for datum, label in train_val:
                # dataum.shape - [1, 34, 34, 50, 130]
                # More data - double split with 2 nested for loops
                splits = np.split(datum, [i], axis=len(datum.shape)-1)

                for split in splits:
                    # try split[0]
                    datum = self.average(arr=split, output_shape=[1, 34, 34, 50, -1])
                    datum = datum[:,:,:,7:11]
                    datum = self.average(datum, [1, 34, 34, -1])
                    X.append(datum)
                    y.append(label)
        del train_val
        #del self.data
        X, y = torch.Tensor(X), torch.Tensor(y).long()-1
        torch.save({'test':[test_X,test_y], 'train/val':[X,y]}, tensor_path)

        print(f"Data saved at {tensor_path}")

preprocessor = Prerocessor()
