import sys
import torch
from tqdm import tqdm
from segmenter import Segmenter

class Prerocessor(Segmenter):
    def __init__(self, tensor_path=None):
        super().__init__()

        if tensor_path is None:
            fn = sys.argv[0]
            fn = fn[fn.find('/')+1:fn.find('.')]
            tensor_path = f"../data/tensor-data/{fn}.pt"

        X, y = [], []
        for datum, label in tqdm(zip(self.data, self.labels)):
            datum = self.average(arr=datum[0], output_shape=[1, 34, 34, 50, -1])
            X.append(datum)
            y.append(label)
        X, y = torch.Tensor(X), torch.Tensor(y).long()-1
        torch.save([X, y], tensor_path)

        print(f"Data saved at {tensor_path}")

preprocessor = Prerocessor()
