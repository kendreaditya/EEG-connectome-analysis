from preprocessor import Preprocessor
from tqdm import tqdm

pp = Preprocessor()
def twoD_10_data(data, labels, path='../data/TensorDataset/twoD_10_data.pt'):
    dataset = {'X':[], 'y':[]}
    for datum, label in tqdm(zip(pp.data, pp.labels)):
        datum = data_segmentation(datum[0], [1, 34,34, -1], 13, 3)
        for X in datum:
            dataset['X'].append(X)
            dataset['y'].append(label-1)
    torch.save(to_TensorDataset(dataset),path)

def twoD_data(data, labels,path='../data/TensorDataset/twoD_data.pt'):
    dataset = {'X':[], 'y':[]}
    for datum, label in tqdm(zip(data, labels)):
        datum = average(datum[0], [1, 34, 34, -1])
        dataset['X'].append(datum)
        dataset['y'].append(label-1)

    #np.save('../data/twoD_data.npy', np.array([dataset['X'], dataset['y']]))
    torch.save(to_TensorDataset(dataset),path)

def twoD_all_data(data, labels,path='../data/TensorDataset/twoD_test_data.pt'):
    dataset = {'X':[], 'y':[]}
    for i in tqdm(range(2,131)):
        for datum, label in zip(data, labels):
            try:
                datum = data_segmentation(datum[0], [1, 34,34, -1], i, 3)
                for x in datum:
                    dataset['X'].append(x)
                    dataset['y'].append(label-1)
            except:
                pass
    torch.save(to_TensorDataset(dataset), path)

def twoD_all_split_data(data, labels):
    sdata = np.array(list(zip(data, labels)))
    np.random.shuffle(sdata)

    test = sdata[90:99]
    twoD_data(test[:,0], test[:,1], path='../data/tensordataset/twoD_test_data.pt')

    train_val = sdata[:90]
    dataset = {'X':[], 'y':[]}
    for i in tqdm(range(1,131)):
        for datum, label in train_val:
            try:
                datum = data_segmentation(datum[0], [1, 34,34, -1], i, 3)
                for x in datum:
                    dataset['X'].append(x)
                    dataset['y'].append(label-1)
            except:
                pass
    torch.save(to_TensorDataset(dataset), '../data/tensordataset/twoD_all_train_data.pt')


