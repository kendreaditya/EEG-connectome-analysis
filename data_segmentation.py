import sqlite3
from tqdm import tqdm
from mat73 import loadmat
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

def segmentation(arr, output_shape):
    output_shape.append(-1)
    return np.average(np.reshape(arr, output_shape), axis=-1)

PATH = "../data/"
data_files = os.listdir(PATH)

conn = sqlite3.connect(PATH+"data.db")
database = conn.cursor()
database.execute("""CREATE TABLE data (
                 twoD text)""")

for fn in data_files:
    labels = loadmat(PATH+fn)["label"]
    data = loadmat(PATH+fn)["data"]
    for datum, label in tqdm(zip(data, labels)):
        datum = segmentation(datum[0], [34, 34])
        database.execute("INSERT INTO data VALUES (:twoD)", {'twoD': repr([datum, label])})
        conn.commit()
conn.close()
