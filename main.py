import sys

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from models.svm import SVM_model
from models.ema import EMA_model
from models.dst import DST_model
from models.combo import Combo

""" TODO:
Implement a dummy agent according to previous experiment:

Models:
    Implement a HMM that looks at the figure of the last three days and estimates the next day. States are -, ^, v, / and \.
    Fitting a none-linear curve and extraploate.
    Implement DT as combiner.
    

"""
def read_data(path, ratio=0.8):
    # Read all data from file
    all_data = np.genfromtxt(path, delimiter=',')
    # Remove header row and date column.
    all_data = all_data[1::,1::]
    # Some data rows may have faulty numbers (np.nan), these are iterpolated with the
    # mean of the row before and after.
    for i in range(np.shape(all_data)[0]):
        if any(np.isnan(all_data[i])):
            for k in range(1, 10):
                if not any(np.isnan(all_data[i+k])):
                    break
            for j in range(np.shape(all_data)[1]):
                all_data[i,j] = (all_data[i-1,j] + all_data[i+k,j])/2.0
    # Add a column for mid values.
    all_data = np.c_[(all_data[:,1]+all_data[:,2])/2.0, all_data]
    # Split into training and test set
    nr_train = int(np.shape(all_data)[0]*ratio)
    train_data = all_data[:nr_train,:]
    test_data = all_data[nr_train:,:]
    # Scale the data to [0, 1]
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    # Convert to dictionary and return
    return mat_2_dic(train_data), mat_2_dic(test_data)

def mat_2_dic(mat):
    return {header:mat[:,i] for i, header in enumerate(["mid","open","high","low","close","adj" "close","volume"])}

def agent(predictions, truths):
    nr_buys = 0
    nr_sales = 0
    purse = 0
    owned_stocks = 0
    bought = False
    for day in range(len(predictions)-1):
        # If we predict a higher price tomorrow and we have not bought -> buy
        if truths[day] < predictions[day + 1] and not bought:
            purse -= 1
            owned_stocks = 1/truths[day] 
            bought = True
            nr_buys += 1
        # If we predict the stock to fall and we have bought -> sell 
        elif truths[day] > predictions[day + 1] and bought:
            purse += owned_stocks*truths[day]
            bought = False
            nr_sales += 1
    purse += truths[len(predictions)]
    return {"purse":purse, "sales":nr_sales, "buys":nr_buys}

files = ["data/HM.csv", "data/AAL.csv", "data/VOLVOB.csv"]
for data_file in files:
    print("_______Running: {}_______".format(data_file))
    train_data, test_data = read_data(data_file)
    combo = Combo(train_data,  test_data)
    combo.fit()
    combo.test()
    print()

"""
for depth in range(1, 10):
    dts = DST_model(max_depth=depth)
    acc = []
    for data_file in files:
        train_data, test_data = read_data(data_file)
        dts.fit(train_data)
        acc.append(dts.test(test_data)*100)
    print("Depth: {} {}".format(depth, ", ".join("{:4.2f}%".format(a) for a in acc)))

train_data, test_data = read_data("data/AAL.csv")
ema = EMA_model()
ema.fit(train_data)
ema.test(test_data, verbose=True)

train_data, test_data = read_data("data/HM.csv")
ema = EMA_model()
ema.fit(train_data)
ema.test(test_data, verbose=True)

train_data, test_data = read_data("data/VOLVOB.csv")
ema = EMA_model()
ema.fit(train_data)
ema.test(test_data, verbose=True)
"""


"""
train_data, test_data = read_data("data/AAL.csv")
svm = SVM_model()
svm.fit(train_data)
svm.test(test_data, verbose=True)

train_data, test_data = read_data("data/HM.csv")
svm = SVM_model()
svm.fit(train_data)
svm.test(test_data, verbose=True)

train_data, test_data = read_data("data/VOLVOB.csv")
svm = SVM_model()
svm.fit(train_data)
svm.test(test_data, verbose=True)
#svm.predict(test_data)
"""
#aal = read_data("data/HM.csv")
#aal = read_data("data/VOLVOB.csv")
#aal = np.genfromtxt('data/AAL.csv', delimiter=',')
#print(aal)

#hmb = np.genfromtxt('data/HM-B.ST.csv', delimiter=',')
#print(hmb)

#vol = np.genfromtxt('data/VOLV-B.ST.csv', delimiter=',')
#print(vol)
