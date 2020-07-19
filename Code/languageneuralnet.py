from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
import splitstring
from sklearn.preprocessing import OneHotEncoder

FileName = "/Users/Aidia/Documents/SummerResearch2020/LanguageNeuralNet/Data/MasterList.csv"
X = splitstring.LetArrayToNumArray(splitstring.CSVToNumpy(FileName))
Y = splitstring.GetLanguageIndex(FileName)

TrainPercent = 0.9
np.random.seed(100)
ShuffleIndex = np.random.permutation(Y.shape[0])
XShuffled, YShuffled = X[ShuffleIndex], Y[ShuffleIndex]
SplitBorder = int(np.round(TrainPercent * X.shape[0]))

XTrain = XShuffled[:SplitBorder]
YTrain = YShuffled[:SplitBorder]
XTest = XShuffled[SplitBorder:]
YTest = YShuffled[SplitBorder:]

# encoder = OneHotEncoder(categories='auto')
# Y_OneHot = encoder.fit_transform(Y.reshape(-1, 1))
# print(Y_OneHot.toarray())

model = Sequential()
model.add(Dense(12, input_dim=20, activation = 'relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(XTrain, YTrain, epochs = 150, batch_size = 10)

# _, accuracy = model.evaluate(X, Y)
# print('Accuracy: %.2f' % (accuracy*100))
#
#
# predictions = model.predict([X[1]])
#
# print(X[1], predictions)
