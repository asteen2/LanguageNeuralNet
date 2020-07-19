from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import splitstring
from sklearn.preprocessing import OneHotEncoder

FileName = "/Users/Aidia/Documents/SummerResearch2020/LanguageNeuralNet/Data/MasterList.csv"
X = splitstring.LetArrayToNumArray(splitstring.CSVToNumpy(FileName))
Y = splitstring.GetLanguageIndex(FileName)



# encoder = OneHotEncoder(categories='auto')
# Y_OneHot = encoder.fit_transform(Y.reshape(-1, 1))
# print(Y_OneHot.toarray())

model = Sequential()
model.add(Dense(12, input_dim=20, activation = 'relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs = 150, batch_size = 10)

# _, accuracy = model.evaluate(X, Y)
# print('Accuracy: %.2f' % (accuracy*100))
#
#
# predictions = model.predict([X[1]])
#
# print(X[1], predictions)
