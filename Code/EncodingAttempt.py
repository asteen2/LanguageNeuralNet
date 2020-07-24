import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv("../Data/MasterList.csv", names=['fullString', 'langID'])
data = data.join(data.fullString.str.split("", expand=True))
columns = ['fullString', 'langID']
columns.extend([f"Char{ind:02d}" for ind in data.columns[2:]])
data.columns = columns
data=data.drop(columns=['fullString', 'Char00', 'Char21'])
dataset = data.values

X = dataset[:, 1:]
Y = dataset[:, 0]

X = X.astype('str')
#Y = Y.reshape(-1,1)

X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size=0.33, random_state=1)

print(Y_train)
print(type(Y_train))

# Need to one-hot encode the input values; either use sklearn OneHotEncoder
# or something like enumerate(np.unique(X_train))

def PrepareInputs(X_train, X_test):
    categoryArray = [np.unique(X_train.flatten())
                            for col in np.arange(X_train.shape[1])]
    OHE = OneHotEncoder(categories = categoryArray)
    TrainShape = X_train.shape
    TestShape = X_test.shape
    OHE.fit(X_train)
    X_train_enc = OHE.transform(X_train)
    X_test_enc = OHE.transform(X_test)
    return(X_train_enc, X_test_enc)

def PrepareOutputs(Y_train, Y_test):
    # Print for debug
    print(Y_train)
    LB = LabelBinarizer()
    LB.fit(Y_train)
    # Print for debug
    print(f"LB type:{LB.y_type_}")
    print(LB.classes_)
    # The following line give us this error:
    # ValueError: Unknown label type: (array([2, 2, 0, ..., 2, 0, 2], dtype=object),)
    Y_train_enc = LB.transform(Y_train)
    Y_test_enc = LB.transform(Y_test)
    return(Y_train_enc, Y_test_enc)


X_train_enc, X_test_enc = PrepareInputs(X_train, X_test)
Y_train_enc, Y_test_enc = PrepareOutputs(Y_train, Y_test)

# model = Sequential()
# model.add(Dense(10, input_dim=X_train_enc.shape[1], activation='relu',
#                         kernel_initializer = "he_normal"))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation="softmax"))
#
# model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
# print(X_train_enc)
# model.fit(X_train_enc, Y_train_enc, epochs=50, batch_size=10, verbose=2)
