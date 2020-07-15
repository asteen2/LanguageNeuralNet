from numpy import loadtxt
#from keras.models import Sequential
#from keras.layers import Dense
import pandas as pd

InputFileName = '../Data/MasterList.csv'

DF = pd.read_csv(InputFileName, names = ["String", "LanguageIndex"])
print(DF)
DF = DF.String.str.split('',expand=True)
print(DF)

X = DF.iloc[:, 1:-1].to_numpy()
#DF

print(X)
