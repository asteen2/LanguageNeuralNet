from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import splitstring

FileName = "../Data/MasterList.csv"
X = splitstring.LetArrayToNumArray(splitstring.CSVToNumpy(FileName))
Y = splitstring.GetLanguageIndex(FileName)
