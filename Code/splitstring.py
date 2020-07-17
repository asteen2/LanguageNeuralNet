import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense
import pandas as pd

# InputFileName = '../Data/MasterList.csv'
#
# DF = pd.read_csv(InputFileName, names = ["String", "LanguageIndex"])
# print(DF)
# SplitDF = DF.String.str.split('',expand=True)
# print(SplitDF)
#
# NumpyArray = DF.iloc[:, 1:-1].to_numpy()
# #DF
#
# print(NumpyArray)


def CSVToNumpy(CSVFileName):
    DF = pd.read_csv(CSVFileName, names = ["String", "LanguageIndex"])
    #print(DF)
    SplitDF = DF.String.str.split('',expand=True)
    #print(SplitDF)
    NumpyArray = SplitDF.iloc[:, 1:-1].to_numpy()
    return(NumpyArray)

def MakeDict(NpArray, ToLetter = False):
    UnqLetters = np.unique(NpArray)
    NumToLetDict = dict(enumerate(UnqLetters))
    LetToNumDict = {value: key for key, value in NumToLetDict.items()}
    if ToLetter == True:
        return(NumToLetDict)
    else:
        return(LetToNumDict)

def LetArrayToNumArray(NpLetArray):
    LetToNumDict = MakeDict(NpLetArray)
    ArrayShape = NpLetArray.shape
    Output = np.array([LetToNumDict[Let] for Let in NpLetArray.flatten()])
    return(Output.reshape(ArrayShape))

# write another function, opposite of LetArrayToNumArray(): Number array to
# number array
