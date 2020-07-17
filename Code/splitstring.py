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
    """
    input: CSV file with values: letter string, language index
    output: numpy array of individual letters from the strings in the input file
    """
    # create pandas DataFrame from input CSV, columns "String" & "LanguageIndex"
    DF = pd.read_csv(CSVFileName, names = ["String", "LanguageIndex"])
    print(DF)
    # ignore DF's 2nd column. Take DF's 1st column, split the string into
    # individual letters, and turn them into a DataFrame
    SplitDF = DF.String.str.split('',expand=True)
    print(SplitDF)
    # turn SplitDF into a numpy array, removing the empty first and last columns
    # (they're created when you split)
    NumpyArray = SplitDF.iloc[:, 1:-1].to_numpy()
    return(NumpyArray)


def GetLanguageIndex(CSVFileName):
    """
    input: CSV file with values: letter string, language index
    output: one-dimensional numpy DataFrame of the language index values from
        the input file
    """
    DF = pd.read_csv(CSVFileName, names = ["String", "LanguageIndex"])
    # print(DF)
    # SplitDF = DF.String.str.split('',expand=True)
    #print(SplitDF)
    # NumpyArray = SplitDF.iloc[:, 1:-1].to_numpy()
    return(DF.LanguageIndex.to_numpy())


def MakeDict(NpArray, ToLetter = False):
    """
    input:
    output:
    """
    UnqLetters = np.unique(NpArray)
    NumToLetDict = dict(enumerate(UnqLetters))
    LetToNumDict = {value: key for key, value in NumToLetDict.items()}
    if ToLetter == True:
        return(NumToLetDict)
    else:
        return(LetToNumDict)


def LetArrayToNumArray(NpLetArray):
    """
    input:
    output:
    """
    LetToNumDict = MakeDict(NpLetArray)
    ArrayShape = NpLetArray.shape
    Output = np.array([LetToNumDict[Let] for Let in NpLetArray.flatten()])
    return(Output.reshape(ArrayShape))

# write another function, opposite of LetArrayToNumArray(): Number array to
# number array
