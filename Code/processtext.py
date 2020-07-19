import re
import numpy as np
from pathlib import Path
import csv

def ProcessText(FilePath):
    """
    input:
    output:
    """
    file = open(FilePath, "r")
    string = file.read().upper()
    file.close()
    pattern = re.compile(r"[^A-Z\u00C0-\u00DF]")
    return(re.sub(pattern, "", string))

def MakeSection(ProcessedText, MinLength = 20, MaxLength = 21):
    """
    input:
        1) string of English, German, or Spanish text (all caps, no punctuation)
            Typically output of ProcessText()
        2) MinLength = minimum length of returned section
        3) MaxLength = maximum length of returned section
    output: a section of that file (string) with a random length

    MaxLength must be > MinLength. If you want all same-sized sections, set
        MaxLength = MinLength + 1
    """
    MaxPosition = len(ProcessedText)
    StringLength = np.random.randint(MinLength, MaxLength)
    Start = np.random.randint(MaxPosition-(StringLength+1))
    End = Start + StringLength
    return(ProcessedText[Start:End])

def MakeTupleList(ProcessedText, LanguageIndex, NumSections = 500):
    """
    input:
        1) string of English, German, or Spanish text (all caps, no punctuation)
            Typically output of ProcessText()
        2) LanguageIndex = integer assigned to the language in LangDict
        3) NumSections = number of tuples in returned list
    output:
        list of (string, index) tuples
            string = section of processed text
            index = number assigned to language in LangDict
    """
    return(
           [(MakeSection(ProcessedText), LanguageIndex)
                        for item in np.arange(NumSections)]
           )

def TxtToTupleList(FilePath, Language):
    """
    input:
        1) filepath of .txt file
        2) Language of that .txt file (must be listed in LangDict)
    output:
        list of (string, index) tuples
            string = section of processed text
            index = number assigned to language in LangDict
    """
    LangDict = {"English":0, "German":1, "Spanish":2}
    if Language not in LangDict.keys():
        raise ValueError(f"Language {Language} not listed in LangDict.\n" + \
            f"Allowed languages are: {', '.join(LangDict.keys())}")
    ProcessedText = ProcessText(FilePath)
    return(MakeTupleList(ProcessedText, LangDict[Language]))

def MakeCSV(TupleList):
    """
    input:
    output:

    temporary function to write the tuple list into a csv
    """
    BaseFilePath = Path("/Users/Aidia/Documents/SummerResearch2020/LanguageNeuralNet/Data/")
    langs = ["English", "German", "Spanish"]
    MasterList = []
    for lang in langs:
            FileName = f"{lang}Text.txt"
            FilePath = BaseFilePath / FileName
            MasterList.extend(TxtToTupleList(FilePath, lang))
    LangCSVFile = open("../Data/MasterList.csv", "w+")
    with LangCSVFile:
        write = csv.writer(LangCSVFile)
        write.writerows(MasterList)
    return(MasterList)

if __name__ == "__main__":

    BaseFilePath = Path("/Users/Aidia/Documents/SummerResearch2020/LanguageNeuralNet/Data/")
    langs = ["English", "German", "Spanish"]

    for lang in langs:
            FileName = f"{lang}Text.txt"
            FilePath = BaseFilePath / FileName
            # print(TxtToTupleList(FilePath, lang))


    # ProcessedText = ProcessText(FilePath)
    # TxtToTupleList(FilePath, "Korean")
