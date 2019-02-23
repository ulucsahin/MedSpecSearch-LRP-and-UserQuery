# coding: utf-8

# In[1]:


import os
import re
from os.path import join, dirname

import numpy as np
from google.cloud import translate

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = join(dirname(__file__), 'data', 'aut.json')


# In[2]:

def checkInputData(inputData):
    charList = [[x] for x in inputData]
    charSet = set(charList)
    uniqChars = len(charSet)

    wordList = inputData.split(" ")
    wordSet = set(wordList)
    uniqWords = len(wordSet)

    if uniqWords < 4 or uniqChars < 7:
        return False
    return True


turkishLabels = ['Deri ve Zührevi Hastalıkları (Cildiye)', 'İç Hastalıkları (Dahiliye)', 'Nöroloji', 'Kadın Hastalıkları ve Doğum', 'Göz Hastalıkları', 'Ortopedi ve Travmatoloji',
                 'Kulak Burun Boğaz Hastalıkları', 'Çocuk Sağlığı ve Hastalıkları', 'Ruh Sağlığı ve Hastalıkları', 'Radyoloji', 'Genel Cerrahi', 'Üroloji']
englishLabels = ['Dermatology', 'Internal Medicine', 'Neurology', 'Obstetrics & Gynecology', 'Ophthalmology', 'Orthopaedic Surgery', 'Otolaryngology', 'Pediatrics', 'Psychiatry',
                 'Radiology-Diagnostic', 'Surgery-General', 'Urology']

labelMatches = zip(turkishLabels, englishLabels)

labelTranslateDict = {y: x for x, y in labelMatches}

print(labelTranslateDict)


class DataHandler:
    translator = translate.Client()
    turkishLabelsDict = labelTranslateDict

    def __init__(self):
        pass

    @staticmethod
    def getUniqueClassMapDict(classList):
        uniques = np.unique(classList)
        count = np.arange(uniques.size)
        listDict = np.hstack((uniques.reshape(-1, 1), count.reshape(-1, 1)))
        uniqueDict = {elem[0]: int(elem[1]) for elem in listDict}
        return uniqueDict

    @staticmethod
    def translateInput(inputTR):
        return DataHandler.translator.translate(inputTR, target_language="en")["translatedText"]

    @staticmethod
    def cleanTextData(textList):

        cleanTextList = []
        for text in textList:
            try:
                text = text.lower()
            except:
                print(text)
                # raise exception("oops")
            text = text.replace("Ãƒâ€šÃ‚Â", "")
            text = re.sub(' +', ' ', text)

            cleanText = ""

            for word in text.split(" "):
                cleanWord = ""
                for char in word:
                    if (ord(char) > 96 and ord(char) < 123):
                        cleanWord += char + ""
                    else:
                        cleanWord += " "

                cleanText += cleanWord + " "
                cleanText = re.sub(' +', ' ', cleanText)
            cleanTextList += [cleanText.strip()]

        return cleanTextList

    @staticmethod
    def idxListToidxDict(idxList):
        idxDict = {}

        for i in range(len(idxList)):
            idxDict[idxList[i]] = i

        return idxDict

    @staticmethod
    def calculateLongestSentence(sentenceList):
        longestSentence = 25

        for elem in sentenceList:
            stcLength = len(elem.split(" "))
            if (stcLength > longestSentence):
                longestSentence = stcLength

        return longestSentence

    @staticmethod
    def fillSentenceArray(sentence, fillSize, maxLength=1500):
        fillCount = fillSize - len(sentence)

        for i in range(fillCount):
            sentence += [np.zeros(300)]

        return sentence[0:maxLength]

    @staticmethod
    def fillWordListArray(sentence, maxLength):
        fillCount = maxLength - len(sentence)

        for i in range(fillCount):
            sentence += ["[None]"]

        return sentence[0:maxLength]

    @staticmethod
    def textIntoWordList(textList, maxLength, embedModel=None):
        embedList = []
        lengthList = []

        for sentence in textList:
            embeddedSentence = []

            for word in sentence.split(" "):
                if (embedModel is not None):
                    if word in embedModel:
                        embedding = word
                        embeddedSentence += [embedding]
                else:
                    embedding = word
                    embeddedSentence += [embedding]

            sentenceLength = len(embeddedSentence)
            embeddedSentence = DataHandler.fillWordListArray(embeddedSentence, maxLength)
            embedList += [embeddedSentence]
            lengthList += [sentenceLength]

        return embedList, sentenceLength

    @staticmethod
    def masterPreprocessor(data, maxLength, shuffle=False, classDict=None):
        if (classDict is None):
            classDict = DataHandler.getUniqueClassMapDict(data[:, 1])
        if (shuffle == True):
            np.random.shuffle(data)

        convertedClasses = np.array([classDict[elem] for elem in data[:, 1]])
        print("Outputs converted to numerical forms")
        cleanedTextData = DataHandler.cleanTextData(data[:, 0])
        print("Input text claned")
        wordList, lengthList = DataHandler.textIntoWordList(cleanedTextData, maxLength)
        print("Input text split into tokens and all inputs padded to maximum length")

        return np.array(wordList), np.array(convertedClasses), classDict

    @staticmethod
    def inputPreprocessor(data, maxLength):
        cleanedTextData = DataHandler.cleanTextData(data)
        wordList, lengthList = DataHandler.textIntoWordList(cleanedTextData, maxLength)
        return wordList, lengthList

    @staticmethod
    def batchIterator(data, target, batchSize):
        dataSize = data.shape[0]

        while (True):
            randomIdx = np.random.randint(dataSize, size=batchSize)

            yield np.take(data, randomIdx, axis=0), np.take(target, randomIdx)
