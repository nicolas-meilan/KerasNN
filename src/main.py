from neuralNetworkUtils.neuralNetwork import createNeuralNetwork, testNeuralNetwork
import numpy as np
import json

trainingDataPath = './example/trainingData/temperature.json'
testDataPath = './example/testData/temperature.json'

def _readJson(path):
    with open(path) as json_data:
        return json.load(json_data)


def _formater(arrayData):
    input = []
    output = []
    for data in arrayData:
        input.append(data['input'])
        output.append(data['output'])
    formatedData = {
        'input': np.array(input, ndmin=2),
        'output': np.array(output, ndmin=2),
    }
    return formatedData

architecture = [8, 16, 8, 4]
trainingData = _formater(_readJson(trainingDataPath)['trainingData'])
testData = _formater(_readJson(testDataPath)['testData'])
neuralNetwork = createNeuralNetwork(trainingData, architecture)
input = np.array([0,0,0,0,1,1,0,0], ndmin=2)
print(testNeuralNetwork(testData, neuralNetwork))

