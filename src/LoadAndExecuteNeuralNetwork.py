from neuralNetworkUtils.neuralNetwork import testNeuralNetwork, loadNeuralNetwork
import numpy as np
import json

trainingDataPath = './example/trainingData/temperature.json'
testDataPath = './example/testData/temperature.json'
neuralNetworkPath = './example/neuralNetwork'
neuralNetworkName = 'temperature'

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
        'output': np.array(output, ndmin=2)
    }
    return formatedData

neuralNetwork = loadNeuralNetwork(neuralNetworkPath, neuralNetworkName)
testData = _formater(_readJson(testDataPath)['testData'])
print(testNeuralNetwork(testData['input'], [testData['output'][::-1], testData['output'][::-1], testData['output'], testData['output'], testData['output']], neuralNetwork, 5))