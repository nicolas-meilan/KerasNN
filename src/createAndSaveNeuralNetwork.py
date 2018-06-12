from neuralNetworkUtils.neuralNetwork import createNeuralNetwork, saveNeuralNetwork
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

# architecture = [8, 16, 8, 4]
# trainingData = _formater(_readJson(trainingDataPath)['trainingData'])
# testData = _formater(_readJson(testDataPath)['testData'])
# neuralNetwork = createNeuralNetwork(trainingData, architecture)
# print(testNeuralNetwork(testData, neuralNetwork))

architectureOneOutputs = [8, 16, 8, 4]
architectureManyOutputs = [8, 1150, 300, 700, 4, 4, 4, 4, 4]
trainingData = _formater(_readJson(trainingDataPath)['trainingData'])
testData = _formater(_readJson(testDataPath)['testData'])
# neuralNetwork = createNeuralNetwork(trainingData['input'], trainingData['output'], architectureOneOutputs)
# print(testNeuralNetwork(testData['input'], [testData['output']], neuralNetwork))
neuralNetwork = createNeuralNetwork(trainingData['input'], [trainingData['output'][::-1], trainingData['output'][::-1], trainingData['output'], trainingData['output'], trainingData['output']], architectureManyOutputs , 5)
saveNeuralNetwork(neuralNetwork, neuralNetworkPath, neuralNetworkName)