from keras.models import Model, model_from_json
from keras.layers import *
import numpy as np

def createNeuralNetwork(inputData, outputData, architecture, output=1):
    inp = Input((architecture[0],))
    x = Dense(architecture[1])(inp)
    for i in range(2, len(architecture) - output):
        x = Dense(architecture[i], activation='relu')(x)
    outputNeurons = []
    for i in range(output, 0, -1):
        outputNeurons.append(Dense(architecture[-i], activation='sigmoid')(x))
    model = Model(inp, outputNeurons)
    model.compile(optimizer='sgd', metrics=['accuracy'], loss='categorical_crossentropy')
    model.fit(inputData, outputData, epochs=10000, batch_size=150)
    return model

def testNeuralNetwork(inputData, outputData, neuralNetwork, outputs=1):
    errors = 0
    lengthTests = len(inputData)
    for iterator in range(0, lengthTests):
        outputAux = []
        outputExpectedAux = []
        result = neuralNetwork.predict(np.array(inputData[iterator], ndmin=2))
        for subIterator in range(0, outputs):
            outputAux.append(result[subIterator].argmax())
            outputExpectedAux.append(outputData[subIterator][iterator].argmax())
        output = np.array(outputAux)
        outputExpected = np.array(outputExpectedAux)
        if np.array_equal(output, outputExpected) == False:
            print(str(inputData[iterator]) +":::::::"+str(outputExpected) + "--->" + str(output))
            errors += 1
    return errors/lengthTests

def saveNeuralNetwork(neuralNetwork, path='./', neuralNetworkName='neuralNetwork'):
    # serialize model to JSON
    path += '/'
    neuralNetwork_json = neuralNetwork.to_json()
    with open(path + neuralNetworkName + '.json', "w") as json_file:
        json_file.write(neuralNetwork_json)
    # serialize weights to HDF5
    neuralNetwork.save_weights(path + neuralNetworkName + '.h5')
    print("Saved model to disk")

def loadNeuralNetwork(path='./', neuralNetworkName='neuralNetwork'):
    path += '/'
    json_file = open(path + neuralNetworkName + '.json', 'r')
    neuralNetwork_json = json_file.read()
    json_file.close()
    neuralNetwork = model_from_json(neuralNetwork_json)
    neuralNetwork.load_weights(path + neuralNetworkName + '.h5')
    print("Loaded model from disk")
    neuralNetwork.compile(optimizer='sgd', metrics=['accuracy'], loss='categorical_crossentropy')
    return neuralNetwork
