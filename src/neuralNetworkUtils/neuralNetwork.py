from keras.models import Model
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
    model.fit(inputData, outputData, epochs=10000, batch_size=150, verbose=False)
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
