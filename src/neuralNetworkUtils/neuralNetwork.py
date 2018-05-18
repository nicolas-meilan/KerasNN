from keras.models import Model
from keras.layers import *
import numpy as np

def createNeuralNetwork(trainingData, architecture, output=1):
    inp = Input((architecture[0],))
    x = Dense(architecture[1])(inp)
    for i in range(2, len(architecture) - output):
        x = Dense(architecture[i])(x)
    outputNeurons = []
    for i in range(output, 0, -1):
        outputNeurons.append(Dense(architecture[-i])(x))
    model = Model(inp, outputNeurons)
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    model.fit(trainingData['input'], trainingData['output'], epochs=1000, batch_size=10, verbose=False)
    return model

def testNeuralNetwork(testData, neuralNetwork):
    errors = 0
    lengthTests = len(testData['input'])
    for i in range(0, lengthTests):
        output = neuralNetwork.predict(np.array(testData['input'][i], ndmin=2)).argmax()
        outputExpected = testData['output'][i].argmax()
        if output != outputExpected:
            errors += 1
    return errors/lengthTests
