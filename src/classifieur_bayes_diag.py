#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pylab

iris = np.loadtxt("data/iris.txt")
#4.1 Mélange les exemples d'Iris et diviser l'ensemble de tous les exemples en 2.
np.random.seed(123)
np.random.shuffle(iris)

#On place 108 exemples dans l'ensemble d'entrainement et 42 dans l'ensemble de validation
trainSetSize = 150
classCount = 3 #Nombre de classes

#d = 4
completeTrainSet = iris[0:trainSetSize, :-1]
completeValidationSet = iris[trainSetSize:150, :-1]

#d = 2
partialTrainSet = iris[0:trainSetSize, 0:2]
partialValidationSet = iris[trainSetSize:150, 0:2]

#4.2 a) Algorithme de classifieur de Bayes basé sur des densités paramétriques Gaussiennes diagonales
class GaussienneDiagonale:
    def __init__(self, d):
        self.d = d
        self.mu = np.zeros((1, self.d))
        self.sigmaSquare = np.ones(self.d) #La matrice de covariance étant diagonale, il ne faut garder en mémoire que les termes de la diagonale

    def train(self, trainSet):
        self.mu = np.mean(trainSet, axis=0)
        self.sigmaSquare = np.sum((trainSet-self.mu)**2.0, axis=0) / len(trainSet)

    def computePredictions(self, trainSet):
        c = -self.d * np.log(2*np.pi) / 2.0 - self.d * np.log(np.prod(self.sigmaSquare)) / 2.0

        logProb = c - np.sum((trainSet - self.mu)**2 / (2.0 * self.sigmaSquare), axis=1)
        return logProb

class ClassifieurBayes:
    def __init__(self, models, priors):
        self.models = models
        self.priors = priors
        self.classCount = len(self.models)

        if len(self.models) != len(self.priors):
            raise Exception('Le nombre de models doit être égal au nombre de priors')

    def computePredictions(self, validationSet):
        logPrediction = np.empty((len(validationSet), self.classCount))

        for k in range(self.classCount):
            logPrediction[:, k] = self.models[k].computePredictions(validationSet) + self.priors[k]

        return logPrediction

#4.2 b) Visualisation des régions de décision pour le sous-ensemble de données (d=2)

def createClassifieur(trainSet):
    propertiesCount = len(trainSet[0])

    #Sépare l'ensemble d'entrainement pour obtenir un sous-ensemble par classe
    limit1 = np.floor(trainSetSize/float(classCount))
    limit2 = 2*np.floor(trainSetSize/float(classCount))

    trainC1 = trainSet[0:limit1]
    modelC1 = GaussienneDiagonale(propertiesCount)
    modelC1.train(trainC1)

    trainC2 = trainSet[limit1:limit2]
    modelC2 = GaussienneDiagonale(propertiesCount)
    modelC2.train(trainC2)

    trainC3 = trainSet[limit2:]
    modelC3 = GaussienneDiagonale(propertiesCount)
    modelC3.train(trainC3)

    models = [modelC1, modelC2, modelC3]
    priors = [len(trainC1)/float(trainSetSize), len(trainC2)/float(trainSetSize), len(trainC3)/float(trainSetSize)]

    classifieur = ClassifieurBayes(models, priors)
    return classifieur

#4.2 b) Entrainement d'un classifieur de Bayes sur l'ensemble d'entrainement (d=2) et visualisation des résultats
classifieur = createClassifieur(partialTrainSet)
logProbabiliteTrain = classifieur.computePredictions(partialTrainSet)
classesPreditesTrain = logProbabiliteTrain.argmax(1)+1

logProbabiliteValidation = classifieur.computePredictions(partialValidationSet)
classesPreditesValidation = logProbabiliteValidation.argmax(1)+1

minX1 = min(iris[:, 0])
maxX1 = max(iris[:, 0])
minX2 = min(iris[:, 1])
maxX2 = max(iris[:, 1])

x1Vals = np.linspace(minX1, maxX1)
x2Vals = np.linspace(minX2, maxX2)

grille = []
step = 0.05
i = minX1
while i < maxX1:
    j = minX2
    while j < maxX2:
        grille.append([i, j])
        j += step
    i += step
grille = np.array(grille)

logProbabiliteGrille = classifieur.computePredictions(grille)
classesPreditesGrille = logProbabiliteGrille.argmax(1)+1

pylab.scatter(grille[:, 0], grille[:, 1], s=50, c=classesPreditesGrille, alpha=0.25)
pylab.scatter(partialTrainSet[:, 0], partialTrainSet[:, 1], c=classesPreditesTrain, marker='v', s=100)
pylab.scatter(partialValidationSet[:, 0], partialValidationSet[:, 1], c=classesPreditesValidation, marker='s', s=100)

#pylab.show()

#4.2 c) Calcul des erreurs en dimension d = 2
def calculateTauxErreur(classesPreditesTrain, classesPreditesValidation, d):
    trainError = 0
    validationError = 0

    for i in range(len(iris)):
        if i >= len(partialTrainSet):
            classePredite = classesPreditesValidation[i-len(partialTrainSet)]

            if iris[i, 4] != classePredite:
                validationError += 1
        else:
            classePredite = classesPreditesTrain[i]

            if iris[i, 4] != classePredite:
                trainError += 1

    tauxErreurTrainSet = (trainError/float(len(partialTrainSet)))*100

    if len(partialValidationSet) > 0:
        tauxErreurValidationSet = (validationError/float(len(partialValidationSet)))*100
    else:
        tauxErreurValidationSet = -1

    print "\n######################"
    print "d="+str(d)
    print "Taux d'erreur sur l'ensemble d'entrainement: %.2f%%" % tauxErreurTrainSet

    if tauxErreurValidationSet != -1:
        print "Taux d'erreur sur l'ensemble de validation: %.2f%%" % tauxErreurValidationSet
    print "######################\n"

calculateTauxErreur(classesPreditesTrain, classesPreditesValidation, 2)

#4.2 d) Calcul des erreurs en dimension d = 4
classifieur = createClassifieur(completeTrainSet)
logProbabiliteTrain = classifieur.computePredictions(completeTrainSet)
classesPreditesTrain = logProbabiliteTrain.argmax(1)+1

logProbabiliteValidation = classifieur.computePredictions(completeValidationSet)
classesPreditesValidation = logProbabiliteValidation.argmax(1)+1

calculateTauxErreur(classesPreditesTrain, classesPreditesValidation, 4)

pylab.show()