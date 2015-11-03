#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pylab
import classifieur_bayes

iris = np.loadtxt("iris.txt")
#4.1 Mélange les exemples d'Iris et diviser l'ensemble de tous les exemples en 2.
np.random.seed(123)
np.random.shuffle(iris)

#On place 108 exemples dans l'ensemble d'entrainement et 42 dans l'ensemble de validation
trainSetSize = 108
classCount = 3 #Nombre de classes

#d = 4
completeTrainSet = iris[0:trainSetSize, :]
completeValidationSet = iris[trainSetSize:, :]

#d = 2
partialTrainSet = np.zeros((trainSetSize, 3))
for i in range(trainSetSize):
    partialTrainSet[i][0] = iris[i, 0]
    partialTrainSet[i][1] = iris[i, 1]
    partialTrainSet[i][2] = iris[i, -1]

partialValidationSet = np.zeros((len(iris)-trainSetSize, 3))
for i in range(len(iris)-trainSetSize):
    partialValidationSet[i][0] = iris[i+trainSetSize, 0]
    partialValidationSet[i][1] = iris[i+trainSetSize, 1]
    partialValidationSet[i][2] = iris[i+trainSetSize, -1]

#4.2 a) Algorithme de classifieur de Bayes basé sur des densités paramétriques Gaussiennes diagonales
# Voir classifieur_bayes.py

#4.2 b) Entrainement d'un classifieur de Bayes sur l'ensemble d'entrainement (d=2) et visualisation des résultats
#classifieur = createClassifieurBayesGaussien(partialTrainSet)
classifieur = classifieur_bayes.creerClassifieur(partialTrainSet, "gaussien", 3)

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
pylab.scatter(partialTrainSet[:, 0], partialTrainSet[:, 1], c=iris[0:trainSetSize, -1], marker='v', s=100)
pylab.scatter(partialValidationSet[:, 0], partialValidationSet[:, 1], c=iris[trainSetSize:, -1], marker='s', s=100)

#4.2 c) Calcul des erreurs en dimension d = 2
logProbabiliteTrain = classifieur.computePredictions(partialTrainSet[:, :-1])
classesPreditesTrain = logProbabiliteTrain.argmax(1)+1

logProbabiliteValidation = classifieur.computePredictions(partialValidationSet[:, :-1])
classesPreditesValidation = logProbabiliteValidation.argmax(1)+1

tauxErreur = classifieur_bayes.calculateTauxErreur(iris, classesPreditesTrain, classesPreditesValidation)
classifieur_bayes.afficherTauxErreur(tauxErreur[0], tauxErreur[1], 2)

#4.2 d) Calcul des erreurs en dimension d = 4
classifieur = classifieur_bayes.creerClassifieur(completeTrainSet, "gaussien", 3)
logProbabiliteTrain = classifieur.computePredictions(completeTrainSet[:, :-1])
classesPreditesTrain = logProbabiliteTrain.argmax(1)+1

logProbabiliteValidation = classifieur.computePredictions(completeValidationSet[:, :-1])
classesPreditesValidation = logProbabiliteValidation.argmax(1)+1

tauxErreur = classifieur_bayes.calculateTauxErreur(iris, classesPreditesTrain, classesPreditesValidation)
classifieur_bayes.afficherTauxErreur(tauxErreur[0], tauxErreur[1], 4)

pylab.show()
pylab.close()