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

#4.3 a) Algorithme de classifieur de Bayes basé sur des densités des Parzen avec noyau isotropique
# Voir classifieur_bayes.py

#4.3 b) Entrainement d'un classifieur de Bayes sur l'ensemble d'entrainement (d=2) et visualisation des résultats
sigmas = [0.08, 0.4, 4]

for sigma in sigmas:
    args = {'sigma': sigma}
    classifieur = classifieur_bayes.creerClassifieur(partialTrainSet, "parzen", 3, args)

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

def calculErreurs(trainSet, validationSet, d, nClasses):
    tauxErreurs = []

    for i in range(1, 101):
        sigma = i/100
        args = {'sigma': sigma}
        classifieur = classifieur_bayes.creerClassifieur(trainSet, "parzen", nClasses, args)
        logProbabiliteTrain = classifieur.computePredictions(trainSet[:, :-1])
        classesPreditesTrain = logProbabiliteTrain.argmax(1)+1

        logProbabiliteValidation = classifieur.computePredictions(validationSet[:, :-1])
        classesPreditesValidation = logProbabiliteValidation.argmax(1)+1

        tauxErreurs.append(classifieur_bayes.calculateTauxErreur(iris, classesPreditesTrain, classesPreditesValidation, trainSetSize))

    tauxErreurs = np.array(tauxErreurs)
    minTrain = np.argmin(tauxErreurs[0, :])+1
    minValid = np.argmin(tauxErreurs[1, :])+1

    print "Meilleur sigma entrainement: "+str(float(minTrain)/100)
    classifieur_bayes.afficherTauxErreur(tauxErreurs[np.argmin(tauxErreurs[:, 0]), 0], tauxErreurs[np.argmin(tauxErreurs[:, 0]), 1], d)

    print "Meilleur sigma validation: "+str(float(minValid)/100)
    classifieur_bayes.afficherTauxErreur(tauxErreurs[np.argmin(tauxErreurs[:, 1]), 0], tauxErreurs[np.argmin(tauxErreurs[:, 1]), 1], d)

    sigmas = range(1, 101)
    for i in range(len(sigmas)):
        sigmas[i] /= 100.0

    pylab.plot(sigmas, tauxErreurs[:, 0])
    pylab.plot(sigmas, tauxErreurs[:, 1])
    pylab.show()

#4.2 c) Calcul des erreurs en dimension d = 2
calculErreurs(partialTrainSet, partialValidationSet, 2, 3)


#4.2 d) Calcul des erreurs en dimension d = 4
calculErreurs(completeTrainSet, completeValidationSet, 4, 3)
