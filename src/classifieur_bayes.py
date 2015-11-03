#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from densite_fonction import *

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
            logPrediction[:, k] = self.models[k].compute_predictions(validationSet) + np.array(self.priors[k])
        return logPrediction

def creerClassifieur(trainSet, type, nbClasses, args=None):
    d = len(trainSet[0])-1

    donneesParClasse = []
    for i in range(nbClasses):
        donneesParClasse.append([])

    #Divise l'ensemble d'entrainement en tableaux contenant les données pour chacune des classes
    for i in range(len(trainSet)):
        classe = int(trainSet[i, -1])-1
        donneesParClasse[classe].append(trainSet[i])

    #Transforme le tout en tableau numpy
    donneesParClasse = np.array(donneesParClasse)
    for i in range(len(donneesParClasse)):
        donneesParClasse[i] = np.array(donneesParClasse[i])

    modeles = []
    if type == "gaussien":
        for i in range(nbClasses):
            modeles.append(DensiteGaussienne(d))
            modeles[i].train(donneesParClasse[i][:, :-1])
    elif type == "parzen":
        if args is not None and args['sigma'] is not None:
            sigma = args['sigma']

            for i in range(nbClasses):
                modeles.append(DensiteParzen(d, sigma))
                modeles[i].train(donneesParClasse[i][:, :-1])
        else:
            raise Exception('Aucune valeur de sigma fournie dans args')
    else:
        raise Exception('Type de classifieur inconnu')

    priors = []
    for i in range(nbClasses):
        priors.append(len(donneesParClasse[i])/float(len(trainSet)))

    classifieur = ClassifieurBayes(modeles, priors)
    return classifieur

def calculateTauxErreur(dataSet, classesPreditesTrain, classesPreditesValidation):
    trainError = 0
    validationError = 0
    trainSetSize = len(classesPreditesTrain)

    for i in range(len(dataSet)):
        if i >= trainSetSize:
            classePredite = classesPreditesValidation[i-trainSetSize]

            if dataSet[i, 4] != classePredite:
                validationError += 1
        else:
            classePredite = classesPreditesTrain[i]

            if dataSet[i, 4] != classePredite:
                trainError += 1

    tauxErreurTrainSet = (trainError/float(len(classesPreditesTrain)))*100

    if len(classesPreditesValidation) > 0:
        tauxErreurValidationSet = (validationError/float(len(classesPreditesValidation)))*100
    else:
        tauxErreurValidationSet = -1

    return [tauxErreurTrainSet, tauxErreurValidationSet]

def afficherTauxErreur(tauxErreurTrainSet, tauxErreurValidationSet, d):
    print "\n######################"
    print "d="+str(d)
    print "Taux d'erreur sur l'ensemble d'entrainement: %.2f%%" % tauxErreurTrainSet

    if tauxErreurValidationSet != -1:
        print "Taux d'erreur sur l'ensemble de validation: %.2f%%" % tauxErreurValidationSet
    print "######################\n"