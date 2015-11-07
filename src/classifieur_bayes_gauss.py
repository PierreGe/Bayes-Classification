#-*- coding: utf-8 -*-
# Mathieu Bouchard && Pierre Gerard

import numpy as np
import pylab
import classifieur_bayes

class ClassifieurBayesGaussien:
    def __init__(self, d):
        self.iris = np.loadtxt("iris.txt")

        if d > len(self.iris[0]):
            raise Exception('Le nombre de dimensions est trop grand!')

        #4.1 Mélange les exemples d'Iris et diviser l'ensemble de tous les exemples en 2.
        np.random.seed(123)
        np.random.shuffle(self.iris)

        #On place 108 exemples dans l'ensemble d'entrainement et 42 dans l'ensemble de validation
        self.trainSetSize = 108
        self.classCount = 3 #Nombre de classes
        self.d = d

        self.trainSet = np.zeros((self.trainSetSize, d+1))
        for i in range(self.trainSetSize):
            for j in range(d):
                self.trainSet[i][j] = self.iris[i, j]
            self.trainSet[i][d] = self.iris[i, -1]

        self.validationSet = np.zeros((len(self.iris)-self.trainSetSize, d+1))
        for i in range(len(self.iris)-self.trainSetSize):
            for j in range(d):
                self.validationSet[i][j] = self.iris[i+self.trainSetSize, j]
            self.validationSet[i][d] = self.iris[i+self.trainSetSize, -1]

        #4.2 a) Algorithme de classifieur de Bayes basé sur des densités paramétriques Gaussiennes diagonales
        # Voir classifieur_bayes.py

        #4.2 b) Entrainement d'un classifieur de Bayes sur l'ensemble d'entrainement...
        self.classifieur = classifieur_bayes.creerClassifieur(self.trainSet, "gaussien", 3)

    def getClassifieurBayesGaussienGraph(self):
        #4.2 b) ... visualisation des résultats
        minX1 = min(self.iris[:, 0])
        maxX1 = max(self.iris[:, 0])
        minX2 = min(self.iris[:, 1])
        maxX2 = max(self.iris[:, 1])

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

        logProbabiliteGrille = self.classifieur.computePredictions(grille)
        classesPreditesGrille = logProbabiliteGrille.argmax(1)+1

        pylab.scatter(grille[:, 0], grille[:, 1], s=50, c=classesPreditesGrille, alpha=0.25)
        pylab.scatter(self.trainSet[:, 0], self.trainSet[:, 1], c=self.iris[0:self.trainSetSize, -1], marker='v', s=100)
        pylab.scatter(self.validationSet[:, 0], self.validationSet[:, 1], c=self.iris[self.trainSetSize:, -1], marker='s', s=100)
        pylab.title("Regions de decision")
        #pylab.show()
        fileTitle = 'bayes_gaussienne.png'
        pylab.savefig(fileTitle)
        print("[Created] file : " + fileTitle)
        pylab.close()

    #4.2 c) d) Calcul des erreurs en dimension d
    def printTauxErreur(self):
        logProbabiliteTrain = self.classifieur.computePredictions(self.trainSet[:, :-1])
        classesPreditesTrain = logProbabiliteTrain.argmax(1)+1

        logProbabiliteValidation = self.classifieur.computePredictions(self.validationSet[:, :-1])
        classesPreditesValidation = logProbabiliteValidation.argmax(1)+1

        tauxErreur = classifieur_bayes.calculateTauxErreur(self.iris, classesPreditesTrain, classesPreditesValidation)
        classifieur_bayes.afficherTauxErreur(tauxErreur[0], tauxErreur[1], self.d)
