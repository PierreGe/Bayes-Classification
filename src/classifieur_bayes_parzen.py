#-*- coding: utf-8 -*-
# Mathieu Bouchard && Pierre Gerard

import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.patches as mpatches
import classifieur_bayes

class ClassifieurBayesParzen:
    def __init__(self, d):
        self.iris = np.loadtxt("iris.txt")
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

        #4.3 a) Algorithme de classifieur de Bayes basé sur des densités des Parzen avec noyau isotropique
        # Voir classifieur_bayes.py

    def getClassifieurBayesGaussienGraphs(self):
        #4.3 b) Entrainement d'un classifieur de Bayes sur l'ensemble d'entrainement visualisation des résultats
        sigmas = [0.01, 0.5, 10]

        for sigma in sigmas:
            args = {'sigma': sigma}
            classifieur = classifieur_bayes.creerClassifieur(self.trainSet, "parzen", 3, args)

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

            logProbabiliteGrille = classifieur.computePredictions(grille)
            classesPreditesGrille = logProbabiliteGrille.argmax(1)+1

            pyplot.scatter(grille[:, 0], grille[:, 1], s=50, c=classesPreditesGrille, alpha=0.25)
            pyplot.scatter(self.trainSet[:, 0], self.trainSet[:, 1], c=self.iris[0:self.trainSetSize, -1], marker='v', s=100)
            pyplot.scatter(self.validationSet[:, 0], self.validationSet[:, 1], c=self.iris[self.trainSetSize:, -1], marker='s', s=100)
            pyplot.title("Regions de decision (sigma = "+str(sigma)+")")
            #pylab.show()
            pyplot.savefig('bayes_parzen_'+str(sigma)+'.png')
            pyplot.close()

    def calculErreurs(self, sigmas):
        tauxErreurs = []

        for i in sigmas:
            sigma = i/100.
            args = {'sigma': sigma}
            classifieur = classifieur_bayes.creerClassifieur(self.trainSet, "parzen", 3, args)
            logProbabiliteTrain = classifieur.computePredictions(self.trainSet[:, :-1])
            classesPreditesTrain = logProbabiliteTrain.argmax(1)+1

            logProbabiliteValidation = classifieur.computePredictions(self.validationSet[:, :-1])
            classesPreditesValidation = logProbabiliteValidation.argmax(1)+1

            tauxErreurs.append(classifieur_bayes.calculateTauxErreur(self.iris, classesPreditesTrain, classesPreditesValidation))

        tauxErreurs = np.array(tauxErreurs)
        sigmaMinIndex = np.argmin(tauxErreurs[:, 1])
        sigmaMin = sigmas[sigmaMinIndex]/100.

        classifieur_bayes.afficherTauxErreur(tauxErreurs[np.argmin(tauxErreurs[:, 1]), 0], tauxErreurs[np.argmin(tauxErreurs[:, 1]), 1], self.d, sigmaMin)

        for i in range(len(sigmas)):
            sigmas[i] /= 100.0

        pyplot.plot(sigmas, tauxErreurs[:, 0], c="red")
        pyplot.plot(sigmas, tauxErreurs[:, 1], c="green")
        pyplot.xlabel("Sigma")
        pyplot.ylabel("Taux d'erreur")
        pyplot.title("Courbes d'apprentissage")
        red = mpatches.Patch(color="red", label="Erreur d'apprentissage")
        green = mpatches.Patch(color="green", label="Erreur de validation")
        pyplot.legend(handles=[red, green])
        pyplot.savefig('bayes_parzen_'+str(self.d)+'d.png')
        #pyplot.show()
        pyplot.close()

