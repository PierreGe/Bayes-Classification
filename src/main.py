#-*- coding: utf-8 -*-
import densite
import classifieur_bayes_gauss
import classifieur_bayes_parzen


def main():
    print("Partie pratique : Estimateur de densité")
    print(" - Densite Gaussienne")
    d1d = densite.Densite1D()
    d1d.getOneDimGraph()
    print(" - Densite de Parzen")
    d2d = densite.Densite2D()
    d2d.getParamDensityGraph()
    d2d.getParzenGraph(0.08)
    d2d.getParzenGraph(0.40)
    d2d.getParzenGraph(20)
    print(" - Classifieur de Bayes basé sur des densités paramétriques Guassiennes diagonales")
    cbg = classifieur_bayes_gauss.ClassifieurBayesGaussien(2)
    cbg.getClassifieurBayesGaussienGraph()
    cbg.printTauxErreur()
    cbg = classifieur_bayes_gauss.ClassifieurBayesGaussien(4)
    cbg.printTauxErreur()
    print(" - Classifieur de Bayes basé sur des densités de Parzen avec noyau Gaussien isotropique")
    cbp = classifieur_bayes_parzen.ClassifieurBayesParzen(2)
    cbp.getClassifieurBayesGaussienGraphs()
    cbp.calculErreurs(range(1, 300))
    cbp = classifieur_bayes_parzen.ClassifieurBayesParzen(4)
    cbp.calculErreurs(range(1, 300))
    print ("Fin de l'exécution")
if __name__ == '__main__':
    main()