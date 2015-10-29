import numpy
import matplotlib.pyplot as plt
import utilitaires
import random
import densite_fonction



class Densite1D:
    def __init__(self):
        self.iris = numpy.loadtxt('iris.txt')
        self.oneDimData = self._getOneDimDate()


    def _getOneDimDate(self):
        chosenClass = 1
        param = 0
        oneDimData = []
        for l in self.iris:
            if l[-1] == chosenClass:
                oneDimData.append(l[param])
        return numpy.array(oneDimData)


    def getOneDimGraph(self):
        trainSetPlot = plt.plot(self.oneDimData,numpy.array([i/500. for i in range(len(self.oneDimData))]), 'g^', label= "Ensemble entrainement")
        dg = densite_fonction.DensiteGaussienne(1)
        dg.train(self.oneDimData)
        mu = dg.getMu()
        sigma = dg.getSigma()
        x = numpy.linspace(3,7,1000)
        densParamPlot2 = plt.plot(x,[dg.p(i) for i in x], 'red', label= "Densite parametrique")

        #plt.legend(handles=[trainSetPlot, densParamPlot])
        plt.axis([3, 7, 0, 5])
        plt.show()


class Densite2D:
    def __init__(self):
        self.iris = numpy.loadtxt('iris.txt')
        self.twoDimData = self.getTwoDimDate()

    def getTwoDimDate(self):
        classe = 1
        param1 = 0
        param2 = 1
        oneDimData = []
        for l in self.iris:
            if l[-1] == classe:
                oneDimData.append([l[param1], l[param2]])
        return numpy.array(oneDimData)

if __name__ == '__main__':
    d1d = Densite1D()
    d1d.getOneDimGraph()
