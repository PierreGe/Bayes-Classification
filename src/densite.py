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
        trainSetPlot, = plt.plot(self.oneDimData,numpy.array([i/500. for i in range(len(self.oneDimData))]), 'bo')

        dg = densite_fonction.DensiteGaussienne(1)
        dg.train(self.oneDimData)
        x = numpy.linspace(3,7,1000)
        densParamPlot, = plt.plot(x,[dg.p(i) for i in x], 'red', label= "Densite parametrique")


        sigma = 0.2
        dg = densite_fonction.DensiteParzen(1,sigma)
        dg.train(self.oneDimData)
        x = numpy.linspace(3,7,1000)
        parzenParamPlotGrand, = plt.plot(x,[dg.p(i) for i in x], 'blue', label= "Parzen sigma petit")

        sigma = 1
        dg = densite_fonction.DensiteParzen(1,sigma)
        dg.train(self.oneDimData)
        x = numpy.linspace(3,7,1000)
        parzenParamPlotPetit, = plt.plot(x,[dg.p(i) for i in x], 'green', label= "Parzen sigma grand")

        sigma = 0.45
        dg = densite_fonction.DensiteParzen(1,sigma)
        dg.train(self.oneDimData)
        x = numpy.linspace(3,7,1000)
        parzenParamPlotAdequat, = plt.plot(x,[dg.p(i) for i in x], 'yellow', label= "Parzen sigma adequat")

        plt.legend([trainSetPlot, densParamPlot, parzenParamPlotGrand,parzenParamPlotPetit,parzenParamPlotAdequat], ["Ensemble entrainement", "Densite parametrique","Parzen sigma petit","Parzen sigma grand","Parzen sigma adequat"])
        plt.axis([3, 7, 0, 1.7])
        plt.title('1D Gaussian and Parzen probability density')
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
