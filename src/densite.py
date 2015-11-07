#-*- coding: utf-8 -*-
# Mathieu Bouchard && Pierre Gerard

import numpy
import matplotlib.pyplot as plt
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
        trainSetPlot, = plt.plot(self.oneDimData, numpy.array([i / 1000. for i in range(len(self.oneDimData))]), 'bo')

        dg = densite_fonction.DensiteGaussienne(1)
        dg.train(self.oneDimData)
        x = numpy.linspace(3, 7, 1000)
        densParamPlot, = plt.plot(x, [dg.p(i) for i in x], 'red', label="Densite parametrique")

        sigma = 0.05
        dg = densite_fonction.DensiteParzen(1, sigma)
        dg.train(self.oneDimData)
        x = numpy.linspace(3, 7, 1000)
        parzenParamPlotGrand, = plt.plot(x, [dg.p(i) for i in x], 'yellow', label="Parzen sigma petit")

        sigma = 1
        dg = densite_fonction.DensiteParzen(1, sigma)
        dg.train(self.oneDimData)
        x = numpy.linspace(3, 7, 1000)
        parzenParamPlotPetit, = plt.plot(x, [dg.p(i) for i in x], 'green', label="Parzen sigma grand")

        sigma = 0.349
        dg = densite_fonction.DensiteParzen(1, sigma)
        dg.train(self.oneDimData)
        x = numpy.linspace(3, 7, 1000)
        parzenParamPlotAdequat, = plt.plot(x, [dg.p(i) for i in x], 'blue', label="Parzen sigma adequat")

        plt.legend([trainSetPlot, densParamPlot, parzenParamPlotGrand, parzenParamPlotPetit, parzenParamPlotAdequat],
                   ["Ensemble entrainement", "Densite parametrique", "Parzen sigma petit", "Parzen sigma grand",
                    "Parzen sigma adequat"])
        plt.axis([3, 7, 0, 1.7])
        title = '1D Gaussian and Parzen probability density'
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('Probability density')
        fileTitle = title.replace(" ", "_") + ".png"
        plt.savefig(fileTitle)
        plt.close()
        print("[Created] file : " + fileTitle)

    def parzenEqualToOne(self):
        sigma = 0.349
        dg = densite_fonction.DensiteParzen(1, sigma)
        dg.train(self.oneDimData)
        input = [ i/100. for i in range(-1000,1000)]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        n, bins, rectangles = ax.hist(input, 50, normed=True)
        print(numpy.sum(n * numpy.diff(bins)))


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

    def getParamDensityGraph(self):
        min_x1 = min([i[0] for i in self.twoDimData])
        max_x1 = max([i[0] for i in self.twoDimData])
        min_x2 = min([i[1] for i in self.twoDimData])
        max_x2 = max(([i[1] for i in self.twoDimData]))
        nDots = 100
        plt.plot(numpy.array([i[0] for i in self.twoDimData]), numpy.array([i[1] for i in self.twoDimData]), 'bo')

        xgrid = numpy.linspace(min_x1, max_x1, num=nDots)
        ygrid = numpy.linspace(min_x2, max_x2, num=nDots)

        dg = densite_fonction.DensiteGaussienne(2)
        dg.train(self.twoDimData)

        resZ = [[0 for i in range(nDots)] for j in range(nDots)]
        for ix, x in enumerate(xgrid):
            for iy, y in enumerate(ygrid):
                resZ[ix][iy] = float(dg.p([x, y]))

        plt.contour(xgrid, ygrid, numpy.array(resZ))

        plt.axis([min_x1, max_x1, min_x2, max_x2])
        title = '2D Gaussian border'
        plt.title(title)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.savefig(title.replace(" ", "_") + ".png")
        plt.close()

    def getParzenGraph(self, sigma):
        min_x1 = min([i[0] for i in self.twoDimData])
        max_x1 = max([i[0] for i in self.twoDimData])
        min_x2 = min([i[1] for i in self.twoDimData])
        max_x2 = max(([i[1] for i in self.twoDimData]))
        nDots = 100
        plt.plot(numpy.array([i[0] for i in self.twoDimData]), numpy.array([i[1] for i in self.twoDimData]), 'bo')

        xgrid = numpy.linspace(min_x1, max_x1, num=nDots)
        ygrid = numpy.linspace(min_x2, max_x2, num=nDots)

        dg = densite_fonction.DensiteParzen(2, sigma)
        dg.train(self.twoDimData)

        resZ = [[0 for i in range(nDots)] for j in range(nDots)]
        for ix, x in enumerate(xgrid):
            for iy, y in enumerate(ygrid):
                resZ[ix][iy] = float(dg.p([x, y]))

        plt.contour(xgrid, ygrid, numpy.array(resZ))

        plt.axis([min_x1, max_x1, min_x2, max_x2])
        title = '2D Parzen sigma=' + str(sigma)
        plt.title(title)
        plt.xlabel('x1')
        plt.ylabel('x2')
        fileTitle = title.replace(" ", "_") + ".png"
        plt.savefig(fileTitle)
        plt.close()
        print("[Created] file : " + fileTitle)


if __name__ == '__main__':
    d1d = Densite1D()
    #d1d.getOneDimGraph()
    d1d.parzenEqualToOne()

    #d2d = Densite2D()
    #d2d.getParamDensityGraph()
    ##d2d.getParzenGraph(0.08)
    ##d2d.getParzenGraph(0.40)
    #d2d.getParzenGraph(4)
