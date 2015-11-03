
import densite



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
    d2d.getParzenGraph(4)


if __name__ == '__main__':
    main()