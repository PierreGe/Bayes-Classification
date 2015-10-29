import numpy
import matplotlib
import utilitaires


# init du seed
numpy.random.seed(123)
# pour faire taire les underflow
# np.seterr(under='ignore')

iris=numpy.loadtxt('iris.txt')
numpy.random.shuffle(iris)


print(iris)

