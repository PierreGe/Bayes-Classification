# coding=utf-8
import numpy
import utilitaires


class DensiteGaussienne:
    def __init__(self, n_dims):
        self._n_dims = n_dims
        self._mu = numpy.zeros((1, n_dims))
        self._sigma_sq = numpy.ones(n_dims)

    def train(self, train_data):
        """Pour un ensemble d'entrainement, la fonction devrait calculer l'estimateur par MV de la moyenne et de
        la matrice de covariance
        """
        self._mu = numpy.mean(train_data, axis=0)
        self._sigma_sq = numpy.sum((train_data - self._mu) ** 2.0, axis=0) / train_data.shape[0]

    def getMu(self):
        return self._mu

    def getSigma(self):
        return numpy.sqrt(self._sigma_sq)

    def p(self, x):
        if self._n_dims == 1:
            sigma = numpy.sqrt(self._sigma_sq)
            expo = -(float(x) - self._mu)**2
            expo /= 2*(float(sigma)**2)
            return 1./(numpy.sqrt(2.*numpy.pi)*sigma)* numpy.exp(expo)
        else:
            sigma = [ numpy.sqrt(i) for i in self._sigma_sq]
            sigma[0] += 0.1
            norma = 1./((2.*numpy.pi)*(self._n_dims/2.) * numpy.sqrt(numpy.prod(sigma)))
            expo1 = numpy.transpose ( numpy.array(x) - self._mu )
            expo2 = numpy.linalg.inv( numpy.diag(sigma))
            expo3 = ( numpy.array(x) - self._mu )
            expo = numpy.dot(expo1, expo2)
            expo = numpy.dot(expo, expo3)
            return norma * numpy.exp(-1/2. *expo)

    def compute_predictions(self, test_data):
        """
        Retourne un vecteur de taille nb. ex. de test contenant les log probabilites de chaque
        exemple de test sous le modèle.
        """
        # on prend le produit du vecteur représentant la diagonale (np.prod(self.sigma)
        c = -self._n_dims * numpy.log(2 * numpy.pi) / 2.0 - self._n_dims * numpy.log(numpy.prod(self._sigma_sq)) / 2.0
        # on somme sur l'axe 1 après avoir divisé par sigma puisque celui ci aussi est
        # de dimension d
        log_prob = c - numpy.sum((test_data - self._mu) ** 2.0 / (2.0 * self._sigma_sq), axis=1)
        return log_prob


class DensiteParzen:
    def __init__(self, n_dims, sigma):
        self._n_dims = n_dims
        self._sigma = sigma
        self._distanceFunction = utilitaires.minkowski

    def train(self, train_data):
        self._data = train_data

    def p(self, x):
        res = 0
        for i,xi in enumerate(self._data):
            expo = -1/2. * self._distanceFunction(x, xi)**2
            expo /= (self._sigma**2)
            res += 1./(((2.*numpy.pi)**(self._n_dims/2.)) * (self._sigma**self._n_dims)) \
            * numpy.exp(expo)
        res /= float(len(self._data))
        return res

    def compute_predictions(self, test_data):
        log_prob = []
        for i in test_data:
            log_prob.append(numpy.log(i))
        return res
