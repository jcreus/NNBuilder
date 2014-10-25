from sklearn import preprocessing
import numpy
import sys

SHUFFLED = 'Shuffled'
UNIFORM = 'Uniform'
REGRESSION = 'Regression'
CLASSIFICATION = 'Classification'

def FoldLabels(Y):

    inds = Y<0
    Y[inds] = -Y[inds]

    return [Y, inds]


def unFoldLabels(Y, inds):
    Y[inds] = -Y[inds]

    return Y

def Preprocess(X, Scaler=None):
    """Preprocess data array, returns transformed array and the scaler."""
    if len(numpy.shape(X)) == 2:
        #scaler = preprocessing.MinMaxScaler().fit(X)
        scaler = preprocessing.StandardScaler().fit(X)
        if Scaler is not None:
            scaler = Scaler
        X = scaler.transform(X)
    elif len(numpy.shape(X)) == 1:
        X = numpy.reshape(X, (len(X), 1))
        #scaler = preprocessing.MinMaxScaler().fit(X)
        scaler = preprocessing.StandardScaler().fit(X)
        if Scaler is not None:
            scaler = Scaler
        X = scaler.transform(X)
        X = numpy.reshape(X, (len(X)))

    return [X, scaler]


def Postprocess(X, scaler):
    """Inverse transforms X using scaler"""
    if len(numpy.shape(X)) == 2:
        X = scaler.inverse_transform(X)
    elif len(numpy.shape(X)) == 1:
        X = numpy.reshape(X, (len(X), 1))
        X = scaler.inverse_transform(X)
        X = numpy.reshape(X, (len(X)))

    return X

def get_error(Y, pred):
    return numpy.mean(abs(Y - pred)**2)

def percent(i, n):
    print str(float(i)/float(n)*100)+'%\r',
    sys.stdout.flush();