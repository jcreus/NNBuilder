from utils import *
import theano.tensor as T
import theano
from sklearn.cross_validation import train_test_split

class Node:
    def __init__(self, path=None, w=None, b=None, a=None, predict=None, lr=None):
        self.path = path
        self.w = w
        self.b = b
        self.a = a
        self.predict = a
        self.lr = None

    def early_stop(self, x_validate, y_validate):
        '''
        Creates validation set
        Evaluates Node's path on validation set
        Chooses optimal w in Node's path based on validation set
        '''
        x = T.matrix("x")
        y = T.vector("y")
        w = T.vector("w")
        b = T.dscalar("b")
        a = T.dscalar("a")
        p_1 = -0.5 + a / (1 + T.exp(-T.dot(x, w) - b))
        xent = 0.5 * (y - p_1)**2
        cost = xent.mean()
        loss = theano.function(inputs=[x, y, w, b, a], outputs=cost)

        Path = self.path.keys()
        Path = map(int, Path)
        Path.sort()
        best_node = {}
        best_node_ind = 0
        best_loss = numpy.mean(y_validate**2)
        losses = []
        for ind in Path:
            node = self.path[str(ind)]
            l = loss(x_validate, y_validate, node['w'], node['b'], node['a'])
            losses.append(l)
            if l < best_loss:
                best_node = node
                best_node_ind = ind
                best_loss = l

        self.w = best_node['w']
        self.b = best_node['b']
        self.a = best_node['a']

    def is_useful(self, layer, X, Y):
        pred_validate = layer._predict(X)
        err_layer = numpy.mean(abs(Y - pred_validate)**2)
        pred_with_node = pred_validate + self.predict(X)*self.lr
        self.validation_err = numpy.mean(abs(Y - pred_with_node)**2)
        
        return self.validation_err < err_layer

class OptimalNode(Node):
    def __init__(self, X, Y, mode=REGRESSION, bias=False, num_iter=5, alpha=0.01,
                minibatch=False):
        '''
        inputs
            x_train: training features
            y_train: response variable
            n_iter: # of iterations for SGD
            alpha: strength of L2 penalty (default penalty for now)
        outputs
            Node: dictionary with Node parameters an predict method
        '''

        rng = numpy.random

        feats = len(X[0, :])
        D = [X, Y]
        training_steps = num_iter
        #print "training steps: ", training_steps
        #print "penalty strength: ", alpha
        #print "Uses bias: ", bias

        # Declare Theano symbolic variables
        x = T.matrix("x")
        y = T.vector("y")
        w = theano.shared(rng.uniform(low=-0.25, high=0.25, size=feats), name="w")
        b = theano.shared(rng.randn(1)[0], name="b")
        a = theano.shared(abs(rng.randn(1)[0]), name="a")
        #print "Initialize node as:"
        #print w.get_value(), b.get_value(), a.get_value()

        # Construct Theano expression graph
        if bias:
            p_1 = -0.5 + a / (1 + T.exp(-T.dot(x, w) - b))
        else:
            p_1 = a / (1 + T.exp(-T.dot(x, w)))
        prediction = p_1 > 0.5
        if mode == CLASSIFICATION:
            xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)  # Cross-entropy loss
        elif mode == REGRESSION:
            xent = 0.5 * (y - p_1)**2
        if alpha == 0:
            cost = xent.mean()  # The cost to minimize
        else:
            cost = xent.mean() + alpha * ((w ** 2).sum())
        if bias:
            gw, gb, ga = T.grad(cost, [w, b, a])
        else:
            gw, ga = T.grad(cost, [w, a])  # Compute the gradient of the cost

        # Compile
        self.path = {}
        if bias:
            train = theano.function(inputs=[x, y], outputs=[prediction, xent],
                                    updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb),
                                             (a, a - 0.1 * ga)))
        else:
            train = theano.function(inputs=[x, y], outputs=[prediction, xent],
                                    updates=((w, w - 0.1 * gw), (a, a - 0.1 * ga)))

        predict = theano.function(inputs=[x], outputs=p_1)

        # Train
        for i in range(training_steps):
            percent(i, training_steps)
            if minibatch:
                batch_split = train_test_split(X, Y, test_size=0.2)
                _, D[0], _, D[1] = batch_split
                pred, err = train(D[0], D[1])
            elif not minibatch:
                pred, err = train(D[0], D[1])
            self.path[str(i)] = {}
            self.path[str(i)]['w'] = w.get_value()
            self.path[str(i)]['b'] = b.get_value()
            self.path[str(i)]['a'] = a.get_value()

        self.w = w.get_value()
        self.b = b.get_value()
        self.a = a.get_value()
        self.predict = predict