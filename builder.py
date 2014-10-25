import logging
from sklearn.cross_validation import train_test_split
import comparison
from layer import Layer
from utils import *
from node import *

class LayerBuilder:

    def __init__(self, X, Y, num_nodes=50, num_iter=100, epsilon=0.01,
            test_size=0.3, boostCV_size=0.2, nodeCV_size=0.1, boost_decay=False,
            ultra_boosting=False, g_final=0.0000001, g_tol=0.01,
            threshold=-0.01, minibatch=False, validation=SHUFFLED,
            symmetric_labels=False, mode=REGRESSION, alpha=0.0):
        print "creating training, validation, and testing sets..."
        train_test = train_test_split(X, Y, test_size=test_size)
        X_nottest, X_test, Y_nottest, Y_test = train_test

        print 'fitting scalers...tranforming data...'
        if symmetric_labels:
            X_nottest, X_nottest_inds = FoldLabels(X_nottest)
            X_test, X_test_inds = FoldLabels(X_test)
        X_nottest, X_nottest_scaler = Preprocess(X_nottest)
        X_test, _ = Preprocess(X_test, Scaler=X_nottest_scaler)
        Y_nottest, Y_nottest_scaler = Preprocess(Y_nottest)

        (self.X_train,
        self.X_validate_layer,
        self.Y_train,
        self.Y_validate_layer) = train_test_split(X_nottest, Y_nottest, test_size=boostCV_size)

        if validation == UNIFORM:
            (self.X_train_node,
            self.X_validate_node,
            self.Y_train_node,
            self.Y_validate_node) = train_test_split(self.X_train, self.Y_train,
                                          test_size=nodeCV_size)
        elif validation == SHUFFLED:
            self.X_train_node = self.X_train
            self.X_validate_node = self.X_validate_layer
            self.Y_train_node = self.Y_train
            self.Y_validate_node = self.Y_validate_layer
        else:
            raise ValueError("What is this validation supposed to mean -.-'")

        self.init_layer(num_iter, alpha, epsilon, minibatch)
        self.build_layer(num_nodes, validation, nodeCV_size, num_iter, alpha, epsilon)

        pred_train = self.layer._predict(self.X_train)
        pred_validate = self.layer._predict(self.X_validate_layer)
        pred_test = self.layer._predict(X_test)

        # stack training+validation sets, inverse transform, separate again
        K = len(self.Y_train)
        x_train = numpy.vstack((self.X_train, self.X_validate_layer))
        y_train = numpy.hstack((self.Y_train, self.Y_validate_layer))

        x_train = Postprocess(x_train, X_nottest_scaler)
        y_train = Postprocess(y_train, Y_nottest_scaler)
        pred_train = Postprocess(pred_train, Y_nottest_scaler)
        pred_validate = Postprocess(pred_validate, Y_nottest_scaler)
        pred_test = Postprocess(pred_test, Y_nottest_scaler)

        self.X_train, self.X_validate_layer = [x_train[:K, :], x_train[K:, :]]
        self.Y_train, self.Y_validate_layer = [y_train[:K], y_train[K:]]

        self.layer.err_train = get_error(self.Y_train, pred_train)
        self.layer.err_validate = get_error(self.Y_validate_layer, pred_validate)
        self.layer.err_test = get_error(Y_test, pred_test)

        self.layer.X_scaler = X_nottest_scaler
        self.layer.Y_scaler = Y_nottest_scaler

        print self.layer.err_train, self.layer.err_validate, self.layer.err_test

    def init_layer(self, num_iter, alpha, epsilon, minibatch):
        """Initializes the layer and adds an initial node to it."""
        self.layer = Layer()

        node = OptimalNode(self.X_train_node, self.Y_train_node, bias=True,
                           num_iter=num_iter, alpha=alpha, minibatch=minibatch)
        node.early_stop(self.X_validate_node, self.Y_validate_node)
        node.lr = epsilon
        node.is_useful(self.layer, self.X_validate_node, self.Y_validate_node)

        self.layer.add_node(node)
        node.train_err = get_error(self.Y_train_node,
                    self.layer._predict(self.X_train_node))

    def build_layer(self, num_nodes, validation, nodeCV_size, num_iter, alpha, epsilon):
        """Builds a Layer by optimizing new nodes and adding them if they are useful.
        Each successive node optimizes w.r.t. residuals of the previous iteration.
        If the new node reduces error, the node is added to the layer. If it increases
        error, it stops (unless a certain number of consecutive bad nodes are allowed)."""

        for i in range(num_nodes):
            if validation=='Shuffled':
                train_validate = train_test_split(self.X_train, self.Y_train,
                                                  test_size=nodeCV_size)
                [self.X_train_node, self.X_validate_node,
                    self.Y_train_node, self.Y_validate_node] = train_validate

            Y_pseudo = self.Y_train_node-self.layer._predict(self.X_train_node)
            Y_pseudo_validate = self.Y_validate_node-self.layer._predict(self.X_validate_node)

            node = OptimalNode(self.X_train_node, Y_pseudo, bias=True,
                               num_iter=num_iter, alpha=alpha)
            node.early_stop(self.X_validate_node, Y_pseudo_validate)
            node.lr = epsilon

            if node.is_useful(self.layer, self.X_validate_layer, self.Y_validate_layer):
                self.layer.add_node(node)
                node.train_err = get_error(self.Y_train_node,
                            self.layer._predict(self.X_train_node))
                print "Adding node %d. Validation error: %f Training error: %f" % \
                        (i,
                        node.validation_err,
                        node.train_err)
            else:
                break

    def get_layer(self):
        return self.layer