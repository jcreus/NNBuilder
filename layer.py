import cPickle as pickle
from utils import *

class Layer:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        """Add a node to the layer."""
        self.nodes.append(node)

    def predict(self, X):
        """Predict Y values for a given X array, returns array of Y values."""
        X, _ = Preprocess(X, Scaler=self.X_scaler)
        return Postprocess(self._predict(X), self.Y_scaler)

    def _predict(self, X):
        """Predict Y values for a given X array without postprocessing results."""
        return sum(map(lambda node: node.predict(X)*node.lr, self.nodes))

    def save(self, path):
        """Save the layer model to a file using cPickle."""
        pickle.dump(self, open("%s.pickle" % path,"wb"), -1)

    @staticmethod
    def save_multiple(layerset, prefix):
        """Save multiple layers ('layer set') to files with a common prefix."""
        for i, layer in enumerate(layerset):
            layer.save("%s-%d" % (prefix, i))

    @classmethod
    def multiple_from_file(cls, prefix):
        """Load multiple layers saved using Layer.save_multiple."""
        out = []
        while True:
            try:
                out.append(Layer.new_from_file("%s-%d" % (prefix, len(out))))
            except IOError:
                break
        return out

    @classmethod
    def new_from_file(cls, path):
        """Load layer from pickled file."""
        return pickle.load(open("%s.pickle" % path,"rb"))

    @classmethod
    def new(cls, *args, **kwargs):
        """Create new Layer object with parameters."""
        from builder import LayerBuilder
        return LayerBuilder(*args, **kwargs).get_layer()

    @classmethod
    def new_multiple(cls, n, *args, **kwargs):
        """Create multiple Layer objects with the same parameters."""
        return [Layer.new(*args, **kwargs) for _ in range(n)]