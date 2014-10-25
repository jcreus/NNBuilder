from layer import Layer
from sklearn import datasets
from plots import Plot
from builder import LayerBuilder

iris = datasets.load_boston()

X = iris.data
Y = iris.target

l = Layer.new_from_file("testing")
p = Plot([l])
p.training_validation()

"""l = Layer.new(X, Y, num_nodes=50, num_iter=5000, epsilon=1.0,
                test_size=0.25, boostCV_size=0.15, nodeCV_size=0.18,
                minibatch=True, validation='Uniform')
l.save("testing")
Layer.save_multiple(l, "boston-set")"""
for a in range(10):
    print l.predict([X[a]]), Y[a]

#Layer.save_multiple(l, "boston-set")
exit()

l2 = Layer.new_multiple(3, X, Y, num_nodes=50, num_iter=1000, epsilon=1.0,
                test_size=0.25, boostCV_size=0.15, nodeCV_size=0.18,
                minibatch=True, validation='Uniform')

p = Plot(l, l2)
p.training_validation()
p.box_plot(["ni=5000", "ni=1000"])