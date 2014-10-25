import matplotlib.pyplot as plt
import matplotlib.lines as mlines

class Plot:
    def __init__(self, *layersets):
        self.layersets = layersets

    def training_validation(self, i=0):
        """Plots error vs node for a given layerset."""
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for color, layer in zip(colors, self.layersets[i]):
            validation = [node.validation_err for node in layer.nodes]
            training = [node.train_err for node in layer.nodes]
            plt.plot(validation, color=color)
            plt.plot(training, linestyle="--", color=color)

        plt.plot([], linestyle="--", color="k", label="Training")
        plt.plot([], color="k", label="Validation")
        plt.legend()

        plt.show()

    def box_plot(self, labels=None):
        """Builds a box plot of a set of layer sets."""
        data = map(lambda plot: map(lambda x: x.err_train, plot), self.layersets)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        bp = plt.boxplot(data, notch=1, sym='+', vert=1, whis=1.5)
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], color='red', marker='+')
        if not labels:
            labels = ["%d (%d)" % (j+1, len(e)) for j, e in enumerate(self.layersets)]
        xtickNames = plt.setp(ax1, xticklabels=labels)
        plt.setp(xtickNames, rotation=15, fontsize=8)
        plt.title('Box plot of test errors')
        plt.show()