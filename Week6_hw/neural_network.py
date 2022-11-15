from losses import Loss
from layers import Layer
from tqdm import trange


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.err_log = []
        self.err_log_val = []

    # add layer to network
    def add(self, layer: Layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss: Loss):
        self.loss = loss

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network

    def fit(self, x_train, y_train, epochs=5, learning_rate=0.1):
        """
        Fit function does the training.
        Training data is passed 1-by-1 through the network layers during forward propagation.
        Loss (error) is calculated for each input and back propagation is performed via partial
        derivatives on each layer.
        """
        # sample dimension first
        samples = len(x_train)
        # samples_val = len(x_val)
        t = trange(epochs, desc="Error | Epoch", leave=True)

        # training loop
        for i in t:
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss(y_train[j], output, derivative=True)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # err_val = 0
            # for j in range(samples_val):
            #     # forward propagation
            #     output = x_val[j]
            #     for layer in self.layers:
            #         output = layer.forward_propagation(output)

            #     # compute loss (for display purpose only)
            #     err_val += self.loss(y_val[j], output)

            # calculate average error on all samples
            err /= samples
            # err_val /= samples_val

            self.err_log.append(err)
            # self.err_log_val.append(err_val)
            t.set_description(
                f"Epoch: {i} | Train loss: {err:.4}",
                refresh=True,
            )
