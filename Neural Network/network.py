#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]#random bias for every index of the second(array) and third layer(array)
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]#random weights from input (x) and output(y) in the second and third layer--creates a 2 dimensional array(x *y) representing every individual connection

    def feedforward(self, a):#output the last layer
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)#Checks if test_data is empty
        n = len(training_data) #50 000
        for j in range(epochs): #epochs 30 
            random.shuffle(training_data) #shuffle training data
            mini_batches = [ # create a mini batch
                training_data[k:k+mini_batch_size] #mini batch size is 10 
                for k in range(0, n, mini_batch_size)] #every 10 
            for mini_batch in mini_batches: #run 50 000 times
                self.update_mini_batch(mini_batch, eta) #for each minibatch
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]#create a 2D with the same shape as the biases array
        nabla_w = [np.zeros(w.shape) for w in self.weights]#create a 2D with the same shape as the weight array
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #gradient descent
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]#adjust the weights
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]#adjusts the biases

    def backprop(self, x, y): 
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] #activation vector
        zs = [] # weighted input vector
        for b, w in zip(self.biases, self.weights): #Define vectors that represent the weighted inputs and activations
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #Output layer
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) #Error for each neuron in the output layer
        nabla_b[-1] = delta # They are equal due to chain rule
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        #First layer
        l=2
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(self.weights[-l+1].transpose(), delta) * sp #Delta calculation with respect to first layer
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data] #Finds the index of whiever neuron in the final layer has the highest activation
        return sum(int(x == y) for (x, y) in test_results)#Sums the number of numbers they guessed correctly
    

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    def individual_test(self, test_data):#Individual tests
        for i in range (10):
            index = random.randint(0,10000)
            x, y = test_data[index]  
            output=(self.feedforward(x))
            predictionNumber = np.argmax(output)
            probability=(output[predictionNumber][0]) * 100 
            print((str(predictionNumber)) +  " with a " + str(probability) + " percent confidence")
            self.show_number(test_data,index)


    def show_number(self, test_data, index):#Display a 28 x 28 MNIST digit
        image_data, label = test_data[index]
        image_data = np.reshape(image_data, (28, 28))
        plt.imshow(image_data, cmap='gray')
        plt.title(f'Label: {label}')
        plt.show()

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

