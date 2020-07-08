In this problem we will investigate handwritten digit classification. The inputs are 16 by 16 grayscale images
of handwritten digits (0 through 9), and the goal is to predict the number of the given the image. If you run
example_neuralNetwork it will load this dataset and train a neural network with stochastic gradient descent,
printing the validation set error as it goes. To handle the 10 classes, there are 10 output units (using a {−1, 1}
encoding of each of the ten labels) and the squared error is used during training. Your task in this question is to
modify this training procedure architecture to optimize performance.
Report the best test error you are able to achieve on this dataset, and report the modifications you made to
achieve this. Please refer to previous instruction of writing the report.
