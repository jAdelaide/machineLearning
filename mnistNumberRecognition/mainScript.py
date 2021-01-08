    # The downloaded MNIST data set has number images and their corresponding labels
import torch
from torchvision import datasets
import matplotlib.pyplot as plt

    # We import the PyTorch library for building our neural networks and the Torchvision library for downloading the MNIST data set
    # The Matplotlib library is used for displaying images from out data set

                # Preparing your data set

mnist = datasets.MNIST('./data', download=True)

threes = mnist.data[(mnist.targets == 3)]/255.0
sevens = mnist.data[(mnist.targets == 7)]/255.0

len(threes), len(sevens)

        # Defining the show_image function
def show_image(img):
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

show_image(threes[3])
show_image(sevens[8])

        # This should print "torch.Size([6131, 28, 28]) torch.Size([6265, 28, 28])"
        # This means there are 6131 images that are 28*28 for 3s and 6265 for 7s
print(threes.shape, sevens.shape)

        # Now we have 2 tensors, we need to combine them to feed into our neural network
combined_data = torch.cat([threes, sevens])
combined_data.shape

        # The images in the dataset now need to be flattened
        # Each of the 28*28 images will be made into rows of 784 columns (28*28=784)
flat_imgs = combined_data.view((-1, 28*28))
flat_imgs.shape

        # We need to create labels for the images in the combined data set
        # Images containing a 3 will get the label 1
        # Images containing a 7 will get the label 0
target = torch.tensor([1]*len(threes)+[2]*len(sevens))
target.shape

                # Training your Neural Network

            # Step 1: Build the model
        # As 0=7 and 1=3, we can create a threshold of 0.5
        # We will use a sigmoid function to get a number between 0 and 1
        # If it is below 0.5 it is a 7 and above will be a 3
def sigmoid(x): return 1/(1+torch.exp(-x))
def simple_nn(data, weights, bias): return sigmoid((data@weights) + bias)

            # Step 2: Defining the loss
        # We need to define a loss function to calculate how far our predicted value is from the ground truth
        # When there is a big difference, the model will update the weights and bias to try and reduce this loss
        # We will be using Mean Squared Error to check the loss value
def error(pred, target): return ((pred-target)**2).mean()

            # Step 3: Initialise the weight and bias values
        # We just initialise these randomly
        # The model will update them during it's training to get more accurate results
w = torch.randn((flat_imgs.shape[1], 1), requires_grad=True)    # weight
b = torch.randn((1, 1), requires_grad=True)                     # bias

        # The best method to decrease loss is to calculate gradients and use a method called gradient descent to update our weight and bias values
        # We need to take the derivative of every w&b with respect to the loss function, the subtract this from our weights and bias
        # requires_grad is a special parameter for calculating the gradients

            # Step 4: Update the weights and bias
        # To update the weights and bias, we put all the above steps in a loop and allow it to run over and over and over again
        # Each loop checks how far off our prediction is and updates the values to do better next time
for i in range(2000):
        # We calculate the predictions and store them in the "pred" variable, using a function created above
    pred = simple_nn(flat_imgs, w, b)
        # Then calculate the mean squared error loss
    loss = error(pred, target.unsqueeze(1))
    loss.backward()

        # Then calculate all the gradients and update our w&b values
        # The gradients are multiplies by a learning rate of 0.001
        # Too high and the model will be slow to learn, too high and it will jump about too much and be unstable
    w.data -= 0.001*w.grad.data
    b.data -= 0.001*b.grad.data

        # Zero out the gradients at the end of each loop to prevent any unwanted accumulation
    w.grad.zero_()
    b.grad.zero_()

print("Loss: ", loss.item())