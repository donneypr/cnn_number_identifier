# MNIST Digit Classification with Convolutional Neural Networks (CNN)
This repository contains a project that classifies handwritten digits from the MNIST dataset using a Convolutional Neural Network (CNN) built with PyTorch. The MNIST dataset consists of 60,000 training images and 10,000 test images, each representing one of the digits from 0 to 9. The goal is to accurately classify these digits using a deep learning model.

## Project Overview
The model is a CNN with the following layers:

- Two convolutional layers followed by max pooling layers for feature extraction.
- Three fully connected layers to map extracted features to digit classifications (0-9).
- ReLU activation functions and log_softmax for the output layer to calculate the class probabilities.

After training the model for 5 epochs, the accuracy of the model on the test dataset reaches approximately 99%.

## Model Architecture
The CNN model consists of the following layers:

- Convolutional Layer 1: 1 input channel (grayscale), 6 output channels, 3x3 kernel.
- Max Pooling Layer 1: 2x2 pooling.
- Convolutional Layer 2: 6 input channels, 16 output channels, 3x3 kernel.
- Max Pooling Layer 2: 2x2 pooling.
- Fully Connected Layer 1: Takes the 5x5x16 flattened feature map and outputs 120 units.
- Fully Connected Layer 2: 120 input units, 84 output units.
- Fully Connected Layer 3: 84 input units, 10 output units (for 10 digit classes).
## Dataset
The MNIST dataset is a benchmark dataset in machine learning and computer vision, consisting of 28x28 pixel grayscale images of handwritten digits. Each image belongs to one of 10 classes (digits 0-9).

## Setup and Installation
1.Clone the repository:

```bash
git clone https://github.com/donneypr/mnist_cnn_pytorch.git
cd mnist_cnn_pytorch
```
2.Install dependencies:

```bash
pip install torch torchvision matplotlib numpy pandas scikit-learn
Run the project in a Jupyter Notebook or Python script.
```
## Training the Model
The model is trained using the Adam optimizer and the Cross-Entropy loss function. The training process runs for 5 epochs, and the model's performance is evaluated on both training and test datasets at each epoch.

### Example Training Output:
```bash
Epoch: 0  Batch: 600  Loss: 0.16495497524738312
Epoch: 0  Batch: 1200  Loss: 0.19736522436141968
...
Training Took: 2.993 mins
```
### Accuracy:
The model achieves approximately 98.93% accuracy on the test dataset.

## Results and Evaluation
The training process includes visualizing the training and testing losses, as well as the accuracy over the epochs. After training, the model achieves high accuracy and generalizes well to the test set.

Example plots:

- Loss Plot: Shows the loss over epochs for both training and test datasets.
- Accuracy Plot: Tracks the accuracy of predictions on the training and test datasets after each epoch.
## Test Predictions
The model can be used to predict the digit class for new handwritten images from the test set. To visualize the test image and pass it through the model:

```python
# Select an image from the test dataset
testing = test_data[1]

# Reshape and display the image
plt.imshow(testing[0].reshape(28,28))

# Predict the class using the trained model
model.eval()
with torch.no_grad():
    new_pred = model(testing[0].view(1,1,28,28))
    print(new_pred.argmax().item())  # Outputs the predicted class
```
## Conclusion
This project demonstrates the use of CNNs for image classification on the MNIST dataset. The model achieves a high level of accuracy and can generalize well to unseen test data.
