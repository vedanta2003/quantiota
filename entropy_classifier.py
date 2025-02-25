import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Load dataset from the local file
with np.load("mnist.npz") as data:
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]
    
# Normalize images to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten the images to be vectors
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Convert digit labels to even/odd labels: Even -> 0, Odd -> 1
y_train_even_odd = np.array([label % 2 for label in y_train])
y_test_even_odd = np.array([label % 2 for label in y_test])

# Initialize parameters
n_features = x_train.shape[1]  # Number of input features (28x28 = 784)
w = np.random.randn(n_features) * 0.01  # Weight vector w
G = np.random.randn(n_features) * 0.01  # Weight vector G
b = np.random.randn() * 0.01  # Bias term
delta = 0.1  # Increased constraint on z changes
eta = 0.1  # Increased learning rate

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Entropy gradient calculation
def entropy_gradient(z, D):
    epsilon = 1e-8  # Small constant to avoid division by zero
    return (-1 / np.log(2)) * z * D * (1 - D + epsilon)

# Custom training loop
epochs = 10  # Increased number of epochs
batch_size = 64  # Increased batch size
n_samples = x_train.shape[0]
accuracy_history = []

for epoch in range(epochs):
    for i in range(0, n_samples, batch_size):
        # Get mini-batch
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train_even_odd[i:i + batch_size]

        # Compute z and D for the batch
        z = np.dot(x_batch, w + G) + b
        D = sigmoid(z)

        # Compute entropy gradient
        grad_H = entropy_gradient(z, D)

        # Update parameters
        w -= eta * np.dot(grad_H, x_batch)
        G -= eta * np.dot(grad_H, x_batch)
        b -= eta * np.sum(grad_H)

        # Enforce z constraint: z_{i+1} - z_i < delta
        z_new = np.dot(x_batch, w + G) + b
        z_diff = z_new - z
        if np.any(z_diff >= delta):
            scale_factor = delta / np.max(z_diff)
            w *= scale_factor
            G *= scale_factor
            b *= scale_factor

    # Evaluate accuracy on test set after each epoch
    z_test = np.dot(x_test, w + G) + b
    D_test = sigmoid(z_test)
    predictions = (D_test >= 0.5).astype(int)
    accuracy = np.mean(predictions == y_test_even_odd)
    accuracy_history.append(accuracy)
    print(f"Epoch {epoch + 1}, Test Accuracy: {accuracy:.4f}")

# Plot accuracy over epochs
plt.plot(range(1, epochs + 1), accuracy_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Epochs")
plt.show()

# Display 10 random test images with predictions
random_indices = np.random.randint(0, x_test.shape[0], size=10)
plt.figure(figsize=(15, 6))
for idx, i in enumerate(random_indices):
    image = x_test[i].reshape(28, 28)
    true_label = y_test_even_odd[i]
    prediction_prob = sigmoid(np.dot(x_test[i], w + G) + b)
    prediction = 1 if prediction_prob >= 0.5 else 0

    true_text = "Odd" if true_label == 1 else "Even"
    pred_text = "Odd" if prediction == 1 else "Even"

    plt.subplot(2, 5, idx + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"True: {true_text}\nPred: {pred_text}")
    plt.axis('off')

plt.tight_layout()
plt.show()