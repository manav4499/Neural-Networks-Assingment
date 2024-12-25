import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(2024)

# Exercise 1: Single Layer Feed-Forward Neural Network
# Step 1 & 2: Generate training data
input_jainam = np.random.uniform(-0.9, 0.9, (35, 2))

# Step 3: Create target data
target = 4 * input_jainam[:, 0] + 2 * input_jainam[:, 1] + 0.45
output_groupnumber = target.reshape(35, 1)

# Step 5: Create a simple neural network with one layer
net = nl.net.newff([[-0.9, 0.9], [-0.9, 0.9]], [10, 1])

# Step 6: Train the network
net.trainf = nl.train.train_gd
error = net.train(input_jainam, output_groupnumber, epochs=100, show=10, goal=0.0001)

# Step 8: Test the network
test_input = np.array([[0.33, 0.42]])
result_1 = net.sim(test_input)
print("Result #1 (Single Layer):", result_1)

# Exercise 2: Multi-Layer Feed-Forward Neural Network
# Step 1 & 2 are the same as Exercise 1

# Step 5: Create a multi-layer neural network with two hidden layers
net_multi = nl.net.newff([[-0.9, 0.9], [-0.9, 0.9]], [8, 3, 1])

# Step 6: Train the network with gradient descent
net_multi.trainf = nl.train.train_gd
error_multi = net_multi.train(input_jainam, output_groupnumber, epochs=150, show=10, goal=0.0001)

# Step 8: Test the network
result_2 = net_multi.sim(test_input)
print("Result #2 (Multi-Layer):", result_2)

# Exercise 3: Single Layer with More Training Data
# Step 1: Generate 150 random samples
input_jainam_more = np.random.uniform(-0.9, 0.9, (150, 2))

# Step 2: Create target data
target_more = 4 * input_jainam_more[:, 0] + 2 * input_jainam_more[:, 1] + 0.45
output_groupnumber_more = target_more.reshape(150, 1)

# Step 5: Create a simple neural network with one layer
net_more = nl.net.newff([[-0.9, 0.9], [-0.9, 0.9]], [10, 1])

# Step 6: Train the network
error_more = net_more.train(input_jainam_more, output_groupnumber_more, epochs=100, show=10, goal=0.0001)

# Step 8: Test the network
result_3 = net_more.sim(test_input)
print("Result #3 (Single Layer with More Training Data):", result_3)

# Exercise 4: Multi-Layer with More Training Data
# Step 5: Create a multi-layer neural network with two hidden layers
net_multi_more = nl.net.newff([[-0.9, 0.9], [-0.9, 0.9]], [8, 2, 1])

# Step 6: Train the network
net_multi_more.trainf = nl.train.train_gd
error_multi_more = net_multi_more.train(input_jainam_more, output_groupnumber_more, epochs=150, show=10, goal=0.0001)

# Step 8: Test the network
result_4 = net_multi_more.sim(test_input)
print("Result #4 (Multi-Layer with More Training Data):", result_4)

# Plotting Error Training Size Graph (Exercise 4)
plt.figure()
plt.plot(error_multi_more)
plt.title('Error Training Size Graph (Exercise 4)')
plt.xlabel('Epochs')
plt.ylabel('Training Error')
plt.grid()
plt.show()

# Exercise 5: Three Input Multi-Layer Neural Network
# Step 1: Generate training data with three inputs
input_jainam_three = np.random.uniform(-0.9, 0.9, (35, 3))

# Step 2: Create target data
target_three = 3 * input_jainam_three[:, 0] + 2 * input_jainam_three[:, 1] + 4 * input_jainam_three[:, 2] + 0.05
output_groupnumber_three = target_three.reshape(35, 1)

# Step 5: Create a multi-layer neural network with three inputs
net_three = nl.net.newff([[-0.9, 0.9], [-0.9, 0.9], [-0.9, 0.9]], [8, 2, 1])

# Step 6: Train the network
net_three.trainf = nl.train.train_gd
error_three = net_three.train(input_jainam_three, output_groupnumber_three, epochs=150, show=10, goal=0.0001)

# Step 8: Test the network
test_input_three = np.array([[0.1, 0.2, 0.1]])
result_5 = net_three.sim(test_input_three)
print("Result #5 (Three Input Multi-Layer):", result_5)

# Step 12: Test for three-input model with more data
input_jainam_more_three = np.random.uniform(-0.9, 0.9, (150, 3))
target_more_three = 3 * input_jainam_more_three[:, 0] + 2 * input_jainam_more_three[:, 1] + 4 * input_jainam_more_three[:, 2] + 0.05
output_groupnumber_more_three = target_more_three.reshape(150, 1)

net_more_three = nl.net.newff([[-0.9, 0.9], [-0.9, 0.9], [-0.9, 0.9]], [8, 2, 1])
net_more_three.trainf = nl.train.train_gd
error_more_three = net_more_three.train(input_jainam_more_three, output_groupnumber_more_three, epochs=150, show=10, goal=0.0001)

result_6 = net_more_three.sim(test_input_three)
print("Result #6 (Three Input Multi-Layer with More Training Data):", result_6)
