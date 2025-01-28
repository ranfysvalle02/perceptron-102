import numpy as np  
   
# Define the activation function  
def activation_function(x):  
    return 1 if x >= 0 else 0  # Step function  
   
# Initialize weights and bias  
weights = np.array([0.0, 0.0])  # Start with weights of 0 for both inputs  
bias = 0.0  # Start with bias of 0  
   
# Set the learning rate  
learning_rate = 0.1  
   
# Prepare the training data  
training_data = [  
    (np.array([0.85, 0.90]), 1),  # Student A passed  
    (np.array([0.60, 0.70]), 0),  # Student B failed  
    (np.array([0.75, 0.80]), 1),  # Student C passed  
    (np.array([0.50, 0.65]), 0),  # Student D failed  
]  
   
# Number of epochs  
epochs = 10  # You can increase this number for better learning  
   
# Training loop  
for epoch in range(epochs):  
    print(f"Epoch {epoch+1}")  
    for inputs, desired_output in training_data:  
        # Calculate the weighted sum and add bias  
        weighted_sum = np.dot(inputs, weights) + bias  
  
        # Get the perceptron's prediction  
        prediction = activation_function(weighted_sum)  
  
        # Calculate the error  
        error = desired_output - prediction  
  
        # Update the weights and bias if there's an error  
        if error != 0:  
            # Adjust weights  
            weights += learning_rate * error * inputs  
            # Adjust bias  
            bias += learning_rate * error  
  
            print(f"Updated weights: {weights}")  
            print(f"Updated bias: {bias}")  
    print("---")  
   
# Function to make predictions  
def predict(input_scores):  
    weighted_sum = np.dot(input_scores, weights) + bias  
    output = activation_function(weighted_sum)  
    return output  
   
# Test the perceptron with a new student's scores  
new_student_scores = np.array([0.70, 0.75])  # Normalized scores  
   
# Make prediction  
result = predict(new_student_scores)  
   
# Display the result  
if result == 1:  
    print("The perceptron predicts the student will pass.")  
else:  
    print("The perceptron predicts the student will fail.")  
