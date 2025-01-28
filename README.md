# perceptron-102

---

# Perceptron Learning in Action
   
---  
   
*Discover how machines learn from data by teaching a perceptron to predict whether a student will pass or fail, using simple Python code and easy-to-understand concepts.*  
   
---  
   
## Introduction  
   
In our previous blog post, **[How Simple Ideas Power Modern AI](https://github.com/ranfysvalle02/perceptron-101)**, we explored the foundational concepts of the perceptron—a simple yet powerful model that forms the basis of modern neural networks. We learned how a perceptron makes decisions by weighing inputs, summing them up, and applying an activation function.  
   
Now, let's take a step further and see the perceptron in action! We'll teach a perceptron to learn from data—specifically, we'll train it to predict whether a student will pass or fail based on their test scores. This journey will demystify the learning process, show how weights and biases are adjusted, and illustrate how machines can improve over time.  
   
---  
   
## The Learning Perceptron: How Machines Learn from Data  
   
### What's New?  
   
In the previous post, we manually set the weights and bias of our perceptron. This time, we'll let the perceptron **learn** these values from training data. We'll use a simple learning algorithm to adjust the weights and bias based on errors between the perceptron's predictions and the actual outcomes.  
   
### The Concept  
   
- **Training Data**: A set of input-output pairs that the perceptron will learn from.  
- **Learning Rate**: A small positive number that determines how much we adjust the weights and bias during learning.  
- **Epochs**: The number of times we go through the entire training dataset.  
- **Error Calculation**: The difference between the desired output and the perceptron's prediction.  
- **Weight and Bias Update**: Adjusting the weights and bias to reduce the error.  
   
---  
   
## Step-by-Step Guide: Teaching the Perceptron to Predict Pass or Fail  
   
Let's dive into the code and understand how the perceptron learns from data.  
   
### Step 1: Import Necessary Libraries  
   
We start by importing the NumPy library, which we'll use for numerical operations.  
   
```python  
import numpy as np  
```  
   
### Step 2: Define the Activation Function  
   
The activation function determines the perceptron's output based on the weighted sum of inputs.  
   
```python  
def activation_function(x):  
    return 1 if x >= 0 else 0  # Step function  
```  
   
- **Logic**: If the weighted sum is greater than or equal to 0, the perceptron outputs 1 (pass); otherwise, it outputs 0 (fail).  
   
### Step 3: Initialize Weights and Bias  
   
We start with weights and bias set to zero.  
   
```python  
weights = np.array([0.0, 0.0])  # Start with weights of 0 for both inputs  
bias = 0.0  # Start with bias of 0  
```  
   
### Step 4: Set the Learning Rate  
   
The learning rate determines how much we adjust the weights and bias during each update.  
   
```python  
learning_rate = 0.1  # A small positive value  
```  
   
### Step 5: Prepare the Training Data  
   
Our training data consists of students' normalized test scores and whether they passed (1) or failed (0).  
   
```python  
training_data = [  
    (np.array([0.85, 0.90]), 1),  # Student A passed  
    (np.array([0.60, 0.70]), 0),  # Student B failed  
    (np.array([0.75, 0.80]), 1),  # Student C passed  
    (np.array([0.50, 0.65]), 0),  # Student D failed  
]  
```  
   
- **Normalization**: We divide the scores by 100 to bring them between 0 and 1.  
   
### Step 6: Set the Number of Epochs  
   
An epoch is one full pass through the training data.  
   
```python  
epochs = 10  # You can increase this number for better learning  
```  
   
### Step 7: Training Loop  
   
Now, we'll train the perceptron over several epochs.  
   
```python  
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
```  
   
#### Breaking Down the Training Loop  
   
1. **For Each Epoch**:  
   - We iterate over the training data.  
   - **Calculate Weighted Sum**: Use the dot product of inputs and weights, and add the bias.  
   - **Make a Prediction**: Apply the activation function to the weighted sum.  
   - **Calculate Error**: `error = desired_output - prediction`.  
   - **Update Weights and Bias**:  
     - If there's an error, adjust the weights and bias in the direction that reduces the error.  
     - **Weights Update**: For each weight, we add a small adjustment. This adjustment is the product of the learning rate, the error, and the corresponding input value.  
     - **Bias Update**: We adjust the bias by adding the product of the learning rate and the error.  
   
2. **Printing Updates**:  
   - We print the updated weights and bias whenever they change.  
   
### Step 8: Define the Prediction Function  
   
After training, we can use the perceptron to make predictions on new data.  
   
```python  
def predict(input_scores):  
    weighted_sum = np.dot(input_scores, weights) + bias  
    output = activation_function(weighted_sum)  
    return output  
```  
   
### Step 9: Test the Perceptron with New Data  
   
We input a new student's normalized test scores.  
   
```python  
new_student_scores = np.array([0.70, 0.75])  # Normalized scores  
   
# Make a prediction  
result = predict(new_student_scores)  
   
# Display the result  
if result == 1:  
    print("The perceptron predicts the student will pass.")  
else:  
    print("The perceptron predicts the student will fail.")  
```  
   
---  
   
## Understanding the Output  
   
### Epochs and Updates  
   
During each epoch, the perceptron goes through the training data and updates the weights and bias whenever its prediction doesn't match the desired output.  
   
Here's an excerpt of what the output might look like:  
   
```  
Epoch 1  
Updated weights: [-0.06 -0.07]  
Updated bias: -0.1  
Updated weights: [0.015 0.01 ]  
Updated bias: 0.0  
Updated weights: [-0.035 -0.055]  
Updated bias: -0.1  
---  
Epoch 2  
Updated weights: [0.05  0.035]  
Updated bias: 0.0  
...  
```  
   
- **Weights Update**: The weights change based on the learning rate, error, and inputs.  
- **Bias Update**: The bias adjusts to shift the decision boundary.  
   
### Final Prediction  
   
After training, the perceptron predicts whether the new student will pass:  
   
```  
The perceptron predicts the student will pass.  
```  
   
---  
   
## Breaking Down the Learning Process with Analogies  
   
### The Learning Rule  
   
Think of the perceptron as an eager student trying to improve at predicting outcomes. Each time it makes a mistake, it adjusts its understanding to do better next time.  
   
- **Error as Feedback**: When the perceptron makes a wrong prediction, it uses this error as feedback.  
- **Adjusting Weights**: The perceptron tweaks its weights by adding or subtracting a small amount. This amount depends on:  
  - **Learning Rate**: How quickly the perceptron is willing to change. A small learning rate means cautious adjustments.  
  - **Error**: The difference between the desired output and the prediction. A larger error leads to a bigger adjustment.  
  - **Inputs**: The values of the inputs at which the error occurred. Inputs that are larger in magnitude will influence the adjustment more.  
   
- **Adjusting Bias**: Similar to weights, the bias is adjusted by adding or subtracting a small amount based on the learning rate and error.  
   
### How Adjustments Happen  
   
- **If the Perceptron Predicts Too Low**:  
  - The error is positive (desired output is 1, prediction is 0).  
  - We increase the weights and bias slightly to make it more likely to predict a 1 next time with similar inputs.  
- **If the Perceptron Predicts Too High**:  
  - The error is negative (desired output is 0, prediction is 1).  
  - We decrease the weights and bias slightly to make it less likely to predict a 1 next time with similar inputs.  
   
### Epochs as Practice Rounds  
   
Each epoch is like a practice session where the perceptron reviews all examples to improve its performance. Over multiple epochs, the perceptron fine-tunes its weights and bias to reduce errors.  
   
---  
   
## Visualizing the Decision Boundary  
   
The perceptron learns a linear decision boundary that separates passing and failing students.  
   
- **Before Training**: The weights are zero, so the perceptron cannot distinguish between any inputs.  
- **After Training**: The weights and bias define a line (in two dimensions) that separates the input space into regions predicting pass or fail.  
   
Imagine plotting test scores on a graph, with one score on the x-axis and the other on the y-axis. The perceptron finds the line that best separates the "pass" points from the "fail" points.  
   
---  
   
## Experiment Yourself!  
   
### Adjusting Parameters  
   
Try modifying the following:  
   
- **Learning Rate**: Increase or decrease it to see how quickly the perceptron learns.  
  - **Higher Learning Rate**: Faster learning but can overshoot the optimal weights.  
  - **Lower Learning Rate**: Slower learning but more precise adjustments.  
- **Epochs**: Increase the number of epochs to allow the perceptron more time to learn.  
- **Training Data**: Add more students or change their scores to see the impact.  
   
### Observing Convergence  
   
- **Overfitting**: With too few examples or too many epochs, the perceptron might memorize the training data and not perform well on new data.  
- **Underfitting**: With too few epochs, the perceptron may not learn enough to make accurate predictions.  
   
---  
   
## Full Code for the Learning Perceptron  
   
```python  
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
```  
   
---  
   
## The Fundamentals: Understanding the Elegant Math  
   
### Weight and Bias Updates Explained  
   
When the perceptron makes a mistake in its prediction, it adjusts its weights and bias to improve future predictions.  
   
#### Adjusting Weights  
   
- **How It Works**:  
  - For each weight, we adjust it by adding a small amount.  
  - This amount is calculated by multiplying:  
    - **Learning Rate**: Determines how aggressive the adjustment is.  
    - **Error**: The difference between the actual output and the predicted output.  
    - **Input Value**: The value of the input corresponding to the weight.  
- **Why It Works**:  
  - If an input contributed to a wrong prediction, adjusting the weight associated with that input helps correct the error.  
  - Inputs with larger values have a more significant impact on the adjustment.  
   
#### Adjusting Bias  
   
- **How It Works**:  
  - The bias is adjusted by adding the product of the learning rate and the error.  
- **Why It Works**:  
  - Adjusting the bias shifts the decision boundary, helping the perceptron make more accurate predictions overall.  
   
#### Example Update  
   
Suppose the perceptron predicted 0 (fail), but the desired output was 1 (pass). Here's how the adjustment happens:  
   
- **Error Calculation**:  
  - Error = Desired Output - Prediction = 1 - 0 = 1  
- **Adjust Weights**:  
  - Multiply the learning rate (0.1), the error (1), and each input value.  
  - For input 1 (e.g., 0.85):  
    - Adjustment = 0.1 * 1 * 0.85 = 0.085  
    - New weight = Old weight + Adjustment  
  - Do the same for input 2.  
- **Adjust Bias**:  
  - Adjustment = 0.1 * 1 = 0.1  
  - New bias = Old bias + Adjustment  
   
### Why It Works  
   
- **Direction of Adjustment**:  
  - If the perceptron needs to predict higher outputs, the weights and bias increase.  
  - If it needs to predict lower outputs, the weights and bias decrease.  
- **Magnitude of Adjustment**:  
  - The size of the adjustment is proportional to the error and the input values.  
  - This ensures that significant errors lead to more substantial adjustments, speeding up learning.  
   
---  
   
## Patterns in Hype Cycles: Embracing the Learning Journey  
   
Just like the perceptron adjusts its understanding over time, the field of AI has evolved through periods of excitement and skepticism.  
   
- **Learning from Mistakes**: Early limitations didn't stop progress; they guided researchers to improve models.  
- **Continuous Improvement**: By embracing new data and techniques, AI models become more accurate and versatile.  
   
---  
   
## Conclusion  
   
We've seen how a perceptron can learn from data to make predictions—transforming from a simple model with zero knowledge to one that can predict student outcomes based on test scores. This process mirrors how humans learn: by making mistakes, receiving feedback, and adjusting our understanding.  
   
The perceptron's ability to adjust weights and bias based on errors is a fundamental concept that extends to more complex neural networks and deep learning models. Understanding this simple learning mechanism provides a foundation for exploring advanced AI topics.  
   
---  
   
## Explore Further  
   
- **Modify the Code**: Add more features (e.g., attendance, homework scores) to the inputs and see how the perceptron adjusts.  
- **Learning Rates and Epochs**: Experiment with different learning rates and numbers of epochs to observe their effects.  
- **Visualize the Learning**: Plot the decision boundary before and after training to visualize how the perceptron learns to separate pass and fail regions.  
   
---  
   
## Connect to Real Life  
   
Consider how you adjust your own beliefs and actions based on feedback. Learning from mistakes and making incremental improvements is a universal process—whether in machines or humans.  
   
---  
   
*By understanding the perceptron and its learning process, we're not just exploring a machine learning model—we're uncovering fundamental principles of learning and adaptation that apply to many areas of life.*  
   
---  
   
# Full Code for Your Reference  
   
```python  
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
```  
   
---  
   
*Happy Learning!*  
   
---  
   
*Note: This code is intended for educational purposes to illustrate how a perceptron learns from data. In real-world applications, more sophisticated models and techniques are used to handle complex data and make accurate predictions.*
