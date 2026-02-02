# Developing a Neural Network Regression Model
## AIM
To develop a neural network regression model for the given dataset

## THEORY

The objective is to design and train a neural network–based regression model that can learn the relationship between input features and a continuous target variable in the given dataset.
The model aims to accurately predict numerical outcomes by minimizing prediction error and capturing underlying data patterns.

## Neural Network Model

![alt text](<Screenshot 2026-02-02 095012.png>)


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: LAKSHMIDHAR N

### Register Number: 212224230138

```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        # Include your code here
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss':[]}
  def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range(epochs):
      optimizer.zero_grad()
      loss = criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()

      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
    

```

### Dataset Information


![alt text](image.png)

### OUTPUT

### Training Loss Vs Iteration Plot

![alt text](image-1.png)

### New Sample Data Prediction


![alt text](image-2.png)

![alt text](image-3.png)

![alt text](image-4.png)

![alt text](image-5.png)


## RESULT
Thus, a neural network regression model w as successfully developed and trained using PyTorch.
