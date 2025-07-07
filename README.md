# PyTorch Neural Network: Predicting Bike Share Demand

This project involves building a neural network from scratch using PyTorch to forecast the hourly demand for a bike-sharing program based on historical data. The goal is to accurately predict the total number of bike rentals (`cnt`) by learning from seasonal, weather, and time-based features.

## Project Overview

The core of this project is the implementation of a multi-layer neural network to solve a regression problem. The process includes:

  * **Data Loading and Preprocessing:** Loading the dataset with Pandas and preparing it for training.
  * **Feature Engineering:** Converting categorical features (like season, month, and hour) into dummy variables and scaling numerical features to a standard range.
  * **Model Architecture:** Designing and building a neural network with multiple hidden layers using the ReLU activation function.
  * **Training and Validation:** Implementing a training loop with a validation step to monitor performance and prevent overfitting. An early stopping technique was used to save the best-performing model.
  * **Evaluation:** Using the trained model to make predictions on an unseen test dataset and visualizing the results against the actual values.

## Technologies Used

  * **Python**
  * **PyTorch:** For building and training the neural network.
  * **Pandas:** For data manipulation and preprocessing.
  * **NumPy:** For numerical operations.
  * **Matplotlib:** For data visualization.
  * **Jupyter Notebook** / **VS Code**

## Setup and Usage

To run this project on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/josemac36/Bike-Sharing-Demand-Prediction.git
    ```
2.  **Create a Conda Environment:**
    ```bash
    conda create --name bike-prediction-env python=3.12
    conda activate bike-prediction-env
    ```
3.  **Install Dependencies:**
    ```bash
    conda install numpy pandas matplotlib pytorch jupyter -c pytorch
    ```
4.  **Run the Notebook:**
    Open the `Your_first_neural_network_Johan_Sebastian_Martinez.ipynb` file in Jupyter or VS Code and run the cells sequentially.

## Results and Evaluation

The model was successfully trained, and both the training and validation losses showed a steady decrease, indicating effective learning.

#### Training and Validation Loss

The following plot shows the decrease in loss over 30 epochs:

#### Prediction on Test Data

The final evaluation on the unseen test data shows that the model's predictions closely track the actual bike rental counts, capturing the daily and weekly patterns in ridership.
