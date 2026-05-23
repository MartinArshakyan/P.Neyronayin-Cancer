# Car Price Prediction — Overall Project Overview

What is this project?
This is an end-to-end machine learning project that predicts the selling price of a car based on its specifications. It uses a deep neural network built with PyTorch, trained on real automotive data containing features like engine size, horsepower, brand, fuel type, body style, and dimensions.
The goal is simple: given the technical details of a car, how accurately can a model estimate its market price?

Why is this problem interesting?
Car pricing is not random. It is driven by a complex mix of factors — a turbocharged engine adds value, poor fuel economy reduces it, a luxury brand commands a premium, and heavier cars tend to cost more to manufacture. These relationships are nonlinear and interact with each other in ways that a simple formula cannot capture. Neural networks are well suited to learning exactly these kinds of complex, multi-variable patterns from data.
This makes it a realistic and practical regression problem — the kind that appears in insurance valuation, used car platforms, dealership pricing tools, and automotive market analysis.

Who would use something like this?

A used car marketplace that wants to flag listings priced too far above or below market value
An insurance company estimating vehicle replacement cost
A dealership building an automated pricing suggestion tool
A buyer trying to figure out if a car is fairly priced before negotiating
A student learning how to apply neural networks to a real structured dataset




What data does it use?
The dataset contains around 200–500 cars, each described by roughly 25 features split into three categories:
Identity and body — brand, body style (sedan, hatchback, wagon, etc.), number of doors, drive wheel type (front, rear, all-wheel)
Dimensions and weight — wheelbase, length, width, height, curb weight
Engine and performance — engine type, number of cylinders, engine size, fuel system, bore ratio, stroke, compression ratio, horsepower, peak RPM, city MPG, highway MPG
Fuel and aspiration — fuel type (gas or diesel), aspiration (standard or turbo), engine location
The target variable is price in US dollars, ranging roughly from $5,000 to $45,000.

How does the model work?
At its core the model is a feedforward neural network — a stack of mathematical layers that progressively transform the input features into a single output number representing the predicted price.
Each layer learns a set of weights during training. The training process feeds the model thousands of examples, compares its predictions to the real prices, measures the error, and nudges the weights slightly in the direction that reduces that error. Repeating this hundreds of times gradually makes the model better and better at the task.
Three techniques are used to make training more stable and prevent the model from simply memorizing the training data:
Batch normalization normalizes the outputs of each layer so that values stay in a reasonable range, which speeds up training significantly.
Dropout randomly switches off a fraction of neurons during each training step, forcing the network to learn redundant representations and become more robust.
Weight decay adds a small penalty for large weights, discouraging the model from becoming overly confident about any single feature.

How is performance measured?
Three metrics are reported:
RMSE (Root Mean Squared Error) measures the average prediction error in dollars. It penalizes large mistakes more heavily than small ones, which is appropriate here because being off by $10,000 is much worse than being off by $500.
MAE (Mean Absolute Error) is the plain average of how many dollars off each prediction is. Easier to interpret — if MAE is $1,500 it means predictions are wrong by about $1,500 on average.
R² (R-squared) measures how much of the natural variation in car prices the model explains. A value of 1.0 means perfect predictions. A value of 0.85 means the model explains 85% of price variation, which is strong performance for this kind of problem.



What makes this a good learning project?
It touches nearly every important concept in applied machine learning:
Data preprocessing — handling messy strings, fixing typos, encoding categories, scaling features and targets
Train/validation/test splitting — understanding why you need three sets, not two, and why the test set must stay completely unseen until the end
Neural network design — choosing layer sizes, activation functions, normalization, and regularization
Hyperparameter tuning — systematically comparing configurations rather than guessing
Evaluation — using multiple metrics and understanding what each one tells you
Model persistence — saving, loading, and running a model in inference mode, which is how models get deployed in real applications

Limitations and possible improvements
Dataset size — with only 200–500 samples the model has limited data to learn from. A larger dataset would almost certainly improve accuracy.
Feature engineering — the raw features are used as-is. Adding derived features like power-to-weight ratio or engine displacement per cylinder could help the model learn more meaningful patterns.
Architecture search — only 5 hyperparameter combinations are tested. A more thorough search or techniques like Bayesian optimization could find better configurations.
Gradient boosting comparison — for tabular data of this size, models like XGBoost or LightGBM often match or outperform neural networks with less tuning effort. Comparing against a baseline like that would give a clearer picture of how much the neural network is actually helping.
No cross-validation — a single train/val split means results depend somewhat on which samples happened to land in which set. K-fold cross-validation would give a more reliable estimate of true performance.
What this project does
This project builds a neural network that predicts the price of a car based on its physical and mechanical characteristics. Given features like engine size, horsepower, brand, fuel type, and body style, the model learns to estimate a realistic market price. It covers the full machine learning pipeline from raw data to a saved, reusable model.

Pipeline structure
Cell 1 — Setup
Installs and imports all required libraries: PyTorch for the neural network, scikit-learn for preprocessing and metrics, pandas and numpy for data handling, and matplotlib for plots.
Cell 2 — Data loading
Loads the car dataset directly from a URL (or generates a synthetic one as fallback). The dataset contains ~500 cars with 25+ columns covering dimensions, engine specs, fuel system, brand, and price.
Cell 3 — Preprocessing
Cleans and transforms the raw data into a numeric format the model can read. This includes extracting the brand name from the car name string, fixing typos in brand names, binary-encoding yes/no columns, and one-hot encoding multi-category columns like body style, drive wheel, and engine type.
Cell 4 — Split and scale
Divides the data into three sets: 72% training, 18% validation, and 10% test. Features and prices are standardized using StandardScaler fitted only on training data to prevent leakage. Everything is converted to PyTorch tensors.
Cell 5 — Model architecture
Defines CarPriceModel, a feedforward neural network with the following structure:
Input → Linear(128) → BatchNorm → ReLU → Dropout
      → Linear(64)  → BatchNorm → ReLU → Dropout
      → Linear(32)  → ReLU
      → Linear(1)   → Price output
Batch normalization stabilizes training, dropout prevents overfitting.
Cell 6 — Training function
A reusable function that builds and trains a model for a given number of epochs using the Adam optimizer, MSE loss, weight decay for regularization, and a step learning rate scheduler that halves the rate every 50 epochs.
Cell 7 — Hyperparameter search
Trains 5 different configurations of learning rate, hidden layer size, and dropout rate for 150 epochs each. Picks the one with the lowest validation RMSE as the winner.
Cell 8 — Full retraining + loss plot
Retrains the best configuration for 300 epochs and plots train vs validation loss curves to confirm the model converged without overfitting.
Cell 9 — Evaluation
Measures final performance on both the validation and test sets using three metrics, and prints a table of 10 individual predictions side by side with their actual prices.
Cell 10 — Scatter plot
Plots predicted vs actual prices on the test set. Points close to the diagonal red line indicate accurate predictions.
Cell 11 — Save and download
Saves the trained model weights to car_model_best.pth and downloads it along with both plots to your machine.
Cell 12 — Inference
Reloads the saved model from disk and runs it on 5 test samples to confirm the save/load cycle works correctly and the model is ready for real-world use.


# CIFAR-10 Image Classification with ResNet18

This notebook fine-tunes a pre-trained ResNet18 model to classify images from the CIFAR-10 dataset (10 classes: planes, cars, birds, etc.).
Here's what it does step by step:

Loads CIFAR-10 — applies augmentation (flips, rotation) to training images and normalizes both splits using ImageNet statistics.
Sets up ResNet18 — takes the ImageNet pre-trained model and swaps out its final layer to output 10 classes instead of 1000.
Trains for 10 epochs — uses Adam optimizer with a step-based learning rate decay. Each epoch runs a full training pass and evaluates on the test set, saving the best model weights.
Plots loss & accuracy curves — so you can visually inspect training progress and spot overfitting.
Evaluates performance — prints a full classification report with precision, recall, and F1 per class. Overall accuracy lands at ~85.5%.
Saves and reloads the model — exports the best weights to a .pth file and shows how to reload it for future inference.

The key idea is transfer learning: rather than training from scratch, it reuses features the model already learned from millions of ImageNet images, which is why it converges quickly and performs well with minimal training time.
