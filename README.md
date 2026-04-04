Car Price Prediction — Overall Project Overview

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
