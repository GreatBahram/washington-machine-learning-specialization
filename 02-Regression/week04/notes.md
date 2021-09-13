# Ridge Regression (Regularization)

### Characteristics of overfit models

In the previous week we talked about the complex model to become overfit to the data and also discussed this idea of bias and variance trade off. Where high-complexity model could have very low bias but high variance; whereas low-complexity models have high bias but low variance. This week we talk about a way to automatically balance between bias and variance using something called ridge regression.

In the below picture, on the left side we have low order polynomial, on the right side we a high-order polynomial which suffers from high variance and low bias; in other words it is being overfit.

<img src="assets/overfit-01.png" style="zoom:50%"/>

**Symptom of overfitting**

* is there some type of quantitative measure that is indicative of when a model is overfit? Well, when a model overfits, the coefficients tend to really large in magnitude!
* This is problem is not unique to polynomial regression, it happens if we have a lot of features (very large $D$).
* If number of observation is small, then it more likely that overfitting happens. But if the number of observation is high, we are not going to quickly become overfit; because we have dense observations.

<img src="assets/overfit-02.png" style="zoom:50%"/>

* Number of inputs influence overfitting: if we only have one input we should a dense input to prevent the model from overfitting which is hard because it might not be possible to gather that much data. and it becomes harder if you have different inputs, because you are required to have examples of all possible combination to avoid overfitting.

<img src="assets/overfit-03.png" style="zoom:50%"/>

### The ridge objective

### Optimizing the ridge objective

### Tying up the loose ends



- [ ] Characteristics of overfit models
- [ ] The ridge objective
- [ ] Optimizing the ridge objective
- [ ] Tying up the loose ends
- [ ] Programming Assignment 1
- [ ] Programming Assignment 2