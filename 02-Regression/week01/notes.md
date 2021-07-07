# week 1

This week we are going to covert a simple regression model, we call it simple as it only utilizes one input (or one feature), then will define the goodness-of-fit to evaluate a model.

To optimize our prediction we will use *Gradient Descent* algorithm. This algorithm helps use to find best value of parameters which in this case of slope and intercept, then we talk about how we can interpret these parameters.

## Regression Fundamentals

The main example/problem in this course is house pricing prediction. The one input is square feet and the value that we want to predict is the value of the house, i.e. we are looking for a relationship between input and output: $Y_i=F(X_i)$. But we should assume there is some error: $Y_i = F(X_i) + E_i$.

The expected value of $E_i$ is zero, expected value itself means all possible values that error can take weighted by how likely is to take any of them. But what it says here it is equally likely that we are going have positive or negative sales price, $Y_i$ is equally likely to be above or below $F(X_i)$.



Which model our function should use? should it be:

* Constant relationship (regardless of sq.ft we have a constant price?)
* Linear model? as we increase sq.ft we expect a higher price?
* Quadratic fit? or higher polynomial model? 

When we choose a model, we must estimate a fit from data! But there are a tons of estimates that we can guess.

<img src="assets/regression-ml-block-diagram.png" style="zoom:40%"/>

## Simple Linear Regression (its use and interpretation)

As we know a equation of line is: $f(x)  = w_0 + w_1 x$ and we know this too: $y_i = w_0 + w_1 x_i + \epsilon_i$, the parameters in this formula ($w_0,w_1$) or intercept and *slope* are *regression coefficients*. For brevity we show these parameters as $\hat{W}$.



What is the cost of using a specified line? The one we talk about is Residual sum of squares (RSS), a residual is the difference between an estimated and the actual value:

<img src="assets/rss.png" style="zoom:30%"/>

or $RSS(w_0, w_1) = \Sigma^{N}_{i=1}(y_i - [w_0 + w_1 x_i])^2$

Now, imagine we have multiple lines and we want to find **the best** line? Out of all lines, I want to choose a line the has the minimum RSS.

Interpreting the coefficients

* we set slope ($w_1$) to zero then we'd have only intercept which shows the predicted price of a house with zero sq.ft, as you can guess, this is not normally that much meaningful.
* the estimated slope ($w_1$) for one sq.ft what is the predicted change in price? it shows the price of one square feet.

<img src="assets/interpretation.png" style="zoom:25%"/>