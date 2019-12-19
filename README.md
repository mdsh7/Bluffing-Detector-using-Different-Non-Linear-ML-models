# Bluffing-Detector-using-Different-Non-Linear-ML-models

Bluffing Detector using Different Non-Linear ML models

## Problem

HR team of a company is going to hire a new employee and is going to make an offer for him. The to be employee says that he had experience of 15+ years and was taking 160k salary in his previous company and wants more than that as he can bring value in the company. He also says that he was working as a Regional Manager. So, the HR team wants to make a model to check if that to be employee is bluffing or telling the truth about his salary.

## Dataset

One of the member of HR decided to explore and was successful in getting that to be employees previous companies salary and level of qualification's data. The dataset consists position, level and salary columns.

## Solution

From the to be employees CV, the conclusion was made that his level is 6.5.To be more accurate and know the better correlation between salary and level, different non-linear models were used (Polynomial, SVR, Decision Tree and Random Forest) and the findings are as below.

### _Polynomial Regression_

Going thru' the polynomial regression model and tuning the degree of polynomial features to 4, the following graph and salary was predicted.

![Truth or Bluff (Polynomial Regression)](https://user-images.githubusercontent.com/14214659/71173167-afd3ea00-226a-11ea-80a7-8852345abdd7.png)

To get how linear regression would have predict the salary, we compare the result between linear and polynomial regression.

![Lin vs Poly Regression](https://user-images.githubusercontent.com/14214659/71173380-40122f00-226b-11ea-8390-2a71b276a92a.png)

From this, we find that if the salary was predicted only with linear regression, it would have been a disaster but polynomial regression shows good result.

### _Support Vector Regression_

Now the same data was fitted in SVR model. The model failed without feature scaling so feature scaling was done and then prediction was made.

![Truth or Bluff (SVR)](https://user-images.githubusercontent.com/14214659/71173203-c67a4100-226a-11ea-9ded-82fc670c5d2d.png)

![svr](https://user-images.githubusercontent.com/14214659/71173415-591ae000-226b-11ea-9192-5acb315dfda0.png)

It shows that SVR is less appropriate non-linear regression model for this specific problem.

### _Decision Tree_

Next, the data was fitted in the decision tree and the prediction was made.

![Truth or Bluff (Decision Tree Regression)](https://user-images.githubusercontent.com/14214659/71173248-dbef6b00-226a-11ea-9093-3c6f9b697d1d.png)

![Decision Tree](https://user-images.githubusercontent.com/14214659/71173594-d2b2ce00-226b-11ea-8bdd-489fa3a1000e.png)

It shows that, decision tree is also not good idea for this kind of data.

### _Random Forest_

As a final try, random forest was used and the data was visualized as well as prediction was made. The number of trees (n_estimators) was used as default and keep on adding. The best result was achieved on 340 trees.

![Truth or Bluff (Random Forest Regression)](https://user-images.githubusercontent.com/14214659/71173307-03dece80-226b-11ea-91d4-757574336498.png)

![Random Forest](https://user-images.githubusercontent.com/14214659/71173518-a26b2f80-226b-11ea-8e7b-394468fe9fb4.png)

## Conclusion

From running different models, the conclusion was made that the to be employee **_was not bluffing_** about his salary. He was honest and the previous salary he was telling was accurate. Best lesson to learn from this series of model prediction is that, before making any decisions patience is needed and data must be explored properly.
