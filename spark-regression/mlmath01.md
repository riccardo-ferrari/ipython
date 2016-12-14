
# Machine Learning Math 1


```python
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.style.use('seaborn-colorblind')
```

## Why Machine Learning Math

We are currently marketing DXP as a digital transformation platform that enables omni-channel experiences and enriches customer relationship insight.

https://www.liferay.com/digital-experience-platform

> Digital Transformation **Platform**
>
> * **Easy Integration**: Give customers secure access to information and apps hidden deep in your systems.
> * **Modular Services**: Quickly implement and evolve digital strategy with reusable and resilient services.
> * **Runtime Platform**: Support any digital business model with our resilient support for cloud and on-premise infrastructures.
>
> Omni Channel **Experiences**
>
> * **Modern Web**: Build web experiences with fast, usable interfaces customers will love.
> * **Mobile**: Create native apps and mobile sites with push notifications and camera and GPS.
> * **Digital Hybrid**: Enhance in-store experiences with digital context and services.
>
> Customer Relationship **Insight**
>
> * **Personalization and Segments**: Refine your understanding of your audience's needs and interests with targeted information and segment identification.
> * **Analytics and Metrics**: Help users get what they need. Use data to evolve touch points and increase click-through and conversion.
> * **Single View of the Customer**: Get a full understanding of what customers are doing across all touch points.

To achieve the last (customer relationship insight), you need to understand what an insightful question might be and how to provide the answers in a way that they generate insight. Rephrased slightly, when building a DXP, its developers and maintainers should be able to answer the following three questions:

1. What types of questions will our users will try to ask with a DXP?
2. How will users attempt to use DXP in order to answer those questions?
3. How will each of these different answers translate to insight?

This series of workshops will focus on the second question, which itself centers around machine learning.

You've likely heard about machine learning as a concept, and solutions rooted in machine learning dominate the subtopics listed in [Gartner's 2016 Hype Cycle](http://www.gartner.com/newsroom/id/3412017).

We won't pretend that this series of workshops can provide a better crash course on performing machine learning than the existing Coursera, Udacity, or EdX courses on the topic. Instead, we'll look at machine learning at a more basic level, focusing on helping you better understand the questions that machine learning attempts to ask and the form that machine learning answers take.

Many machine learning courses begin with linear regression, because the end result is very easy to visualize in two dimensions. An added benefit is that due to its simplicity, it forces you to think carefully about the form of the data. You end up learning that choosing different data formats influences both the reliability of your machine learning model and even changes what your machine learning model actually predicts.

Given that 80% of data science involves getting your data into the right format (*data cleaning*), we too will start with linear regression. The focus in this first session will be on recognizing different types of variables and how these different variable types relate to *interpreting* a linear regression.

## Machine Learning Intuition

### Plain English

In a lot of computer science, we understand algorithms by assuming an oracle machine where you feed it with inputs and it produces the correct output every time. Essentially, black magic hiding inside of black boxes.

* [Oracle Machine](https://en.wikipedia.org/wiki/Oracle_machine)

Can we create something that isn't a black box that will produce the correct output every time? We can't be sure. Maybe no oracle machine can be built. Maybe building such an oracle machine is extremely complex and time-consuming.

What if we could step back and ask the question, "What if an almost-oracle is good enough?"

At a very abstract level, if we have many data points, we can theoretically summarize the data points with a model that says something meaningful or illuminating about the data. While it isn't necessarily 100% correct, this model can be used to understand both the data points that we've seen as well as data points we've not yet seen.

* [Obligatory George Box quote](https://en.wikipedia.org/wiki/All_models_are_wrong)

The process of creating these almost-oracles based on data points is known as machine learning.

In machine learning, you choose some class of models you have reason to believe will provide you with almost correct outputs given what you understand about the inputs. You feed your model with the input data, you provide some hints, and then you ask the model to train itself. By observing the outputs it creates, the model adjusts itself to provide increasingly correct answers as outputs.

To give you a better sense of what this means in practice, we'll talk about the model you created in the machine learning workshop: linear regression.

### Math English

Let's start by pretending that you want to describe an object, say a house. To create this description, you will want to capture many aspects of the house. You might record values like the size of the house, the size of the backyard, the number of bedrooms and bathrooms in the house. Each of these values that you choose to collect is referred to as a *feature*.

Assume you have $m$ different objects that you've described with $n$ features. You can label the specific feature of a specific object as $x_{i,j}$, where $i$ corresponds to the ID of the object, and $j$ corresponds to the index of the feature. This allows us to represent these $m$ data points as an $m \times n$ matrix.

$$
\begin{bmatrix}
x_{1,1} & \cdots & x_{1,n} \\
x_{2,1} & \cdots & x_{2,n} \\
\vdots & \ddots & \vdots \\
x_{m,1} & \cdots & x_{m,n}
\end{bmatrix}
$$

Let's assume that there is an additional feature that is intuitively related to the features we collected. In our case, it might be the price of the house in question. Because we have $m$ different values for this additional feature, we can represent these as an $m \times 1$ column vector.

$$
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_m
\end{bmatrix}
$$

So to repeat what we just said about the almost-oracles in a more mathematical way, machine learning is the process of taking the input matrix, performing various [matrix operations](https://en.wikipedia.org/wiki/Matrix_operation), and ultimately ending up with the output matrix.

## Define the Task

Now that you know what machine learning is, the next step is to recognize that machine learning is a tool that serves to answer important questions. Therefore, the first step in machine learning is to decide what question you wish to answer.

To help you understand what these questions might look like in the real world, let's do some roleplay.

Something that happens occasionally in dramas is there will be an evil corporation that is looking to buy out neighborhoods in order to construct resorts or hotels that will pull in tourists and make lots of money for the corporation.

* [Princess Jellyfish](https://myanimelist.net/anime/8129/Kuragehime)

Imagine that you are an executive at one such evil hotel corporation, Hotelray.

You'd like to find places for your hotels that will make you a lot of money (surrounded by high value homes or tourist attractions) but wouldn't cost you very much money in terms of construction (low cost of labor, relatively inexpensive to buy out).

## Operationalize the Task

The next step is to identify what it is that you wish to predict in order to answer your question and identify data that you can bring together that might help in this prediction. This is known as *operationalizing the question*.

* [Operationalization](https://en.wikipedia.org/wiki/Operationalization)

In our case, we can answer a substantial amount of Hotelray's business question by simply finding out what the prices of the houses are. Unfortunately, not every house will publish their market value, and sending out an appraiser to every house of every possible neighborhood would be extremely expensive and most likely unwelcomed by the current residents.

However, an "almost-oracle" that had reasonable guesses about the price of every home would be good enough.

We're in luck! It turns out that we have data from the [House Sales in King County, USA](https://www.kaggle.com/harlfoxem/housesalesprediction), which is available from Kaggle data sets. If we can create a model that had reasonable guesses about the price of every home in that data set, it might also have reasonable guesses about every house in every neighborhood we're interested in.

I've taken advantage of the [CC0: Public Domain License](https://creativecommons.org/publicdomain/zero/1.0/) which permits copying the data set for any purpose, and I've pushed it to the static hosting service of [WeDeploy](https://www.wedeploy.com/) so it's accessible without sign-in.


```python
df = pd.read_csv('http://hosting.mlmath.wedeploy.io/kc_house_data.csv')
```

To summarize, our goal now is to create a model that will predict the prices of houses. From the machine learning perspective, we'd like to simply present the data to the model, and the model will figure out the appropriate way to guess at the price of our houses.

These are the features we have available:


```python
df.columns
```




    Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
           'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
           'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
           'lat', 'long', 'sqft_living15', 'sqft_lot15'],
          dtype='object')



## Define Incorrectness

So, let's take a quick step back. You will be asking the model to train itself, and it will provide increasingly correct answers as outputs. How does the model know what to do in order to know how to get increasing correctness?

Essentially, it knows because you have a definition of "degree of incorrectness". In machine learning, this is referred to as a *cost function*.

* [Machine Learning Yearning, Chapters 8-9](https://gallery.mailchimp.com/dc3a7ef4d750c0abfc19202a3/files/Machine_Learning_Yearning_V0.5_01.pdf)

Intuitively, the cost function provides a way for the model to detect whether it got better or if it got worse.

There are a variety of considerations that you take into account when saying one model is better than another, but one of the more obvious elements of the cost function is the difference between what the model guessed $\hat{y_i}$ as compared to the true correct answer $y_i$.

The notion of a guess can vary between machine learning models. Some guess at a number in the range $(-\infty, +\infty)$. Some guess in a more bounded range, such as $[0, 1]$ or $[-1, 1]$. Still others might have multiple outputs. In all cases, you define your sense of right and wrong for any given example, and then you apply a function (sum and average are common) and declare that to be the degree of incorrectness of the model.

The simplest definition of right and wrong is binary. In this world, every incorrect answer is weighted equally. Either you got it exactly right or you got it wrong. As a cost function, this is known as **zero-one loss**.

$$
L(y_i, \hat{y_i}) =
\begin{cases}
0 & y = \hat{y_i} \\
1 & y \ne \hat{y_i}
\end{cases}
$$

Another simple definition is to take the absolute value of the difference between $y_i$ and $\hat{y_i}$. In this world, every incorrect answer is weighted according to how far away it was from the true value. As a cost function, this is known as **absolute loss**.

$$
L(y_i, \hat{y_i}) = | y_i - \hat{y_i} |
$$

One of the more common loss functions is to take the square of the difference between $y_i$ and $\hat{y_i}$. In this world, the farther you are away from the true value, the more dramatic the penalty for that incorrectness. As a cost function, this is known as **squared loss**.

$$
L(y_i, \hat{y_i}) = ( y_i - \hat{y_i} )^2
$$

### Checkpoint: Define Incorrectness

In our case, we are predicting the prices of houses. We don't expect to have this guess be perfect, but we'd like to penalize guesses that are way off.

Because we wish to penalize guesses differently based on how far off they are, we rule out zero-one loss. Because we would like our loss function to penalize really wrong guesses more heavily, **squared loss** makes more sense for us than absolute loss.

After summing the squared losses, we'll wind up with the sum of squared errors, or SSE.

$$
\text{SSE} = \sum\limits_{i=1}^m(y_i - \hat{y_i})^2
$$

However, this number is huge, and we aren't able to relate it back to what it is that we're predicting, so we don't have a good sense of just how wrong we are, just that smaller numbers are preferred. How can we fix that?

The first transformation is to compute the average of these squared differences rather than just their sum. This will give us the average of the squared errors, which reins in the values so that even if you add more data points, the value does not increase if the errors are about the same across the board. This transformation is known as the mean-square error, or MSE.

$$
\text{MSE} = \frac{\text{SSE}}{m} = \frac{1}{m} \sum\limits_{i=1}^m(y_i - \hat{y_i})^2
$$

If we have the average squared error, then if we take the square root, we have something that we can relate to our output value in a more obvious way. This transformation is known as root-mean-square error, or RMSE.

$$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{m} \sum\limits_{i=1}^m(y_i - \hat{y_i})^2}
$$

This is the number we will use to define the incorrectness of our model.

## Baseline Model

Once you've defined a cost function, all machine learning follows with the following question. What is the simplest almost-oracle I can think of related to this degree of incorrectness?

* [How to Get Baseline Results and Why They Matter](http://machinelearningmastery.com/how-to-get-baseline-results-and-why-they-matter/)

Answering this question gives you a meaningful starting point and it allows you to say whether your self-adjusting almost-oracle is actually any better than the simplest almost-oracle.

The simplest almost-oracle in many cases is simply always guessing the exact same value, no matter what the input values are. For that reason, you will often choose an almost-oracle that is equivalent to the central tendency measure connected to your definition of incorrectness.

* [Modes, Medians, Means: A Unifying Perspective](http://www.johnmyleswhite.com/notebook/2013/03/22/modes-medians-and-means-an-unifying-perspective/)

To summarize the article above:

* For zero-one loss, you would choose the mode of the output values as the baseline model.
* For absolute loss, you would choose the median of the output values.
* For squared loss, which encompasses the [sum of squared errors](https://en.wikipedia.org/wiki/Residual_sum_of_squares), the [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error), or the [root-mean-square error](https://en.wikipedia.org/wiki/Root-mean-square_deviation), you would choose the mean of the output values.

### Checkpoint: Create the Baseline Model

First, let's create our baseline model by computing the mean of all prices.


```python
baselineModel = df['price'].mean()

baselineModel
```




    540088.14176652941



To put what this baseline model means in plain English, we have now created a model that will simply always guess that the price of the house is \$540,088 USD, no matter what.

Now that we have our baseline model, we'll want to compute the incorrectness of this baseline model. Since we chose RMSE as our degree of incorrectness measure, all we have to do is compute it. The first step is to find out how off we are, which we can compute simply by subtracting the true price $y_i$ and our model's guess $\hat{y_i}$.


```python
baselineModelPredictions = np.repeat([baselineModel], len(df))
baselineModelError = df['price'] - baselineModelPredictions
```

If we square each of these errors and compute the mean of the squared values, we will have the mean squared error, MSE. If we compute the square root of that, then we have the root mean squared error, RMSE.


```python
baselineModelSquaredError = baselineModelError ** 2
baselineModelRMSE = np.sqrt(baselineModelSquaredError.mean())
baselineModelRMSE
```




    367118.70318137232



As a side note, in machine learning, it's important that you do not create a model based on the whole data set and then re-evaluate based on the same data set you used to create the model, which is what we did just now in creating a baseline model.

We're going to ignore that formality until we can explain the math behind why it's important, which won't be for a few more lessons.

## Linear Model Overview

When you choose linear regression as a model, you essentially say that the output variable is a linear combination of the input variables. More explicitly, we have a vector of *weights* that contains $n$ entries (for each of the features) and an additional intercept term $\beta_0$:

$$
\begin{bmatrix}
\beta_0 \\
\vdots \\
\beta_n
\end{bmatrix}
$$

If we were to take the numerical value we gave to each feature $x_{i,j}$ and multiply it by the corresponding $\beta_j$ and then sum these values along with the intercept term $\beta_0$, we have an estimate of the result variable $\hat{y_i}$.

$$
\hat{y_i} =
\begin{bmatrix}
1 & x_{i,1} & \cdots & x_{i,n}
\end{bmatrix}
\times
\begin{bmatrix}
\beta_0 \\
\vdots \\
\beta_n
\end{bmatrix}
$$

Choosing a linear regression model is equivalent to saying that we can estimate the output vector by transforming the input matrix with a single matrix multiplication.

$$
\begin{bmatrix}
\hat{y_1} \\
\hat{y_2} \\
\vdots \\
\hat{y_m}
\end{bmatrix}
=
\begin{bmatrix}
1 & x_{1,1} & \cdots & x_{1,n} \\
1 & x_{2,1} & \cdots & x_{2,n} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{m,1} & \cdots & x_{m,n}
\end{bmatrix}
\times
\begin{bmatrix}
\beta_0 \\
\vdots \\
\beta_n
\end{bmatrix}
$$

## Linear Model Assumptions

Some of the basic assumptions you make with a linear model are summarized in the following discussion.

* [Linear Models](http://www.stat.berkeley.edu/~aditya/resources/LectureFOUR.pdf)

We'll go into the more advanced understanding of these assumptions in our next lesson, but for now, we will only rely on the simplest assumption, which is that we assume that the input variables $x_{i,j}$ are non-random and that the result variables $y_i$ are random.

If the idea of a "random variable" as a statistical concept is unfamiliar to you, it's good to do a quick refresher of drawing samples from a probability distribution.

* [Chapter 1 of Introduction to Stochastic Processes](https://www.ma.utexas.edu/users/gordanz/notes/introduction_to_stochastic_processes.pdf)

Essentially, we assume that our measurements of each of the $x_{i,j}$ variables are precise values and are not random, but $y_i$ will vary from the model estimate $\hat{y_i}$ by an error term $\epsilon_i$.

In an ideal world where all of the linear model assumptions are satisfied, you can say something special about $\epsilon_i$, which is that while we don't actually expect our guesses to be exactly right, we will be wrong in a way that looks like a zero-mean random variable.

* [White Noise](https://en.wikipedia.org/wiki/White_noise#Mathematical_definitions)

We aren't going to assume ideal world today, so what we will instead say for now is that if we know that our $x_{i,j}$ values are non-random and our $y_i$ values are random, we can say that the true $y_i$ values relate to our model's guesses $\hat{y_i}$ values in the following way, though we know very little about the nature of $e_i$.

$$
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_m
\end{bmatrix}
=
\begin{bmatrix}
\hat{y_1} \\
\hat{y_2} \\
\vdots \\
\hat{y_m}
\end{bmatrix}
+
\begin{bmatrix}
\epsilon_1 \\
\epsilon_2 \\
\vdots \\
\epsilon_m
\end{bmatrix}
$$

## Indicator/Boolean Variables

Our journey begins with the idea of an indicator variable, which is a variable whose only values are 0 and 1. If you're coming from a programming background, you can think of these as `false` and `true`, respectively.

When you include a boolean or indicator variable $x_{i,j}$ in a linear model, what you are effectively saying is that, all else being equal, we expect that activating the indicator value (setting it to 1 instead of 0) will result in a change in $y_i$ equal to the coefficient $\beta_j$.

### Checkpoint: Indicator/Boolean Variables

In our data set, the only boolean indicator variable is `waterfront`, which represents whether the house sits adjacent to a body of water.

Before you include a boolean indicator variable, you'll want to know if it there is actually a difference between the result value when the variable is 0 vs. the result value when the variable is 1.

One way to do that is to look at the quantiles (also known as percentiles) after we've divided the data into waterfront properties and non-waterfront properties.

* [Percentile](http://www.r-tutor.com/elementary-statistics/numerical-measures/percentile)


```python
quantiles = np.linspace(0, 0.95, 20)

for key, grp in df.groupby(['waterfront']):
    grp['price'].quantile(quantiles).plot(label='waterfront = %d' % key)

plt.legend(loc='best')
plt.show()
```


![png](mlmath01_files/mlmath01_38_0.png)


From the raw quantile information, we can see that the data for non-waterfront properties shows a median of \$450,000 (the 0.5 or 50% quantile), while the prices for waterfront properties have a median of \$1,400,000 USD. The differences at other quantiles is also dramatic.

Another visualization which might help us understand the distribution of the prices between waterfront properties and non-waterfront properties is to instead create a histogram of all the price values, with each value of the `waterfront` treated as a facet value.


```python
for key, grp in df.groupby(['waterfront']):
    grp['price'].plot(kind='kde', label='waterfront = %d' % key)

plt.legend(loc='best')
plt.show()
```


![png](mlmath01_files/mlmath01_40_0.png)


From the visualization, we see that the concentration at the lower values is more pronounced for non-waterfront properties than for waterfront properties, which also gives us a sense that the two types of properties have fundamental price differences.

Combining these two visualizations of how the price relates to whether or not a house is a waterfront property gives us confidence that there is some amount of $y_i$ that is explained by our $x_{i,j}$, and our model is likely to improve if we include the variable.

## Linear Regression, Take 1

When adding the `waterfront` variable to our regression, we effectively have the following regression, where $\beta_1$ is the change in $y_i$ we expect from activating the waterfront indicator variable, all else being equal, and $x_{i,1}$ is the value of our waterfront indicator variable.

$$
y_i = \widehat{\beta_0} + \widehat{\beta_1} x_{i,1} + \widehat{\epsilon_i}
$$


```python
take1Model = LinearRegression(fit_intercept=True)
take1Model.fit(X=df[['waterfront']], y=df['price'])

take1Model.intercept_, *take1Model.coef_
```




    (531563.59981351998, 1130312.4247263379)



Up until now, we haven't mentioned $\beta_0$ at all. So you might be wondering, aside from resembling an intercept term in a standard linear equation of $y = mx + b$, what does $\beta_0$ mean in English?

Think back to the matrix multiplication used to define $\hat{y_i}$.

$$
\hat{y_i} =
\begin{bmatrix}
1 & x_{i,1} & \cdots & x_{i,n}
\end{bmatrix}
\times
\begin{bmatrix}
\beta_0 \\
\vdots \\
\beta_n
\end{bmatrix}
$$

It follows from the matrix multiplication that $\beta_0$ is the value we will predict for $\hat{y_i}$ when all of the $x_{i,j}$ values are 0. In this specific example, it is the value we will predict as the price of a house when it is not a waterfront property.

Now using one housing feature (waterfront or not waterfront), we have expanded our regression model to produce variable results.

This is a slight improvement to the previous baseline model that will always guess the same value (the output is always the mean,\$540,088 USD).
It will now output two values when guessing housing price, based on whether the house is on the waterfront of not.

Guess for a non-waterfront house: 

$$
y_i = \widehat{\beta_0} + \widehat{\epsilon_i}
$$

Guess for a waterfront house: 
$$
y_i = \widehat{\beta_0} + \widehat{\beta_1} x_{i,1} + \widehat{\epsilon_i}
$$

So what does this model actually say? It says that if a house is not a waterfront property, the model predicts that it will be \$531,563 USD, and if it is a waterfront property, the model predicts that it will be worth an additional \$1,130,312.

However, our regression is not that helpful because there many other features that could add or detract from a house's value.
This leads into the assessment of the model. 

Next, we evaluate the model's performance compared to the baseline model.


```python
take1ModelPredictions = take1Model.predict(X=df[['waterfront']])
take1ModelError = df['price'] - take1ModelPredictions

take1ModelSquaredError = take1ModelError ** 2
take1ModelRMSE = np.sqrt(take1ModelSquaredError.mean())

baselineModelRMSE, take1ModelRMSE
```




    (367118.70318137232, 353855.07535782689)



The RMSE is lowered very slightly. Since we're using RMSE as our measure of whether the model is improving, by our definition, our single-variable linear model using a single indicator variable provides a very slight improvement over our baseline model.

## Meaningful Numeric Variables

Our journey continues with the idea of numeric variables, which is a variable that has a much wider range than just 0 and 1 (it might even be all floating point numbers between 0 and 1). However, just because you have lots of values doesn't mean that all of those values are meaningful as numbers in the context of regression.

If a value $x_{i,j}$ is meaningful as a number in the context of regression, that means that when we include the numeric value $x_{i,j}$ in a linear model, all else being equal, we expect that an increase $\delta_k$ in $x_{i,j}$ will increase the value of $y_i$ by $\delta_k \beta_j$ for all values of $\delta_k$.

More explicitly, if we choose some $\delta_k$ to use as our increment, every time we increase $x_{i,j}$ by that increment $\delta_k$, we expect $y_i$ to change by the same amount $\delta_k \beta_j$ with each increment, subject to white noise. This constant increase for each unit increase is what makes linear regression "linear".

### Checkpoint: Meaningful Numeric Variables


```python

```

## Less Meaningful Numeric Variables



### Checkpoint: Less Meaningful Numeric Variables


```python

```

## Linear Regression, Take 2

Let's go ahead and expand our basic linear regression model to include $n$ variables (other variables could be features such as bedrooms, bathrooms, sqft_living, sqft_lot, floors, condition, yr_built, yr_renovated, zipcode, etc).

$$
y_i = \widehat{\beta_0} + \widehat{\beta_1} x_1 + \dotsb + \widehat{\beta_n} x_n + \widehat{\epsilon_i}
$$
Starting off with some factors that we instictively know add and dectract value, let's first consider the square footage, bedrooms, and bathrooms.

## Really Wrong Answers

After you've got your first model that isn't a baseline model and you've determined that you've improved, the next step is to decide whether you should keep going or if you can stop (if you're familiar with blackjack, it's essentially like deciding whether you should hit or stand).

* [Machine Learning Yearning, Chapter 13](https://gallery.mailchimp.com/dc3a7ef4d750c0abfc19202a3/files/Machine_Learning_Yearning_V0.5_02.pdf)
* [Machine Learning Yearning, Chapter 14](https://gallery.mailchimp.com/dc3a7ef4d750c0abfc19202a3/files/Machine_Learning_Yearning_V0.5_03.pdf)

In this process, you look at samples where you scored poorly on your cost function and ask, "How can I update my model so that it doesn't do poorly on these examples?"

Perhaps there are features that you might simply be missing in your model. Perhaps you should adjust your cost function or more aggressively sample rare but important data points from your data set. Perhaps you simply need more examples with specific characteristics.

Put more explicitly, after each incremental improvement of your machine learning model, it's a good idea to evaluate your model in a way that's related to your cost function and determine if additional model refinement is achievable with the data you already have, and how much effort is involved in acquiring new data if it's not achievable.

### Checkpoint: Really Wrong Answers


```python

```

## Transformed Variables



### Checkpoint: Transformed Variables


```python

```

## Interaction Terms



### Checkpoint: Interaction Terms


```python

```

## Linear Regression, Take 3

$$
y_i = \widehat{\beta_0} + \widehat{\beta_1} x_1 + \dotsb + \widehat{\beta_n} x_n + \widehat{\epsilon_i}
$$


```python

```

## Text Vectorization



### Checkpoint: Text Vectorization


```python

```

## Linear Regression, Take 4

$$
y_i = \widehat{\beta_0} + \widehat{\beta_1} x_1 + \dotsb + \widehat{\beta_n} x_n + \widehat{\epsilon_i}
$$


```python

```

## Closing Thoughts

Hopefully you have now become curious about linear regression.

You might wonder about simple extensions to linear regression, such as linear spline regression where you have boundary points where the coefficients completely change.

* [An Introduction to Splines](http://www.statpower.net/Content/313/Lecture%20Notes/Splines.pdf)

You might also wonder why we talked about cost functions at the start and if choosing a different cost function might change the way the regression works.

* [Quantile Regression: An Introduction](http://www.econ.uiuc.edu/~roger/research/intro/rq3.pdf)

It's likely that you've also been wondering about applying transformations of the input and output variables in order to overcome the constraints of the linear relationship between variables.

* [Transformations in Regression](http://people.stern.nyu.edu/jsimonof/classes/2301/pdf/transfrm.pdf)

You might be able to follow examples of people looking at input types that we haven't talked about (such as geospatial data) that will allow you to apply linear regression to other problems that involve the prediction of a continuous variable with range $(-\infty, \infty)$.

* [AirBnb Properties in Boston](https://github.com/ResidentMario/boston-airbnb-geo/blob/master/notebooks/boston-airbnb-geo.ipynb)

However, before we let you get into any of that, the next important thing we want you to understand is the math behind the cost function(s) that might be applied when performing linear regression. That will be our next lesson.

* [Usual assumptions for linear regression](http://stats.stackexchange.com/questions/16381/what-is-a-complete-list-of-the-usual-assumptions-for-linear-regression)
* Convexity of cost functions
* Stochastic gradient descent
* Different interpretations of distance (euclidean, etc.)
* Regularization

## Additional Resources




```python

```
