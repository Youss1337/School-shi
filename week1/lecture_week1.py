"""
Fintech P1 - Week 1: Linear Models & The Risk of Overfitting
A (Finance) Tale of Linear Regression, Ridge, and Lasso

Spyder version of the lecture notebook (converted from the .ipynb).

HOW TO USE THIS FILE IN SPYDER
    Every "# %%" line starts a new cell. Put the cursor inside a cell and press
    Ctrl+Enter to run just that cell, or Shift+Enter to run it and jump to the
    next one. Run the cells top to bottom the first time: later cells depend on
    variables created by earlier ones.
"""

# %% [markdown]
# # Fintech P1: Week 1: Linear Models & The Risk of Overfitting
# # A (Finance) Tale of Linear Regression, Ridge, and Lasso

# %% [markdown]
# This notebook demonstrates the core concepts of Week 1. We will not just learn how to implement linear models, but why they are used—and feared—in finance and economics.
#
# We will cover:
# *   The Problem: Simulating a noisy economic cycle.
# *   The "Too Simple" Model: Why a basic Linear Regression fails.
# *   The "Perfect Backtest" Trap: How overfitting creates a high-risk, dangerous model.
# *   The Stability Tool: Using Ridge Regression to manage risk and create a robust model.
# *   The Insight Tool: Using Lasso Regression to gain insight and build an interpretable model.

# %% [markdown]
# ## Part 1: The "Market Cycle" Dataset
# Let's imagine the data we are about to create isn't just a mathematical function. It's a seasonal economic indicator or a market cycle (e.g., quarterly earnings, commodity prices, etc.).
#
# The blue dots are the real, noisy, daily data points.
#
# **Our Goal**: Build a model that can understand this cycle. A good model could be used for a profitable trading strategy or for making accurate economic forecasts.
#
# ### Library Import

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 160)

# %% [markdown]
# ### Function to simulate

# %% [markdown]
# Create a function of X to simulate a dataset:
# $$f(x)=\cos(\frac{3}{2}\pi x)$$

# %%
def true_fun(X):
    return np.cos(1.5 * np.pi * X)


np.random.seed(100)
n_samples = 50

# %% [markdown]
# ### Simulation

# %% [markdown]
# We need three separate things, and it is worth being precise about them because students
# (and practitioners) routinely collapse them into one:
#
# *   a **training sample** - the noisy points the model is allowed to learn from;
# *   a **test sample** - *different* points, drawn from the same process and carrying their own
#     noise, which the model never sees during fitting;
# *   a **smooth grid** of the true function, used only to draw the underlying cycle on charts.
#
# The third one is a luxury of simulated data: in real life the true function does not exist in
# a column of your dataframe. If we scored our models against it we would be marking our own
# homework against the answer sheet, and the test error would look far better than anything
# achievable on real data.

# %%
# 1. The TRAINING sample - what we actually observe, noise included
X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.2

# 2. A genuine HELD-OUT TEST sample - new draws from the same process,
#    with their own noise. This is what "unseen data" means.
rng_test = np.random.RandomState(7)
X_test = np.sort(rng_test.rand(200))
y_test = true_fun(X_test) + rng_test.randn(200) * 0.2

# 3. A smooth grid used ONLY for drawing the true curve on the charts.
#    It is not data and we never score against it.
X_curve = np.linspace(0, 1, 100)
y_curve = true_fun(X_curve)

print(f"train: {len(X)} noisy points")
print(f"test : {len(X_test)} noisy points, held out")
print(f"noise sd = 0.2  ->  no model can do better than RMSE = 0.20 on the test set")

# %%
plt.figure(figsize=[8, 5])
plt.scatter(X, y, label='Training data')
plt.xlabel('x'); plt.ylabel('y')
plt.title('The "market cycle": 50 noisy observations')
plt.legend()
plt.show()

# %% [markdown]
# ## Part 2: The "Too Simple" Model (Standard Linear Regression)
#
# Let's start with our most basic tool: a standard Linear Regression model. We will try to fit a straight line to our "market cycle" data.
#
# ### Library Import

# %%
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# We keep results in dictionaries keyed by model name rather than in lists.
# Re-running a cell then OVERWRITES that model's row instead of silently
# appending a duplicate - which matters, because we will re-run these cells.
scores = {}
coef_rows = {}


def score(name, train_pred, test_pred):
    """Score a model on the training sample and on the held-out test sample."""
    scores[name] = {
        'Model': name,
        'R-squared train': r2_score(y, train_pred),
        'R-squared test': r2_score(y_test, test_pred),
        'RMSE train': sqrt(mean_squared_error(y, train_pred)),
        'RMSE test': sqrt(mean_squared_error(y_test, test_pred)),
        'MAE train': mean_absolute_error(y, train_pred),
        'MAE test': mean_absolute_error(y_test, test_pred),
    }
    return pd.DataFrame(scores.values())


def record_coefs(name, estimator):
    """Store the fitted coefficients, whether estimator is a model or a pipeline."""
    model = estimator.steps[-1][1] if hasattr(estimator, 'steps') else estimator
    row = {'Model': name, 'Intercept': model.intercept_}
    for j, c in enumerate(model.coef_, start=1):
        row[f'X^{j}'] = c
    coef_rows[name] = row

# %% [markdown]
# ### Fitting the Linear Regression

# %%
linear = LinearRegression()
linear.fit(X.reshape(-1, 1), y)

linear_train_prediction = linear.predict(X.reshape(-1, 1))
linear_test_prediction = linear.predict(X_test.reshape(-1, 1))
linear_curve = linear.predict(X_curve.reshape(-1, 1))

# %%
plt.figure(figsize=[8, 5])
plt.scatter(X, y, label='Training data')
plt.plot(X_curve, y_curve, label='True function')
plt.plot(X_curve, linear_curve, label='Linear')
plt.legend()
plt.show()

# %% [markdown]
# ### Calculating performance metrics

# %%
print(score('Linear', linear_train_prediction, linear_test_prediction))

# %% [markdown]
# **The Verdict: A Failed Model**
#
# As we can see, the straight line completely misses the point. It fails to capture the seasonal "cycle" in our data.
#
# In financial terms, this model is too simple (it has high bias). It sees limited pattern (only the downward trend but not the reverse after) and would be completely useless for trading or forecasting.

# %% [markdown]
# ## Part 3: The "Perfect Backtest" Trap (Overfitting)
# "Okay," a junior analyst says, "if a simple line is bad, let's use a complex line!"
#
# They use a high-degree polynomial (e.g., 15 or 30 degrees) to create new, complex features. This allows the model to "wiggle" as much as it needs to fit the data perfectly.
#
# Let's see what happens.

# %% [markdown]
# ### Library Import

# %%
from sklearn.preprocessing import PolynomialFeatures

# %% [markdown]
# ### Polynomial Transformation

# %%
# include_bias=False: PolynomialFeatures would otherwise add a constant column,
# and LinearRegression already fits its own intercept. Keeping both gives you a
# useless "X^0" column whose coefficient is always 0.
poly_transform = PolynomialFeatures(degree=15, include_bias=False)

# fit_transform on the TRAINING inputs only, then transform everything else
X_poly = poly_transform.fit_transform(X.reshape(-1, 1))
X_poly_test = poly_transform.transform(X_test.reshape(-1, 1))
X_poly_curve = poly_transform.transform(X_curve.reshape(-1, 1))

# %%
linear_poly = LinearRegression()
linear_poly.fit(X_poly, y)

linear_poly_train_prediction = linear_poly.predict(X_poly)
linear_poly_test_prediction = linear_poly.predict(X_poly_test)
linear_poly_curve = linear_poly.predict(X_poly_curve)

print(f"largest coefficient in this model: {np.abs(linear_poly.coef_).max():,.0f}")

# %%
plt.figure(figsize=[8, 5])
plt.scatter(X, y, label='Training data')
plt.plot(X_curve, y_curve, label='True function')
plt.plot(X_curve, linear_poly_curve, label='Linear Poly (degree 15)')
plt.ylim(-2, 2)
plt.legend()
plt.show()

# %% [markdown]
# ### Calculating performance metrics

# %%
print(score('Linear Poly', linear_poly_train_prediction, linear_poly_test_prediction))

# %% [markdown]
# ### Plotting based on different degrees

# %%
degrees = [1, 5, 10, 15, 30]

plt.figure(figsize=(16, 4))

for i, degree in enumerate(degrees):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    # Transform - fitted on the training inputs only
    pf = PolynomialFeatures(degree=degree, include_bias=False)
    Xd = pf.fit_transform(X.reshape(-1, 1))
    Xd_test = pf.transform(X_test.reshape(-1, 1))
    Xd_curve = pf.transform(X_curve.reshape(-1, 1))

    # Fit
    model = LinearRegression()
    model.fit(Xd, y)

    # Score on the training sample and on the held-out test sample
    name = f'Poly degree {degree}'
    score(name, model.predict(Xd), model.predict(Xd_test))
    record_coefs(name, model)

    # Plot
    plt.scatter(X, y)
    plt.plot(X_curve, y_curve, label='True')
    plt.plot(X_curve, model.predict(Xd_curve), label=f'Degree {degree}')
    plt.ylim(-2, 2)
    plt.legend(fontsize=8)
    plt.title(f'Degree {degree}\nmax|coef| = {np.abs(model.coef_).max():,.0f}', fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# **The Verdict: A DANGEROUS Model**
#
# Now look at the table below, and read the two R-squared columns *against each other*.
#
# Up to degree 10, adding complexity helps on both. From degree 15 onwards the two columns
# start moving in **opposite directions**: the training score keeps creeping up while the test
# score falls. That divergence is the signature of overfitting, and it is the only reliable way
# to detect it - you cannot see it from the training score alone.
#
# Degree 30 is the one to dwell on. Its largest coefficient is over 2.5 x 10^11, and its test
# R-squared is **-52.6**. A negative R-squared means the model is worse than predicting
# the average of the data every single time.
#
# This is the "perfect backtest" trap. The model has not learned the cycle, it has memorised
# the noise in these particular 50 points. It would look defensible in a backtest and lose
# money on the first day of live trading. In finance, overfitting is a multi-million dollar risk.
#
# This is where regularization comes in.

# %% [markdown]
# ### Performance Metrics

# %%
print(pd.DataFrame(scores.values()))

# %% [markdown]
# ### Coefficients

# %%
print(pd.DataFrame(coef_rows.values()).fillna(0))

# %% [markdown]
# ## Part 4: The Stability Tool (Ridge Regression - L2)
# How do we find a balance? We need a model that is complex enough to capture the cycle, but not so complex that it memorizes noise.
#
# First, let's look at Ridge Regression.
#
# Ridge is a risk management tool. It works by applying a penalty (the L2 norm) to the model, preventing any single feature from having a 'exploding' coefficient.
#
# It's forcing the model to be more stable and less sensitive to noise. In finance, we love stability. This is especially true when we have many correlated features (multicollinearity), which is common in economic data.

# %% [markdown]
# ### Library Import

# %%
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# %% [markdown]
# ### Building the pipeline
#
# From here on we stop transforming the data by hand and use a **Pipeline** instead.
#
# A pipeline chains the steps into a single estimator, and `fit` applies every step to the
# training data only. That matters for two reasons: it is the habit that stops data leaking
# from the test set into training, and it is what lets us cross-validate honestly later.
#
# We add a `StandardScaler`. Ridge and Lasso penalise the size of the coefficients, so a
# feature measured on a large scale gets an unfairly small coefficient and escapes the
# penalty. Scaling puts every polynomial term on the same footing before the penalty is
# applied.

# %%
def poly_ridge(alpha, degree=15):
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha),
    )

# %% [markdown]
# ### Fitting Ridge Model

# %%
ridge = poly_ridge(alpha=0.01)
ridge.fit(X.reshape(-1, 1), y)          # the pipeline takes the RAW x

ridge_train_prediction = ridge.predict(X.reshape(-1, 1))
ridge_test_prediction = ridge.predict(X_test.reshape(-1, 1))
ridge_curve = ridge.predict(X_curve.reshape(-1, 1))

print(f"largest coefficient: {np.abs(ridge.steps[-1][1].coef_).max():,.2f}"
      f"   (unregularised degree 15 was {np.abs(linear_poly.coef_).max():,.0f})")

# %%
plt.figure(figsize=[8, 5])
plt.scatter(X, y, label='Training data')
plt.plot(X_curve, y_curve, label='True function')
plt.plot(X_curve, ridge_curve, label='Ridge')
plt.plot(X_curve, linear_poly_curve, label='Unregularised poly')
plt.ylim(-2, 2)
plt.legend()
plt.show()

# %% [markdown]
# ### Calculating performance metrics

# %%
print(score('Ridge Poly', ridge_train_prediction, ridge_test_prediction))

# %% [markdown]
# ### Ridge with different lambdas

# %%
lambda_list = [0, 0.01, 0.1, 1, 10]

plt.figure(figsize=(16, 4))

for i, lambdai in enumerate(lambda_list):
    ax = plt.subplot(1, len(lambda_list), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    model = poly_ridge(alpha=lambdai)
    model.fit(X.reshape(-1, 1), y)

    name = f'Ridge lambda={lambdai}'
    score(name, model.predict(X.reshape(-1, 1)), model.predict(X_test.reshape(-1, 1)))
    record_coefs(name, model)

    plt.scatter(X, y)
    plt.plot(X_curve, y_curve, label='True')
    plt.plot(X_curve, model.predict(X_curve.reshape(-1, 1)), label=f'lambda={lambdai}')
    plt.ylim(-2, 2)
    plt.legend(fontsize=8)
    plt.title(f'Ridge lambda={lambdai}\nmax|coef| = {np.abs(model.steps[-1][1].coef_).max():,.2f}',
              fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Performance Metrics

# %%
print(pd.DataFrame(scores.values()))

# %% [markdown]
# **The Verdict: A Robust Model**
#
# The Ridge model captures the true signal (the smooth cycle) while ignoring the extreme,
# noisy wiggles.
#
# Look at what lambda does across the five panels. At `lambda=0` there is no penalty at all,
# so this is exactly the unregularised model from Part 3 - same coefficients, same test score.
# A small penalty is enough to collapse the largest coefficient from 1.25 billion to 1.85, and
# the test score jumps from 0.891 back to 0.912. Push lambda too far and the model becomes too rigid
# and underfits.
#
# It is not "perfect" on the training data, but it is robust. We traded a little
# backtest-perfection for a lot of real-world stability. That is a winning trade.

# %% [markdown]
# ### Coefficients

# %%
print(pd.DataFrame(coef_rows.values()).fillna(0))

# %% [markdown]
# ## Part 5: The Insight Tool (Lasso Regression - L1)
#
# Lasso is another, powerful type of regularization. It also adds a penalty (the L1 norm), but
# it has a unique and incredibly useful side-effect.
#
# Lasso is an insight and feature selection tool. It drives some coefficients to **exactly**
# zero, so the fitted model tells you which features it is actually using.
#
# Imagine you have a credit model with 200 features. Lasso might keep only 15 of them. That is
# incredibly valuable for explaining your model to a boss, a risk manager, or a bank regulator.
#
# One caution to carry with you: *which* coefficients get zeroed depends on the data and on
# lambda. Where two features are strongly correlated, Lasso tends to keep one and drop the
# other more or less arbitrarily, and a small change in the data can flip which one survives.
# A zero coefficient is not proof that a variable does not matter.

# %% [markdown]
# ### Library Import

# %%
from sklearn.linear_model import Lasso

# %% [markdown]
# ### Building the pipeline
#
# Same structure as Ridge, with two additions.
#
# `max_iter` is raised because Lasso is solved by coordinate descent, and the polynomial design
# matrix here is severely ill-conditioned (its condition number is about 2 x 10^11, since
# `x`, `x^2`, `x^3` ... are all nearly the same shape on [0, 1]). At the default 1000 iterations
# scikit-learn stops early and prints a `ConvergenceWarning`. Note that scaling alone does not
# fix this - the collinearity is real, not a units problem.

# %%
def poly_lasso(alpha, degree=15):
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Lasso(alpha=alpha, max_iter=100000),
    )

# %%
lasso = poly_lasso(alpha=0.001)
lasso.fit(X.reshape(-1, 1), y)

lasso_train_prediction = lasso.predict(X.reshape(-1, 1))
lasso_test_prediction = lasso.predict(X_test.reshape(-1, 1))
lasso_curve = lasso.predict(X_curve.reshape(-1, 1))

lasso_coefs = lasso.steps[-1][1].coef_
print(f"{np.sum(lasso_coefs == 0)} of {len(lasso_coefs)} coefficients are exactly zero")

# %%
plt.figure(figsize=[8, 5])
plt.scatter(X, y, label='Training data')
plt.plot(X_curve, y_curve, label='True function')
plt.plot(X_curve, ridge_curve, label='Ridge')
plt.plot(X_curve, lasso_curve, label='Lasso')
plt.plot(X_curve, linear_poly_curve, label='Unregularised poly')
plt.ylim(-2, 2)
plt.legend()
plt.show()

# %% [markdown]
# ### Calculating performance metrics

# %%
print(score('Lasso Poly', lasso_train_prediction, lasso_test_prediction))

# %%
lambda_list = [0.0001, 0.001, 0.01, 0.1]

plt.figure(figsize=(16, 4))

for i, lambdai in enumerate(lambda_list):
    ax = plt.subplot(1, len(lambda_list), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    model = poly_lasso(alpha=lambdai)
    model.fit(X.reshape(-1, 1), y)

    name = f'Lasso lambda={lambdai}'
    score(name, model.predict(X.reshape(-1, 1)), model.predict(X_test.reshape(-1, 1)))
    record_coefs(name, model)

    coefs = model.steps[-1][1].coef_
    plt.scatter(X, y)
    plt.plot(X_curve, y_curve, label='True')
    plt.plot(X_curve, model.predict(X_curve.reshape(-1, 1)), label=f'lambda={lambdai}')
    plt.ylim(-2, 2)
    plt.legend(fontsize=8)
    plt.title(f'Lasso lambda={lambdai}\n{np.sum(coefs == 0)} of {len(coefs)} coefficients zeroed',
              fontsize=9)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Performance Metrics

# %%
print(pd.DataFrame(scores.values()))

# %% [markdown]
# **The Verdict: An Interpretable Model**
#
# The Lasso model also gives us a smooth, robust curve, and its test score is competitive with
# Ridge.
#
# Now look at the coefficients in the table below. Lasso has set most of the fifteen polynomial
# terms to **exactly 0.0** - not "small", not "1e-9", but zero. It has effectively turned those
# features off, leaving a simpler model that a human can read.
#
# Compare that with the Ridge row directly above it: every one of the fifteen Ridge
# coefficients is non-zero. Ridge shrank them, Lasso selected among them. That difference is
# the whole reason both exist.

# %% [markdown]
# ### Coefficients

# %%
print(pd.DataFrame(coef_rows.values()).fillna(0))

# %% [markdown]
# ## Part 6: Choosing lambda Properly
#
# Every lambda in this notebook so far was picked by hand, by looking at the charts. That is
# fine for teaching and unacceptable in practice - we chose it after seeing how it performed.
#
# The honest procedure is cross-validation on the **training set only**: split the training data
# into folds, fit on some and validate on the rest, and let the average validation error choose
# lambda. The test set stays sealed until the very end.
#
# Because the whole pipeline sits inside `GridSearchCV`, the polynomial expansion and the scaler
# are refitted on each fold, so no fold can leak into another.

# %%
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    poly_ridge(alpha=1.0),                              # pipeline; alpha is overwritten below
    {'ridge__alpha': np.logspace(-4, 3, 50)},
    cv=5,
    scoring='neg_root_mean_squared_error',
)
search.fit(X.reshape(-1, 1), y)                         # training data only

best_alpha = search.best_params_['ridge__alpha']
print(f"cross-validation chose lambda = {best_alpha:.4f}")
print(f"   (we had hand-picked 0.01)")

score('Ridge (CV-tuned)',
      search.predict(X.reshape(-1, 1)),
      search.predict(X_test.reshape(-1, 1)))

# %% [markdown]
# The tuned model lands at an RMSE of roughly 0.21 on the test set, against a noise floor of
# 0.20. In other words, cross-validation has taken us to within a couple of percent of the best
# score any model could possibly achieve on this data - and it got there without ever looking at
# the test set.
#
# That is the workflow the final project expects.

# %% [markdown]
# ## Final Verdict: From Engineering to Finance
# In this notebook, we saw four types of models:
#
# *   **The Too-Simple Model**: a straight line through a cycle. Consistently wrong (high bias).
#
# *   **The Overfit Model**: Looked best on the training data, and at degree 30 scored a
#     *negative* test R-squared. Dangerously unusable.
#
# *   **The Ridge Model**: Gave us a stable and robust prediction. Ideal for managing correlated
#     features and creating reliable forecasts.
#
# *   **The Lasso Model**: Gave us an interpretable and simple prediction. Ideal for
#     understanding what drives the model and explaining it to stakeholders.
#
# Two habits to take away, because they are worth more than any single algorithm:
#
# 1.  **Judge every model on data it has never seen.** Every honest number in this notebook came
#     from the held-out test sample, never from the training data.
# 2.  **Nothing beats the noise.** We simulated the data, so we know the noise had a standard
#     deviation of 0.2 - and the best model above lands at an RMSE of about 0.21. That gap is
#     irreducible error, and no amount of extra complexity removes it. With real data you never
#     know where that floor is, which is exactly why a model that appears to beat it should
#     worry you rather than please you.
#
# This is the key lesson: in finance, we don't just want the model with the lowest error. We
# want the model that is the most reliable, stable, and interpretable.

# %% [markdown]
# # Homework: Week 1 - The Overfitting Trap in Marketing ROI
#
# ## Dataset:
# https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Advertising.csv
#
# Load it with `pd.read_csv(url, index_col=0)`. The file carries an unnamed first column that
# is just the row number - if you forget `index_col=0` you will silently train on it as a
# feature.
#
# ## Part 1: The "Simple" Model (Baseline)
# 1.   **Split Your Data:** Before you do anything else, split your data into a training set and a test set using train_test_split. (Use a test_size=0.3 and random_state=1). You will use the training set to fit all your models and the test set to evaluate them.
#
# 2.   **Fit & Interpret:** Fit a simple LinearRegression using only the three original features (TV, Radio, Newspaper).
#
# 3.   **Write Down Coefficients and Performance metrics:** Note the coefficients for TV, Radio, and Newspaper and its performance metrics on both train set and test set. This will be your baseline model.
#
# ## Part 2: The "Overly Complex" Model (The Trap)
#
# 1.   **Create Polynomial Features:** Use PolynomialFeatures (try degree=5) to create a new, high-dimensional training set. This will create many new features (e.g., TV^2, Radio^3, TV * Radio). Make sure to fit_transform on your training data and only .transform your test data.
# 2.   **Scale Your Features:** Use StandardScaler. fit_transform on the polynomial training data and just .transform on the polynomial test data. (This is critical for regularization to work.)
# 3.   **Fit the Overfit Model:** Fit a LinearRegression on this new, scaled, polynomial training set.
# 4.   **Check the Coefficients and Metrics:** Print the model.coef_ and calculate its performance metrics.
#
# > Steps 1 and 2 are exactly what a `Pipeline` does for you, as we did in Parts 4 and 5. Do it
# > by hand once so you can see the order, then use a pipeline for the rest.
#
# ### Question 1 (Observation):
# * What do you observe about the coefficients? Are they large or small? Do they make any intuitive sense? What does this tell you about the risk of this model? (Hint: They will likely be huge and non-sensical, a classic sign of overfitting).
# * Print the metrics for this complex model on the training set.
# * Print the R-squared score for this same model on the test set.
# * What do you observe? What does the difference between these two scores (and the baseline score from Part 1) tell you about this model? Is this a good model?
#
# ## Part 3: The Regularization Fix (Ridge & Lasso)
# Now, let's fix the model from Part 2.
# 1.   Fit Ridge: Fit a Ridge model on the same scaled, polynomial training data.
# 2.   Fit Lasso: Fit a Lasso model on the same scaled, polynomial training data.
#
# ### Question 2 (Analysis & Performance):
# * What do you observe about the coefficients from the two new models now? Are they still large or small? Make some comments about the changes.
# * Look at the coefficients from your Lasso model. How many features did it set to zero? What does this tell you about the 'true' drivers of sales?
# * What is the performance metrics for your Ridge model on the train set and test set?
# * What is the performance metrics for your Lasso model on the train set and test set?
# * How do these scores compare to the 'overfit' model's test score? What does this prove about the value of regularization?
#
# ### Question 3 (The Verdict):
# In the end, after trying a simple model, an overfit complex model, and two regularized models, what is your final recommendation to the CMO? Which channels (TV, Radio, Newspaper) are the most reliable drivers of sales?
#
# ### Question 4 (Choosing lambda):
# You picked the Ridge and Lasso `alpha` by hand. Use `GridSearchCV` with `cv=5` on the
# *training set only* to choose it properly, then report the test score of the tuned model.
# Did the tuned model beat your hand-picked one?
