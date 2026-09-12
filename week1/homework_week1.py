"""
Fintech P1 - Week 1 Homework: The Overfitting Trap in Marketing ROI

This is a SCAFFOLD, not a solution. The plumbing is done (imports, data load,
train/test split, a metrics helper). Each homework step is marked TODO with a
pointer to the lecture cell that shows the same pattern.

HOW TO USE THIS FILE IN SPYDER
    Every "# %%" line starts a new cell. Put the cursor inside a cell and press
    Ctrl+Enter to run just that cell. Work top to bottom - later cells need
    variables from earlier ones. Nothing here is a Jupyter-only feature, so it
    all runs in a plain Spyder console.

    Written answers go in the triple-quoted ANSWER blocks. Keep them in the file:
    the file is your homework submission.
"""

# %% [markdown]
# # Setup: imports
#
# Everything the homework needs, imported once at the top. This is the normal
# way to write a .py script (in the notebook the imports were scattered across
# the parts, which is a notebook habit, not a Python one).

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import sqrt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 160)

# %% [markdown]
# # Setup: the scoring helper
#
# Same idea as `score()` in the lecture notebook: results live in a dict keyed
# by model name, so re-running a cell overwrites that model's row instead of
# quietly appending a duplicate.

# %%
scores = {}


def score(name, train_pred, test_pred):
    """Score a model on the training sample and on the held-out test sample."""
    scores[name] = {
        'Model': name,
        'R-squared train': r2_score(y_train, train_pred),
        'R-squared test': r2_score(y_test, test_pred),
        'RMSE train': sqrt(mean_squared_error(y_train, train_pred)),
        'RMSE test': sqrt(mean_squared_error(y_test, test_pred)),
        'MAE train': mean_absolute_error(y_train, train_pred),
        'MAE test': mean_absolute_error(y_test, test_pred),
    }
    return pd.DataFrame(scores.values())


def results_table():
    """All models scored so far, side by side."""
    return pd.DataFrame(scores.values())


# %% [markdown]
# # Part 1: The "Simple" Model (Baseline)
#
# ## 1.1 Load the data
#
# `index_col=0` matters: the file carries an unnamed first column that is just
# the row number. Forget it and you silently train on it as a feature.

# %%
URL = ('https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/'
       'master/Notebooks/Data/Advertising.csv')

advertising = pd.read_csv(URL, index_col=0)

print(advertising.head())
print(advertising.shape)
print(advertising.describe())

# %% [markdown]
# ## 1.2 Split into features and target, then train/test
#
# The brief is explicit: `test_size=0.3`, `random_state=1`. The random_state
# makes the split reproducible - without it you get a different split every run
# and your numbers will not match what you wrote down.

# %%
X = advertising[['TV', 'Radio', 'Newspaper']]
y = advertising['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

print(f"train: {X_train.shape[0]} rows")
print(f"test : {X_test.shape[0]} rows")

# %% [markdown]
# ## 1.3 Fit the baseline LinearRegression
#
# TODO: fit a plain `LinearRegression` on the three original features.
# Pattern: lecture cell "Fitting the Linear Regression".
#   - create the model, `.fit(X_train, y_train)`
#   - predict on X_train and on X_test
#   - `score('Baseline', <train prediction>, <test prediction>)`

# %%
# TODO: your code here


# %% [markdown]
# ## 1.4 Write down the coefficients
#
# TODO: print the intercept and the coefficient on each of TV, Radio, Newspaper.
# Hint: `pd.Series(baseline.coef_, index=X_train.columns)` gives you a labelled
# series instead of a bare array - much easier to read and to quote in an answer.

# %%
# TODO: your code here


# %% [markdown]
# ANSWER - Part 1 interpretation
#
# Write, in the block below: what each coefficient means in business terms
# (one extra unit of spend on that channel is associated with how much extra
# Sales?), and the baseline train/test metrics.

# %%
"""
ANSWER (Part 1):

TODO

"""

# %% [markdown]
# # Part 2: The "Overly Complex" Model (The Trap)
#
# Do this part BY HAND (separate PolynomialFeatures -> StandardScaler ->
# LinearRegression) so you see the order of operations. Parts 3 and 4 then use a
# Pipeline, which does exactly this for you.
#
# The rule that matters: `fit_transform` on the TRAINING data, `transform` only
# on the test data. Fitting the transformer on the test set leaks information
# from it into training and inflates every number you report afterwards.

# %% [markdown]
# ## 2.1 Polynomial features, degree 5
#
# TODO: build `PolynomialFeatures(degree=5, include_bias=False)`,
# `fit_transform` on X_train, `transform` on X_test.
# Pattern: lecture cell "Polynomial Transformation".
# Print the resulting shape - note how three features became many.

# %%
# TODO: your code here


# %% [markdown]
# ## 2.2 Scale
#
# TODO: `StandardScaler()`, `fit_transform` on the polynomial TRAINING data,
# `transform` on the polynomial test data.

# %%
# TODO: your code here


# %% [markdown]
# ## 2.3 Fit the overfit model and score it
#
# TODO: `LinearRegression` on the scaled polynomial training data.
# Score it as 'Poly degree 5 (unregularised)'.
# Also print `np.abs(model.coef_).max()` - the size of the largest coefficient
# is the tell.

# %%
# TODO: your code here


# %% [markdown]
# ANSWER - Question 1 (Observation)
#
# * What do you observe about the coefficients? Large or small? Do they make
#   intuitive sense? What does that tell you about the risk of this model?
# * The metrics on the training set.
# * The R-squared on the test set.
# * What does the gap between the two - and the comparison with the Part 1
#   baseline - tell you? Is this a good model?

# %%
"""
ANSWER (Question 1):

TODO

"""

# %% [markdown]
# # Part 3: The Regularization Fix (Ridge & Lasso)
#
# Fit Ridge and Lasso on the same scaled, polynomial training data.
#
# From here you can use a pipeline instead of transforming by hand:
#
#     ridge = make_pipeline(
#         PolynomialFeatures(degree=5, include_bias=False),
#         StandardScaler(),
#         Ridge(alpha=...),
#     )
#     ridge.fit(X_train, y_train)        # note: the RAW X_train
#
# Reach the final estimator with `ridge.steps[-1][1]` to get `.coef_`.
# Give Lasso `max_iter=100000` - the polynomial design matrix is ill-conditioned
# and the default 1000 iterations stops early with a ConvergenceWarning.

# %% [markdown]
# ## 3.1 Ridge
#
# TODO: fit a Ridge model, score it, print its coefficients and the largest one.

# %%
# TODO: your code here


# %% [markdown]
# ## 3.2 Lasso
#
# TODO: fit a Lasso model, score it, print its coefficients.
# Count the zeros: `np.sum(coefs == 0)` out of `len(coefs)`.

# %%
# TODO: your code here


# %% [markdown]
# ## 3.3 Compare everything side by side

# %%
print(results_table())

# %% [markdown]
# ANSWER - Question 2 (Analysis & Performance)
#
# * What do you observe about the coefficients from the two new models? Still
#   large? Comment on the change.
# * How many features did Lasso set to exactly zero? What does that say about
#   the 'true' drivers of sales?
# * Ridge metrics on train and test.
# * Lasso metrics on train and test.
# * How do these compare to the overfit model's test score? What does that prove
#   about the value of regularization?

# %%
"""
ANSWER (Question 2):

TODO

"""

# %% [markdown]
# ANSWER - Question 3 (The Verdict)
#
# After a simple model, an overfit complex one, and two regularized ones: what
# is your final recommendation to the CMO? Which of TV, Radio, Newspaper are the
# most reliable drivers of sales?
#
# Tip: back this with evidence from BOTH the baseline coefficients (Part 1) and
# which features survived Lasso (Part 3) - and say what the test metrics were
# for the model you are recommending.

# %%
"""
ANSWER (Question 3):

TODO

"""

# %% [markdown]
# # Part 4: Choosing lambda properly (Question 4)
#
# You picked alpha by hand. Now let cross-validation pick it, on the TRAINING
# set only - the test set stays sealed until the very end.
#
# Pattern: lecture cell in "Part 6: Choosing lambda Properly".
#
#     search = GridSearchCV(
#         <your pipeline>,
#         {'ridge__alpha': np.logspace(-4, 3, 50)},   # step name __ param name
#         cv=5,
#         scoring='neg_root_mean_squared_error',
#     )
#     search.fit(X_train, y_train)
#     search.best_params_
#
# The grid key is "<step name>__<parameter>". `make_pipeline` names steps after
# the class, lowercased: 'ridge__alpha', 'lasso__alpha'. Run
# `print(search.estimator.get_params().keys())` if you are unsure.

# %% [markdown]
# ## 4.1 Tune Ridge with GridSearchCV
#
# TODO: your code here. Score the tuned model as 'Ridge (CV-tuned)'.

# %%
# TODO: your code here


# %% [markdown]
# ## 4.2 Tune Lasso with GridSearchCV
#
# TODO: same for Lasso. Score it as 'Lasso (CV-tuned)'.

# %%
# TODO: your code here


# %% [markdown]
# ## 4.3 Final table

# %%
print(results_table())

# %% [markdown]
# ANSWER - Question 4 (Choosing lambda)
#
# What alpha did cross-validation choose for each model? Report the test score
# of the tuned models. Did the tuned model beat your hand-picked one?

# %%
"""
ANSWER (Question 4):

TODO

"""
