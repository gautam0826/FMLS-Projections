# FMLS Projections
-----

## Basic Usage:
------
The project follows the basic [Cookiecutter for Data Science template](https://github.com/drivendata/cookiecutter-data-science), with a few modifications taking inspiration from the [Kedro project structure](https://kedro.readthedocs.io/en/stable/02_getting_started/04_hello_world.html). The `scrape_data` script saves FMLS JSON data to the `data/raw` folder. The `make_database` script creates a SQLite database populated with FMLS statistics, updating the database if it already exists. The `build_features` script creates views for machine learning models. The final five scripts, `train_simple_linear_model`, `train_robust_simple_linear_model`, `train_gbm_ranking_model`, `train_rnn_ordinal_model` and `train_cnn_ordinal_model` train machine learning models on the data and output predictions for the upcoming week.

Here are the commands to be run from the project's root directory
```
python -m src.data.scrape_data
python -m src.data.make_database
python -m src.data.build_features
python -m src.models.train_simple_linear_model
python -m src.models.train_robust_simple_linear_model
python -m src.models.train_gbm_ranking_model
python -m src.models.train_rnn_ordinal_model
python -m src.models.train_cnn_ordinal_model
```
Program parameters default values are set in `conf/base/parameters.yaml` and can be overidden by creating a local parameters file in `conf/local/parameters.yaml` or by including command line arguments like shown
```
python -m src.models.train_rnn_ordinal_model --normal_layer_size=64
```

## Objectives:
------
 - Create a system to aid the decision-making of fantasy players by distinguishing MLS players who are likely to perform well in Fantasy MLS from those who will not.
 - Create a system where it is easy to try and robustly compare different approaches to achieve objective #1. Be able to compare state-of-the-art machine learning models with simple heuristics-based models, and classification approaches with regression approaches. People can comfortably plug-in and integrate their own ideas for data transformations and modeling.
 - Make sure predictions are explainable and informative. Using techniques like [Shap values](https://github.com/slundberg/shap) can help tell fantasy players why the model predicted the values it predicted, which is more helpful than telling players to blindly trust the model's judgement. Also knowing a predictive distribution instead of just a number greatly helps assess potential risk.
 - Reduce the difficulty of people trying to run or work on this project. The project should be able to run completely locally on a laptop and the python scripts should all be individually executable without much setup.

## Details:
------
The project follows the [OSEMN](http://www.dataists.com/2010/09/a-taxonomy-of-data-science/) taxonomy of Data Science projects
### Obtain: 
Raw JSON data is taken from the hidden FMLS API and is saved in the raw data folder with separate folders for each season. New data overrides any old data for the current season.
### Scrub:
Data from the raw data folder is wrangled into a table where every row contains an observation of one player’s statistics for one game and dumped to a SQLite database. Statistics are aggregated into views suitable for machine learning models. For example, the `advanced_position` view assigns players to “advanced” positions like Attacking Midfielder, or Wing-back based on their average statistics in a season. The `player_lagging_stats` view computes the previous 6 game’s statistics for each game each player has appeared in. This approach saves space compared to writing out each view’s data to disk every new week and any views used by multiple models get to be reused.
### Explore:
Exploratory Jupyter notebooks will be ported over to Python scripts soon. Visualizations were made to understand how each opponent impacts each position and how different models compare in predictive strength.
### Model:
Coupled to each model is a SQL query that joins all the necessary data from the database. That data is loaded into a pandas dataframe to allow for non-SQL operations to be done on the data, like calculating standard deviation, and written to a Parquet file for fast reading. The model template also logs back metrics, saves, and loads models using the [MLFlow library](https://github.com/mlflow/mlflow). Right now, the first approach is adopted in the files `train_simple_linear_model` and `train_robust_simple_linear_model`, while the second approach is adopted in `train_rnn_ordinal_model` and `train_cnn_ordinal_model`.

##### Approach #1: Pure machine learning
Features being used by the model:
 - Lagged statistics(goals, assists, shots, …) for each player last 8 games
 - Lagged statistics for player’s team for last 8(total goals, assists, shots, ...)
 - Lagged statistics for player’s team’s opponent for last 8
 - Lagged statistics for opponent last 8
 - Lagged statistics for opponent’s opponents last 8
 - Lagged statistics for players in the same cluster against opponent last 8
 - Binary DGW variables for player’s team and opponent (whether the team is playing twice in a week)
 - Binary Home variable
 - \# of matches in given round(to spot international breaks)

Additionally, instead of predicting just a player's fantasy points, the model is trained on adjusted points, filtering out low frequency noisy events like own goals and red cards. Adjusted points are further winsorized into a range of 2 to 15 since discriminating between low and extremely low or high and extremely high points is not necessary. The model also predicts the probability of each score between 2 and 15 instead of just predicting a number, giving a more complete assessment of players’ floors and ceilings. Neural networks were used because of their ability to learn complex features from very “raw” data like image pixels. Recurrent neural networks and Convolutional neural networks were chosen specifically because of how they deal with time series data. Potentially other machine learning algorithms can be used, although doing PCA on the lagged statistics or replacing lagged statistics with averaged statistics will probably be needed to reduce the data’s dimensionality. The way data is fed into the model also preserves the relation between columns, essentially making the model "see" the data the same way humans see each player's scoring history in a table, instead of seeing a bunch of unrelated columns.

##### Approach #2: Time weighted models
Features being used by the model:
 - Binary DGW variables for player’s team and opponent
 - Binary Home variable
 - \# of matches in given round
 - Player’s last 6 adjusted points average
 - One-hot-encoded opponent variable
 - One-hot encoded team variable
 - Day of week (Wednesday, Friday, Saturday, Sunday)
 - Days since player’s last game
 - Interaction terms between advanced position and opponent, home and team, home and opponent

This means using sample weights(with the weight in proportion to how close the previous week is to current week) and keeping dummy columns for different opponents instead of using lagging statistics. Currently, a simple GLM is the model being used, due to the easily interpretable nature of coefficients. Using Multilevel models instead of simple GLMs could also be interesting.

##### Approach #3: 'Component' models
This means instead of just predicting points, predict attacking bonus points, defending bonus points, goals, assists, and clean sheet chances separately and simulate each player's fantasy points using those predicted values. This approach creates a distribution of points that should be more useful for evaluating players than just a single number. Additionally this will ensure defenders and goalkeepers from the same team will have the exact same clean sheet chance as part of their projected points.

### Interpret:
##### Evaluation:
The evaluation procedure is designed to work to compare all models, regardless of what they are predicting and how they are predicting it, all that's necessary is a way to sort players. For each of the pervious five weeks models are re-run with all the data leading up to that week, ensuring that the predictions for the testing dataset are the predictions that would have been output at the time. Then, metrics are calculated based on comparing the model's sorted players list and actual player scores.

##### Metrics:
The problem can be thought of as a mix of a ranking problem, classification and a regression problem. While predicting points is the most direct and straightforward to solve the problem, one can also frame the problem as determining how likely every indidivual player will score well or as ranking the players by probability of scoring well, turning the problem from a regression problem into ranking and classification problems. Since the ultimate objective is to determine the highest scoring players each week, averaging the scores or calculating the hit rate of the top N players selected by sorting on model recommendation works as an easily-interpretable, rudimentary metric. Since this measure might be noisy, other ranking metrics are also calculated like the [Kendall rank correlation coefficient](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient) and [Spearman's rank correlation coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient). However, just using these metrics may obfuscate certain models performing well on subsets of the data and not well on others; for example, linear models might be able to perform well on forwards but not for goalkeepers. Therefore, every metric is calculated for each position as well as the whole set of players each week and averaged. Metrics also can be calculated for each output for models with multiple outputs.

## Future plans:
------
 - Use the [featuretools](https://github.com/FeatureLabs/featuretools) and/or [tsfresh](https://github.com/blue-yonder/tsfresh) libraries to automatically generate features
 - New feature for if a player is team’s attacker/defender with the most/second-most shots/key passes (maybe some teams let up more to secondary attacker)
 - New feature for out-of-position players (Ex: forward listed as defender)
 - Scrape betting odds (may be a little out of scope for this project)
 - Indirectly predicting performance
   - Predict the difference between points and last 6 average
   - Predict whether points will be greater than 7/8/10 (turn into classification problem)
   - Use Quantile Regression to estimate floor/ceiling
 - Make sure model doesn't try too hard to fit in very high outliers, since we care more about discerning between 2 point and 10 point performances than between 15 point and 22 point performances (Currently trying to do this via winsorizing points)
   - Using a log transform or square root transform on points
   - Use class imbalancing techniques like SMOTE
   - Use sample weights
 - Write an optimizer that automates lineup building given a set of projections
   - When one score or one probability is output, simply maximize sum of scores or sum of probabilities
   - For optimizing when there are probabilities for each individual score, use simulations to optimize lineups probability of getting > N points total
 - Add unit tests
 - Have an interactive viz where users can look through player's projected points and shapley values.
