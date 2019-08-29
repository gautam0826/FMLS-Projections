# Fantasy MLS Projections


### Usage instructions
First, run scrape_data to scrape the latest data from the Fantasy MLS site, and then extract to create a csv where each row will be a record of a player's fantasy stats from 2017 onwards. Next, run transform_position_modeling and train_positional_model to infer player's detailed positions. Finally, run transform_ml_regression_modeling and train_ml_regression_nn_model to make predictions on the current week's data. You can see the predictions in the file 2019_current_nn_predictions.csv inside data\processed.
From the src directory:
```
python data\scrape_data.py
python data\extract.py
python features\transform_position_modeling.py
python models\train_positional_model.py
python features\transform_ml_regression_modeling.py
python models\train_ml_regression_nn_model.py
```

### High-level Goals
There are a few important philosophical goals of this project
* Make it very easy to try out different approaches and compare them.
* Make sure predictions are explainable. The goal of this project is to aid the decision-making of fantasy players, not to make all decisions for them. Using techniques like Shapley values can help tell fantasy players why the model predicted the values it predicted, which is more useful to players blindly trusting the model's judgement.

### Future work
The end goal is to create a module where people can write their own pipelines based off the data that comes from the extracted csv file. Currently, pipelines are being planned to support three differing modeling approaches. So far, work is finished on writing two programs to transform and model data using a neural network, but the scripts will need refinement. Most of the work being done now is fixing technical debt issues in order to streamline future work. In order to compare completely different modeling approaches, a program will compare predictions made on the previous five rounds to evaluate how each approach would have done according to different metrics and positions.

### Approach #1: Simple machine learning
The current features being used by the neural network are: 
* Detailed position (A model is trained on some rough rules where high assists means attacking midfielder, high clearances means center back improve on the rough rules' classifications)
* Lagged statistics(goals, assists, shots, …) for each player last 8 games (all stats taken from FMLS website)
* Lagged statistics for player’s team for last 8(sum of goals, assists, shots)
* Lagged statistics for player’s team’s opponent for last 8
* Lagged statistics for opponent last 8
* Lagged statistics for opponent’s opponents last 8
* Lagged statistics for players in the same cluster against opponent last 8
* Binary DGW variable (whether player is playing twice in a week)
* Binary Home variable
* \# of matches in given round(to spot international breaks)

Additionally, instead of predicting just a players fantasy points, the model is trained on adjusted points, filtering out low frequency noisy events like own goals and red cards. Potentially other machine learning algorithms can be used, although doing PCA on the lagged statistics or replacing lagged statistics with summed statistics will probably be needed to reduce dimensionality.

### Approach #2: 'Component' models
This means instead of just predicting points, predict attacking bonus points, defending bonus points, goals, assists, and clean sheet chances separately and simulate each player's fantasy points using those predicted values. This approach creates a distribution of points that should be more useful for evaluating players than just a single number. Additionally this will ensure defenders and goalkeepers from the same team will have the exact same clean sheet chance as part of their projected points.

### Approach #3: Time weighted models
This means using sample weights(with the weight in proportion to how close the week is to current day) and keeping dummy columns for different opponents instead of using lagging statistics. This could still be used with more trendy machine learning methods like Gradient Boosting and Neural Networks but using a Mixed Effects could be interesting. The coefficients would also make it easier to interpret.

### TODOs
* Include match timestamps when extracting from raw data
* Try week/day of week/month as feature
* Try using tsfresh library to generate features
* Try to spot out-of-position players (Ex: forward listed as defender)
* Try adding feature if player is team’s attacker/defender with the most/second-most shots/key passes (maybe some teams let up more to secondary attacker)
* Make feature generation scripts keep track of columns to use for or exclude from modeling (will be passed in to model instead of currently way of hardcoding it)
* Write script inside models directory that runs all different approaches
* Rewrite files to use configuration files and command line arguments
* Try winsorizing target variable (so model doesn't try too hard to fit in 27-point matches, we care more about discerning between 2 point and 10 point performances than between 15 point and 22 point performances)
* Try out importance weights adjusting for time(for the Time weighted models) and points(for normal ML models to tackle imbalance)
* Try classifiers for 7+/10+ points instead of using regression models
* Look at Spearman's rank correlation coefficient across each different position as the final metric
* Have linting and unit tests setup
* Have an interactive viz where users can look through player's projected points and shapley values.