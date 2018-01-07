# FMLS Projections

The scraping script used to download the raw data is not included right now.

In order, run:
```
python preprocessing-clustering.py
python clustering.py
python preprocessing-modeling.py
python modeling.py
python screening.py
python optimizer.py
```
The clustering files are to create more nuanced positions(attacking midfielder, defensive midfielder, etc) instead of just using the basic positions(forward, midfielder, etc).

The screening file is to get rid of players unlikely to play.

The optimizing file is to create multiple lineups based on certain changeable conditions.

Currently the settings in optimizer.py as well as the preprocessing files written are for GW 33, and need to be modified a bit when the new season comes out. 

So far the best achieved values for R^2 and MAE are .17 and 2.3 respectively.

Next major steps for improving models:
* Transform points so Nikolic's 27 point week does not have so much of a pull(Log transform or square root probably)
* Create separate models for forwards, midfielders, defenders, and goalkeepers
* Scrape betting odds
