import pandas as pd
import numpy as np
import math
import pulp
from pulp import *

#constraints
num_lineups = 20
budget = 105
num_players = 11
num_gk = 1
min_df = 3
max_df = 5
min_mf = 3
max_mf = 5
min_fw = 1
max_fw = 3

#list of player ids projected highly but whom are still undesirable for mass team selection(using player ids because they are unique unlike player names)
player_exposure_constraints = [
(119, 0.2) #Javier Morales
]

#list of player ids who are to be avoided completely(using player ids because they are unique unlike player names)
players_avoid = [
386, #Shea Salinas
389, #Jackson Yueill
388, #Tommy Thompson
308 #Alejandro Bedoya
]

projections_column = 'Top Ridge'
floor_column = 'Top GBM 0.30'

#TODO: have a player id list of 'must haves' and possibly min exposure
#TODO: keep track of lineup with highest 'floor'

loc_projections = 'screened current predictions.csv'
loc_lineups = 'top ' + str(num_lineups) + ' lineups.csv'

#combine players playing two games into one combined total
#even though we're not using floor column for optimization, we can use it to pick out the lineup with the highest floor out of all the top lineups
df = pd.read_csv(loc_projections, sep=',', encoding='ISO-8859-1', usecols=['player name', 'player id', 'cost', projections_column, floor_column, 'team id', 'position id_1', 'position id_2', 'position id_3', 'position id_4'])
df = df.reset_index().groupby(['player name', 'player id', 'cost', 'team id', 'position id_1', 'position id_2', 'position id_3', 'position id_4'], as_index=False).sum()
df['team id'] = df['team id'].astype(np.int64) #getting numpy.float64 cannot be interpreted as an integer error on line 35[range(max(df['team id']))] so makes sure to convert team id to integer
df = df[~df['player id'].isin(players_avoid)] #select all players not in avoid list
df['lineup exposure'] = 0 #new column for lineup usage for each player
df['max lineups'] = num_lineups #new column for maximum number of lineups selected in for each player
for player_id, exposure_pct in player_exposure_constraints:
	df.loc[df['player id'] == player_id, 'max lineups'] = math.ceil(exposure_pct * num_lineups)

#create the 'prob' variable to contain the problem data
prob = pulp.LpProblem('FMLSLineups', pulp.LpMaximize)

#define objective function(projected points), the number of players chosen constraint, cost constraint, position constraints, and team constraints
objective_function = ''
num_players_constraint = ''
cost_constraint = ''
gk_constraint = ''
df_constraint = ''
mf_constraint = ''
fw_constraint = ''
team_constraints = ['' for i in range(max(df['team id']))]

#create pulp variables for each player(note the variable corresponds to row number in dataframe, not player id) and build constraints
for row_num, row in df.iterrows():
	variable = pulp.LpVariable('p' + str(row_num), lowBound=0, upBound=1, cat=pulp.LpInteger) #make binary player variable(0 for not in lineup, 1 for in lineup)

	#update constraints with player's info
	objective_function += row[projections_column] * variable
	num_players_constraint += variable
	cost_constraint += row['cost'] * variable
	if row['position id_1'] == 1:
		gk_constraint += variable
	elif row['position id_2'] == 1:
		df_constraint += variable
	elif row['position id_3'] == 1:
		mf_constraint += variable
	elif row['position id_4'] == 1:
		fw_constraint += variable
	team_constraints[row['team id'] - 1] += variable #subtract 1 since team id values start at 1

#add objective function(projected points), the number of players chosen constraint, cost constraint, position constraints, and team constraints to problem
prob += objective_function
prob += (num_players_constraint == num_players)
prob += (cost_constraint <= budget)
prob += (gk_constraint == num_gk)
prob += (df_constraint >= min_df)
prob += (df_constraint <= max_df)
prob += (mf_constraint >= min_mf)
prob += (mf_constraint <= max_mf)
prob += (fw_constraint >= min_fw)
prob += (fw_constraint <= max_fw)
for team_constraint in team_constraints:
	prob += (team_constraint <= 3)

#solve for the specified number of lineups
for i in range(1, num_lineups+1):
	#find solution
	prob.solve()

	#create binary column for this specific lineup
	lineup_col = 'lineup_' + str(i)
	df[lineup_col] = 0

	#add solution players to this lineup's column
	#also add a new constraint so the solver doesn't select same lineup again
	new_constraint = ''
	for variable in prob.variables():
		if variable.varValue == 1:
			row_num = int(variable.name[1:])
			df.ix[row_num, lineup_col] = 1
			df.ix[row_num, 'lineup exposure'] += 1 #right now lineup exposure is just the number of lineups player is in - at the end it will be divided by the number of lineups
			if df.ix[row_num, 'lineup exposure'] == df.ix[row_num, 'max lineups']: #check if player is in maximum number of lineups
				prob += (variable == 0) #if so, set to 0 to exclude from selection permamently
			new_constraint +=  variable
	prob += (new_constraint <= num_players - 1)

	print('optimized lineup no. ' + str(i))

#change lineup exposure from number of lineups appeared in to ratio of lineups included in to number of lineups
df['lineup exposure'] /= num_lineups

#delete max lineups column
df.drop('max lineups', axis=1, inplace=True)

#first sort by projections, then sort by lineups
df = df.sort_values(by=projections_column, ascending=False)
df = df.sort_values(by=['lineup_' + str(i) for i in range(1, num_lineups+1)], ascending=False)

#write to csv
df.to_csv(loc_lineups, sep=',', index=False)
