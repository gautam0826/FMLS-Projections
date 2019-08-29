import os
import sys
SRC_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(SRC_DIRECTORY))
os.chdir(SRC_DIRECTORY)
import re
import json
import requests
import unidecode
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.utilities import data_utilities as utilities

current_subdir = utilities.subdir_nineteen
base_url = 'https://fgp-data-us.s3.amazonaws.com/json/mls_mls/'

def download_data(url, headers, outfile_path, timeout=15):
	r = requests.get(url, headers=headers, timeout=timeout)
	data = r.text
	outfile = open(outfile_path, 'w')
	outfile.write(unidecode.unidecode(data))
	outfile.close()
	#tqdm.write(url)
	return r.status_code

def download_match_data(match_id, overwrite=False):
	url = base_url + 'stats/' + str(match_id) + '.json'
	headers = None
	outfile_path = os.path.join(utilities.path_to_raw, current_subdir, 'Matches', 'Match' + str(match_id) +  '.json')
	if not overwrite and os.path.exists(outfile_path):
		return (200, match_id)
	return (download_data(url, headers, outfile_path), match_id)

def download_player_data(player_id):
	url = base_url + 'stats/players/' + str(player_id) + '.json'
	headers = None
	outfile_path = os.path.join(utilities.path_to_raw, current_subdir, 'Players', 'Player' + str(player_id) +  '.json')
	return (download_data(url, headers, outfile_path), player_id)

def download_general_json():
	pages = ['Squads', 'Players', 'Rounds', 'Venues']
	for page in pages:
		file = page + '.json'
		download_data(base_url + file.lower(), None, os.path.join(utilities.path_to_raw, current_subdir, file))

def get_match_ids():
	match_ids = []
	round_data = json.load(open(os.path.join(utilities.path_to_raw, current_subdir, 'Rounds.json'), 'r'))
	for match_round in round_data:
		round_id = match_round['id']
		if match_round['status'] == 'complete' or match_round['status'] == 'active':
			matches = match_round['matches']
			for match in matches:
				match_id = match['id']
				match_ids.append(match_id)
	return match_ids

def get_player_ids():
	player_ids = []
	player_data = json.load(open(os.path.join(utilities.path_to_raw, current_subdir, 'Players.json'), 'r'))
	for player_entry in player_data:
		player_id = player_entry['id']
		player_ids.append(player_id)
	return player_ids

def download_all_match_data(connections=25):
	match_ids = get_match_ids()
	kwargs = {
		'total': len(match_ids),
		'unit': 'match',
		'unit_scale': True,
		'leave': True
	}

	with ThreadPoolExecutor(max_workers=connections) as executor:
		time1 = time.time()
		futures = (executor.submit(download_match_data, match_id) for match_id in match_ids)
		for future in tqdm(as_completed(futures), **kwargs):
			try:
				status_code, match_id = future.result()
				tqdm.write('finished match ' + str(match_id) + ' status code: ' + str(status_code))
			except Exception as exc:
				exception = str(type(exc))
				tqdm.write('exception: ' + exception)
		time2 = time.time()

	print('Took %f s'%(time2-time1))

def download_all_player_data(connections=25):
	player_ids = get_player_ids()
	kwargs = {
		'total': len(player_ids),
		'unit': 'player',
		'unit_scale': True,
		'leave': True
	}

	with ThreadPoolExecutor(max_workers=connections) as executor:
		time1 = time.time()
		futures = (executor.submit(download_player_data, player_id) for player_id in player_ids)
		for future in tqdm(as_completed(futures), **kwargs):
			try:
				status_code, player_id = future.result()
				tqdm.write('finished player ' + str(player_id) + ' status code: ' + str(status_code))
			except Exception as exc:
				exception = str(type(exc))
				tqdm.write('exception: ' + exception)
		time2 = time.time()

	print('Took %f s'%(time2-time1))

def scrape_fmls():
	if not os.path.exists(os.path.join(utilities.path_to_raw, current_subdir)):
		os.makedirs(os.path.join(utilities.path_to_raw, current_subdir, 'Matches'))
		os.makedirs(os.path.join(utilities.path_to_raw, current_subdir, 'Players'))

	download_general_json()
	download_all_match_data()
	download_all_player_data()

if __name__ == '__main__':
	scrape_fmls()
