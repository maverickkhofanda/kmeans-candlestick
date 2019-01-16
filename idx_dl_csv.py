import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas_datareader.data as web
import sys
import os
import collections
from io import StringIO
# dev tools
'''
import cProfile
import pstats
'''
# functionality: download, update, 

data_source = 'yahoo'
start='2008-01-01'
now = datetime.now()
end = now.strftime("%Y-%m-%d")
#end = '2018-11-10'

directory = 'IDX'
if not os.path.exists(directory):
    os.makedirs(directory)

tickers = []

def get_tickers(ticker_list_txt):
	with open(ticker_list_txt) as f:
		next(f)
		for ticker in f.readlines():
			tickers.append(ticker.replace('\n', ''))
		#print(tickers)

def is_market_open_hours():
	open_market = 9
	close_market = 17

	day_open = [0,1,2,3,4] # Monday thru Friday
	# check if weekday
	if now.weekday() in day_open:
		if now.hour >= 9 and now.hour <= 17:
			#print("it's business hour, updating now may produce unspecified & unwanted result")
			return True
		else:
			print("it's weekday, but not business hour")
	else:
		print("it's weekend! Do whatever you want!")

	print("updating...")
	return False
	
def ticker_download(ticker_list_txt=None):
	fail_count = 0
	ok_count = 0
	update_count = 0

	if is_market_open_hours():
		raise Exception("it's business hour, updating now may produce unspecified & unwanted result")
	
	#raise Exception("test exception")

	if ticker_list_txt is not None:
		get_tickers(ticker_list_txt)
	
	for ticker in tickers:
		try:
			ticker_dir = "./"+directory+"/"+ticker+".csv"

			# if file doesn't exist, create new file
			if not os.path.exists(ticker_dir):
				result = web.DataReader(ticker, data_source, start, end)
				print("got data for " + ticker + " " + str(result.shape))
				
				result.to_csv(ticker_dir)
				ok_count += 1

		    # else, update file
			else:
				ticker_update_latest(ticker, ticker_dir)
				update_count += 1
		
		except Exception as ex:
			print(ex)
			print ("did not get data for " + ticker)
			fail_count += 1
	    
	print(ok_count, "loads,", update_count, "updated,", fail_count, "failures")

def ticker_update_latest(ticker, ticker_dir):
	tail_date = None
	with open(ticker_dir, 'r') as f:
		q = collections.deque(f, 1)
		tail_data = pd.read_csv(StringIO(''.join(q)), index_col=0, header=None)
		tail_date = tail_data.index.values[0]
		#print(tail_date)

	# do tail
	try:
		data_aft_tail = web.DataReader(ticker, data_source, tail_date, end)
		data_aft_tail = data_aft_tail.iloc[1:] # delete top row
		tail_date_is_equal = data_aft_tail.head(1).index.strftime("%Y-%m-%d").values[0] == tail_date
		
		if not tail_date_is_equal:
			#pd.read_csv(ticker_dir, index_col=0).tail(1)
			data_aft_tail.to_csv(ticker_dir, mode='a', header=False)
			print(ticker, "updated")

	except Exception as ex:
		if ex == 'start must be an earlier date than end' or str(ex) == 'index 0 is out of bounds for axis 0 with size 0':
				print(ticker, "need not updating")
		else:
			print('ticker_update_latest Exception:',ex)



if __name__ == '__main__':
	arg_names = ['script','ticker_list_txt']
	args = dict(zip(arg_names, sys.argv))

	Arg_list = collections.namedtuple('Arg_list', arg_names)
	args = Arg_list(*(args.get(arg, None) for arg in arg_names))
	#print(args)

	ticker_download(args.ticker_list_txt)
	'''
	# debug
	debug_file_name = 'cProfile'
	cProfile.run('ticker_download(args.ticker_list_txt)', debug_file_name)
	p = pstats.Stats(debug_file_name)
	p.sort_stats('cumulative').print_stats(25)
	'''