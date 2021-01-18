import pandas as pd
import os
import time
from datetime import datetime

    # Sets location to find all relevant stock data in the Key Stats directory
path = "/home/Jordan/machineLearning/sentdex/documents/intraQuarter"
def Key_Stats(gather = "Total Debt/Equity (mrq)"):
    statspath = path + '/_KeyStats'
    stock_list = [x[0] for x in os.walk(statspath)]
        # Define data frame and specify starting columns
    df = pd.DataFrame(columns = ['Date', 'Unix', 'Ticker', 'DE Ratio'])

        # [1:] Removes the root directory from the list
    for each_dir in stock_list[1:]:
            # Gets the name of each file in the directories
        each_file = os.listdir(each_dir)
            # Define the stock ticker
        ticker = each_dir.split('/documents/')[1]
            # Only runs for files that actually have data
        if len(each_file) > 0:
            for file in each_file:
                date_stamp = datetime.strptime(file, '%Y%m%d%H%M%S.html')
                unix_time = time.mktime(date_stamp.timetuple())
                full_file_path = each_dir + '/' + file
                source = open(full_file_path, 'r').read()
                
                try:
                        # Splits the data at the source code segments that come before and after the Total Debt/Equity value to just get the value
                        # Turn it into a float to check the correct data is available, if not it becomes an Exception and gets passed
                    value = float(source.split(gather+':</td><td class="yfnc_tabledata1">')[1].split('</td>')[0])
                    df = df.append({'Date':date_stamp,'Unix':unix_time,'Ticker':ticker, 'DE Ratio':value}, ignore_index = True)
                except Exception as e:
                    pass

        # Clean the file and save as a CSV
    save = gather.replace(' ','').replace('(','').replace(')','').replace('/','') + '.csv'
    print(save)
    df.to_csv(save)

Key_Stats()

