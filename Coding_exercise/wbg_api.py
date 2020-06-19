import requests 
import pandas as pd
import numpy as np
import json
import time


def get_indicator_res(reqs):
    """
    Helper function to extract data from API response and returns a dataframe
    """
    df = pd.DataFrame()
    for ids in reqs:
        idx = ids['indicator']['id']
        df_con = pd.DataFrame(index=[idx])
        df_con.loc[idx, 'iso3'] = ids['countryiso3code']
        df_con.loc[idx, 'country'] = ids['country']['value']
        df_con.loc[idx, 'date'] = ids['date']
        df_con.loc[idx, 'value'] = ids['value']

        df = pd.concat([df, df_con])
    return df


def ind_data(reeq, indicator):
    """
    Helper function to get indicator data
    """
    master_df = pd.DataFrame() 
    
    for i in range(reeq.json()[0]['pages']):
            url = "http://api.worldbank.org/v2/country/all/indicator/{}?format=json&per_page={}&page={}".format(indicator, per_page, i+1)
            r = requests.get(url)

            if r.status_code == 200:
                print("{} of {} pages processed.".format(r.json()[0]['page'],
                                                        r.json()[0]['pages']))
                res = r.json()[1]

                df = get_indicator_res(res)

                master_df = pd.concat([master_df, df])

                time.sleep(10)
            elif r.status_code == 502:
                time.sleep(10)
                r = retry(url)
                try:
                    print("{} of {} pages processed.".format(r.json()[0]['page'],
                                                        r.json()[0]['pages']))
                    res = r.json()[1]

                    df = get_indicator_res(res)

                    master_df = pd.concat([master_df, df])
                except AttributeError:
                    time.sleep(10)
                    r = retry(url)
                    print("{} of {} pages processed.".format(r.json()[0]['page'],
                                                        r.json()[0]['pages']))
                    res = r.json()[1]

                    df = get_indicator_res(res)

                    master_df = pd.concat([master_df, df])
                    

            else:
                print(i+1, r.status_code)
        
    return master_df


def retry(url):
    
    reeq = requests.get(url)
    
    if reeq.status_code == 200:
        return reeq
    elif reeq.status_code == 502:
        time.sleep(10)
        retry(url)
    else:
        print("Ran into 502 error agin. Please wait for a while and re-run the script.")


def get_indicator_data(indicator):
    """
    Function to get indicator data from World Bank API for all years and all countries/aggregates. 
    
    Input:
    indicator: indicator code for the dataset you want to download
    
    Returns:
    Raw data as returned by WBG API
    """
    master = pd.DataFrame()
    
    #iso3, cid, country, date, value = [], [], [], [], []

    url = "http://api.worldbank.org/v2/country/all/indicator/{}?format=json&per_page={}".format(indicator, per_page)
    reeq = requests.get(url)
    
    if reeq.status_code == 200:
        df = ind_data(reeq, indicator)
        master = pd.concat([master, df])
    
    elif reeq.status_code == 502:
        reeq = retry(url)
        df = ind_data(reeq, indicator)
        master = pd.concat([master, df])
    else:
        print(i+1, reeq.staus_code)
    
    master['date_new'] = ['YR'+i for i in master.date]
    return master




def get_country_data(reqs):
    """
    Helper function to extract data from API response and returns a dataframe
    """
    
    df = pd.DataFrame()
    for rows in reqs:
        df_con = pd.DataFrame(index=[rows['id']])
        df_con.loc[rows['id'], 'Country'] = rows['name']
        df_con.loc[rows['id'], 'Region_id'] = rows['region']['id']
        df_con.loc[rows['id'], 'Region_name'] = rows['region']['value']
        df_con.loc[rows['id'], 'incomeLevel'] = rows['incomeLevel']['id']
        df_con.loc[rows['id'], 'lendingType'] = rows['lendingType']['id']
        df_con.loc[rows['id'], 'lon'] = rows['longitude']
        df_con.loc[rows['id'], 'lat'] = rows['latitude']
        df = pd.concat([df, df_con])
    return df






def get_country_df():
    """
    Function to get a list of all countries and aggreegates along with their income level and lending type.
    
    Returns:
    A dataframe with all information returned by countries/all API
    """
    master_con = pd.DataFrame()
    url_con = "https://api.worldbank.org/v2/country/all?format=json&per_page=100"
    r_con = requests.get(url_con)

    for i in range(r_con.json()[0]['pages']):
        url_con = "https://api.worldbank.org/v2/country/all?format=json&per_page=100&page={}".format(i+1)
        req = requests.get(url_con)

        if req.status_code == 200:
            reqs = req.json()[1]

            df_con = get_country_data(reqs)

            master_con = pd.concat([master_con, df_con])

        elif req.status_code == 502:
            time.sleep(10)

            url_con = "https://api.worldbank.org/v2/country/all?format=json&per_page=100&page={}".format(i+1)
            req = requests.get(url_con)

            if req.status_code == 200:
                reqs = req.json()[1]

                df_con = get_country_data(reqs)

                master_con = pd.concat([master_con, df_con])

        else:
            print(i+1, req.status_code)
        
    return master_con

    




def main(indicator):
    """
    Input:
    indicator : indicator code for the dataset you want to download
    
    Returns: None
    
    Saves 3 files for country level data, aggregarte level data and complete country list. 
    """
    global per_page
    per_page = 800                  #### Change this number if you get 502 errors (choose a lower number)
    
    
    df = get_indicator_data(indicator)
    cols = df.date_new.unique().tolist()
    ind = df.iso3.unique().tolist()
    dd = pd.DataFrame(columns=cols, index=ind)
    
    for iso in dd.index:
        for year in dd.columns:
            dd.loc[iso, year] = df.query("iso3 == '{}' and date_new == '{}'".format(iso, year))['value'].iloc[0]
    
    for iso in dd.index:
        dd.loc[iso, 'Country'] = df.query("iso3 == '{}'".format(iso))['country'].iloc[0]

    master_con = get_country_df()
    
    dd_con = dd[~dd.index.isin(master_con[master_con.Region_name == 'Aggregates'].index.tolist())]
    
    dd_agg = dd[dd.index.isin(master_con[master_con.Region_name == 'Aggregates'].index.tolist())]
    
    dd_con.to_csv("Sanitation_fac_nonAgg.csv")
    
    dd_agg.to_csv("Sanitation_fac_Agg.csv")
    
    master_con.to_csv("master_country.csv")
    
    return None
    
 
    
if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
