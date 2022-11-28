#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:22:04 2022

@author: kieran.brophyarabesque.com
"""
from dotenv import load_dotenv
load_dotenv("/Users/kieranbrophy/.env_prod")

import pandas as pd
import numpy as np

import math 
from scipy import stats
from scipy.interpolate import interp1d

from sray_db.apps import apps

from sray_db.apps.pk import PrimaryKey
import input_scoring_toolbox.loading_tools as lt

from datetime import date
from dateutil.relativedelta import relativedelta

today = date.today()
start = today - relativedelta(days=7)

'''
Load emissions data
'''
assetid_df = lt.get_meta_data(apps['assetinfo_entity'][(1,0,0,0)]).reset_index().rename(columns={PrimaryKey.assetid:"assetid"})

em_now = lt.get_app_data(apps['temperature_emissions'][(1, 1, 0, 0)], start, today).reset_index().rename(columns={PrimaryKey.assetid:"assetid"}).drop_duplicates(subset = 'assetid')
em_back_1 = lt.get_app_data(apps['temperature_emissions'][(1, 1, 0, 0)], start - relativedelta(years=1), today - relativedelta(years=1)).reset_index().rename(columns={PrimaryKey.assetid:"assetid"}).drop_duplicates(subset = 'assetid').rename(columns={"em_1": "em_1_1"})
em_back_2 = lt.get_app_data(apps['temperature_emissions'][(1, 1, 0, 0)], start - relativedelta(years=2), today - relativedelta(years=2)).reset_index().rename(columns={PrimaryKey.assetid:"assetid"}).drop_duplicates(subset = 'assetid').rename(columns={"em_1": "em_1_2"})

emissions_df = em_now.merge(em_back_1[['assetid','em_1_1']], on='assetid', how='inner').merge(em_back_2[['assetid','em_1_2']], on='assetid', how='inner')

'''
Load industry classification
'''
industry_df = lt.get_meta_data(apps['assetinfo_activity'][(1,0,0,0)], columns = ['industry','economic_sector']).reset_index()
ind_df = industry_df.merge(pd.read_csv('weo_to_factset_industry_mapping.csv')[['industry', 'iea_sector']], on='industry', how='inner').rename(columns={PrimaryKey.assetid:"assetid"})

STEPS_df = pd.read_csv('WEO_scenarios/STEPS/scope1.csv')
APS_df = pd.read_csv('WEO_scenarios/APS/scope1.csv')
NZE_df = pd.read_csv('WEO_scenarios/NZE/scope1.csv')

'''
Merge into one dataframe
'''
input_df = assetid_df.merge(ind_df, on='assetid', how='outer').merge(emissions_df, on='assetid', how='inner')

'''
Create industry benchmarks
'''
benchmarks_df = input_df[['assetid','industry','iea_sector','em_1']].merge(STEPS_df,on='iea_sector').merge(APS_df,on='iea_sector').merge(NZE_df,on='iea_sector')

benchmarks_df['STEPS_benchmark_2030'] = benchmarks_df.em_1 * benchmarks_df['STEPS_scaling_2030']
benchmarks_df['STEPS_benchmark_2050'] = benchmarks_df.em_1 * benchmarks_df['STEPS_scaling_2050']

benchmarks_df['APS_benchmark_2030'] = benchmarks_df.em_1 * benchmarks_df['APS_scaling_2030']
benchmarks_df['APS_benchmark_2050'] = benchmarks_df.em_1 * benchmarks_df['APS_scaling_2050']

benchmarks_df['NZE_benchmark_2030'] = benchmarks_df.em_1 * benchmarks_df['NZE_scaling_2030']
benchmarks_df['NZE_benchmark_2050'] = benchmarks_df.em_1 * benchmarks_df['NZE_scaling_2050']

'''
Output benchmarks
'''
output_df = benchmarks_df[['assetid','industry','iea_sector','em_1',
                           'STEPS_scaling_2030','STEPS_scaling_2050',
                           'APS_scaling_2030','APS_scaling_2050',
                           'NZE_scaling_2030','NZE_scaling_2050',
                           'STEPS_benchmark_2030','STEPS_benchmark_2050',
                           'APS_benchmark_2030','APS_benchmark_2050',
                           'NZE_benchmark_2030','NZE_benchmark_2050']]

output_df.to_csv('benchmarks_sample.csv')

def calc_trend(input_df):
    """
    Applies function to calculate the trend in weekly emissions over the last 3 years
    
    Parameters
    ----------
        weekly_emissions: dataframe containing weekly emissions data
        var: variable to use to calculate the trend
        
    Returns
    -------
        trend: dataframe containing the percent change in emissions over the previous 3 years
    """    
    
    input_df['em_1'] = input_df['em_1'].replace(0, np.nan)
    input_df['em_1_1'] = input_df['em_1_1'].replace(0, np.nan)
    input_df['em_1_2'] = input_df['em_1_2'].replace(0, np.nan)
    
    trend = input_df.groupby('assetid').apply(get_trend)
        
    return trend


def get_trend(input_df) -> pd.Series:
    """
    Calculates the 3 year running trend in the percent change in emissions
    
    Parameters
    ----------
        gi: timeseries of emissions for a company
        
    Returns
    -------
        gi: series with additional fields of the annual percent change in emissions and the 3 year running mean
    """
        
    input_df['p_change_em'] = np.nanmean(input_df[['em_1','em_1_1','em_1_2']].pct_change(axis='columns', fill_method = 'pad',periods=-1),axis = 1)
    
    return input_df

comp_df = calc_trend(input_df)

'''
Apply trend to project emissions in 2030 and 2050
'''
comp_df['em_2030'] = comp_df.em_1 * ((comp_df['p_change_em'] + 1)**8)
comp_df['em_2050'] = comp_df.em_1 * ((comp_df['p_change_em'] + 1)**28)

'''
Combine into one dataframe
'''
all_df = benchmarks_df[['assetid','industry','iea_sector',
                           'STEPS_benchmark_2030','STEPS_benchmark_2050',
                           'APS_benchmark_2030','APS_benchmark_2050',
                           'NZE_benchmark_2030','NZE_benchmark_2050']].merge(
                               comp_df[['assetid','em_1','em_2030','em_2050']])

'''
Calculate Implied Rise in Temperature in 2100
'''                           
all_df['temperature_2030'] = np.nan
all_df['temperature_2050'] = np.nan

'''
2030
'''
for corp in all_df.index[np.isnan(all_df.em_1) == False]:

        x = np.array([all_df['NZE_benchmark_2030'][corp],all_df['APS_benchmark_2030'][corp],all_df['STEPS_benchmark_2030'][corp]])
        y = np.array([1.5, 1.7, 2.5])
            
        int_temp = interp1d(x, y, fill_value="extrapolate")
        
        all_df['temperature_2030'][corp] = int_temp(all_df['em_2030'][corp])
                               
'''
2050
'''                               
for corp in all_df.index[np.isnan(all_df.em_1) == False]:

        x = np.array([all_df['NZE_benchmark_2050'][corp],all_df['APS_benchmark_2050'][corp],all_df['STEPS_benchmark_2050'][corp]])
        y = np.array([1.5, 1.7, 2.5])
                    
        int_temp = interp1d(x, y, fill_value="extrapolate")
        
        all_df['temperature_2050'][corp] = int_temp(all_df['em_2050'][corp])       

all_df['temperature_2030'] = round(all_df['temperature_2030'], 1)
all_df['temperature_2050'] = round(all_df['temperature_2050'], 1)

'''
As the world has already warmed by 1.1 degrees, this is the minimum temperature
'''
all_df['temperature_2030'].loc[all_df['temperature_2030'] < 1.1] = 1.1               
all_df['temperature_2050'].loc[all_df['temperature_2050'] < 1.1] = 1.1                  
                               
                               
'''
As the world has already warmed by 1.1 degrees, this is the minimum temperature
'''
all_df['temperature_2030'].loc[all_df['temperature_2030'] > 10] = 10              
all_df['temperature_2050'].loc[all_df['temperature_2050'] > 10] = 10                         

