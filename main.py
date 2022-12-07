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

assetid_df = lt.get_meta_data(apps['assetinfo_entity'][(1,0,0,0)]).reset_index().rename(columns={PrimaryKey.assetid:"assetid"})

'''
Config variables
'''

'''
Include  estimates from the Emissions Estimation Model?
'''
inc_ests = True

'''
Load disclosed emissions data
'''
em_now = lt.get_app_data(apps['temperature_emissions'][(1, 1, 0, 0)], start, today).reset_index().rename(columns={PrimaryKey.assetid:"assetid"}).drop_duplicates(subset = 'assetid')
em_back_1 = lt.get_app_data(apps['temperature_emissions'][(1, 1, 0, 0)], start - relativedelta(years=1), today - relativedelta(years=1)).reset_index().rename(columns={PrimaryKey.assetid:"assetid"}).drop_duplicates(subset = 'assetid').rename(columns={"em_1": "em_1_1","em_2": "em_2_1"})
em_back_2 = lt.get_app_data(apps['temperature_emissions'][(1, 1, 0, 0)], start - relativedelta(years=2), today - relativedelta(years=2)).reset_index().rename(columns={PrimaryKey.assetid:"assetid"}).drop_duplicates(subset = 'assetid').rename(columns={"em_1": "em_1_2","em_2": "em_2_2"})

dis_em = em_now.merge(em_back_1[['assetid','em_1_1','em_2_1']], on='assetid', how='inner').merge(em_back_2[['assetid','em_1_2','em_2_2']], on='assetid', how='inner')

'''
Load estimated emissions data
'''
if inc_ests == True:
    est_em_1 = pd.read_csv('estimated_emissions/estimated_scope_1.csv').drop_duplicates(subset = 'assetid')
    est_em_2 = pd.read_csv('estimated_emissions/estimated_scope_2.csv').drop_duplicates(subset = 'assetid')

    all_emissions = dis_em.merge(est_em_1, on='assetid',how='outer').merge(est_em_2, on='assetid',how='outer')
    all_emissions['em_1'] = all_emissions['em_1'].fillna(all_emissions['estimated_scope_1_emissions'])
    all_emissions['em_2'] = all_emissions['em_2'].fillna(all_emissions['estimated_scope_2_emissions'])

else:
    all_emissions = dis_em
    
all_emissions['em_12'] = all_emissions['em_1'] + all_emissions['em_2']
all_emissions['em_12_1'] = all_emissions['em_1_1'] + all_emissions['em_2_1']
all_emissions['em_12_2'] = all_emissions['em_1_2'] + all_emissions['em_2_2']

'''
Load industry classification
'''
industry_df = lt.get_meta_data(apps['assetinfo_activity'][(1,0,0,0)], columns = ['industry','economic_sector']).reset_index()
ind_df = industry_df.merge(pd.read_csv('weo_to_factset_industry_mapping.csv')[['industry', 'iea_sector']], on='industry', how='outer').rename(columns={PrimaryKey.assetid:"assetid"})

'''
Merge into one dataframe
'''
input_df = assetid_df.merge(ind_df, on='assetid', how='outer').merge(all_emissions, on='assetid', how='inner')

'''
Read in scope reduction targets
'''
STEPS_S1 = pd.read_csv('WEO_scenarios/STEPS/scope1.csv')
APS_S1 = pd.read_csv('WEO_scenarios/APS/scope1.csv')
NZE_S1 = pd.read_csv('WEO_scenarios/NZE/scope1.csv')

STEPS_S2 = pd.read_csv('WEO_scenarios/STEPS/scope2.csv')
APS_S2 = pd.read_csv('WEO_scenarios/APS/scope2.csv')
NZE_S2 = pd.read_csv('WEO_scenarios/NZE/scope2.csv')

'''
Create company specific benchmarks per scope
'''
benchmarks_S1 = input_df[['assetid','industry','iea_sector']].merge(STEPS_S1,on='iea_sector').merge(APS_S1,on='iea_sector').merge(NZE_S1,on='iea_sector')
benchmarks_S2 = input_df[['assetid','industry','iea_sector']].merge(STEPS_S2,on='iea_sector').merge(APS_S2,on='iea_sector').merge(NZE_S2,on='iea_sector')

benchmarks_df = input_df[['assetid','industry','iea_sector','em_1','em_2']]
benchmarks_df['em_12'] = benchmarks_df['em_1'] + benchmarks_df['em_2']
'''
STEPS
'''
benchmarks_df['STEPS_benchmark_2030_em_1'] = benchmarks_df['em_1'] * benchmarks_S1['STEPS_scaling_2030']
benchmarks_df['STEPS_benchmark_2050_em_1'] = benchmarks_df['em_1'] * benchmarks_S1['STEPS_scaling_2050']

benchmarks_df['STEPS_benchmark_2030_em_2'] = benchmarks_df['em_2'] * benchmarks_S2['STEPS_scaling_2030']
benchmarks_df['STEPS_benchmark_2050_em_2'] = benchmarks_df['em_2'] * benchmarks_S2['STEPS_scaling_2050']

benchmarks_df['STEPS_benchmark_2030_em_12'] = benchmarks_df['STEPS_benchmark_2030_em_1'] + benchmarks_df['STEPS_benchmark_2030_em_2']
benchmarks_df['STEPS_benchmark_2050_em_12'] = benchmarks_df['STEPS_benchmark_2050_em_1'] + benchmarks_df['STEPS_benchmark_2050_em_2']
'''
APS
'''
benchmarks_df['APS_benchmark_2030_em_1'] = benchmarks_df['em_1'] * benchmarks_S1['APS_scaling_2030']
benchmarks_df['APS_benchmark_2050_em_1'] = benchmarks_df['em_1'] * benchmarks_S1['APS_scaling_2050']

benchmarks_df['APS_benchmark_2030_em_2'] = benchmarks_df['em_2'] * benchmarks_S2['APS_scaling_2030']
benchmarks_df['APS_benchmark_2050_em_2'] = benchmarks_df['em_2'] * benchmarks_S2['APS_scaling_2050']

benchmarks_df['APS_benchmark_2030_em_12'] = benchmarks_df['APS_benchmark_2030_em_1'] + benchmarks_df['APS_benchmark_2030_em_2']
benchmarks_df['APS_benchmark_2050_em_12'] = benchmarks_df['APS_benchmark_2050_em_1'] + benchmarks_df['APS_benchmark_2050_em_2']

'''
NZE
'''
benchmarks_df['NZE_benchmark_2030_em_1'] = benchmarks_df['em_1'] * benchmarks_S1['NZE_scaling_2030']
benchmarks_df['NZE_benchmark_2050_em_1'] = benchmarks_df['em_1'] * benchmarks_S1['NZE_scaling_2050']

benchmarks_df['NZE_benchmark_2030_em_2'] = benchmarks_df['em_2'] * benchmarks_S2['NZE_scaling_2030']
benchmarks_df['NZE_benchmark_2050_em_2'] = benchmarks_df['em_2'] * benchmarks_S2['NZE_scaling_2050']

benchmarks_df['NZE_benchmark_2030_em_12'] = benchmarks_df['NZE_benchmark_2030_em_1'] + benchmarks_df['NZE_benchmark_2030_em_2']
benchmarks_df['NZE_benchmark_2050_em_12'] = benchmarks_df['NZE_benchmark_2050_em_1'] + benchmarks_df['NZE_benchmark_2050_em_2']

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
    
    input_df['em_2'] = input_df['em_2'].replace(0, np.nan)
    input_df['em_2_1'] = input_df['em_2_1'].replace(0, np.nan)
    input_df['em_2_2'] = input_df['em_2_2'].replace(0, np.nan)
    
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
        
    input_df['p_change_em_1'] = np.nanmean(input_df[['em_1','em_1_1','em_1_2']].pct_change(axis='columns', fill_method = 'pad',periods=-1),axis = 1)
    input_df['p_change_em_2'] = np.nanmean(input_df[['em_2','em_2_1','em_2_2']].pct_change(axis='columns', fill_method = 'pad',periods=-1),axis = 1)
    
    return input_df

comp_df = calc_trend(input_df)

'''
Estimated emissions use industry averaged trend
'''
def calc_indtrend(comp_df):
    
    median_indtrend_em_1 = np.nanmedian(comp_df['p_change_em_1'].loc[~(comp_df['p_change_em_1']==0)])
    median_indtrend_em_2 = np.nanmedian(comp_df['p_change_em_2'].loc[~(comp_df['p_change_em_2']==0)])
    
    return median_indtrend_em_1, median_indtrend_em_2
    
comp_df_ind = comp_df.groupby('industry').apply(lambda x: calc_indtrend(x)).reset_index()

comp_df_ind['p_change_ind_em_1'] = np.nan
comp_df_ind['p_change_ind_em_2'] = np.nan

for corp in comp_df_ind.index:
    comp_df_ind['p_change_ind_em_1'][corp] = comp_df_ind[0][corp][0]
    comp_df_ind['p_change_ind_em_2'][corp] = comp_df_ind[0][corp][1]

comp_df = comp_df.merge(comp_df_ind, on='industry', how='outer')

comp_df['p_change_em_1'].loc[comp_df['p_change_em_1'] == 0] = comp_df['p_change_ind_em_1']
comp_df['p_change_em_2'].loc[comp_df['p_change_em_2'] == 0] = comp_df['p_change_ind_em_2']

'''
Apply trend to project emissions in 2030 and 2050
'''
comp_df['em_1_2030'] = comp_df.em_1 * ((comp_df['p_change_em_1'] + 1)**8)
comp_df['em_1_2050'] = comp_df.em_1 * ((comp_df['p_change_em_1'] + 1)**28)

comp_df['em_2_2030'] = comp_df.em_2 * ((comp_df['p_change_em_2'] + 1)**8)
comp_df['em_2_2050'] = comp_df.em_2 * ((comp_df['p_change_em_2'] + 1)**28)

comp_df['em_12_2030'] = comp_df['em_1_2030'] + comp_df['em_2_2030']
comp_df['em_12_2050'] = comp_df['em_1_2050'] + comp_df['em_2_2050']

'''
Combine into one dataframe
'''
all_df = benchmarks_df[['assetid','industry','iea_sector',
                           'STEPS_benchmark_2030_em_1','STEPS_benchmark_2050_em_1',
                           'STEPS_benchmark_2030_em_2','STEPS_benchmark_2050_em_2',
                           'STEPS_benchmark_2030_em_12','STEPS_benchmark_2050_em_12',
                           'APS_benchmark_2030_em_1','APS_benchmark_2050_em_1',
                           'APS_benchmark_2030_em_2','APS_benchmark_2050_em_2',
                           'APS_benchmark_2030_em_12','APS_benchmark_2050_em_12',
                           'NZE_benchmark_2030_em_1','NZE_benchmark_2050_em_1',
                           'NZE_benchmark_2030_em_2','NZE_benchmark_2050_em_2',
                           'NZE_benchmark_2030_em_12','NZE_benchmark_2050_em_12']].merge(
                               comp_df[['assetid','em_1','em_1_2030','em_1_2050',
                                        'em_2','em_2_2030','em_2_2050',
                                        'em_12','em_12_2030','em_12_2050']])

'''
Calculate Implied Rise in Temperature in 2100
'''                           
all_df['temperature_2030_em_1'] = np.nan
all_df['temperature_2050_em_1'] = np.nan

all_df['temperature_2030_em_2'] = np.nan
all_df['temperature_2050_em_2'] = np.nan

all_df['temperature_2030_em_12'] = np.nan
all_df['temperature_2050_em_12'] = np.nan

'''
2030
'''
for corp in all_df.index[np.isnan(all_df.em_1) == False]:

        '''
        Scope 1
        '''
        x_1 = np.array([all_df['NZE_benchmark_2030_em_1'][corp],all_df['APS_benchmark_2030_em_1'][corp],all_df['STEPS_benchmark_2030_em_1'][corp]])
        y_1 = np.array([1.5, 1.7, 2.5])
            
        int_temp_1 = interp1d(x_1, y_1, fill_value="extrapolate")
        
        all_df['temperature_2030_em_1'][corp] = int_temp_1(all_df['em_1_2030'][corp])
        
        '''
        Scope 2
        '''
        x_2 = np.array([all_df['NZE_benchmark_2030_em_2'][corp],all_df['APS_benchmark_2030_em_2'][corp],all_df['STEPS_benchmark_2030_em_2'][corp]])
        y_2 = np.array([1.5, 1.7, 2.5])
            
        int_temp_2 = interp1d(x_2, y_2, fill_value="extrapolate")
        
        all_df['temperature_2030_em_2'][corp] = int_temp_2(all_df['em_2_2030'][corp])
        
        '''
        Scope 1 and 2
        '''
        x_12 = np.array([all_df['NZE_benchmark_2030_em_12'][corp],all_df['APS_benchmark_2030_em_12'][corp],all_df['STEPS_benchmark_2030_em_12'][corp]])
        y_12 = np.array([1.5, 1.7, 2.5])
            
        int_temp_12 = interp1d(x_12, y_12, fill_value="extrapolate")
        
        all_df['temperature_2030_em_12'][corp] = int_temp_12(all_df['em_12_2030'][corp])
                               
'''
2050
'''                               
for corp in all_df.index[np.isnan(all_df.em_1) == False]:

        '''
        Scope 1
        '''
        x_1 = np.array([all_df['NZE_benchmark_2050_em_1'][corp],all_df['APS_benchmark_2050_em_1'][corp],all_df['STEPS_benchmark_2050_em_1'][corp]])
        y_1 = np.array([1.5, 1.7, 2.5])
                    
        int_temp_1 = interp1d(x_1, y_1, fill_value="extrapolate")
        
        all_df['temperature_2050_em_1'][corp] = int_temp_1(all_df['em_1_2050'][corp])  
        
        '''
        Scope 2
        '''
        x_2 = np.array([all_df['NZE_benchmark_2050_em_2'][corp],all_df['APS_benchmark_2050_em_2'][corp],all_df['STEPS_benchmark_2050_em_2'][corp]])
        y_2 = np.array([1.5, 1.7, 2.5])
                    
        int_temp_2 = interp1d(x_2, y_2, fill_value="extrapolate")
        
        all_df['temperature_2050_em_2'][corp] = int_temp_2(all_df['em_2_2050'][corp]) 
        
        '''
        Scope 1 and 2
        '''
        x_12 = np.array([all_df['NZE_benchmark_2050_em_12'][corp],all_df['APS_benchmark_2050_em_12'][corp],all_df['STEPS_benchmark_2050_em_12'][corp]])
        y_12 = np.array([1.5, 1.7, 2.5])
                    
        int_temp_12 = interp1d(x_12, y_12, fill_value="extrapolate")
        
        all_df['temperature_2050_em_12'][corp] = int_temp_12(all_df['em_12_2050'][corp])
        

all_df['temperature_2030_em_1'] = round(all_df['temperature_2030_em_1'], 1)
all_df['temperature_2050_em_1'] = round(all_df['temperature_2050_em_1'], 1)

all_df['temperature_2030_em_2'] = round(all_df['temperature_2030_em_2'], 1)
all_df['temperature_2050_em_2'] = round(all_df['temperature_2050_em_2'], 1)

all_df['temperature_2030_em_12'] = round(all_df['temperature_2030_em_12'], 1)
all_df['temperature_2050_em_12'] = round(all_df['temperature_2050_em_12'], 1)

'''
As the world has already warmed by 1.1 degrees, this is the minimum temperature
'''
all_df['temperature_2030_em_1'].loc[all_df['temperature_2030_em_1'] < 1.1] = 1.1               
all_df['temperature_2050_em_1'].loc[all_df['temperature_2050_em_1'] < 1.1] = 1.1                  
 
all_df['temperature_2030_em_2'].loc[all_df['temperature_2030_em_2'] < 1.1] = 1.1               
all_df['temperature_2050_em_2'].loc[all_df['temperature_2050_em_2'] < 1.1] = 1.1                                 
        
all_df['temperature_2030_em_12'].loc[all_df['temperature_2030_em_12'] < 1.1] = 1.1               
all_df['temperature_2050_em_12'].loc[all_df['temperature_2050_em_12'] < 1.1] = 1.1   
                       
'''
As the world has already warmed by 1.1 degrees, this is the minimum temperature
'''
all_df['temperature_2030_em_1'].loc[all_df['temperature_2030_em_1'] > 10] = 10              
all_df['temperature_2050_em_1'].loc[all_df['temperature_2050_em_1'] > 10] = 10                         

all_df['temperature_2030_em_2'].loc[all_df['temperature_2030_em_2'] > 10] = 10              
all_df['temperature_2050_em_2'].loc[all_df['temperature_2050_em_2'] > 10] = 10   

all_df['temperature_2030_em_12'].loc[all_df['temperature_2030_em_12'] > 10] = 10              
all_df['temperature_2050_em_12'].loc[all_df['temperature_2050_em_12'] > 10] = 10   

'''
Plot results
'''
import matplotlib.pyplot as plt

'''
Scope 1
'''
plt.scatter(all_df['assetid'],all_df['temperature_2030_em_1'], alpha=0.75, s=10)
plt.show()

plt.hist(all_df['temperature_2030_em_1'], 100, rwidth=0.7)
plt.hist(all_df['temperature_2050_em_1'], 100, rwidth=0.7)
plt.show()

'''
Scope 2
'''
plt.scatter(all_df['assetid'],all_df['temperature_2030_em_2'], alpha=0.75, s=10)
plt.show()

plt.hist(all_df['temperature_2030_em_2'], 100, rwidth=0.7)
plt.hist(all_df['temperature_2050_em_2'], 100, rwidth=0.7)
plt.show()

'''
Scope 1 and 2
'''
plt.scatter(all_df['assetid'],all_df['temperature_2030_em_12'], alpha=0.75, s=10)
plt.show()

plt.hist(all_df['temperature_2030_em_12'], 100, rwidth=0.7)
plt.hist(all_df['temperature_2050_em_12'], 100, rwidth=0.7)
plt.show()

'''
Compare scopes
'''
plt.hist(all_df['temperature_2050_em_1'], 100, rwidth=0.7)
plt.hist(all_df['temperature_2050_em_2'], 100, rwidth=0.7)
plt.hist(all_df['temperature_2050_em_12'], 100, rwidth=0.7)
plt.show()



