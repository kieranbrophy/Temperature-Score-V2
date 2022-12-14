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

from scipy.interpolate import interp1d

from sray_db.apps import apps

from sray_db.apps.pk import PrimaryKey
import input_scoring_toolbox.loading_tools as lt

from datetime import date
from dateutil.relativedelta import relativedelta

today = date.today()
start = today - relativedelta(days=8)

years = ['2030','2050']
scenarios = ['STEPS','APS','NZE']
scopes = ['1','2','3','12','23','13','123']
    
"""
Config variables
"""

'''
Based on cumulative emissions?
'''
cumul = True

'''
Include  estimates from the Emissions Estimation Model?
'''
inc_ests = True

"""
Load in data
"""

'''
Load in assetid's
'''
assetid_df = lt.get_meta_data(apps['assetinfo_entity'][(1,0,0,0)]).reset_index().rename(columns={PrimaryKey.assetid:"assetid"})

'''
Load disclosed emissions data
'''
em_now = lt.get_app_data(apps['temperature_emissions'][(1, 1, 0, 0)], start, today).reset_index().rename(columns={PrimaryKey.assetid:"assetid"}).drop_duplicates(subset = 'assetid')
em_back_1 = lt.get_app_data(apps['temperature_emissions'][(1, 1, 0, 0)], start - relativedelta(years=1), today - relativedelta(years=1)).reset_index().rename(columns={PrimaryKey.assetid:"assetid"}).drop_duplicates(subset = 'assetid').rename(columns={"em_1": "em_1_1","em_2": "em_2_1","em_3": "em_3_1"})
em_back_2 = lt.get_app_data(apps['temperature_emissions'][(1, 1, 0, 0)], start - relativedelta(years=2), today - relativedelta(years=2)).reset_index().rename(columns={PrimaryKey.assetid:"assetid"}).drop_duplicates(subset = 'assetid').rename(columns={"em_1": "em_1_2","em_2": "em_2_2","em_3": "em_3_2"})

dis_em = em_now.merge(em_back_1[['assetid','em_1_1','em_2_1','em_3_1']], on='assetid', how='inner').merge(em_back_2[['assetid','em_1_2','em_2_2','em_3_2']], on='assetid', how='inner')

'''
Load estimated emissions data
'''
if inc_ests == True:
    est_em_1 = pd.read_csv('estimated_emissions/estimated_scope_1.csv').drop_duplicates(subset = 'assetid')
    est_em_2 = pd.read_csv('estimated_emissions/estimated_scope_2.csv').drop_duplicates(subset = 'assetid')
    est_em_3 = pd.read_csv('estimated_emissions/estimated_scope_3.csv').drop_duplicates(subset = 'assetid')

    all_emissions = dis_em.merge(est_em_1, on='assetid',how='outer').merge(est_em_2, on='assetid',how='outer').merge(est_em_3, on='assetid',how='outer')
    all_emissions['em_1'] = all_emissions['em_1'].fillna(all_emissions['estimated_scope_1_emissions'])
    all_emissions['em_2'] = all_emissions['em_2'].fillna(all_emissions['estimated_scope_2_emissions'])
    all_emissions['em_3'] = all_emissions['em_3'].fillna(all_emissions['estimated_scope_3_emissions'])

else:
    all_emissions = dis_em

'''
Scope 1 and 2
'''
all_emissions['em_12'] = all_emissions['em_1'] + all_emissions['em_2']
all_emissions['em_12_1'] = all_emissions['em_1_1'] + all_emissions['em_2_1']
all_emissions['em_12_2'] = all_emissions['em_1_2'] + all_emissions['em_2_2']

'''
Scope 2 and 3
'''
all_emissions['em_23'] = all_emissions['em_2'] + all_emissions['em_3']
all_emissions['em_23_1'] = all_emissions['em_2_1'] + all_emissions['em_3_1']
all_emissions['em_23_2'] = all_emissions['em_2_2'] + all_emissions['em_3_2']

'''
Scope 1 and 3
'''
all_emissions['em_13'] = all_emissions['em_1'] + all_emissions['em_3']
all_emissions['em_13_1'] = all_emissions['em_1_1'] + all_emissions['em_3_1']
all_emissions['em_13_2'] = all_emissions['em_1_2'] + all_emissions['em_3_2']

'''
Scope 1 and 2 and 3
'''
all_emissions['em_123'] = all_emissions['em_1'] + all_emissions['em_2'] + all_emissions['em_3']
all_emissions['em_123_1'] = all_emissions['em_1_1'] + all_emissions['em_2_1'] + all_emissions['em_3_1']
all_emissions['em_123_2'] = all_emissions['em_1_2'] + all_emissions['em_2_2'] + all_emissions['em_3_2']

'''
Load industry classification
'''
industry_df = lt.get_meta_data(apps['assetinfo_activity'][(1,0,0,0)], columns = ['industry','economic_sector']).reset_index()
ind_df = industry_df.merge(pd.read_csv('weo_to_factset_industry_mapping.csv')[['industry', 'iea_sector', 'iea_scope3']], on='industry', how='outer').rename(columns={PrimaryKey.assetid:"assetid"})

'''
Merge into one dataframe
'''
input_df = assetid_df.merge(ind_df, on='assetid', how='outer').merge(all_emissions, on='assetid', how='inner')

"""
Read in scope reduction targets
"""

STEPS_S1 = pd.read_csv('WEO_scenarios/STEPS/scope1.csv')
APS_S1 = pd.read_csv('WEO_scenarios/APS/scope1.csv')
NZE_S1 = pd.read_csv('WEO_scenarios/NZE/scope1.csv')

STEPS_S2 = pd.read_csv('WEO_scenarios/STEPS/scope2.csv')
APS_S2 = pd.read_csv('WEO_scenarios/APS/scope2.csv')
NZE_S2 = pd.read_csv('WEO_scenarios/NZE/scope2.csv')

STEPS_S3 = pd.read_csv('WEO_scenarios/STEPS/scope3.csv')
APS_S3 = pd.read_csv('WEO_scenarios/APS/scope3.csv')
NZE_S3 = pd.read_csv('WEO_scenarios/NZE/scope3.csv')

'''
Create company specific benchmarks per scope
'''
benchmarks_S1 = input_df[['assetid','industry','iea_sector']].merge(STEPS_S1,on='iea_sector').merge(APS_S1,on='iea_sector').merge(NZE_S1,on='iea_sector')
benchmarks_S2 = input_df[['assetid','industry','iea_sector']].merge(STEPS_S2,on='iea_sector').merge(APS_S2,on='iea_sector').merge(NZE_S2,on='iea_sector')
benchmarks_S3 = input_df[['assetid','industry','iea_scope3']].merge(STEPS_S3,on='iea_scope3').merge(APS_S3,on='iea_scope3').merge(NZE_S3,on='iea_scope3')

benchmarks_df = input_df[['assetid','industry','iea_sector','em_1','em_2','em_3']]

'''
STEPS
'''
benchmarks_df['STEPS_benchmark_2030_em_1'] = benchmarks_df['em_1'] * benchmarks_S1['STEPS_scaling_2030']
benchmarks_df['STEPS_benchmark_2050_em_1'] = benchmarks_df['em_1'] * benchmarks_S1['STEPS_scaling_2050']

benchmarks_df['STEPS_benchmark_2030_em_2'] = benchmarks_df['em_2'] * benchmarks_S2['STEPS_scaling_2030']
benchmarks_df['STEPS_benchmark_2050_em_2'] = benchmarks_df['em_2'] * benchmarks_S2['STEPS_scaling_2050']

benchmarks_df['STEPS_benchmark_2030_em_3'] = benchmarks_df['em_3'] * benchmarks_S3['STEPS_scaling_2030']
benchmarks_df['STEPS_benchmark_2050_em_3'] = benchmarks_df['em_3'] * benchmarks_S3['STEPS_scaling_2050']

benchmarks_df['STEPS_benchmark_2030_em_12'] = benchmarks_df['STEPS_benchmark_2030_em_1'] + benchmarks_df['STEPS_benchmark_2030_em_2']
benchmarks_df['STEPS_benchmark_2050_em_12'] = benchmarks_df['STEPS_benchmark_2050_em_1'] + benchmarks_df['STEPS_benchmark_2050_em_2']

benchmarks_df['STEPS_benchmark_2030_em_23'] = benchmarks_df['STEPS_benchmark_2030_em_2'] + benchmarks_df['STEPS_benchmark_2030_em_3']
benchmarks_df['STEPS_benchmark_2050_em_23'] = benchmarks_df['STEPS_benchmark_2050_em_2'] + benchmarks_df['STEPS_benchmark_2050_em_3']

benchmarks_df['STEPS_benchmark_2030_em_13'] = benchmarks_df['STEPS_benchmark_2030_em_1'] + benchmarks_df['STEPS_benchmark_2030_em_3']
benchmarks_df['STEPS_benchmark_2050_em_13'] = benchmarks_df['STEPS_benchmark_2050_em_1'] + benchmarks_df['STEPS_benchmark_2050_em_3']

benchmarks_df['STEPS_benchmark_2030_em_123'] = benchmarks_df['STEPS_benchmark_2030_em_1'] + benchmarks_df['STEPS_benchmark_2030_em_2'] + benchmarks_df['STEPS_benchmark_2030_em_3']
benchmarks_df['STEPS_benchmark_2050_em_123'] = benchmarks_df['STEPS_benchmark_2050_em_1'] + benchmarks_df['STEPS_benchmark_2050_em_2'] + benchmarks_df['STEPS_benchmark_2050_em_3']

'''
APS
'''
benchmarks_df['APS_benchmark_2030_em_1'] = benchmarks_df['em_1'] * benchmarks_S1['APS_scaling_2030']
benchmarks_df['APS_benchmark_2050_em_1'] = benchmarks_df['em_1'] * benchmarks_S1['APS_scaling_2050']

benchmarks_df['APS_benchmark_2030_em_2'] = benchmarks_df['em_2'] * benchmarks_S2['APS_scaling_2030']
benchmarks_df['APS_benchmark_2050_em_2'] = benchmarks_df['em_2'] * benchmarks_S2['APS_scaling_2050']

benchmarks_df['APS_benchmark_2030_em_3'] = benchmarks_df['em_3'] * benchmarks_S3['APS_scaling_2030']
benchmarks_df['APS_benchmark_2050_em_3'] = benchmarks_df['em_3'] * benchmarks_S3['APS_scaling_2050']

benchmarks_df['APS_benchmark_2030_em_12'] = benchmarks_df['APS_benchmark_2030_em_1'] + benchmarks_df['APS_benchmark_2030_em_2']
benchmarks_df['APS_benchmark_2050_em_12'] = benchmarks_df['APS_benchmark_2050_em_1'] + benchmarks_df['APS_benchmark_2050_em_2']

benchmarks_df['APS_benchmark_2030_em_23'] = benchmarks_df['APS_benchmark_2030_em_2'] + benchmarks_df['APS_benchmark_2030_em_3']
benchmarks_df['APS_benchmark_2050_em_23'] = benchmarks_df['APS_benchmark_2050_em_2'] + benchmarks_df['APS_benchmark_2050_em_3']

benchmarks_df['APS_benchmark_2030_em_13'] = benchmarks_df['APS_benchmark_2030_em_1'] + benchmarks_df['APS_benchmark_2030_em_3']
benchmarks_df['APS_benchmark_2050_em_13'] = benchmarks_df['APS_benchmark_2050_em_1'] + benchmarks_df['APS_benchmark_2050_em_3']

benchmarks_df['APS_benchmark_2030_em_123'] = benchmarks_df['APS_benchmark_2030_em_1'] + benchmarks_df['APS_benchmark_2030_em_2'] + benchmarks_df['APS_benchmark_2030_em_3']
benchmarks_df['APS_benchmark_2050_em_123'] = benchmarks_df['APS_benchmark_2050_em_1'] + benchmarks_df['APS_benchmark_2050_em_2'] + benchmarks_df['APS_benchmark_2050_em_3']

'''
NZE
'''
benchmarks_df['NZE_benchmark_2030_em_1'] = benchmarks_df['em_1'] * benchmarks_S1['NZE_scaling_2030']
benchmarks_df['NZE_benchmark_2050_em_1'] = benchmarks_df['em_1'] * benchmarks_S1['NZE_scaling_2050']

benchmarks_df['NZE_benchmark_2030_em_2'] = benchmarks_df['em_2'] * benchmarks_S2['NZE_scaling_2030']
benchmarks_df['NZE_benchmark_2050_em_2'] = benchmarks_df['em_2'] * benchmarks_S2['NZE_scaling_2050']

benchmarks_df['NZE_benchmark_2030_em_3'] = benchmarks_df['em_3'] * benchmarks_S3['NZE_scaling_2030']
benchmarks_df['NZE_benchmark_2050_em_3'] = benchmarks_df['em_3'] * benchmarks_S3['NZE_scaling_2050']

benchmarks_df['NZE_benchmark_2030_em_12'] = benchmarks_df['NZE_benchmark_2030_em_1'] + benchmarks_df['NZE_benchmark_2030_em_2']
benchmarks_df['NZE_benchmark_2050_em_12'] = benchmarks_df['NZE_benchmark_2050_em_1'] + benchmarks_df['NZE_benchmark_2050_em_2']

benchmarks_df['NZE_benchmark_2030_em_23'] = benchmarks_df['NZE_benchmark_2030_em_2'] + benchmarks_df['NZE_benchmark_2030_em_3']
benchmarks_df['NZE_benchmark_2050_em_23'] = benchmarks_df['NZE_benchmark_2050_em_2'] + benchmarks_df['NZE_benchmark_2050_em_3']

benchmarks_df['NZE_benchmark_2030_em_13'] = benchmarks_df['NZE_benchmark_2030_em_1'] + benchmarks_df['NZE_benchmark_2030_em_3']
benchmarks_df['NZE_benchmark_2050_em_13'] = benchmarks_df['NZE_benchmark_2050_em_1'] + benchmarks_df['NZE_benchmark_2050_em_3']

benchmarks_df['NZE_benchmark_2030_em_123'] = benchmarks_df['NZE_benchmark_2030_em_1'] + benchmarks_df['NZE_benchmark_2030_em_2'] + benchmarks_df['NZE_benchmark_2030_em_3']
benchmarks_df['NZE_benchmark_2050_em_123'] = benchmarks_df['NZE_benchmark_2050_em_1'] + benchmarks_df['NZE_benchmark_2050_em_2'] + benchmarks_df['NZE_benchmark_2050_em_3']

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
    
    input_df['em_3'] = input_df['em_3'].replace(0, np.nan)
    input_df['em_3_1'] = input_df['em_3_1'].replace(0, np.nan)
    input_df['em_3_2'] = input_df['em_3_2'].replace(0, np.nan)
    
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
    input_df['p_change_em_3'] = np.nanmean(input_df[['em_3','em_3_1','em_3_2']].pct_change(axis='columns', fill_method = 'pad',periods=-1),axis = 1)

    return input_df

comp_df = calc_trend(input_df)

'''
Estimated emissions use industry averaged trend
'''
def calc_indtrend(comp_df):
    
    median_indtrend_em_1 = np.nanmedian(comp_df['p_change_em_1'].loc[~(comp_df['p_change_em_1']==0)])
    median_indtrend_em_2 = np.nanmedian(comp_df['p_change_em_2'].loc[~(comp_df['p_change_em_2']==0)])
    median_indtrend_em_3 = np.nanmedian(comp_df['p_change_em_3'].loc[~(comp_df['p_change_em_3']==0)])
    
    return median_indtrend_em_1, median_indtrend_em_2, median_indtrend_em_3
    
comp_df_ind = comp_df.groupby('industry').apply(lambda x: calc_indtrend(x)).reset_index()

comp_df_ind['p_change_ind_em_1'] = np.nan
comp_df_ind['p_change_ind_em_2'] = np.nan
comp_df_ind['p_change_ind_em_3'] = np.nan

for corp in comp_df_ind.index:
    comp_df_ind['p_change_ind_em_1'][corp] = comp_df_ind[0][corp][0]
    comp_df_ind['p_change_ind_em_2'][corp] = comp_df_ind[0][corp][1]
    comp_df_ind['p_change_ind_em_3'][corp] = comp_df_ind[0][corp][2]

comp_df = comp_df.merge(comp_df_ind, on='industry', how='outer')

comp_df['p_change_em_1'].loc[comp_df['p_change_em_1'] == 0] = comp_df['p_change_ind_em_1']
comp_df['p_change_em_2'].loc[comp_df['p_change_em_2'] == 0] = comp_df['p_change_ind_em_2']
comp_df['p_change_em_3'].loc[comp_df['p_change_em_3'] == 0] = comp_df['p_change_ind_em_3']

'''
Apply trend to project emissions in 2030 and 2050
'''
comp_df['em_1_2030'] = comp_df.em_1 * ((comp_df['p_change_em_1'] + 1)**8)
comp_df['em_1_2050'] = comp_df.em_1 * ((comp_df['p_change_em_1'] + 1)**28)

comp_df['em_2_2030'] = comp_df.em_2 * ((comp_df['p_change_em_2'] + 1)**8)
comp_df['em_2_2050'] = comp_df.em_2 * ((comp_df['p_change_em_2'] + 1)**28)

comp_df['em_3_2030'] = comp_df.em_3 * ((comp_df['p_change_em_3'] + 1)**8)
comp_df['em_3_2050'] = comp_df.em_3 * ((comp_df['p_change_em_3'] + 1)**28)

comp_df['em_12_2030'] = comp_df['em_1_2030'] + comp_df['em_2_2030']
comp_df['em_12_2050'] = comp_df['em_1_2050'] + comp_df['em_2_2050']

comp_df['em_23_2030'] = comp_df['em_2_2030'] + comp_df['em_3_2030']
comp_df['em_23_2050'] = comp_df['em_2_2050'] + comp_df['em_3_2050']

comp_df['em_13_2030'] = comp_df['em_1_2030'] + comp_df['em_3_2030']
comp_df['em_13_2050'] = comp_df['em_1_2050'] + comp_df['em_3_2050']

comp_df['em_123_2030'] = comp_df['em_1_2030'] + comp_df['em_2_2030'] + comp_df['em_3_2030']
comp_df['em_123_2050'] = comp_df['em_1_2050'] + comp_df['em_2_2050'] + comp_df['em_3_2050']

'''
Combine into one dataframe
'''
all_df = benchmarks_df[['assetid','industry','iea_sector',
                           'STEPS_benchmark_2030_em_1','STEPS_benchmark_2050_em_1',
                           'STEPS_benchmark_2030_em_2','STEPS_benchmark_2050_em_2',
                           'STEPS_benchmark_2030_em_3','STEPS_benchmark_2050_em_3',
                           'STEPS_benchmark_2030_em_12','STEPS_benchmark_2050_em_12',
                           'STEPS_benchmark_2030_em_23','STEPS_benchmark_2050_em_23',
                           'STEPS_benchmark_2030_em_13','STEPS_benchmark_2050_em_13',
                           'STEPS_benchmark_2030_em_123','STEPS_benchmark_2050_em_123',
                           'APS_benchmark_2030_em_1','APS_benchmark_2050_em_1',
                           'APS_benchmark_2030_em_2','APS_benchmark_2050_em_2',
                           'APS_benchmark_2030_em_3','APS_benchmark_2050_em_3',
                           'APS_benchmark_2030_em_12','APS_benchmark_2050_em_12',
                           'APS_benchmark_2030_em_23','APS_benchmark_2050_em_23',
                           'APS_benchmark_2030_em_13','APS_benchmark_2050_em_13',
                           'APS_benchmark_2030_em_123','APS_benchmark_2050_em_123',
                           'NZE_benchmark_2030_em_1','NZE_benchmark_2050_em_1',
                           'NZE_benchmark_2030_em_2','NZE_benchmark_2050_em_2',
                           'NZE_benchmark_2030_em_3','NZE_benchmark_2050_em_3',
                           'NZE_benchmark_2030_em_12','NZE_benchmark_2050_em_12',
                           'NZE_benchmark_2030_em_23','NZE_benchmark_2050_em_23',
                           'NZE_benchmark_2030_em_13','NZE_benchmark_2050_em_13',
                           'NZE_benchmark_2030_em_123','NZE_benchmark_2050_em_123']].merge(
                               comp_df[['assetid','em_1','em_1_2030','em_1_2050',
                                        'em_2','em_2_2030','em_2_2050',
                                        'em_3','em_3_2030','em_3_2050',
                                        'em_12','em_12_2030','em_12_2050',
                                        'em_23','em_23_2030','em_23_2050',
                                        'em_13','em_13_2030','em_13_2050',
                                        'em_123','em_123_2030','em_123_2050']])

'''
Cumulative emissions
'''

if cumul == True:

    for sco in scopes:
        for corp in all_df.index:
            
            all_df['em_'+ str(sco) +'_2030'][corp] = all_df['em_'+ str(sco) +''][corp]*8 - (((all_df['em_'+ str(sco) +''][corp] - all_df['em_'+ str(sco) +'_2030'][corp])*8)/2)
            all_df['em_'+ str(sco) +'_2050'][corp] = all_df['em_'+ str(sco) +'_2030'][corp] + all_df['em_'+ str(sco) +'_2030'][corp]*20 - (((all_df['em_'+ str(sco) +'_2030'][corp] - all_df['em_'+ str(sco) +'_2050'][corp])*20)/2)
        
            for sce in scenarios:
            
                all_df[''+ str(sce) +'_benchmark_2030_em_'+ str(sco) +''][corp] = all_df['em_'+ str(sco) +''][corp]*8 - (((all_df['em_'+ str(sco) +''][corp] - all_df[''+ str(sce) +'_benchmark_2030_em_'+ str(sco) +''][corp])*8)/2)
                all_df[''+ str(sce) +'_benchmark_2050_em_'+ str(sco) +''][corp] = all_df[''+ str(sce) +'_benchmark_2030_em_'+ str(sco) +''][corp] + all_df[''+ str(sce) +'_benchmark_2030_em_'+ str(sco) +''][corp]*20 - (((all_df[''+ str(sce) +'_benchmark_2030_em_'+ str(sco) +''][corp] - all_df[''+ str(sce) +'_benchmark_2050_em_'+ str(sco) +''][corp])*20)/2)
        
                               
'''
Calculate Implied Rise in Temperature in 2100
'''                           
all_df['temperature_2030_em_1'] = np.nan
all_df['temperature_2050_em_1'] = np.nan

all_df['temperature_2030_em_2'] = np.nan
all_df['temperature_2050_em_2'] = np.nan

all_df['temperature_2030_em_3'] = np.nan
all_df['temperature_2050_em_3'] = np.nan

all_df['temperature_2030_em_12'] = np.nan
all_df['temperature_2050_em_12'] = np.nan

all_df['temperature_2030_em_23'] = np.nan
all_df['temperature_2050_em_23'] = np.nan

all_df['temperature_2030_em_13'] = np.nan
all_df['temperature_2050_em_13'] = np.nan

all_df['temperature_2030_em_123'] = np.nan
all_df['temperature_2050_em_123'] = np.nan

for year in years:
    for sco in scopes:
        for corp in all_df.index:

            x = np.array([all_df['NZE_benchmark_'+ str(year) +'_em_'+ str(sco) +''][corp],all_df['APS_benchmark_'+ str(year) +'_em_'+ str(sco) +''][corp],all_df['STEPS_benchmark_'+ str(year) +'_em_'+ str(sco) +''][corp]])
            y = np.array([1.5, 1.7, 2.5])
            
            int_temp = interp1d(x, y, fill_value="extrapolate")
        
            all_df.loc[corp, 'temperature_'+ str(year) +'_em_'+ str(sco) +''] = int_temp(all_df['em_'+ str(sco) +'_'+ str(year) +''][corp])
        
        '''
        Round to 1 decimal point
        '''
        all_df['temperature_'+ str(year) +'_em_'+ str(sco) +''] = round(all_df['temperature_'+ str(year) +'_em_'+ str(sco) +''], 1)
            
        '''
        As the world has already warmed by 1.1 degrees, this is the minimum temperature
        '''
        all_df['temperature_'+ str(year) +'_em_'+ str(sco) +''].loc[all_df['temperature_'+ str(year) +'_em_'+ str(sco) +''] < 1.1] = 1.1          
     
        '''
        It is very highly unlikely the world will warm by 10 degrees or more
        '''
        all_df['temperature_'+ str(year) +'_em_'+ str(sco) +''].loc[all_df['temperature_'+ str(year) +'_em_'+ str(sco) +''] > 10] = 10  
                       
 
output_df = all_df[['assetid','temperature_2030_em_1','temperature_2050_em_1','temperature_2030_em_2','temperature_2050_em_2',
                    'temperature_2030_em_3','temperature_2050_em_3','temperature_2030_em_12','temperature_2050_em_12',
                    'temperature_2030_em_23','temperature_2050_em_23','temperature_2030_em_13','temperature_2050_em_13',
                    'temperature_2030_em_123','temperature_2050_em_123']]


'''
Plot results
'''
import matplotlib.pyplot as plt

'''
Scope 1
'''
plt.hist(output_df['temperature_2030_em_1'], 100, rwidth=0.7)
plt.hist(output_df['temperature_2050_em_1'], 100, rwidth=0.7)
plt.show()

'''
Scope 2
'''
plt.hist(all_df['temperature_2030_em_2'], 100, rwidth=0.7)
plt.hist(all_df['temperature_2050_em_2'], 100, rwidth=0.7)
plt.show()

'''
Scope 3
'''
plt.hist(output_df['temperature_2030_em_3'], 100, rwidth=0.7)
plt.hist(output_df['temperature_2050_em_3'], 100, rwidth=0.7)
plt.show()

'''
Scope 1 and 2 and 3
'''
plt.hist(all_df['temperature_2030_em_123'], 100, rwidth=0.7)
plt.hist(all_df['temperature_2050_em_123'], 100, rwidth=0.7)
plt.show()


