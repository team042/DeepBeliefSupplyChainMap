# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 11:09:36 2022

@author: Jacob Harris
"""
import pandas as pd
import requests
from geopy.distance import geodesic

def get_conflicts():
    conflicts = pd.DataFrame(columns={'id', 'relid', 'year', 'active_year', 'code_status', 'type_of_violence', 
                                      'conflict_dset_id', 'conflict_new_id', 'conflict_name', 'dyad_dset_id', 
                                      'dyad_new_id', 'dyad_name', 'side_a_dset_id', 'side_a_new_id', 'side_a', 
                                      'side_b_dset_id', 'side_b_new_id', 'side_b', 'number_of_sources', 'source_article', 
                                      'source_office', 'source_date', 'source_headline', 'source_original', 'where_prec', 
                                      'where_coordinates', 'where_description', 'adm_1', 'adm_2', 'latitude', 'longitude', 
                                      'geom_wkt', 'priogrid_gid', 'country', 'country_id', 'region', 'event_clarity', 
                                      'date_prec', 'date_start', 'date_end', 'deaths_a', 'deaths_b', 'deaths_civilians', 
                                      'deaths_unknown', 'best', 'high', 'low', 'gwnoa', 'gwnob'})
    
    url = "https://ucdpapi.pcr.uu.se/api/gedevents/21.1?pagesize=1000&page=0"

    while url != "":
        data = requests.get(url)
        jsondata = data.json()
        url = jsondata["NextPageUrl"]
        new_conflicts = pd.DataFrame(jsondata["Result"])
        conflicts = conflicts.append(new_conflicts)
    
    conflicts['date_start'] = pd.to_datetime(conflicts['date_start'])
    conflicts['date_end'] = pd.to_datetime(conflicts['date_end'])
    
    centers = conflicts.groupby(['dyad_name']).agg({'latitude':'mean', 'longitude':'mean'})
    conflicts = conflicts.merge(centers, how='inner', on='dyad_name')
    
    conflicts['center_distance_from_ind_event'] = conflicts.apply(lambda row: geodesic((row['latitude_x'], row['longitude_x']),
                                                                                       (row['latitude_y'], row['longitude_y'])).miles, axis=1)
    
    conflicts_final = conflicts.groupby(['dyad_name', 'type_of_violence'], as_index=False).agg({'latitude_y': 'first', 
                                                        'longitude_y': 'first',
                                                        'center_distance_from_ind_event': 'var',
                                                        'deaths_a': 'sum',
                                                        'deaths_b': 'sum',
                                                        'deaths_civilians': 'sum',
                                                        'deaths_unknown': 'sum',
                                                        'date_start': 'first',
                                                        'date_end': 'last',
                                                        'country': 'first',
                                                        'region': 'first'})
    
    conflicts_final['conflict_duration_days'] = conflicts_final['date_end'] - conflicts_final['date_start']
    conflicts_final['conflict_duration_days'] = conflicts_final['conflict_duration_days'].dt.days.astype('int16')
    conflicts_final['scale'] = conflicts_final.apply(lambda row: 1.96 * row['center_distance_from_ind_event'], axis=1)
    conflicts_final[['center_distance_from_ind_event', 'scale']] = conflicts_final[['center_distance_from_ind_event', 'scale']].fillna(1)
    conflicts_final.loc[conflicts_final["type_of_violence"] == 1, "type_of_violence"] = 'State-Based Violence'
    conflicts_final.loc[conflicts_final["type_of_violence"] == 2, "type_of_violence"] = 'Non-State Violence'
    conflicts_final.loc[conflicts_final["type_of_violence"] == 3, "type_of_violence"] = 'One-Sided Violence'
    conflicts_final['Total_deaths'] = conflicts_final['deaths_a'] + conflicts_final['deaths_b']
    conflicts_final = conflicts_final.rename(columns={'latitude_y': 'latitude', 'longitude_y': 'longitude'})
    conflicts_final = conflicts_final[['dyad_name', 'latitude', 'longitude', 'Total_deaths', 
                                   'deaths_civilians', 'deaths_unknown', 'type_of_violence', 'date_start', 'date_end',
                                   'country', 'region', 'conflict_duration_days', 'scale']]
    
    conflicts_final.to_csv('events.csv', index=False)
