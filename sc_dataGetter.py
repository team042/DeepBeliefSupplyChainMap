"""
@author: Seth Parker (sparker33)
"""

# Need these libraries
import pandas as pd
import numpy as np
import geopandas as gpd
from geopy.distance import geodesic
import networkx as nx

import requests
import time
import os
import json

# Need these static route data files; they are provided in the Git repo
BORDERS = ".\\Data\\EEZ_land_union_v3_202003\\EEZ_Land_v3_202030.shp"
ROADS = ".\\Data\\ne_10m_roads\\ne_10m_roads.shp"
RAILS = ".\\Data\\ne_10m_railroads\\ne_10m_railroads.shp"
CITIES = ".\\Data\\simplemaps_worldcities_basicv1.75\\worldcities.csv"
EUROSTATS = ".\\Data\\Eurostat_Data.csv"
HS2_KEY = ".\\Data\\HS2_Codes.csv"
COUNTRY_CODES = ".\\Data\\country_codes.csv"

class sc_dataGetter:
    """Class to collect, update, and write out supply chain data. The minimum functional use case is to call sc_dataGetter.build_data_files(period_range),
        passing a list of string years (ex. period_range=['2019','2018','2016']) to get trade data for. This will save a csv with all trade and route data
        for the indicated years in the current directory.
    
    Attributes
    ----------
    period_mode : str
        String indicating whether the current_period is a year (period_mode == 'YEAR'), quarter (period_mode == 'QUARTER'), or month (period_mode == 'MONTH')
    current_period : str
        Time period of data currently held in trade dataframe, formatted as YYYY (period_mode == 'YEAR'), YYYYQ (period_mode == 'QUARTER'), or YYYYMM (period_mode == 'MONTH')

    Methods
    -------
    build_routes() -> None
        Loads in all static route data and builds geographic routes
    update_trade(period: str=None) -> None
        Retreives trade quantities data for the indicated time period and updates internal class data. Updates current_period if applicable.
    write_data(fname: str='.\\')
        Writes current supply chain routes data to a csv file at the provided location
    """

    # Locations of canal ends as lat/lng pairs
    CANAL_COUNTRIES = ['PAN','EGY']
    CANAL_TOL = 15 # miles
    PANAMA_N = (9.4,-79.9)
    PANAMA_S = (8.9,-79.5)
    SUEZ_N = (30.4,32.3)
    SUEZ_S = (29.9,32.5)
    # Snap distance tolerance for cities to roadmap
    SNAP_TOL = 10 # miles

    def __init__(self, period_mode: str='YEAR', current_period: str='2019') -> None:
        """
        Paramters
        ---------
        period_mode: str, optional
            Initializes use of yearly (YEAR), quarterly (QUARTER), or monthly (MONTH) trade data (default YEAR)
        current_period: str, optional
            Initializes period for trade data (default 2021); must match format indicated by period_mode
        """

        self.period_mode = period_mode
        self.current_period = current_period
        self._routes_data = pd.DataFrame()
        self.sc_data = pd.DataFrame()
        self._country_codes = pd.DataFrame()
        self._commodities = pd.DataFrame()
        self._mode_wts = pd.DataFrame()
        self._import_country_codes()
        self._import_commodities()
        self._import_mode_wts()
    
    
    def build_data_files(self, period_range: list, out_file_path: str=".\\sc_data.csv") -> None:
        """Assembles all supply chain route and quantities data for the indicated list of periods, saving the result to a csv file
        
        Paramters
        ---------
        period_range: list (of str)
            Time period for retreived trade data; must match format indicated by current period_mode (default uses value of current_period)
        out_file_path: str, optional
            File path to write output data to; default is current directory, file name sc_data.csv
        """
        master_df = pd.DataFrame()
        # Iterate over all periods, retreiving data, writing incremental backup file, and appending to master dataset
        for y in period_range:
            self.update_trade(y)
            backup_file_path = ".\\sc_data_{}.csv".format(y)
            try:
                os.remove(backup_file_path)
            except OSError:
                pass
            self.sc_data.to_csv(backup_file_path)
            master_df = pd.concat([master_df,self.sc_data])
        # Write out complete assembled dataset (all periods)
        try:
            os.remove(out_file_path)
        except OSError:
            pass
        # Remove checkpoint backups now that full file has been written out
        master_df.to_csv(out_file_path)
        for y in period_range:
            os.remove(".\\sc_data_{}.csv".format(y))
        return


    def data_csv_to_geojson(csv_path: str=".\\sc_data.csv", out_file_path: str=".\\sc_data_geojson.geojson") -> None:
        sc_data = pd.read_csv(csv_path)
        sc_data = sc_data.drop(columns=['Unnamed: 0'])
        sc_data_json_features = []
        attr_cols = ['route_ID','orig_ISO','dest_ISO','weight','route_points','route_type']
        for _,row in sc_data.iterrows():
            qtys = {}
            for (columnName, columnData) in row.iteritems():
                if columnName in attr_cols: continue
                qtys[columnName] = columnData
            gj_feature = {"type": "Feature",
                "properties": {"route_ID": row['route_ID'],
                    "orig_ISO": row['orig_ISO'],
                    "dest_ISO": row['dest_ISO'],
                    "weight": row['weight'],
                    "route_type": row['route_type'],
                    "period": row['period'],
                    "trade_goods": qtys},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[p.split(', ')[0],p.split(', ')[1]] for p in row['route_points'][2:-2].split('], [')]
                    }
                }
            sc_data_json_features.append(gj_feature)

        sc_data_json = {"type":"FeatureCollection",
            "properties":{"description": "Geometry for trade routes"},
            "features": sc_data_json_features
            }

        try:
            os.remove(out_file_path)
        except OSError:
            pass
        with open(out_file_path, 'w') as f:
            json.dump(sc_data_json, f)
        return


    def build_routes(self) -> None:
        """Loads in all static route data and builds geographic routes"""
        ## Recommend adding functionality to optionally load the data from "backup" files to speed up route building
        sea_routes = sc_dataGetter.import_sea_routes()
        try:
            os.remove('sea_routes_backup.csv')
        except OSError:
            pass
        sea_routes.to_csv('sea_routes_backup.csv')

        air_routes = sc_dataGetter.import_air_routes()
        try:
            os.remove('air_routes_backup.csv')
        except OSError:
            pass
        air_routes.to_csv('air_routes_backup.csv')

        road_routes = sc_dataGetter.import_road_routes()
        try:
            os.remove('road_routes_backup.csv')
        except OSError:
            pass
        road_routes.to_csv('road_routes_backup.csv')

        self._routes_data = pd.concat([sea_routes,air_routes,road_routes])
        return


    def update_trade(self, period: str=None) -> None:
        """Retreives trade quantities data for the indicated time period and updates internal class data. Updates current_period if applicable.
        
        Paramters
        ---------
        period: str, optional
            Time period for retreived trade data; must match format indicated by current period_mode (default uses value of current_period)
        """

        if self._routes_data.empty:
            self.build_routes()

        if period is None: period = self.current_period
        
        self.sc_data = pd.DataFrame(columns=['route_ID','orig_ISO','dest_ISO','weight','route_points','route_type'])
        progress_tot = self._routes_data.groupby(['orig_ISO','dest_ISO']).count().shape[0]
        progress_count = 0
        for orig in self._routes_data['orig_ISO'].unique():
            for dest in self._routes_data['dest_ISO'].unique():
                temp_routes = self._routes_data[(self._routes_data['orig_ISO']==orig) & (self._routes_data['dest_ISO']==dest)]
                if temp_routes.empty: continue
                temp_routes = self._import_trade_quantities(orig, dest, period, temp_routes)
                progress_count += 1
                print("Progress: {}%".format(round(100.0*(progress_count/progress_tot),3)))
                if temp_routes is None: continue
                self.sc_data = pd.concat([self.sc_data,temp_routes])

        self.sc_data['period'] = period
        return


    def import_sea_routes(trip_lim: int=0):
        # Import and assemlbe maritime route data
        SEA_ROUTES_S3 = "http://worldroutes.s3.amazonaws.com/routes.csv"
        cols = ['point_index','trip_count','orig_port_ID','dest_port_ID','lat','lon','frequency']
        sea_routes = pd.read_csv(SEA_ROUTES_S3, header=0, names=cols).drop(columns='frequency')
        sea_routes = sea_routes.loc[sea_routes['trip_count'] > trip_lim]
        orig_pts = sea_routes.groupby('orig_port_ID', as_index=False).agg({'lat':'first','lon':'first'})
        dest_pts = sea_routes.groupby('dest_port_ID', as_index=False).agg({'lat':'last','lon':'last'})
        countries_gdf = gpd.read_file(BORDERS)
        orig_gdf = gpd.GeoDataFrame(orig_pts, geometry=gpd.points_from_xy(orig_pts.lon, orig_pts.lat))
        dest_gdf = gpd.GeoDataFrame(dest_pts, geometry=gpd.points_from_xy(dest_pts.lon, dest_pts.lat))
        orig_countries = pd.DataFrame(gpd.sjoin(orig_gdf, countries_gdf, how='inner', op='within')).reset_index(drop=True)[['TERRITORY1','ISO_TER1','orig_port_ID']].rename(columns={'TERRITORY1':'orig_territory','ISO_TER1':'orig_ISO'})
        dest_countries = pd.DataFrame(gpd.sjoin(dest_gdf, countries_gdf, how='inner', op='within')).reset_index(drop=True)[['TERRITORY1','ISO_TER1','dest_port_ID']].rename(columns={'TERRITORY1':'dest_territory','ISO_TER1':'dest_ISO'})
        sea_routes = sea_routes.merge(orig_countries, how='inner', on='orig_port_ID')
        sea_routes = sea_routes.merge(dest_countries, how='inner', on='dest_port_ID')
        sea_routes.dropna(inplace=True)
        sea_routes = sea_routes.loc[(sea_routes['orig_ISO'] != sea_routes['dest_ISO'])]
        sea_routes['route_ID'] = sea_routes.apply(lambda row: (str(row['orig_port_ID'])+str(row['dest_port_ID'])+str(row['trip_count'])), axis=1)
        total_trips = sea_routes.groupby('route_ID', as_index=False).agg({'trip_count':'first','orig_ISO':'first','dest_ISO':'first'})
        total_trips = total_trips.groupby(['orig_ISO','dest_ISO'], as_index=False)['trip_count'].sum().rename(columns={'trip_count':'trade_tot'})
        sea_routes = sea_routes.merge(total_trips, how='inner', on=['orig_ISO','dest_ISO'])
        sea_routes['weight'] = sea_routes.apply(lambda row: row['trip_count']/row['trade_tot'], axis=1)
        sea_routes['route_points'] = pd.Series([list(tup) for tup in zip(sea_routes['lat'], sea_routes['lon'])])
        sea_routes = sea_routes.groupby('route_ID', as_index=False).agg({'orig_ISO':'first','dest_ISO':'first','weight':'first','route_points':list})
        sea_routes['route_type'] = 'sea'
        return sea_routes


    #### THIS VERSION CONNECTS ROUTES THROUGH CANALS, BUT HAS BUGS WHICH RESULT IN BAD ROUTES ####
    # def import_sea_routes():
    #     # Suppress warnings to avoid type warnings from geopandas which don't matter
    #     warnings.filterwarnings('ignore')
    #     ## Import and assemble maritime route data
    #     SEA_ROUTES_S3 = "http://worldroutes.s3.amazonaws.com/routes.csv"
    #     cols = ['point_index','trip_count','orig_port_ID','dest_port_ID','lat','lon','frequency']
    #     sea_routes = pd.read_csv(SEA_ROUTES_S3, header=0, names=cols).drop(columns=['frequency','point_index'])
    #     sea_routes = sea_routes.reset_index().rename(columns={'index':'point_index'})
    #     sea_routes.dropna(inplace=True)
    #     ## Identify orig/dest countries
    #     # Extract orig/dest lat/lon pts
    #     orig_pts = sea_routes.groupby('orig_port_ID', as_index=False).agg({'lat':'first','lon':'first'})
    #     dest_pts = sea_routes.groupby('dest_port_ID', as_index=False).agg({'lat':'last','lon':'last'})
    #     # Import country profiles and gpd merge w/ orig/dest pts to ID orig/dest countries
    #     countries_gdf = gpd.read_file(BORDERS)
    #     orig_gdf = gpd.GeoDataFrame(orig_pts, geometry=gpd.points_from_xy(orig_pts.lon, orig_pts.lat))
    #     dest_gdf = gpd.GeoDataFrame(dest_pts, geometry=gpd.points_from_xy(dest_pts.lon, dest_pts.lat))
    #     orig_countries = pd.DataFrame(gpd.sjoin(orig_gdf, countries_gdf, how='inner', op='within')).reset_index(drop=True)[['TERRITORY1','ISO_TER1','orig_port_ID']].rename(columns={'TERRITORY1':'orig_territory','ISO_TER1':'orig_ISO'})
    #     dest_countries = pd.DataFrame(gpd.sjoin(dest_gdf, countries_gdf, how='inner', op='within')).reset_index(drop=True)[['TERRITORY1','ISO_TER1','dest_port_ID']].rename(columns={'TERRITORY1':'dest_territory','ISO_TER1':'dest_ISO'})
    #     # Merge orig/dest countries back into root dataset and remove internal routes or undefined routes
    #     sea_routes = sea_routes.merge(orig_countries, how='inner', on='orig_port_ID')
    #     sea_routes = sea_routes.merge(dest_countries, how='inner', on='dest_port_ID')
    #     sea_routes.dropna(inplace=True)
    #     sea_routes = sea_routes.loc[(sea_routes['orig_ISO'] != sea_routes['dest_ISO'])]
    #     ## Remove duplicate routes with same orig and dest; only keep route with highest traffic for each orig/dest pair
    #     # Assign unique route ID to all rows in each route
    #     sea_routes['route_ID'] = sea_routes.apply(lambda row: (str(row['orig_port_ID'])+str(row['dest_port_ID'])+str(row['trip_count'])), axis=1)
    #     # Identify top routes and inner merge with root dataset to remove all other routes
    #     keep_routes = sea_routes.sort_values('trip_count',ascending=False).groupby(['orig_port_ID','dest_port_ID'], as_index=False).agg({'route_ID':'first'})
    #     sea_routes = keep_routes.merge(sea_routes, how='inner', on='route_ID').drop(columns=['orig_port_ID_y','dest_port_ID_y']).rename(columns={'orig_port_ID_x':'orig_port_ID','dest_port_ID_x':'dest_port_ID'})
    #     sea_routes.dropna(inplace=True)
    #     sea_routes = sea_routes.sort_values('point_index')
    #     ## Consolidate routes to put each on single line and create route points list
    #     sea_routes['route_points'] = pd.Series([list(tup) for tup in zip(sea_routes['lat'], sea_routes['lon'])])
    #     sea_routes.dropna(inplace=True)
    #     sea_routes = sea_routes.groupby('route_ID', as_index=False).agg({'orig_ISO':'first','dest_ISO':'first','trip_count':'first','route_points':list})
    #     ## Correct routes through Canals
    #     print("Fixing sea route canal connections.")
    #     canal_country_routes = sea_routes[sea_routes['orig_ISO'].isin(sc_dataGetter.CANAL_COUNTRIES)]
    #     canal_country_routes = canal_country_routes.append(sea_routes[sea_routes['dest_ISO'].isin(sc_dataGetter.CANAL_COUNTRIES)])
    #     canal_country_routes['canal'] = canal_country_routes.apply(lambda row: sc_dataGetter._classify_canal(tuple(row['route_points'][0]), tuple(row['route_points'][-1])), axis=1)
    #     transPan_NS_routes = canal_country_routes[(canal_country_routes['canal'] == 'to_PAN_N')].merge(canal_country_routes[(canal_country_routes['canal'] == 'from_PAN_S')], how='inner', left_on='dest_ISO', right_on='orig_ISO')
    #     transPan_SN_routes = canal_country_routes[(canal_country_routes['canal'] == 'to_PAN_S')].merge(canal_country_routes[(canal_country_routes['canal'] == 'from_PAN_N')], how='inner', left_on='dest_ISO', right_on='orig_ISO')
    #     transSuez_NS_routes = canal_country_routes[(canal_country_routes['canal'] == 'to_SUEZ_N')].merge(canal_country_routes[(canal_country_routes['canal'] == 'from_SUEZ_S')], how='inner', left_on='dest_ISO', right_on='orig_ISO')
    #     transSuez_SN_routes = canal_country_routes[(canal_country_routes['canal'] == 'to_SUEZ_S')].merge(canal_country_routes[(canal_country_routes['canal'] == 'from_SUEZ_N')], how='inner', left_on='dest_ISO', right_on='orig_ISO')
    #     canal_routes = pd.concat([transPan_NS_routes, transPan_SN_routes, transSuez_NS_routes, transSuez_SN_routes])
    #     canal_country_routes.to_csv('canals_check.csv') #delete this line
    #     canal_routes['route_ID'] = canal_routes.apply(lambda row: str(row['route_ID_x']) + str(row['route_ID_y']), axis=1)
    #     canal_routes['trip_count'] = canal_routes.apply(lambda row: (float(row['trip_count_x']) + float(row['trip_count_y'])) / 2.0, axis=1)
    #     canal_routes.rename(columns={'orig_ISO_x':'orig_ISO','dest_ISO_y':'dest_ISO'}, inplace=True)
    #     canal_routes['route_points'] = canal_routes.apply(lambda row: (row['route_points_x']).extend(row['route_points_y']), axis=1)
    #     sea_routes = sea_routes.append(canal_routes[['route_ID','trip_count','orig_ISO','dest_ISO','route_points']])
    #     ## Assign route weights
    #     # Calculate route weights
    #     total_trips = sea_routes.groupby(['orig_ISO','dest_ISO'], as_index=False)['trip_count'].sum().rename(columns={'trip_count':'trade_tot'})
    #     sea_routes = sea_routes.merge(total_trips, how='inner', on=['orig_ISO','dest_ISO'])
    #     sea_routes['weight'] = sea_routes.apply(lambda row: row['trip_count']/row['trade_tot'], axis=1)
    #     ## Add route type identifier
    #     sea_routes['route_type'] = 'Sea'
    #     ## Return dataframe with required fields
    #     return sea_routes[['orig_ISO','dest_ISO','weight','route_points','route_type']]


    # def _classify_canal(orig, dest):
    #     if geodesic(dest,sc_dataGetter.PANAMA_N).miles < sc_dataGetter.CANAL_TOL:
    #         return 'to_PAN_N'
    #     if geodesic(orig,sc_dataGetter.PANAMA_S).miles < sc_dataGetter.CANAL_TOL:
    #         return 'from_PAN_S'
    #     if geodesic(dest,sc_dataGetter.PANAMA_S).miles < sc_dataGetter.CANAL_TOL:
    #         return 'to_PAN_S'
    #     if geodesic(orig,sc_dataGetter.PANAMA_N).miles < sc_dataGetter.CANAL_TOL:
    #         return 'from_PAN_N'            
    #     if geodesic(dest,sc_dataGetter.SUEZ_N).miles < sc_dataGetter.CANAL_TOL:
    #         return 'to_SUEZ_N'
    #     if geodesic(orig,sc_dataGetter.SUEZ_S).miles < sc_dataGetter.CANAL_TOL:
    #         return 'from_SUEZ_S'
    #     if geodesic(dest,sc_dataGetter.SUEZ_S).miles < sc_dataGetter.CANAL_TOL:
    #         return 'to_SUEZ_S'
    #     if geodesic(orig,sc_dataGetter.SUEZ_N).miles < sc_dataGetter.CANAL_TOL:
    #         return 'from_SUEZ_N'
    #     return None


    def import_air_routes(pop_lim: int=0):
        air_routes = sc_dataGetter.get_cities_pairs(pop_lim)
        air_routes['route_ID'] = air_routes.apply(lambda row: (str(row['orig_id'])+str(row['dest_id'])), axis=1)
        air_routes['route_points'] = air_routes.apply(lambda row: [row['orig_lat_lng'], row['dest_lat_lng']], axis=1)
        air_routes['weight'] = air_routes.apply(lambda row: (int(row['orig_population'])+int(row['dest_population'])) / geodesic(tuple(row['route_points'][0]), tuple(row['route_points'][1])).miles, axis=1)
        weight_norms = air_routes.groupby('orig_ISO', as_index=False)['weight'].sum().rename(columns={'weight':'wt_norm'})
        air_routes = air_routes.merge(weight_norms, how='inner', on='orig_ISO')
        air_routes['weight'] = air_routes.apply(lambda row: row['weight'] / row['wt_norm'], axis=1)
        air_routes = air_routes[['route_ID', 'orig_ISO', 'dest_ISO', 'weight', 'route_points']]
        air_routes['route_type'] = 'Air'
        return air_routes


    def import_road_routes(pop_lim: int=0):
        # # Suppress warnings to avoid deprecation warning from networkx which don't matter
        # warnings.filterwarnings('ignore')
        # # Import shapefile and build into correct lat/lon network architecture
        # roadmap = sc_dataGetter.build_latlng_network(nx.read_shp(ROADS))
        # # Create city pairs with road route connections
        # road_routes = sc_dataGetter.get_cities_pairs(pop_lim, snap_map=roadmap)
        # road_routes['route_ID'] = road_routes.apply(lambda row: (str(row['orig_id'])+str(row['dest_id'])), axis=1)

        road_routes = sc_dataGetter.get_cities_pairs(pop_lim)
        road_routes['route_ID'] = road_routes.apply(lambda row: (str(row['orig_id'])+str(row['dest_id'])+"rd"), axis=1)

        road_routes['route_points'] = road_routes.apply(lambda row: [row['orig_lat_lng'], row['dest_lat_lng']], axis=1)
        road_routes['route_length'] = road_routes.apply(lambda row: geodesic(tuple(row['route_points'][0]), tuple(row['route_points'][1])).miles, axis=1)
        road_routes = road_routes[road_routes['route_length'] < 500]
        road_routes['weight'] = road_routes.apply(lambda row: (int(row['orig_population'])+int(row['dest_population'])) / row['route_length'], axis=1)
        weight_norms = road_routes.groupby('orig_ISO', as_index=False)['weight'].sum().rename(columns={'weight':'wt_norm'})
        road_routes = road_routes.merge(weight_norms, how='inner', on='orig_ISO')
        road_routes['weight'] = road_routes.apply(lambda row: row['weight'] / row['wt_norm'], axis=1)
        road_routes = road_routes[['route_ID', 'orig_ISO', 'dest_ISO', 'weight', 'route_points']]
        road_routes['route_type'] = 'Road'
        return road_routes


    def get_cities_pairs(pop_lim: int=0, snap_map=None):
        cities = pd.read_csv(CITIES)[['id','city','lat','lng','iso3','population']]
        cities = cities.loc[cities['population'] > pop_lim]
        if snap_map != None:
            cities['lat_lng'] = cities.apply(lambda row: sc_dataGetter._get_nearest_map_point((row['lat'],row['lng']),snap_map), axis=1)
        else:
            cities['lat_lng'] = cities.apply(lambda row: [row['lat'], row['lng']], axis=1)
        cities = cities.drop(columns=['lat','lng'])
        cities_pairs_o = pd.concat([cities]*cities.shape[0]).sort_index().reset_index(drop=True).rename(columns={'id':'orig_id','city':'orig_city','lat_lng':'orig_lat_lng','iso3':'orig_ISO','population':'orig_population'})
        cities_pairs_d = pd.concat([cities]*cities.shape[0]).reset_index(drop=True).rename(columns={'id':'dest_id','city':'dest_city','lat_lng':'dest_lat_lng','iso3':'dest_ISO','population':'dest_population'})
        cities_pairs = cities_pairs_o.join(cities_pairs_d)
        cities_pairs = cities_pairs.loc[cities_pairs['orig_ISO'] != cities_pairs['dest_ISO']]
        return cities_pairs


    def build_latlng_network(shp_data):
        nodes_list = []
        edges_list = []
        for e in shp_data.edges:
            v0 = [e[0][1],e[0][0]]
            v1 = [e[1][1],e[1][0]]
            if v0 not in nodes_list:
                nodes_list.append(v0)
            if v1 not in nodes_list:
                nodes_list.append(v1)
            new_edge = (nodes_list.index(v0), nodes_list.index(v1), {'weight':geodesic(v0,v1).miles})
            if new_edge not in edges_list:
                edges_list.append(new_edge)
        roadmap = nx.Graph()
        roadmap.add_nodes_from(nodes_list)
        roadmap.add_edges_from(edges_list)
        return roadmap


    def _get_nearest_map_point(point, map):
        distances = []
        for v in map.nodes:
            d = geodesic(point, v).miles
            if d < sc_dataGetter.SNAP_TOL:
                print("Tol: {}".format(list(v)))
                return list(v)
            distances.append(d)
        print("Min: {}".format(list(map.nodes[np.argmin(np.array(distances))])))
        return list(map.nodes[np.argmin(np.array(distances))])


    def generate_Goods_Lookup(outfile=".\\Data\\HS6_Lookup.csv"):
        commodities_url = "https://api.census.gov/data/timeseries/intltrade/exports/statehs?get=E_COMMODITY_LDESC,E_COMMODITY&time=2020-02&COMM_LVL=HS6"
        commodities_response = requests.get(commodities_url)
        lns = commodities_response.text.split('\n')
        header = lns[0][2:-2].replace('\"','').split(',')
        content = []
        for ln in lns[1:]:
            content.append(ln[2:-3].split('","'))
        commodities = pd.DataFrame(data=content, columns=header)[['E_COMMODITY', 'E_COMMODITY_LDESC']].rename(columns={'E_COMMODITY':'product_code', 'E_COMMODITY_LDESC':'product_name'})
        commodities.to_csv(outfile)
        return


    def _import_commodities(self):
        # Read in list of all commodity IDs
        commodities_url = "https://api.census.gov/data/timeseries/intltrade/exports/statehs?get=E_COMMODITY_LDESC,E_COMMODITY&time=2020-02&COMM_LVL=HS6"
        commodities_response = requests.get(commodities_url)
        lns = commodities_response.text.split('\n')
        header = lns[0][2:-2].replace('\"','').split(',')
        content = []
        for ln in lns[1:]:
            content.append(ln[2:-3].split('","'))
        self._commodities = pd.DataFrame(data=content, columns=header)
        return


    def _import_country_codes(self):
        # Read in country codes decoder file
        self._country_codes = pd.read_csv(COUNTRY_CODES,index_col='3letter')
        return


    def _import_mode_wts(self):
        # Read in transport mode weights file
        mode_stats = pd.read_csv(EUROSTATS)
        mode_stats['HS'] = mode_stats['HS'].astype(int)
        mode_stats = pd.read_csv(EUROSTATS)
        mode_stats['Value'] = mode_stats.apply(lambda row: int(row['Value'].replace(',','').replace(':','0')), axis=1)
        val_by_HS = mode_stats.groupby('HS', as_index=False)['Value'].sum().rename(columns={'Value':'Tot_Val'})
        mode_stats = mode_stats.merge(val_by_HS, how='inner', on='HS')
        mode_stats['Mode_wt'] = mode_stats.apply(lambda row: row['Value'] / row['Tot_Val'], axis=1)
        self._mode_wts = mode_stats[['TRANSPORT_MODE','HS','Mode_wt']]
        return


    def _import_trade_quantities(self, orig: str, dest: str, period: str, routes_set: pd.DataFrame):
        # Import trade data between trading partners
        trade_params = {'fmt':'json','px':'HS','rg':'2','head':'M','cc':'AG6','max':'100000'}
        trade_params['r'] = self._country_codes.loc[orig]['number']
        trade_params['p'] = self._country_codes.loc[dest]['number']
        trade_params['ps'] = period
        tradedata_url = "https://comtrade.un.org/api/get"
        # print('orig: {}, dest: {}'.format(trade_params['r'],trade_params['p']))
        attempt_lim = 2
        attempts = 0
        jsondata = {'dataset':[], 'validation': {'status': {'name':''}}}
        while attempts < attempt_lim:
            attempts += 1
            try:
                tradedata_response = requests.get(tradedata_url, params=trade_params)
                jsondata = tradedata_response.json()
                break
            except requests.exceptions.JSONDecodeError:
                print('Exceeded rate limit, waiting 1 second.')
                time.sleep(1)
                try:
                    tradedata_response = requests.get(tradedata_url, params=trade_params)
                    jsondata = tradedata_response.json()
                    break
                except requests.exceptions.JSONDecodeError:
                    print('Exceeded rate limit, waiting 1 hour.')
                    time.sleep(3600)
        if 'validation' not in jsondata.keys(): return None
        if (jsondata['validation']['status']['name'] != "Ok") | (jsondata['dataset'] == []): return None
        tradedata = pd.DataFrame(jsondata['dataset'])[['cmdCode','TradeValue']]
        trade_amounts = self._commodities.merge(tradedata, how='left', left_on='E_COMMODITY', right_on='cmdCode')[['E_COMMODITY','TradeValue']].fillna(0.0)
        # Distribute trade over routes in set
        for _,r in routes_set.iterrows():
            trade_amounts[r['route_ID']] = trade_amounts.apply(lambda row: self._weight_helper(r,row), axis=1)
        trade_amounts = trade_amounts.drop(columns='TradeValue').T
        trade_amounts.columns = trade_amounts.loc['E_COMMODITY']
        trade_amounts.drop(labels='E_COMMODITY', inplace=True)
        trade_amounts.reset_index(inplace=True)
        routes_qtys = routes_set.merge(trade_amounts, how='inner', left_on='route_ID', right_on='index').drop(columns=['index'])
        return routes_qtys

    def _weight_helper(self, r, row):
        mode_wt = self._mode_wts.loc[(self._mode_wts['TRANSPORT_MODE'] == r['route_type']) & (self._mode_wts['HS'] == int(row['E_COMMODITY'][0:2]))]['Mode_wt']
        if len(mode_wt) == 0:
            return 0
        return r['weight'] * mode_wt.iloc[0] * row['TradeValue']
