#!/usr/bin/python

import requests
import urllib
import json
import pandas as pd
import numpy as np
import scipy.ndimage as ndimage
import geopandas as gpd
import shapely
from shapely.geometry import Point, MultiPoint
from shapely import wkt
import shapely.speedups
import pycrs
from shapely.ops import transform, nearest_points
import plotly.express as px
from pyproj import crs
import os
import gdal
from functools import reduce
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import glob
from functools import partial
import pyproj
import osmnx as ox
### These three files are local python programs. Make sure to paste them wherever you're running this notebook
import polygonize as pz
##
import geocoder
from pandana.loaders import osm
import pandana
import pylab as pl
ox.config(log_console=True, use_cache=True)





def get_city_proj_crs(to_crs, val=0):

    """
    Function to indentify local projection for cities dynamically

    Input:
    to_crs : name of city / country; epsg if known

    Returns:
    Local epsg (in string)
    """

    if isinstance(to_crs, int):
        to_crs = to_crs
    elif isinstance(to_crs, str):
        city, country = to_crs.split(',')
        url = "http://epsg.io/?q={}&format=json&trans=1&callback=jsonpFunction".format(city)
        r = requests.get(url)
        if r.status_code == 200:
            js = json.loads(r.text[14:-1])

            if js['number_result'] != 0:
                lis = []
                for i in js['results']:
                    res = i
                    if (res['unit'] == 'metre') and (res['accuracy'] == 1.0):
                        lis.append(res['code'])
                if len(lis) == 0:
                    for i in js['results']:
                        res = i
                        if res['unit'] == 'metre':
                            lis.append(res['code'])
                    return lis[val]
                else:
                    return lis[val]

            else:
                if country.strip() == 'United Kingdom of Great Britain and Northern Ireland':
                    country = 'United Kingdom'
                elif country.strip() == 'Venezuela (Bolivarian Republic of)':
                    country = 'Venezuela'
                elif country.strip() == 'Viet Nam':
                    country = 'Vietnam'

                url = "http://epsg.io/?q={}&format=json&trans=1&callback=jsonpFunction".format(country)
                r = requests.get(url)
                if r.status_code == 200:
                    js = json.loads(r.text[14:-1])

                    if js['number_result'] != 0:
                        lis = []
                        for i in js['results']:
                            res = i
                            if (res['unit'] == 'metre') and (res['accuracy'] == 1.0):
                                lis.append(res['code'])
                        if len(lis) == 0:
                            for i in js['results']:
                                res = i
                                if res['unit'] == 'metre':
                                    lis.append(res['code'])
                            return lis[val]
                        else:
                            return lis[val]




def convert_geom_to_shp(shapely_polygon, city, out_crs=None):
    string = city.split(',')[0]

    df = pd.DataFrame(
    {'City': [string],
     'geometry': [wkt.dumps(shapely_polygon)]})

    df['geometry'] = df['geometry'].apply(wkt.loads)

    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    if out_crs:
        gdf.crs = {'init' : 'epsg:{}'.format(out_crs)}

        gdf.to_crs(epsg=4326, inplace=True)
    else:
        gdf.crs = {'init' : 'epsg:{}'.format(4326)}

    return gdf




def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio accepts them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]




def get_iso(city):
    """
    Function to get ISO-3 codes for countries

    Input:
    city: city name (Ideally in (city, country) format)

    Returns:
    ISO-3 code for the country
    """

    try:
        country = city.split(',')[1].strip().lower()
        if country == 'south korea':  ### incorrect output for South Korea's ISO code with API
            return 'kor'
        elif country == 'india':
            return 'ind'
        elif country == 'iran':
            return 'irn'
        elif country == "republic of congo":
            return 'cod'
        elif country == "democratic republic of congo":
            return 'cog'
        else:
            url = "https://restcountries.eu/rest/v2/name/{}".format(country)
            r = requests.get(url)
            if len(r.json())>1 :
                for i in range(len(r.json())):
                    if country in r.json()[i]['name'].lower():
                        return r.json()[i]['alpha3Code'].lower()
            else:
                return r.json()[0]['alpha3Code'].lower()
    except IndexError:
        url = "https://restcountries.eu/rest/v2/capital/{}".format(city)
        r = requests.get(url)
        return r.json()[0]['alpha3Code'].lower()





def factors(n):
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))





def test_func(values):
    #print (values)
    return values.sum()


def get_footprint(dim=3):
    footprint = np.array([[1,1,1],
                      [1,1,1],
                      [1,1,1]])
    return footprint



def add_newRow(shp_arr) :
    y_add = shp_arr[-2][0].y - shp_arr[-1][0].y
    x_add = shp_arr[0][-1].x - shp_arr[0][-2].x

    lis_y = []
    for i in range(shp_arr.shape[0]):
        x = shp_arr[i][-1].x + x_add
        y = shp_arr[i][-1].y + 0

        lis_y.append([Point(x,y)])

    lis_x = []
    for i in range(shp_arr.shape[1]):
        x = shp_arr[-1][i].x + 0
        y = shp_arr[-1][i].y + y_add

        lis_x.append(Point(x,y))

    shp_x = shp_arr.shape[0]-1
    shp_y = shp_arr.shape[1]-1

    newp = Point(shp_arr[shp_x][shp_y].x + x_add, shp_arr[shp_x][shp_y].y + y_add)

    lis_x.append(newp)

    append_y = np.hstack((shp_arr, np.atleast_2d(np.array(lis_y, dtype=Point))))

    append_x = np.vstack((append_y, np.atleast_2d(np.array(lis_x, dtype=Point))))

    return append_x





def polygonize_raster(ras_path, shp_path, string):
    """
    Function to polygonize a raster based on the pixel size of base raster.

    Inputs:
    ras_path: path to base raster location that is to be polygonized
    shp_path: path to where the shapefile will be saved
    string: name of the city

    Returns:
    Geodataframe with polygons equivalent to raster pixels.
    """

    print("Polygonizing Raster!!")

    import polygonize as pz

    path = r"M:\Gaurav\GPSUR\Data\Analysis"
    #rasterfn = path+"\\"+ras_path


    #outSHPfn = path+"\\shapefiles\\{}".format(shp_path)
    outSHPfn = shp_path
    lat, lon = pz.main(ras_path,outSHPfn)

    #sh = gpd.read_file(path+"\\shapefiles\\{}".format(shp_path))
    sh = gpd.read_file(shp_path)

    rio = rasterio.open(ras_path)

    sh.crs = rio.meta['crs']

    shp_arr = np.array(sh.geometry).reshape(rio.shape[0], rio.shape[1])

    shp_arr = add_newRow(shp_arr)

    pols = []
    for row in range(shp_arr.shape[0]-1):
        for col in range(shp_arr.shape[1]-1):
            pols.append(shapely.geometry.box(shp_arr[row+1][col].x, shp_arr[row+1][col].y, shp_arr[row][col+1].x, shp_arr[row][col+1].y ))

    gdf = gpd.GeoDataFrame()
    gdf['ID'] = [i for i in range(len(pols))]
    gdf['geometry'] = pols
    gdf.set_geometry('geometry', inplace=True)
    #gdf.crs = {'init':'epsg:4326'}
    gdf.crs = rio.crs
    print("Populating avearge height!!")

    av_h = []
    for i in gdf.geometry:
        coords = getFeatures(convert_geom_to_shp(i, string))
        out_img, out_transform = mask(dataset=rio, shapes=coords, crop=True)
        out_img[out_img<0] = 0
        av_h.append(np.mean(out_img.flatten()))

    gdf['avg_height'] = av_h
    gdf['avg_height'] = [i if i>0 else 0 for i in gdf.avg_height]
    gdf.to_crs(epsg=4326, inplace=True)
    gdf['Lon'] = [i.centroid.x for i in gdf.geometry]
    gdf['Lat'] = [i.centroid.y for i in gdf.geometry]


    return gdf





def main(city, from_pop = "wp"):
    string = city.split(',')[0]

    print("Getting data for {}".format(string))

    try:
        dest_path = r"M:\Gaurav\GPSUR\Data\DLR Data\{}_WSF3D_AW3D30.tif".format(string)
        ras =  rasterio.open(dest_path)
    except:
        dest_path = r"C:\Users\wb542830\OneDrive - WBG\GPSUR\COVID\raster_files\DLR\{}_new_WSF3D_AW3D30.tif".format(string)
        ras =  rasterio.open(dest_path)

    shp_path = r'shapefiles\{}__clip.shp'.format(string)

    gdf = polygonize_raster(dest_path, shp_path, string)

    if string == "Maputo":
        out_crs = 32737
    else:
        out_crs = int(get_city_proj_crs(city))

    gdf_copy = gdf.to_crs(epsg=out_crs)
    gdf['pixel_area'] = [i.area for i in gdf_copy.geometry]

    print("Preparing population data")

    if from_pop == "wp":
        from osgeo import gdal, gdalconst
        iso_ = get_iso(city)
        # Source
        src_filename = r"M:\Gaurav\GPSUR\Data\WorldPop_2019\{}_ppp_2019.tif".format(iso_)
        src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
        src_proj = src.GetProjection()
        src_geotrans = src.GetGeoTransform()

        # We want a section of source that matches this:
        match_filename = dest_path
        match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
        match_proj = match_ds.GetProjection()
        match_geotrans = match_ds.GetGeoTransform()
        wide = match_ds.RasterXSize
        high = match_ds.RasterYSize

        # Output / destination
        dst_filename = r"C:\Users\wb542830\OneDrive - WBG\GPSUR\COVID\raster_files\{}_snap.tif".format(string)
        dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdalconst.GDT_Float32)
        dst.SetGeoTransform( match_geotrans )
        dst.SetProjection( match_proj)

        # Do the work
        gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)

        del dst # Flush


        pop = rasterio.open(dst_filename)

    elif from_pop == "fb":
        iso_ = get_iso(city)
        filename = r"C:\Users\wb542830\OneDrive - WBG\Facebook\population_{}_2018-10-01.tif".format(iso_)
        pop = rasterio.open(filename)

    big_pol =  shapely.geometry.box(pop.bounds[0], pop.bounds[1], pop.bounds[2], pop.bounds[3])

    small_pol =  shapely.geometry.box(gdf.geometry[100].bounds[0], gdf.geometry[100].bounds[1], gdf.geometry[100].bounds[2], gdf.geometry[100].bounds[3])

    if small_pol.intersects(big_pol) == False:
        gdf['geometry'] = gdf.geometry.map(lambda polygon:  shapely.ops.transform(lambda x, y: (y, x), polygon))



    wp_pop = []

    #pop = rasterio.open(r"M:\Gaurav\GPSUR\Data\WorldPop_2019\cod_ppp_2019.tif")
    for i in gdf.index:
        if i%5000 == 0:
            print('Processed {} of {} rows'.format(i, len(gdf)))
        _gdf = gdf[gdf.index == i]
        #_gdf.to_crs(pop.meta['crs'], inplace=True)

        _coords =  getFeatures(_gdf)
        try:
            _out_img, _out_transform =  mask(dataset=pop, shapes=_coords, crop=True)

            outimg =  np.nan_to_num(_out_img)
            #outimg = outimg.reshape(outimg.shape[1], outimg.shape[2])
            wp_pop.append(outimg.sum())
        except ValueError:
            wp_pop.append(0)


    gdf['pop_2019'] = wp_pop
    gdf['pop_2019'] = [i if i>0 else 0 for i in gdf.pop_2019]

    print('Total Population of {} if {}'.format(string, gdf.pop_2019.sum()))

    gdf['tfa'] = [(gdf.avg_height[i] * gdf.pixel_area[i]) / 3 for i in gdf.index]
    gdf['pop_tfa'] = [(gdf.pop_2019[i]) / gdf.tfa[i] for i in gdf.index]

    gdf['pop_tfa'] = [0 if  pd.isna(i) else i for i in gdf.pop_tfa]
    gdf['pop_tfa'] = [0 if i ==  np.inf else i for i in gdf.pop_tfa]
    gdf['pop_tfa'] = [0 if i == -np.inf else i for i in gdf.pop_tfa]

    footprint = [[1,1,1],
            [1,1,1],
            [1,1,1]]

    results =  ndimage.generic_filter( np.array(gdf.pop_tfa).reshape(ras.shape[0], ras.shape[1]),  test_func, footprint=footprint)

    gdf['poptfa_all'] = results.flatten()

    gdf.to_file("{}_Hotspots.shp".format(string))
