#!/usr/bin/python

#import covid_analysis as ca
from covid_analysis import *
from glob import glob
import os, sys
from osgeo import gdal, gdalconst


def resample_raster(city, iso, dest_path):
    """
    Input:
    city : city name (Dhaka, Bangladesh)
    iso : ISO-3 code for country
    dest_path : path where reprojected raster will be saved

    Returns:
    path to reprojected file
    """
    from osgeo import gdal, gdalconst
    string = city.split(",")[0]

    src_filename = r"raster_files\{}_ppp_2020.tif".format(iso)
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()

    match_filename = dest_path
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    # Output / destination
    dst_filename = os.path.join(os.getcwd(), "raster_files/{}_snap.tif".format(string))
    #dst_filename = r"C:\Users\wb542830\OneDrive - WBG\GPSUR\COVID\raster_files\{}_snap.tif".format(string)
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdalconst.GDT_Float32)
    dst.SetGeoTransform( match_geotrans )
    dst.SetProjection( match_proj)

    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Bilinear)

    del dst
    return dst_filename


def polygonize_raster_short(ras_path, shp_path, string):
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
    gdf.crs = {'init':'epsg:4326'}
    #gdf.crs = rio.crs
    return gdf


def merge_raster(ras_path, string):
    """
    Merges raster to create a sinle raster

    Input:
    ras_path = list of raster files path
    string : city name

    Returns:
    Merged raster
    """
    from rasterio.merge import merge
    if len(ras_path)>1:
        ras_all = []
        for i in ras_path:
            r = rasterio.open(i)
            ras_all.append(r)
        mosaic, out_trans = merge(ras_all)
        out_meta = r.meta.copy()
        out_meta.update({"driver": "GTiff",
                      "height": mosaic.shape[1],
                      "width": mosaic.shape[2],
                      "transform": out_trans
                      }
                     )
        out_fp = r"final_outputs\DLR New Data\{}_WSF3D.tif".format(string)
        with rasterio.open(out_fp, "w", **out_meta) as dest:
             dest.write(mosaic)

        ras = rasterio.open(out_fp)
    else:
        ras = rasterio.open(ras_path[0])

    return ras


def save_rasters(raster, city, raster_type):

    """
    Saves result raster in current directory

    Input:
    raster : raster file
    city : city name (e.g. Nairobi)
    raster_type : name extension for file (e.g. hotspots, hotspots_wWater)

    returns:
    saved path
    """

    path = os.path.join(os.getcwd() , "{}_{}.tif".format(city, raster_type))

    with rasterio.Env():
        profile = ras.profile

        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw',
        height=raster.shape[0],
        width=raster.shape[1],)

        with rasterio.open(dest_path, 'w', **profile) as dst:
            dst.write(raster.astype(rasterio.float32), 1)

    return path



def get_service_hotspots(service, hotspot_path):
    """
    Creates service based hotspots raster

    Input:
    service : Name of the service (water, toilets, shops etc.)
    hotspot_path : path where original hotspot file is saved

    Returns:
    None
    """

    if service == 'water':
        amenities = ['water_points', 'drinking_water', 'pumps', 'water_pumps', 'well']
        osm_tags = '"amenity"~"{}"'.format('|'.join(amenities))
        serv = osm.node_query(ras.bounds[1], ras.bounds[0], ras.bounds[3], ras.bounds[2],tags=osm_tags)
    elif service == 'toilet':
        amenities = ['toilet', 'restroom', 'washroom', 'public_services', 'urinal']
        osm_tags = '"amenity"~"{}"'.format('|'.join(amenities))
        serv = osm.node_query(ras.bounds[1], ras.bounds[0], ras.bounds[3], ras.bounds[2],tags=osm_tags)
    elif service == 'shops':
        amenities = ['supermarket', 'convenience', 'general', 'department_stores', 'wholesale', 'grocery', 'general']
        osm_tags = '"shop"~"{}"'.format('|'.join(amenities))
        serv = osm.node_query(ras.bounds[1], ras.bounds[0], ras.bounds[3], ras.bounds[2],tags=osm_tags)
    else:
        serv = None

    serv['geometry'] = (list(zip(serv.lon,serv.lat)))
    serv['geometry'] = serv.geometry.apply(lambda x: Point(x))
    serv = gpd.GeoDataFrame(serv, geometry='geometry')
    serv.crs = {'init':'epsg:4326'}

    serv.to_crs(epsg = coord_sys, inplace=True)
    serv_mult = MultiPoint([i for i in serv.geometry])

    gdf = polygonize_raster_short(hotspot_path, 'shapefiles/{}_short.shp'.format(string), string)
    gdf_copy = gdf.to_crs(epsg = coord_sys)

    dist_serv = []
    for i in gdf_copy.index:
        temp_cent = gdf_copy.geometry.iloc[i].centroid
        nearest_geoms = nearest_points(temp_cent, serv_mult)
        dist_serv.append(nearest_geoms[0].distance(nearest_geoms[1]))

    serv_arr = np.array(dist_serv).reshape(ras.shape[0], ras.shape[1])
    weight_serv = 1 / np.sqrt(serv_arr)
    weight_serv[serv_arr<100] = 1
    pop_weight_serv = (poptfa * weight_serv) / 8

    footprint = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]

    serv_result = ndimage.generic_filter( pop_weight_serv[0,:,:], test_func, footprint=footprint)
    poptfa_serv = results + serv_result
    poptfa_serv[np.isnan(poptfa_serv)] = 0
    poptfa_serv[np.isposinf(poptfa_serv)] = 0
    poptfa_serv[np.isneginf(poptfa_serv)] = 0

    service_path = save_rasters(poptfa_serv, string, 'hotspots_{}'.format(service))
    del service_path



def main(city):
    global ras, coord_sys, iso, string, poptfa, results

    string = city.split(',')[0]
    coord_sys = int(get_city_proj_crs(city.split(',')[1], val=0))
    iso = get_iso(city)

    ras_path = glob(r"raster_files\DLR\*{}*AW3D30.tif".format(string))
    ras = merge_raster(ras_path, string)
    ras_arr = ras.read()

    dst_filename = resample_raster(city , iso, out_path)
    new_ras_arr = np.round(ras_arr, 2)

    pop = rasterio.open(dst_filename)
    pop_arr = pop.read()

    pop_arr = pop_arr.reshape(pop_arr.shape[1], pop_arr.shape[2])
    pop_arr[np.isnan(pop_arr)] = 0
    pop_arr[np.isposinf(pop_arr)] = 0
    pop_arr[np.isneginf(pop_arr)] = 0
    pop_arr[pop_arr < 0] = 0

    denom = new_ras_arr*(10000/3)
    poptfa = pop_arr/denom

    poptfa[np.isnan(poptfa)] = 0
    poptfa[np.isposinf(poptfa)] = 0
    poptfa[np.isneginf(poptfa)] = 0
    poptfa[poptfa<0] = 0

    footprint = [[1,1,1],[1,1,1],[1,1,1]]
    results =  ndimage.generic_filter( poptfa[0,:,],  test_func, footprint=footprint)
    results = results[0,:,:]

    ##saving hotspots raster
    hotspot_path = save_rasters(results, string, 'hotspots')

    ### Running for services

    serv_lis = ['water', 'toilets', 'shops']

    for service in serv_lis:
        get_service_hotspots(service, hotspot_path)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
