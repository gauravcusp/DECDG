#!/usr/bin/python

import ogr, gdal, osr, os
import numpy as np
import itertools

def pixelOffset2coord(raster, xOffset,yOffset):
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    coordX = originX+pixelWidth*xOffset
    coordY = originY+pixelHeight*yOffset
    return coordX, coordY

def raster2array(rasterfn):
    raster = gdal.Open(rasterfn)
    array = raster.ReadAsArray()
    return array

def array2shp(array,outSHPfn,rasterfn):

    # max distance between points
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    pixelWidth = geotransform[1]

    # wkbPoint
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outSHPfn):
        shpDriver.DeleteDataSource(outSHPfn)
    outDataSource = shpDriver.CreateDataSource(outSHPfn)
    outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbPoint )
    featureDefn = outLayer.GetLayerDefn()
    outLayer.CreateField(ogr.FieldDefn("ID", ogr.OFTInteger))

    lat, lon = [], []
    # array2dict
    point = ogr.Geometry(ogr.wkbPoint)
    row_count = array.shape[0]
    for ridx, row in enumerate(array):
        if ridx % 100 == 0:
            print("{0} of {1} rows processed" .format(ridx, row_count))
        for cidx, value in enumerate(row):
            Xcoord, Ycoord = pixelOffset2coord(raster,cidx,ridx)
            lat.append(Xcoord)
            lon.append(Ycoord)
            point.AddPoint(Xcoord, Ycoord)
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(point)
            outLayer.CreateFeature(outFeature)
            outFeature.SetField("ID", outFeature.GetFID())
            outLayer.SetFeature(outFeature)
            outFeature.Destroy()
    outDataSource.Destroy()
    return lat, lon


def main(rasterfn,outSHPfn):
    array = raster2array(rasterfn)
    lat, lon = array2shp(array,outSHPfn,rasterfn)
    return lat, lon
