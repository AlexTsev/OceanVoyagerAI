# -*- coding: utf-8 -*-
from math import radians, degrees, sin, cos, asin, atan2, sqrt, log, tan, atan, exp, fabs, pi

EARTH_MEAN_RADIUS = 6371008.8
EARTH_EQUATORIAL_RADIUS = 6378137.0
EARTH_EQUATORIAL_METERS_PER_DEGREE = pi * EARTH_EQUATORIAL_RADIUS / 180
I_EARTH_EQUATORIAL_METERS_PER_DEGREE = 1 / EARTH_EQUATORIAL_METERS_PER_DEGREE
HALF_PI = pi / 2.0
QUARTER_PI = pi / 4.0

# ---------------------------
# Αποστάσεις
# ---------------------------
def distance(point1, point2):
    """Great-circle distance"""
    lon1, lat1 = radians(point1[0]), radians(point1[1])
    lon2, lat2 = radians(point2[0]), radians(point2[1])

    dlon = fabs(lon1 - lon2)
    dlat = fabs(lat1 - lat2)

    numerator = sqrt((cos(lat2)*sin(dlon))**2 + ((cos(lat1)*sin(lat2)) - (sin(lat1)*cos(lat2)*cos(dlon)))**2)
    denominator = (sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(dlon))
    c = atan2(numerator, denominator)
    return EARTH_MEAN_RADIUS * c

# ---------------------------
# Bearing / Κατεύθυνση
# ---------------------------
def bearing(point1, point2):
    lon1, lat1 = radians(point1[0]), radians(point1[1])
    lon2, lat2 = radians(point2[0]), radians(point2[1])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    theta = atan2(x, y)
    return (degrees(theta) + 360) % 360

def final_bearing(point1, point2):
    return (bearing(point2, point1) + 180) % 360

# ---------------------------
# Προορισμός δεδομένης απόστασης και bearing
# ---------------------------
def destination(point, distance_m, bearing_deg):
    lon1, lat1 = radians(point[0]), radians(point[1])
    theta = radians(bearing_deg)
    delta = distance_m / EARTH_MEAN_RADIUS

    lat2 = asin(sin(lat1)*cos(delta) + cos(lat1)*sin(delta)*cos(theta))
    lon2 = lon1 + atan2(sin(theta)*sin(delta)*cos(lat1), cos(delta)-sin(lat1)*sin(lat2))
    lon2_deg = (degrees(lon2) + 540) % 360 - 180
    lat2_deg = degrees(lat2)
    return (lon2_deg, lat2_deg)

# ---------------------------
# Projection helpers (προαιρετικά)
# ---------------------------
def from4326_to3857(point):
    lon, lat = point
    x = lon * EARTH_EQUATORIAL_METERS_PER_DEGREE
    y = log(tan(radians(45 + lat / 2.0))) * EARTH_EQUATORIAL_RADIUS
    return (x, y)

def from3857_to4326(point):
    x, y = point
    lon = x / EARTH_EQUATORIAL_METERS_PER_DEGREE
    lat = degrees(2.0 * atan(exp(y/EARTH_EQUATORIAL_RADIUS)) - HALF_PI)
    return (lon, lat)

def along_track_distance(start, end, point):
    # Compute along-track distance from start->end line to point
    # This is more advanced, using haversine formula or vector projection
    # For now, use simple Euclidean approximation if lat/lon are in small area
    import numpy as np
    start = np.array(start)
    end = np.array(end)
    point = np.array(point)
    line_vec = end - start
    point_vec = point - start
    line_len = np.dot(line_vec, line_vec)
    if line_len == 0:
        return 0.0
    t = np.dot(point_vec, line_vec) / line_len
    t = np.clip(t, 0, 1)
    proj = start + t * line_vec
    return np.linalg.norm(point - proj)