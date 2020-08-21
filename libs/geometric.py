"""
Copyright (C) 2020 Intel Corporation

SPDX-License-Identifier: BSD-3-Clause
"""

from shapely.geometry import LineString, Point, Polygon


def get_polygon(point_list):
    return Polygon(point_list)


def get_line(data):
    return LineString(data)


def get_point(data):
    return Point(data)

