#!/usr/bin/env python

from xml.dom import minidom
from svg.path import parse_path, Path
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from scipy.io.wavfile import write as writewav

def parse_args():
    parser = ArgumentParser(
        prog="Whooops",
        description="Turn svgs into xy data for oscilloscope music",
        epilog="bottom text")

    parser.add_argument("infile", type=str)
    parser.add_argument("outfile", type=str)
    parser.add_argument("length", type=float)
    parser.add_argument("-r", "--refresh_rate", default=25, type=float)
    parser.add_argument("-s", "--sample_rate", default=44100, type=int)
    return parser.parse_args()

def path_to_points(path: Path, point_density: float):
    x = []
    y = []
    for segment in path:
        num_points = max(point_density * segment.length(), 2)
        sweep = np.arange(0, 1.0, 1/num_points)
        #noise = np.random.rand(*np.shape(sweep)) / 10
        coords = [segment.point(point) for point in sweep]
        xs = [coord.real for coord in coords]
        ys = [-coord.imag for coord in coords]
        x += xs
        y += ys

    return x, y


def normalize_point_clouds(point_clouds):
    x_max = np.max(point_clouds[0])
    x_min = np.min(point_clouds[0])
    y_max = np.max(point_clouds[1])
    y_min = np.min(point_clouds[1])
    x_span = x_max - x_min
    y_span = y_max - y_min
    offset_x = point_clouds[0] - x_min - (x_span / 2)
    offset_y = point_clouds[1] - y_min - (y_span / 2)

    overall_max = max(x_max, y_max)
    scaled_x = offset_x / overall_max
    scaled_y = offset_y / overall_max

    return np.array(scaled_x, dtype=np.float32), np.array(scaled_y, dtype=np.float32)
    

def main():
    args = parse_args()

    with open(args.infile) as f:
        svg_string = f.read()

    svg_dom = minidom.parseString(svg_string)

    path_strings = [path.getAttribute("d") for path in svg_dom.getElementsByTagName("path")]

    paths = [parse_path(path) for path in path_strings]
    paths_sorted = sorted(paths, key=lambda p: [round(p.point(0).real), round(p.point(0).imag / 10)])
    total_length = sum([path.length() for path in paths])
    print(f"{total_length=}")

    point_density = args.sample_rate / (args.refresh_rate * total_length)

    x = []
    y = []
    for path in paths_sorted:
        new_x, new_y = path_to_points(path, point_density)
        x += new_x
        y += new_y
    normalized = normalize_point_clouds((x, y))

    single_length = len(normalized[0])
    required_length = float(args.length) * args.sample_rate
    iterations = round(required_length/single_length)

    stacked = np.column_stack((normalized[0], normalized[1]))
    print(f"{single_length=}, {required_length=}, {iterations=}")

    tiled = np.tile(stacked, (iterations,1))
    noise = np.random.rand(*np.shape(tiled)) / 1000

    writewav(args.outfile, args.sample_rate, tiled+noise)


if __name__ == "__main__":
    main()
