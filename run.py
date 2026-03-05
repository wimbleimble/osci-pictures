#! /usr/bin/env python

from xml.dom import minidom
from svg.path import parse_path, Path
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from scipy.io.wavfile import write as writewav

POINT_DENSITY = 15

def parse_args():
    parser = ArgumentParser(
        prog="Whooops",
        description="Turn svgs into xy data for oscilloscope music",
        epilog="bottom text")

    parser.add_argument("infile", type=str)
    parser.add_argument("outfile", type=str)
    parser.add_argument("length", type=int)
    parser.add_argument("-d", "--point_density", default=15.0, type=float)
    return parser.parse_args()

def path_to_points(path: Path, point_density: float):
    x = []
    y = []
    for segment in path:
        num_points = point_density * segment.length() + 2
        sweep = np.arange(0, 1.0, 1/num_points)
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

    x = []
    y = []
    for path in paths:
        new_x, new_y = path_to_points(path, args.point_density)
        x += new_x
        y += new_y
    normalized = normalize_point_clouds((x, y))

    single_length = len(normalized[0])
    required_length = float(args.length) * 44100
    iterations = round(required_length/single_length)

    stacked = np.column_stack((normalized[0], normalized[1]))
    print(stacked)
    print(f"{single_length=}, {required_length=}, {iterations=}")

    to_write = np.tile(stacked, (iterations,1))

    writewav(args.outfile, 44100, to_write)

    #plt.scatter(as_pcm[0], as_pcm[1])
    #plt.show()
    


if __name__ == "__main__":
    main()
