#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimize the problem described by the given json file.
Save the resulting image at the given location.
"""

# System libs
import os
import json
import argparse
import importlib

# Third party libs
import numpy
import scipy.misc
import scipy

# Local libs
from parameters import Parameters
import multires


def run(params):
    print("ndes: " + str(params.nelx) + " x " + str(params.nely))
    print("volfrac: " + str(params.volumeFracMax) + ", rmin: " +
          str(params.filterRadius) + ", penal: " + str(params.penalty))
    print("Filter method: " + params.filterType.name)

    # Load problem boundary conditions dynamically
    imported_problem = importlib.import_module("problems." + params.problemModule)
    bc = imported_problem.BoundaryConditions()

    # Solve the multires problem
    (x, nelx, nely, results) = multires.multires(params.nelx, params.nely, params, bc)

    # Convert to RGB
    x_rgb = 1.0 - numpy.reshape(numpy.repeat(x.reshape((nelx, nely)), 3, axis=1), (nelx, nely, 3))
    return (x_rgb, nelx, nely, results)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", help="parameter file in .json format")
    parser.add_argument("output_image", help="output image filename")
    return parser.parse_args()


def main(args):
    input_params = Parameters.loads(args.input_json)
    # print(os.path.splitext(args.input_json)[0] + "_results.json")
    (x_rgb_out, nelx_out, nely_out, results_out) = run(input_params)
    print(results_out, nelx_out, nely_out)
    # with open(os.path.splitext(args.input_json)[0] + "_results.json", 'w') as json_file:
    #     json_file.write(json.dumps(results_out))
    json.dumps(results_out, indent=4)
    scipy.misc.toimage(x_rgb_out.T, cmin=0.0, cmax=1.0).save(args.output_image)


if __name__ == "__main__":
    main(parse_args())
