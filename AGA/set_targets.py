from optparse import OptionParser
import subprocess
import tempfile
import numpy as np
import yaml
import h5py
import sys
import os


def setup_parser():
    """
    Setup the CLI parsing.
    """
    parser = OptionParser()
    parser.add_option("-i", "--input_file",                                         help="ASCII input file.")
    parser.add_option("-o", "--output_file",                                        help="YAML output file.")
    parser.add_option("-v", action="store_true", default=False, dest="verbose",     help="Verbose output.")
    parser.add_option("-t", "--attribute", default="pose",                          help="Attribute (pose|depth)")
    
    return parser


def set_phi_targets(info, range=np.arange(1, 5, 0.5)):
    """
    Configure attribute target values.
    """
    for interval in info['intervals']:
        lo, hi = interval
        targets = []
        for r in range:
            if r >= lo and r <= hi:
                continue
            targets.append(r)
        info['PHI_targets'].append(tuple(targets))


def main():
    data = None
    info = None

    parser = setup_parser()
    (options, args) = parser.parse_args()


    with open( options.input_file ) as fid:
        data = fid.readlines()
        data = [l.rstrip() for l in data]

    for line in data:
        parts = line.split(',')

        lo = float(parts[0]) # lower interval boundary
        hi = float(parts[1]) # upper interval boundary

        phi_train_file = parts[-1]

        if info is None:
            info = {
                'intervals':            [],                 # [l_i, h_i]
                'PHI_train_files' :     [],                 # Training data file for phi 
                'PHI_pre' :             'PHI_pre.t7',       # Name of pretrained phi model
                'GAMMA_agnostic_model': 'agnosticGAMMA.t7', # Name of pretrained gamma model
                'PHI_models':           [],                 # Will hold the final trained phi model
                'PHI_targets' :         []                  # Will hold the attribute target values for the phi models
                }
        info['intervals'].append((lo,hi))
        info['PHI_train_files'].append(phi_train_file)

    if options.attribute == "depth":
        set_phi_targets(info)
    elif options.attribute == "pose":
        set_phi_targets(info, np.arange(np.deg2rad(45), np.pi, np.deg2rad(25)))
    else:
        print("Attribute %s unknown!", options.attribute)
        sys.exit(-1)

    fid = open(options.output_file, 'w')
    yaml.dump(info, fid)
    fid.close()


if __name__ == "__main__":
    main()