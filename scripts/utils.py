import yaml


def read_file(file, verbose=False):
    """
    Read file line by line into list.

    Input: filename

    Output: list of whitespace-stripped lines
    """
    lines = None
    with open( file ) as fid:
        lines = fid.readlines()
    if verbose:
        print "Read {} with {} entries.".format(file, len(lines))
    return [tmp.rstrip() for tmp in lines]
    

def read_config(file, verbose=False):
    """
    Read system config YAML file.

    Input: YAML config file

    Returns: dict
    """
    if verbose:
        print "Read config file {}".format(file)
    fid = open(file, "r")
    config = yaml.load(fid)
    fid.close()
    return config
