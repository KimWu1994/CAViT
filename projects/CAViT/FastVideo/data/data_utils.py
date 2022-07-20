import logging
import json
import errno
import os
import os.path as osp
logger = logging.getLogger("fastreid."+__name__)

def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
            logger.info(f"making dir {dirname}")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        logger.info(f"loading json from {fpath}")
        obj = json.load(f)
    return obj



def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        logger.info(f"dumping json from {fpath}")
        json.dump(obj, f, indent=4, separators=(',', ': '))




