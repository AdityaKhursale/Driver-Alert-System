import os
import re

def remove_files(dir, pattern, raise_exception=False):
    try:
        for f in os.listdir(dir):
            if re.search(pattern, f):
                os.remove(os.path.join(dir, f))
    except FileNotFoundError as e:
        if raise_exception:
            raise(e)

