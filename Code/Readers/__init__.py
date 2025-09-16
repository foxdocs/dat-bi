# Special Python file used to mark a directory as a Python package
import pandas as pd
import numpy as np
import json, os

# modules to import in case of "from mypackage import *"
__all__ = ['readfile', 'readweb']


from . import readfile, readweb
from .readfile import jsonio
from .readweb import api

# to be printed at import of this package
print("package has been imported")
