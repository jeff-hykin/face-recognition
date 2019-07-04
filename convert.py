# generic tools
import sys
import os
from os.path import isabs, isfile, isdir, join, dirname, basename, exists, splitext
from os import remove, getcwd, makedirs, listdir, rename, rmdir
from shutil import move
import glob
import dlib
import numpy as np

# local tools
import utils.main as utils


utils.convert_all("./pictures","./faces-aligned", picture_extension=".jpg")

