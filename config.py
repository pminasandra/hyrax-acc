
# Pranav Minasandra
# pminasandra.github.io
# January 21, 2025

import os
import os.path


#Directories
PROJECTROOT = open(".cw", "r").read().rstrip()
DATA = os.path.join(PROJECTROOT, "Data")
FIGURES = os.path.join(PROJECTROOT, "Figures")
ACC_DIR = os.path.join(DATA, "ACC")

formats=['png', 'pdf', 'svg']


#Miscellaneous
ACC_FILE_SEP = ";"
SUPPRESS_INFORMATIVE_PRINT = False
