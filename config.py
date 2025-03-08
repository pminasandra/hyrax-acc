
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
AUDIT_DIR = os.path.join(DATA, "Readable_Audits") #Re-worked using conversions.py

formats=['png', 'pdf', 'svg']


INDIVIDUALS = ["JN", "TD", "AK", "F6", "PJ", "XP", "QM",
                "C6", "CF", "HW", "BL"]

#Miscellaneous
ACC_FILE_SEP = ";"
ACC_AUDIT_OFFSET = 0 #FIXME
SUPPRESS_INFORMATIVE_PRINT = False
