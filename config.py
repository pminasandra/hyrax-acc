
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
FEATURES_DIR = os.path.join(DATA, "Features")

formats=['png', 'pdf', 'svg']


INDIVIDUALS = ["JN", "TD", "AK", "F6", "PJ", "XP", "QM",
                "C6", "CF", "HW", "BL"]
DROP_BEHS = ["Time Synch", "END", "Out of Sight"]
IGNORE_BEHS = ["Self-grooming", "Chewing"]
MAP_BEHS = {
    'Climbing': 'Locomotion',
    'Rolling': 'Locomotion',
    'Walking': 'Locomotion',
    'Running': 'Locomotion',
    'Lying': 'Lying',
    'Scrabbling': 'Scrabbling',
    'Sitting': 'Sitting',
    'Standing': 'Standing',
    'Vocalising': 'Vocalising'
}
SCALE_DATA = True
LOG_TRANSFORM_VEDBA = True


#Context-classifier
timescales = [1, 3, 5, 7, 9, 11]

#Miscellaneous
ACC_FILE_SEP = ";"
AUDIT_GPS_OFFSET = 18 #FIXME
ACC_GPS_OFFSET = 18
TIMEZONE = 3 # IDT
SUPPRESS_INFORMATIVE_PRINT = False
