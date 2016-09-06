#! /bin/bash
#
# This script is run inside the neuropythy docker and simply invokes neuropythy's main function.
# By Noah C. Benson

if [ "$1" = "README" ] || [ "$1" == "readme" ]
then exec cat /README.md
elif [ "$1" = "LICENSE" ] || [ "$1" == "license" ]
then exec cat /LICENSE.txt
elif [ "$1" = "bash" ]
then exec /bin/bash
fi

# A few things we do first:
# (1) Make sure SUBJECTS_DIR is setup correctly
SUBJECTS_DIR=""
[ -d /subjects ] && SUBJECTS_DIR="/subjects"
[ -d /freesurfer_subjects ] && SUBJECTS_DIR="$SUBJECTS_DIR:/freesurfer_subjects"

# Okay, now invoke neuropythy
export SUBJECTS_DIR
exec python -m neuropythy.__main__ "$@"
