#! /bin/bash
#
# This script is run inside the neuropythy docker and simply invokes neuropythy's main function.
# By Noah C. Benson

set -eo pipefail

# A few things we do first:
# (1) Make sure SUBJECTS_DIR is setup correctly
SUBJECTS_DIR=""
[ -d /data/required_subjects ]   && SUBJECTS_DIR="/data/required_subjects"
[ -d /data/freesurfer_subjects ] && SUBJECTS_DIR="$SUBJECTS_DIR:/data/freesurfer_subjects"
# (2) Make sure the HCP_SUBJECTS_DIR is set correctly
HCP_SUBJECTS_DIR=""
[ -d /data/hcp/subjects ] && HCP_SUBJECTS_DIR="$HCP_SUBJECTS_DIR:/data/hcp/subjects"
# (3) Make sure the cache is set correctly
NPYTHY_DATA_CACHE_ROOT=""
[ -d /data/cache ] && NPYTHY_DATA_CACHE_ROOT="/data/cache"

export SUBJECTS_DIR
export HCP_SUBJECTS_DIR
export HCP_CREDENTIALS
export NPYTHY_DATA_CACHE_ROOT

# Okay, now interpret the inputs/args
if [ "$1" = "README" ] || [ "$1" == "readme" ]
then exec cat /README.md
elif [ "$1" = "LICENSE" ] || [ "$1" == "license" ]
then exec cat /LICENSE.txt
elif [ "$1" = "bash" ]
then exec /bin/bash
elif [ "$1" = "notebook" ] || [ -z "$1" ]
then exec /usr/local/bin/start-notebook.sh
fi

# Okay, now invoke neuropythy
exec python -m neuropythy.__main__ "$@"
