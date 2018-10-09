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
elif [ "$1" = "notebook" ]
then exec /usr/local/bin/start-notebook.sh
fi

# A few things we do first:
# (1) Make sure SUBJECTS_DIR is setup correctly
[ -d "$SUBJECTS_DIR" ] || SUBJECTS_DIR=""
[ -d /subjects ]            && SUBJECTS_DIR="$SUBJECTS_DIR:/subjects"
[ -d /freesurfer_subjects ] && SUBJECTS_DIR="$SUBJECTS_DIR:/freesurfer_subjects"
[ -d /required_subjects ]   && SUBJECTS_DIR="$SUBJECTS_DIR:/required_subjects"
# (2) Make sure the HCP_SUBJECTS_DIR is set correctly
[ -d "$HCP_SUBJECTS_DIR" ] || HCP_SUBJECTS_DIR=""
[ -d /hcp_subjects ] && HCP_SUBJECTS_DIR="$HCP_SUBJECTS_DIR:/hcp_subjects"
[ -d /HCP_subjects ] && HCP_SUBJECTS_DIR="$HCP_SUBJECTS_DIR:/HCP_subjects"
[ -d /hcp/subjects ] && HCP_SUBJECTS_DIR="$HCP_SUBJECTS_DIR:/hcp/subjects"
[ -d /HCP/subjects ] && HCP_SUBJECTS_DIR="$HCP_SUBJECTS_DIR:/HCP/subjects"

# Okay, now invoke neuropythy
export SUBJECTS_DIR
export HCP_SUBJECTS_DIR
export HCP_CREDENTIALS

exec python -m neuropythy.__main__ "$@"
