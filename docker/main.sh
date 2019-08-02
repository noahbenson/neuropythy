#! /bin/bash
#
# This script is run inside the neuropythy docker and simply invokes neuropythy's main function.
# By Noah C. Benson

set -eo pipefail

# A few things we do first:
# (1) Make sure SUBJECTS_DIR is setup correctly
if ! [ -d /data/required_subjects ]
then SUBJECTS_DIR="/data/required_subjects"
else SUBJECTS_DIR=""
fi
if   [ -d /data/freesurfer_subjects ]
then SUBJECTS_DIR="/data/freesurfer_subjects:$SUBJECTS_DIR"
else mkdir -p /data/local/freesurfer_subjects
     SUBJECTS_DIR="/data/local/freesurfer_subjects:$SUBJECTS_DIR"
fi
if   [ -d /freesurfer_subjects ]
then SUBJECTS_DIR="/freesurfer_subjects:$SUBJECTS_DIR"
fi
if   [ -d /subjects ]
then SUBJECTS_DIR="/subjects:$SUBJECTS_DIR"
fi

# (2) Make sure the HCP_SUBJECTS_DIR is set correctly
if   [ -d /data/hcp/subjects ]
then HCP_SUBJECTS_DIR="/data/hcp/subjects"
else mkdir -p /data/local/hcp/subjects
     HCP_SUBJECTS_DIR="/data/local/hcp/subjects"
fi
if   [ -d /hcp_subjects ]
then HCP_SUBJECTS_DIR="/hcp_subjects:$HCP_SUBJECTS_DIR"
fi
# (3) Make sure the cache is set correctly
NPYTHY_DATA_CACHE_ROOT="/data/cache"
if   [ -d /data/cache ]
then NPYTHY_DATA_CACHE_ROOT="/data/cache"
else mkdir -p /data/local/cache
     NPYTHY_DATA_CACHE_ROOT="/data/local/cache"
fi

export SUBJECTS_DIR
export HCP_SUBJECTS_DIR
export HCP_CREDENTIALS
export NPYTHY_DATA_CACHE_ROOT

# Okay, now interpret the inputs/args
if   [ "$1" = "help" ] || [ "$1" = "-h" ] || [ "$1" = "-help" ] || [ "$1" = "--help" ]
then exec more /help.txt
elif [ "$1" = "README" ] || [ "$1" == "readme" ]
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
