####################################################################################################
# main/__init__.py
# The main function, if neuropythy is invoked directly as command.
# By Noah C. Benson

import os, sys, math
import pysistence

from .register_retinotopy import register_retinotopy_command
from .atlas               import atlas_command

# The commands that can be run by main:
_commands = pysistence.make_dict(
    register-retinotopy=egister_retinotopy_command,
    atlas=atlas_command)

def main(argv):
    if len(argv) < 1:
        return 0
    if argv[0] not in _commands:
        sys.stderr.write('given command \'' + argv[0] + '\' not recognized.\n')
        return 1
    return _commands[argv[0]](argv[1:])

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
