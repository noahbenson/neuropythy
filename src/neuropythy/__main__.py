####################################################################################################
# __main__.py
# The main function, if neuropythy is invoked directly as command.
# By Noah C. Benson

import sys

from neuropythy.commands import (commands)

def main(argv):
    if len(argv) < 1:
        return 0
    if argv[0] not in commands:
        sys.stderr.write('The given command \'' + argv[0] + '\' not recognized.\n')
        return 1
    return commands[argv[0]](argv[1:])

# Run the main function
sys.exit(main(sys.argv[1:]))
