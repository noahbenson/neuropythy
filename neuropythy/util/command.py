####################################################################################################
# neuropythy/util/command.py
# This file implements the command-line tools that are available as part of neuropythy.

from pysistence import make_dict

class CommandLineParser(object):
    '''
    CommandLineParser()
    '''

    def __init__(self, instructions):
        'See help(CommandLineParser).'
        wflags = {}
        cflags = {}
        wargs = {}
        cargs = {}
        defaults = {}
        for row in instructions:
            if not hasattr(row, '__iter__') or len(row) < 3 or len(row) > 4 or \
               any(x is not None and not isinstance(x, basestring) for x in row[:3]):
                raise ValueError('Invalid instruction row: %s ' % row)
            (c, w, var, dflt) = row if len(row) == 4 else (list(row) + [None])
            defaults[var] = dflt
            if dflt is True or dflt is False:
                if c is not None: cflags[c] = var
                if w is not None: wflags[w] = var
            else:
                if c is not None: cargs[c] = var
                if w is not None: wargs[w] = var
        self.default_values = make_dict(defaults)
        self.flag_words = make_dict(wflags)
        self.flag_characters = make_dict(cflags)
        self.option_words = make_dict(wargs)
        self.option_characters = make_dict(cargs)

    def __call__(self, *args):
        if len(args) > 0 and not isinstance(args[0], basestring) and hasattr(args[0], '__iter__'):
            args = list(args)
            return self.__call__(*(args[0] + args[1:]))
        parse_state = None
        more_opts = True
        remaining_args = []
        opts = dict(self.default_values)
        wflags = self.flag_words
        cflags = self.flag_characters
        wargs  = self.option_words
        cargs  = self.option_characters
        dflts  = self.default_values
        for arg in args:
            larg = arg.lower()
            if parse_state is not None:
                opts[parse_state] = arg
                parse_state = None
            else:
                if arg == '': pass
                elif more_opts and arg[0] == '-':
                    if len(arg) == 1:
                        remaining_args.append(arg)
                    elif arg[1] == '-':
                        trimmed = arg[2:]
                        if trimmed == '':     more_opts = False
                        if trimmed in wflags: opts[wflags[trimmed]] = not dflts[wflags[trimmed]]
                        else:
                            parts = trimmed.split('=')
                            if len(parts) == 1:
                                if trimmed not in wargs:
                                    raise ValueError('Unrecognized flag/option: %s' % trimmed)
                                # the next argument specifies this one
                                parse_state = wargs[trimmed]
                            else:
                                k = parts[0]
                                if k not in wargs:
                                    raise ValueError('Unrecognized option: %s' % k)
                                opts[wargs[k]] = trimmed[(len(k) + 1):]
                    else:
                        trimmed = arg[1:]
                        for (k,c) in enumerate(trimmed):
                            if c in cflags: opts[cflags[c]] = not dflts[cflags[c]]
                            elif c in cargs:
                                remainder = trimmed[(k+1):]
                                if len(remainder) > 0: opts[cargs[c]] = remainder
                                else:
                                    # next argument...
                                    parse_state = cargs[c]
                                break
                else:
                    remaining_args.append(arg)
        if parse_state is not None:
            raise ValueError('Ran out of arguments while awaiting value for %s' % parse_state)
        # that's done; all args are parsed
        return (remaining_args, opts)
