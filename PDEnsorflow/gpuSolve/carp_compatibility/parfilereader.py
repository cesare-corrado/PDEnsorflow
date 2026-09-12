#!/usr/bin/env python
"""
    ParFileReader: reads a `.par` parameter file into an ordered list of
    assignments.

    A `.par` file is a flat sequence of `key = value` lines. This class
    implements the *lexical* rules of that format only. It does not know which
    keys exist, what they mean, or what type they carry: that vocabulary lives
    in ParameterMapper, so the same lexer can be reused to inspect a file
    without pulling in the whole simulation vocabulary.

    The rules, in the order they are applied to each physical line:

      1. a `#` starts a comment that runs to the end of the line, UNLESS it
         sits inside a double-quoted string. Values such as
         `im_param = "a#b"` therefore survive intact;
      2. inside a quoted string, a backslash escapes the next character, so a
         quote can appear in a value;
      3. a line whose last non-blank character is a backslash continues on the
         following line. The pieces are joined with a single space, which is
         why a continuation may be indented freely;
      4. what remains is split on the FIRST `=`. Everything before it is the
         key, everything after is the value. Keys carry their own structure
         (`gregion[0].g_il`, `stim[0].elec.p0[2]`) which is left as text here
         and decoded by ParameterMapper;
      5. a value wrapped in double quotes is unquoted; a bare value keeps its
         internal whitespace but loses the surrounding blanks.

    Order is preserved and repeated keys are NOT collapsed, because the
    resolution rule is "last assignment wins" and the caller replays the list
    to apply it.

    Copyright 2022-2023 Cesare Corrado (c.corrado@imperial.ac.uk)
"""


def strip_comments(line: str) -> str:
    """ strip_comments(line) removes a trailing `#` comment from line.
        A `#` inside a double-quoted string is data, not a comment, so the scan
        tracks the quoting state instead of searching for the character.
    """
    escaped  : bool = False
    in_quote : bool = False
    for ipos, char in enumerate(line):
        if in_quote:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_quote = False
            continue
        if char == '"':
            in_quote = True
        elif char == '#':
            return(line[:ipos])
    return(line)


def has_line_continuation(line: str) -> bool:
    """ has_line_continuation(line) tells whether line ends with a backslash and
        therefore continues on the next one.
    """
    trimmed = line.strip()
    return(len(trimmed) > 0 and trimmed[-1] == '\\')


def strip_line_continuation(line: str) -> str:
    """ strip_line_continuation(line) returns line without its trailing
        backslash, trimmed.
    """
    trimmed = line.strip()
    if len(trimmed) > 0 and trimmed[-1] == '\\':
        return(trimmed[:-1].strip())
    return(trimmed)


def decode_quoted(raw: str) -> str:
    """ decode_quoted(raw) unquotes a double-quoted value and resolves its
        backslash escapes. A bare value is returned unchanged, so callers do not
        have to know which form the file used.
    """
    if len(raw) < 2 or raw[0] != '"' or raw[-1] != '"':
        return(raw)
    body    : str  = raw[1:-1]
    out     : list = []
    escaped : bool = False
    for char in body:
        if escaped:
            out.append(char)
            escaped = False
        elif char == '\\':
            escaped = True
        else:
            out.append(char)
    return(''.join(out))


def split_assignment(statement: str) -> tuple:
    """ split_assignment(statement) splits `key = value` on the FIRST `=` and
        returns the (key, value) pair. Splitting on the first one matters:
        values such as "V_gate=0.1,a_crit=0.1" contain further `=` signs that
        belong to the value.
    """
    ieq = statement.find('=')
    if ieq < 0:
        raise ValueError('not an assignment: {}'.format(statement))
    key   = statement[:ieq].strip()
    value = decode_quoted(statement[ieq + 1:].strip())
    if len(key) == 0:
        raise ValueError('empty key in: {}'.format(statement))
    return((key, value))


class ParFileReader:
    """
    class ParFileReader: reads a `.par` parameter file.
    Provided a file name, this class returns the assignments it contains, in
    order. NOTE: this validates the syntax only, not the parameter names.
    """

    def __init__(self):
        self.__fname : str        = None
        self.__assignments : list = None

    def fname(self) -> str:
        """ fname() returns the name of the file that was read """
        return(self.__fname)

    def assignments(self) -> list:
        """ assignments() returns the list of (key, value) pairs of the last
            read, in file order; None if nothing was read yet
        """
        return(self.__assignments)

    def read(self, fname: str) -> list:
        """ read(fname) reads the parameter file fname and returns its
            assignments as a list of (key, value) pairs, in file order
        """
        try:
            with open(fname, 'r') as fpar:
                lines = fpar.readlines()
            self.__fname       = fname
            self.__assignments = self.__parse(lines, fname)
            return(self.__assignments)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    def __parse(self, lines: list, fname: str) -> list:
        """ turns the raw lines into (key, value) pairs, joining continuations
            and reporting the line where a bad statement starts
        """
        assignments  : list = []
        pending      : str  = ''
        pending_line : int  = 0
        for ilin, line in enumerate(lines):
            uncommented = strip_comments(line)
            continued   = has_line_continuation(uncommented)
            chunk       = strip_line_continuation(uncommented)
            if len(pending) == 0:
                pending_line = 1 + ilin
            if len(chunk) > 0:
                pending = chunk if len(pending) == 0 else '{} {}'.format(pending, chunk)
            if continued:
                continue
            if len(pending) > 0:
                try:
                    assignments.append(split_assignment(pending))
                except ValueError as err:
                    raise ValueError('{}:{}: {}'.format(fname, pending_line, err))
                pending = ''
        # a file that ends on a continuation still carries a statement
        if len(pending) > 0:
            assignments.append(split_assignment(pending))
        return(assignments)
