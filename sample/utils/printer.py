from __future__ import division

import re
import os
import curses
import fcntl
import termios
import struct

from pyparsing import *


def print_sub_header(header, bold=False):
  '''
  Prints header as a sub header
  '''
  print_header(header, size=1/4, buffer=1, bold=bold)

def print_header(header, size=1, buffer=2, filler='=', bold=False):
  '''
  Print 'header' in the center of the terminal and prints a dividing line
  below it.

  Parameters
  ----------
  header: the header to print in the center of the header
  size: the proportion of the terminal that the header should extend over
  buffer: the number of new lines below the header
  bold: whether or not to print the header in bold

  TODO: KNOWN BUG: If there is formatting within the header (aka a color.END tag)
        then the bold will not carry over to the filler after the header
  '''
  # set accent(s)
  if bold:
    wrapper_start = color.BOLD
    wrapper_end = color.END
  else:
    wrapper_start = ''
    wrapper_end = ''

  rows, columns = _getTerminalSize()

  #scale by desired size
  rows = int(size * int(rows))
  columns = int(columns)
  #build string
  build_header = ''
  build_break = ''
  temp = header
  # pad back in the ansii formatting string length so that the header isn't short
  padding = len(temp) - len(nonAnsiString(temp))
  header_len = len(header)
  for i in range(rows + padding):
    start_writing = rows // 2 - header_len // 2
    end_writing = rows // 2 + header_len // 2 + 1
    # spaces to pad the header string
    if i == (start_writing - 1) or (i - start_writing) == header_len:
      build_header += ' '
    elif i < start_writing or i - start_writing >= header_len:
      build_header += filler
    else:
      build_header += header[i - start_writing]
  # add formatting if it is present
  build_header = wrapper_start + build_header + wrapper_end
  print(build_header)
  for i in range(buffer):
    print('')

def print_line_break(size=1, break_char='=', bold=False):
  '''
  Prints a line break in terminal. ie =================================.
  '''
  # set accent(s)
  if bold:
    wrapper_start = color.BOLD
    wrapper_end = color.END
  else:
    wrapper_start = ''
    wrapper_end = ''
  
  rows, columns = _getTerminalSize()
  #scale by desired size
  rows = int(size * int(rows))
  columns = int(columns)
  print('')
  #build string
  build_header = ''
  build_break = ''
  for i in range(rows // len(break_char)):
    #NOTE: This will look a bit funky if it doesn't divide evenly
    build_header += break_char
  build_header = wrapper_start + build_header + wrapper_end
  print(build_header)
  print('')

#found at https://gist.github.com/acaranta/e4fbfbbd25a9cd720ef0
def _getTerminalSize():
  env = os.environ
  def ioctl_GWINSZ(fd):
    try:
      cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
    '1234'))
    except:
        return
    return cr
  cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
  if not cr:
    try:
        fd = os.open(os.ctermid(), os.O_RDONLY)
        cr = ioctl_GWINSZ(fd)
        os.close(fd)
    except:
        pass
  if not cr:
    cr = (env.get('LINES', 25), env.get('COLUMNS', 80))
  return int(cr[1]), int(cr[0])

def nonAnsiString(a_str):
  '''
  Returns a string with ANSII codes stripped out. Useful for calculating length.
  Copied from:
  https://stackoverflow.com/questions/2186919/getting-correct-string-length-in-python-for-strings-with-ansi-color-codes.
  '''
  ESC = Literal('\x1b')
  integer = Word(nums)
  escapeSeq = Combine(ESC + '[' + Optional(delimitedList(integer,';')) + 
                  oneOf(list(alphas)))
  return Suppress(escapeSeq).transformString(a_str)

class color:
  PURPLE = '\033[95m'
  CYAN = '\033[96m'
  DARKCYAN = '\033[36m'
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  RED = '\033[91m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'
  END = '\033[0m'
  #TODO: Create wrapper method ie color.make_bold()