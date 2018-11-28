#!/usr/bin/env python

# This file was taken from:
# https://github.com/Kronuz/ansi2html/blob/master/ansi2html.py
# Copyright © 2013 German M. Bravo (Kronuz)
# License: https://github.com/Kronuz/ansi2html/blob/master/LICENSE
# Changes made to original file: removed main, added run() function.
from __future__ import print_function

import re
import sys

_ANSI2HTML_STYLES = {}
# pylint: disable=anomalous-backslash-in-string
ANSI2HTML_CODES_RE = re.compile('(?:\033\\[(\\d+(?:;\\d+)*)?([cnRhlABCDfsurgKJipm]))')
ANSI2HTML_PALETTE = {
    # See http://ethanschoonover.com/solarized
    'solarized': ['#073642', '#D30102', '#859900', '#B58900', '#268BD2', '#D33682', '#2AA198',
                  '#EEE8D5', '#002B36', '#CB4B16', '#586E75', '#657B83', '#839496', '#6C71C4',
                  '#93A1A1', '#FDF6E3'],
    # Above mapped onto the xterm 256 color palette
    'solarized-xterm': ['#262626', '#AF0000', '#5F8700', '#AF8700', '#0087FF', '#AF005F',
                        '#00AFAF', '#E4E4E4', '#1C1C1C', '#D75F00', '#585858', '#626262',
                        '#808080', '#5F5FAF', '#8A8A8A', '#FFFFD7'],
    # Gnome default:
    'tango': ['#000000', '#CC0000', '#4E9A06', '#C4A000', '#3465A4', '#75507B', '#06989A',
              '#D3D7CF', '#555753', '#EF2929', '#8AE234', '#FCE94F', '#729FCF', '#AD7FA8',
              '#34E2E2', '#EEEEEC'],
    # xterm:
    'xterm': ['#000000', '#CD0000', '#00CD00', '#CDCD00', '#0000EE', '#CD00CD', '#00CDCD',
              '#E5E5E5', '#7F7F7F', '#FF0000', '#00FF00', '#FFFF00', '#5C5CFF', '#FF00FF',
              '#00FFFF', '#FFFFFF'],
    'console': ['#000000', '#AA0000', '#00AA00', '#AA5500', '#0000AA', '#AA00AA', '#00AAAA',
                '#AAAAAA', '#555555', '#FF5555', '#55FF55', '#FFFF55', '#5555FF', '#FF55FF',
                '#55FFFF', '#FFFFFF'],
}


def _ansi2html_get_styles(palette):
    if palette not in _ANSI2HTML_STYLES:
        p = ANSI2HTML_PALETTE.get(palette, ANSI2HTML_PALETTE['console'])

        regular_style = {
            '1': '',  # bold
            '2': 'opacity:0.5',
            '4': 'text-decoration:underline',
            '5': 'font-weight:bold',
            '7': '',
            '8': 'display:none',
        }
        bold_style = regular_style.copy()
        for i in range(8):
            regular_style['3%s' % i] = 'color:%s' % p[i]
            regular_style['4%s' % i] = 'background-color:%s' % p[i]

            bold_style['3%s' % i] = 'color:%s' % p[i + 8]
            bold_style['4%s' % i] = 'background-color:%s' % p[i + 8]

        # The default xterm 256 colour p:
        indexed_style = {}
        for i in range(16):
            indexed_style['%s' % i] = p[i]

        for rr in range(6):
            for gg in range(6):
                for bb in range(6):
                    i = 16 + rr * 36 + gg * 6 + bb
                    r = (rr * 40 + 55) if rr else 0
                    g = (gg * 40 + 55) if gg else 0
                    b = (bb * 40 + 55) if bb else 0
                    indexed_style['%s' % i] = ''.join(
                        '%02X' % c if 0 <= c <= 255 else None for c in (r, g, b))

        for g in range(24):
            i = g + 232
            j = g * 10 + 8
            indexed_style['%s' % i] = ''.join(
                '%02X' % c if 0 <= c <= 255 else None for c in (j, j, j))

        _ANSI2HTML_STYLES[palette] = (regular_style, bold_style, indexed_style)
    return _ANSI2HTML_STYLES[palette]


def ansi2html(text, palette='solarized'):
    # pylint: disable=too-many-branches, too-many-statements
    def _ansi2html(m):
        if m.group(2) != 'm':
            return ''
        state = None
        sub = ''
        cs = m.group(1)
        cs = cs.strip() if cs else ''
        for c in cs.split(';'):
            c = c.strip().lstrip('0') or '0'
            if c == '0':
                while stack:
                    sub += '</span>'
                    stack.pop()
            elif c in ('38', '48'):
                extra = [c]
                state = 'extra'
            elif state == 'extra':
                if c == '5':
                    state = 'idx'
                elif c == '2':
                    state = 'r'
            elif state:
                if state == 'idx':
                    extra.append(c)
                    state = None
                    color = indexed_style.get(c)  # 256 colors
                    if color is not None:
                        sub += \
                            '<span style="%s:%s">' % \
                            ('color' if extra[0] == '38' else 'background-color', color)
                        stack.append(extra)
                elif state in ('r', 'g', 'b'):
                    extra.append(c)
                    if state == 'r':
                        state = 'g'
                    elif state == 'g':
                        state = 'b'
                    else:
                        state = None
                        try:
                            color = '#' + ''.join(
                                '%02X' % c if 0 <= c <= 255 else None for x in extra for c in
                                [int(x)])
                        except (ValueError, TypeError):
                            pass
                        else:
                            sub += \
                                '<span style="%s:%s">' % \
                                ('color' if extra[0] == '38' else 'background-color', color)
                            stack.append(extra)
            else:
                if '1' in stack:
                    style = bold_style.get(c)
                else:
                    style = regular_style.get(c)
                if style is not None:
                    sub += '<span style="%s">' % style
                    stack.append(c)  # Still needs to be added to the stack even if style is empty
                    #  (so it can check '1' in stack above, for example)
        return sub
    stack = []
    regular_style, bold_style, indexed_style = _ansi2html_get_styles(palette)
    sub = ANSI2HTML_CODES_RE.sub(_ansi2html, text)
    while stack:
        sub += '</span>'
        stack.pop()
    return sub


def run(file, out):
    try:
        html = ''
        try:
            with open(file) as fp:
                html += '<pre>' + ansi2html(fp.read()) + '</pre>'
        except IOError:
            print("File not found: %r" % 'pylint.txt', file=sys.stderr)

        with open(out, 'w') as out_fp:
            out_fp.write(html)
    except KeyboardInterrupt:
        pass
