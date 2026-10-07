#!/usr/bin/env python3
"""hxc: compile a small structured register language (json2dir.hxs) into a Hexagony program.

Pipeline:
  1. parse the source: registers, assignments with infix expressions, read/out/print,
     if/elif/else/end, while/end, fail, halt (no break/continue, no procedures)
  2. memory: every register is one edge of Hexagony's hexagonal memory grid; "home" is a
     fixed edge T whose two neighbours A (left) and B (right) form the ALU:
     x = y op z  ==  copy y to A, copy z to B, T = A op B, copy T to x.
     Copies walk value-by-value along a path of scratch edges ("letter &" per hop); the
     MP moves are found by breadth-first search over memory-pointer states.
     Values are Hexagony's unbounded integers, which the source uses as byte strings.
  3. control flow is laid out structurally, as tiles in the parallelogram of the hexagon
     spanned by the E and SE directions (a square grid where SE = "down", NW = "up"):
       straight code   one row, IP moving east
       if              '<' branch: positive -> lower arm, otherwise -> upper arm; both
                       arms merge again at a '>' that turns SE and NE into E
       while           a loop head '>' merging entry and back edge, condition code, '<'
       sequences       folded into lines of bounded width, odd lines rotated by 180
                       degrees (Hexagony's mirrors and branches are symmetric under that
                       rotation, with '<' and '>' swapped)
     Every straight piece of code starts and ends with the MP at home, conditions are
     computed into T, so all merging paths agree on the memory-pointer state.
  4. the parallelogram is embedded in the smallest regular hexagon whose top-left corner
     (where the IP starts, moving east) is the tile's top-left corner.
Only stdlib.
"""
import re
import sys
from collections import deque

WMAX = 150   # target width of folded sequences

# ---------------------------------------------------------------------------
# parsing

TOKEN = re.compile(r"\s*(?:(0x[0-9A-Fa-f]+|\d+)|('(?:\\.|[^'\\])')|([A-Za-z_]\w*)|(==|!=|<=|>=|[-+*/%<>()]))")
ESC = {'n': '\n', 'r': '\r', 't': '\t', '\\': '\\', '"': '"', "'": "'", '0': '\0'}


def unescape(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == '\\':
            i += 1
            c = ESC[s[i]]
        out.append(c)
        i += 1
    return ''.join(out)


def tokenize(src):
    toks, pos = [], 0
    src = src.rstrip()
    while pos < len(src):
        m = TOKEN.match(src, pos)
        if not m:
            raise SyntaxError('bad expression near: ' + src[pos:])
        num, ch, name, op = m.groups()
        if num:
            toks.append(('num', int(num, 0)))
        elif ch:
            toks.append(('num', ord(unescape(ch[1:-1]))))
        elif name:
            toks.append(('name', name))
        else:
            toks.append(('op', op))
        pos = m.end()
    return toks


class Expr:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def parse(self):
        e = self.add()
        if self.i != len(self.t):
            raise SyntaxError('trailing tokens %r' % (self.t[self.i:],))
        return e

    def add(self):
        e = self.mul()
        while self.peek() in (('op', '+'), ('op', '-')):
            e = fold(self.take()[1], e, self.mul())
        return e

    def mul(self):
        e = self.unary()
        while self.peek() in (('op', '*'), ('op', '/'), ('op', '%')):
            e = fold(self.take()[1], e, self.unary())
        return e

    def unary(self):
        if self.peek() == ('op', '-'):
            self.take()
            return fold('-', ('num', 0), self.unary())
        tok = self.take()
        if tok[0] == 'num':
            return ('num', tok[1])
        if tok[0] == 'name':
            return ('reg', tok[1])
        if tok == ('op', '('):
            e = self.add()
            if self.take() != ('op', ')'):
                raise SyntaxError('missing )')
            return e
        raise SyntaxError('unexpected %r' % (tok,))


def fold(op, a, b):
    if a[0] == 'num' and b[0] == 'num':
        x, y = a[1], b[1]
        return ('num', {'+': x + y, '-': x - y, '*': x * y, '/': x // y if y else 0, '%': x % y if y else 0}[op])
    return ('bin', op, a, b)


def expr(s):
    return Expr(tokenize(s)).parse()


RELOPS = ('==', '!=', '<=', '>=', '<', '>')


def cond(s):
    for op in RELOPS:
        k = s.find(op)
        if k >= 0:
            return (op, expr(s[:k]), expr(s[k + len(op):]))
    raise SyntaxError('condition needs a comparison: ' + s)


STR = re.compile(r'^"((?:\\.|[^"\\])*)"$')


def parse_program(text):
    lines = []
    for n, raw in enumerate(text.split('\n'), 1):
        s = raw.strip()
        if s and not s.startswith('#'):
            lines.append((n, s))
    pos = 0

    def string(s):
        m = STR.match(s)
        if not m:
            raise SyntaxError('expected a string literal: ' + s)
        return unescape(m.group(1))

    def body(enders):
        nonlocal pos
        out = []
        while True:
            if pos >= len(lines):
                raise SyntaxError('unexpected end of file')
            n, s = lines[pos]
            word = s.split()[0]
            if word in enders:
                return out, word
            pos += 1
            try:
                out.append(stmt(s, word))
            except SyntaxError as e:
                raise SyntaxError('line %d: %s' % (n, e))

    def stmt(s, word):
        nonlocal pos
        rest = s[len(word):].strip()
        if word == 'if':
            arms = [cond(rest)]
            bodies = []
            while True:
                b, end = body(('elif', 'else', 'end'))
                bodies.append(b)
                s2 = lines[pos][1]
                pos += 1
                if end == 'elif':
                    arms.append(cond(s2[4:].strip()))
                    continue
                els = None
                if end == 'else':
                    els, _ = body(('end',))
                    pos += 1
                # elif chains become nested ifs
                for c, b in reversed(list(zip(arms, bodies))):
                    els = [('if', c, b, els or [])]
                return els[0]
        if word == 'while':
            b, _ = body(('end',))
            pos += 1
            return ('while', cond(rest), b)
        if word == 'print':
            return ('print', string(rest))
        if word == 'fail':
            return ('fail', string(rest))
        if word == 'out':
            return ('out', rest)
        if word == 'halt':
            return ('halt',)
        if word == 'reg':
            return ('nop',)
        m = re.match(r'^([A-Za-z_]\w*)\s*=\s*(.*)$', s)
        if m:
            if m.group(2) == 'read':
                return ('read', m.group(1))
            return ('set', m.group(1), expr(m.group(2)))
        raise SyntaxError('unknown statement: ' + s)

    out = []
    while pos < len(lines):
        n, s = lines[pos]
        pos += 1
        try:
            out.append(stmt(s, s.split()[0]))
        except SyntaxError as e:
            raise SyntaxError('line %d: %s' % (n, e))
    return out


# ---------------------------------------------------------------------------
# memory model (identical to the reference interpreter's memory.rb)

def mleft(s):
    q, r, e, cw = s
    if e == 'NE':
        return (q, r - 1, 'SE', True) if not cw else (q + 1, r - 1, 'SE', False)
    if e == 'E':
        return (q, r, 'NE', cw) if not cw else (q, r + 1, 'NE', cw)
    return (q, r, 'E', cw) if not cw else (q - 1, r + 1, 'E', cw)


def mright(s):
    q, r, e, cw = s
    if e == 'NE':
        return (q, r - 1, 'E', cw) if not cw else (q, r, 'E', cw)
    if e == 'E':
        return (q + 1, r - 1, 'SE', cw) if not cw else (q, r, 'SE', cw)
    return (q, r + 1, 'NE', True) if not cw else (q - 1, r + 1, 'NE', False)


def mrev(s):
    return s[:3] + (not s[3],)


MOVES = [('{', mleft), ('}', mright), ('=', mrev),
         ('"', lambda s: mrev(mright(mrev(s)))), ("'", lambda s: mrev(mleft(mrev(s))))]


def edge(s):
    return s[:3]


def vertex(s):
    return frozenset([s[:3], mleft(s)[:3], mright(s)[:3]])


def neighbours(e):
    out = []
    for cw in (False, True):
        s = e + (cw,)
        out += [mleft(s)[:3], mright(s)[:3]]
    return out


HOME = (0, 0, 'E', False)
T = edge(HOME)
A = edge(mleft(HOME))
B = edge(mright(HOME))
ALU = vertex(HOME)


def candidate_slots():
    """Register edges: every other SE edge, so that the scratch edges stay connected."""
    regs = set((q, r, 'SE') for q in range(-14, 15) for r in range(-14, 15) if (q - r) % 2 == 0)
    regs -= {A, B, T}
    dist = {T: 0}
    dq = deque([T])
    while dq:
        e = dq.popleft()
        for f in neighbours(e):
            if f in dist:
                continue
            dist[f] = dist[e] + 1
            if f not in regs and abs(f[0]) < 16 and abs(f[1]) < 16:
                dq.append(f)
    return sorted((e for e in regs if e in dist), key=lambda e: (dist[e], e)), regs


LETTERS = [c for c in map(chr, range(33, 127)) if c.isalpha()]


class Mem:
    """Generates straight-line code; tracks the memory pointer state."""

    def __init__(self, slots, regedges):
        self.slot = slots              # register name -> edge
        self.regedges = regedges       # every edge that might hold a register
        self.state = HOME
        self.code = []
        self.navcache = {}

    def emit(self, s):
        for c in s:
            for m, f in MOVES:
                if c == m:
                    self.state = f(self.state)
        self.code.append(s)

    def goto(self, goal):
        """Move the MP to a state satisfying goal(state) (shortest move sequence)."""
        key = (self.state, goal)
        if key in self.navcache:
            self.emit(self.navcache[key])
            return
        test = goal[1]
        start = self.state
        seen = {start: ''}
        dq = deque([start])
        while dq:
            s = dq.popleft()
            if test(s):
                self.navcache[key] = seen[s]
                self.emit(seen[s])
                return
            for m, f in MOVES:
                t = f(s)
                if t not in seen:
                    seen[t] = seen[s] + m
                    dq.append(t)
        raise RuntimeError('unreachable')

    def go_edge(self, e, v=None):
        if v is None:
            self.goto((('e', e), lambda s: edge(s) == e))
        else:
            self.goto((('ev', e, v), lambda s: edge(s) == e and vertex(s) == v))

    def go_home(self):
        self.goto((('home',), lambda s: s == HOME))

    def scratch(self, e, avoid):
        return e not in self.regedges and e not in avoid

    def path(self, src, dst, avoid):
        prev = {src: None}
        dq = deque([src])
        while dq:
            e = dq.popleft()
            if e == dst:
                break
            for f in neighbours(e):
                if f not in prev and (f == dst or self.scratch(f, avoid)) and abs(f[0]) < 20 and abs(f[1]) < 20:
                    prev[f] = e
                    dq.append(f)
        p = [dst]
        while prev[p[-1]] is not None:
            p.append(prev[p[-1]])
        return p[::-1]

    def copy_from(self, src):
        """MP is on an edge adjacent to src (facing their common vertex): copy src here."""
        if edge(mright(self.state)) == src:
            self.emit('A&')
        else:
            assert edge(mleft(self.state)) == src
            self.emit('A~&')

    def transfer(self, src, dst, avoid=()):
        if src == dst:
            return
        p = self.path(src, dst, set(avoid))
        for a, b in zip(p, p[1:]):
            v = (set(vertex_of(a)) & set(vertex_of(b)))
            self.go_edge(b, v.pop())
            self.copy_from(a)

    def const_here(self, k, avoid=()):
        """Set the current edge to k (any value)."""
        neg = k < 0
        k = abs(k)
        s = const_unary(k)
        if s is None:
            # two neighbours at one vertex of this edge, set to letters, combined
            e = edge(self.state)
            for cw in (False, True):
                st = e + (cw,)
                l, r = edge(mleft(st)), edge(mright(st))
                if self.scratch(l, avoid) and self.scratch(r, avoid):
                    break
            else:
                raise RuntimeError('no scratch neighbours for a constant at %r' % (e,))
            a, op, b, digits = const_binary(k)
            v = vertex(st)
            self.go_edge(l)
            self.emit(a)
            self.go_edge(r)
            self.emit(b)
            self.go_edge(e, v)
            assert edge(mleft(self.state)) == l
            self.emit(op + digits)
        else:
            self.emit(s)
        if neg:
            self.emit('~')

    def operand(self, x, dst, avoid=()):
        if x[0] == 'num':
            self.go_edge(dst)
            self.const_here(x[1], avoid)
        else:
            self.transfer(self.slot[x[1]], dst, avoid)


def vertex_of(e):
    return [vertex(e + (False,)), vertex(e + (True,))]


def const_unary(k):
    if 0 <= k < 127 and chr(k).isalpha():
        return chr(k)
    s = str(k)
    for n in (2, 3):
        if len(s) > n and int(s[:n]) < 127 and chr(int(s[:n])).isalpha() and s[:n][0] != '0':
            return chr(int(s[:n])) + s[n:]
    for d in (1, 2, 3):
        if k - d >= 0 and k - d < 127 and chr(k - d).isalpha():
            return chr(k - d) + ')' * d
        if k + d < 127 and chr(k + d).isalpha():
            return chr(k + d) + '(' * d
    return None


def const_binary(k):
    """k = (letter op letter) followed by decimal digits."""
    s = str(k)
    best = None
    for cut in range(len(s), 0, -1):
        head, digits = int(s[:cut]), s[cut:]
        if cut > 1 and s[0] == '0':
            continue
        for a in LETTERS:
            for b in LETTERS:
                for op, f in (('-', lambda x, y: x - y), ('+', lambda x, y: x + y), ('*', lambda x, y: x * y)):
                    if f(ord(a), ord(b)) == head:
                        return a, op, b, digits
    return 'A', '-', 'A', s   # zero, then all digits


# ---------------------------------------------------------------------------
# code generation: statements -> tiles

class Tile:
    """A rectangle of cells; the IP enters (0, ey) moving east and leaves (w-1, xy) east."""

    def __init__(self, w, h, ey, xy, cells=None):
        self.w, self.h, self.ey, self.xy = w, h, ey, xy
        self.cells = cells if cells is not None else {}


def straight(code):
    return Tile(len(code), 1, 0, 0, {(i, 0): c for i, c in enumerate(code)})


class Builder:
    """Places tiles at offsets into one cell map and computes the bounding box."""

    def __init__(self):
        self.cells = {}

    def put(self, x, y, c):
        old = self.cells.get((x, y))
        if old is not None and old != c:
            raise RuntimeError('layout conflict at %r: %r vs %r' % ((x, y), old, c))
        self.cells[(x, y)] = c

    def place(self, t, x, y):
        """Put tile t with its top-left corner at (x, y)."""
        for (i, j), c in t.cells.items():
            self.put(x + i, y + j, c)
        for j in range(t.h):     # reserve the tile's whole box (nops) to catch overlaps
            for i in range(t.w):
                self.cells.setdefault((x + i, y + j), '.')

    def line(self, x0, x1, y, c='.'):
        for x in range(x0, x1 + 1):
            self.put(x, y, c)

    def col(self, x, y0, y1, c='.'):
        for y in range(y0, y1 + 1):
            self.put(x, y, c)

    def tile(self, ex, ey, xx, xy):
        """Normalize: (ex, ey) is the entry cell, (xx, xy) the exit cell."""
        xs = [p[0] for p in self.cells]
        ys = [p[1] for p in self.cells]
        x0, y0 = min(xs), min(ys)
        assert ex == x0, (ex, x0)
        w = max(xs) - x0 + 1
        assert xx == max(xs), (xx, max(xs))
        cells = {(x - x0, y - y0): c for (x, y), c in self.cells.items() if c != '.'}
        return Tile(w, max(ys) - y0 + 1, ey - y0, xy - y0, cells)


def rotate(t):
    sw = {'<': '>', '>': '<'}
    cells = {(t.w - 1 - x, t.h - 1 - y): sw.get(c, c) for (x, y), c in t.cells.items()}
    return Tile(t.w, t.h, t.h - 1 - t.ey, t.h - 1 - t.xy, cells)


def if_tile(U, L):
    """'<' at (0,0): positive -> L (lower), otherwise -> U (upper)."""
    b = Builder()
    yU = -2 - (U.h - 1) + U.ey       # U's box bottom at row -2
    yL = 2 + L.ey                    # L's box top at row 2
    m = max(2 + U.w, 2 + L.w)
    yUx = yU - U.ey + U.xy
    yLx = yL - L.ey + L.xy
    b.put(0, 0, '<')
    b.put(1, -1, '|')
    b.col(1, yU + 1, -2)
    b.put(1, yU, '/')
    b.place(U, 2, yU - U.ey)
    b.line(2 + U.w, m, yUx)
    b.put(m + 1, yUx, '\\')
    b.put(m, yUx + 1, '|')
    b.col(m, yUx + 2, -1)
    b.put(m, 0, '>')
    b.put(m + 1, 0, '.')
    b.col(0, 1, yL - 1)
    b.put(0, yL, '>')
    b.place(L, 1, yL - L.ey)
    b.line(1 + L.w, m - 2, yLx)
    b.put(m - 1, yLx, '/')
    b.col(m - 1, 2, yLx - 1)
    b.put(m - 1, 1, '|')
    b.line(1, m - 1, 0)
    return b.tile(0, 0, m + 1, 0)


def while_tile(C, Bd):
    """Loop: entry (0,0); condition C (one row) computes T; positive -> body Bd."""
    assert C.h == 1
    if Bd.w == 0:
        Bd = straight('.')
    b = Builder()
    b.put(0, 0, '/')
    b.put(0, -1, '|')
    b.put(1, -2, '>')
    b.place(C, 2, -2)
    x = 2 + C.w
    b.put(x, -2, '<')
    yB = -1 + Bd.ey
    yBx = -1 + Bd.xy
    b.col(x, -1, yB - 1)
    b.put(x, yB, '>')
    b.place(Bd, x + 1, -1)
    xr = x + 1 + Bd.w
    b.line(x + 1 + Bd.w, xr - 1, yBx)
    b.put(xr, yBx, '/')
    b.col(xr, -3, yBx - 1)
    b.put(xr, -4, '<')
    b.line(2, xr - 1, -4)
    b.put(1, -4, '/')
    b.put(1, -3, '.')
    b.put(x + 1, -3, '>')
    b.line(x + 2, xr + 1, -3)
    b.line(x + 1, xr, -2)
    return b.tile(0, 0, xr + 1, -3)


def concat(tiles):
    b = Builder()
    x, y = 0, 0
    first = None
    for t in tiles:
        if t.w == 0:
            continue
        b.place(t, x, y - t.ey)
        if first is None:
            first = (x, y)
        y = y - t.ey + t.xy
        x += t.w
    if first is None:
        return Tile(0, 1, 0, 0)
    return b.tile(0, first[1], x - 1, y)


def fold_seq(tiles):
    tiles = [t for t in tiles if t.w > 0]
    if not tiles:
        return Tile(0, 1, 0, 0)
    if sum(t.w for t in tiles) <= WMAX:
        return concat(tiles)
    lines, cur, cw = [], [], 0
    for t in tiles:
        if cur and cw + t.w > WMAX:
            lines.append(cur)
            cur, cw = [], 0
        cur.append(t)
        cw += t.w
    lines.append(cur)
    lines = [concat(l) for l in lines]
    lines = [l if i % 2 == 0 else rotate(l) for i, l in enumerate(lines)]
    xr = max(l.w for l in lines) + 2       # column of the right-hand '\'
    b = Builder()
    y = 0          # top of the current line
    prev_bottom = None
    prev_exit = None
    entry = None
    for i, l in enumerate(lines):
        # choose y: below the previous line, and the U-turn needs room
        if prev_bottom is not None:
            y = prev_bottom + 1
            if i % 2 == 1:       # entering a rotated line at its right side
                y = max(y, prev_exit + 2 - l.ey)
            else:
                y = max(y, prev_exit + 1 - l.ey)
        if i % 2 == 0:
            b.place(l, 1, y)
            if i == 0:
                entry = y + l.ey
                b.put(0, entry, '.')
            else:
                b.put(0, y + l.ey, '>')
                b.col(0, prev_exit + 1, y + l.ey - 1)
            ex = y + l.xy
            b.line(1 + l.w, xr - 1, ex)
            if i + 1 < len(lines):
                b.put(xr, ex, '\\')
                b.put(xr - 1, ex + 1, '|')
            else:
                b.put(xr, ex, '.')
        else:
            b.place(l, xr - 1 - l.w, y)
            ee = y + l.ey
            b.put(xr - 1, ee, '/')
            b.col(xr - 1, prev_exit + 2, ee - 1)
            ex = y + l.xy
            # leave at the left: run to column 1, '/' at column 0
            for xx in range(1, xr - 1 - l.w):
                b.put(xx, ex, '.')
            b.put(0, ex, '/')
        prev_exit = ex
        prev_bottom = y + l.h - 1
    if len(lines) % 2 == 0:
        yf = prev_bottom + 1
        b.col(0, prev_exit + 1, yf - 1)
        b.put(0, yf, '>')
        b.line(1, xr, yf)
        return b.tile(0, entry, xr, yf)
    return b.tile(0, entry, xr, prev_exit)


class Gen:
    def __init__(self, prog):
        self.prog = prog
        regs = {}
        self.count(prog, 1, regs)
        self.ntemps = 0
        self.temps_needed(prog)
        for i in range(self.ntemps):
            regs['_t%d' % i] = regs.get('_t%d' % i, 0) + 10 ** 6
        slots, regedges = candidate_slots()
        names = sorted(regs, key=lambda r: -regs[r])
        if len(names) > len(slots):
            raise RuntimeError('too many registers')
        self.slots = {n: slots[i] for i, n in enumerate(names)}
        self.regedges = regedges
        self.navcache = {}

    def count(self, body, w, regs):
        def ex(e):
            if e[0] == 'reg':
                regs[e[1]] = regs.get(e[1], 0) + w
            elif e[0] == 'bin':
                ex(e[2])
                ex(e[3])
        for s in body:
            k = s[0]
            if k == 'set':
                regs[s[1]] = regs.get(s[1], 0) + w
                ex(s[2])
            elif k in ('read', 'out'):
                regs[s[1]] = regs.get(s[1], 0) + w
            elif k == 'if':
                ex(s[1][1])
                ex(s[1][2])
                self.count(s[2], w, regs)
                self.count(s[3], w, regs)
            elif k == 'while':
                ex(s[1][1])
                ex(s[1][2])
                self.count(s[2], w * 10, regs)

    def temps_needed(self, body):
        def depth(e):
            if e[0] != 'bin':
                return 0
            a, b = e[2], e[3]
            n = 0
            if a[0] == 'bin':
                n = 1 + depth(a)
            if b[0] == 'bin':
                n = max(n, (1 if a[0] == 'bin' else 0) + 1 + depth(b))
            return n
        for s in body:
            es = []
            if s[0] == 'set':
                es = [s[2]]
            elif s[0] in ('if', 'while'):
                es = [('bin', '-', s[1][1], s[1][2])]
                self.temps_needed(s[2])
                if s[0] == 'if':
                    self.temps_needed(s[3])
            for e in es:
                self.ntemps = max(self.ntemps, depth(e) + 2)

    # --- straight code
    def mem(self):
        m = Mem(self.slots, self.regedges)
        m.navcache = self.navcache
        return m

    def compile_expr(self, m, e, dst, free):
        """Evaluate e into edge dst; free: list of free temp names."""
        if e[0] == 'num':
            m.go_edge(dst)
            m.const_here(e[1])
            return
        if e[0] == 'reg':
            m.transfer(self.slots[e[1]], dst)
            return
        op, a, b = e[1], e[2], e[3]
        # x = x + 1 style shortcuts
        if dst != T and a[0] == 'reg' and self.slots[a[1]] == dst and b[0] == 'num' and op in '+-' and abs(b[1]) <= 3:
            k = b[1] if op == '+' else -b[1]
            m.go_edge(dst)
            m.emit((')' if k > 0 else '(') * abs(k))
            return
        if a[0] == 'num' and a[1] == 0 and op == '-' and b[0] == 'reg':
            m.transfer(self.slots[b[1]], dst)
            m.go_edge(dst)
            m.emit('~')
            return
        used = []
        ops = []
        for x in (a, b):
            if x[0] == 'bin':
                t = free.pop(0)
                used.append(t)
                self.compile_expr(m, x, self.slots[t], free)
                ops.append(('reg', t))
            else:
                ops.append(x)
        m.operand(ops[0], A)
        m.operand(ops[1], B, avoid=(A,))
        m.go_edge(T, ALU)
        m.emit(':' if op == '/' else op)
        m.transfer(T, dst)
        free[:0] = used

    def straight_code(self, stmts):
        m = self.mem()
        for s in stmts:
            self.simple(m, s)
        m.go_home()
        return ''.join(m.code)

    def simple(self, m, s):
        k = s[0]
        temps = ['_t%d' % i for i in range(self.ntemps)]
        if k == 'set':
            self.compile_expr(m, s[2], self.slots[s[1]], temps)
        elif k == 'read':
            m.go_edge(self.slots[s[1]])
            m.emit(',')
        elif k == 'out':
            m.go_edge(self.slots[s[1]])
            m.emit(';')
        elif k == 'print':
            m.go_edge(T, ALU)
            for c in s[1].encode('utf-8'):
                m.go_edge(T, ALU)
                m.const_here(c)
                m.emit(';')
        elif k == 'fail':
            self.simple(m, ('print', 'json2dir: ' + s[1] + '\n'))
            m.emit('@')
        elif k == 'halt':
            m.emit('@')
        elif k == 'nop':
            pass
        else:
            raise RuntimeError(k)

    def cond_code(self, c):
        """Returns (code, then_is_positive)."""
        op, a, b = c
        if op in ('<', '<='):
            a, b = b, a
            op = {'<': '>', '<=': '>='}[op]
        d = a if b == ('num', 0) else fold('-', a, b)
        temps = ['_t%d' % i for i in range(self.ntemps)]
        m = self.mem()
        if op == '>':
            self.compile_expr(m, d, T, temps)
            pos = True
        elif op == '>=':
            self.compile_expr(m, fold('+', d, ('num', 1)), T, temps)
            pos = True
        else:
            if d[0] == 'reg':
                sq = ('bin', '*', d, d)
                self.compile_expr(m, sq, T, temps)
            else:
                t = temps.pop(0)
                self.compile_expr(m, d, self.slots[t], temps)
                self.compile_expr(m, ('bin', '*', ('reg', t), ('reg', t)), T, temps)
            pos = (op == '!=')
        m.go_home()
        return ''.join(m.code), pos

    def chunks(self, code):
        return [straight(code[i:i + WMAX]) for i in range(0, len(code), WMAX)]

    def block(self, body):
        tiles = []
        run = []

        def flush():
            if run:
                tiles.extend(self.chunks(self.straight_code(run)))
                run.clear()
        for s in body:
            if s[0] in ('if', 'while'):
                flush()
                code, pos = self.cond_code(s[1])
                if s[0] == 'if':
                    th, el = self.block(s[2]), self.block(s[3])
                    U, L = (el, th) if pos else (th, el)
                    tiles.extend(self.chunks(code))
                    tiles.append(if_tile(U, L))
                else:
                    if not pos:
                        raise RuntimeError('while needs a > or >= style condition')
                    tiles.append(while_tile(straight(code), self.block(s[2])))
            else:
                run.append(s)
        flush()
        return fold_seq(tiles)


# ---------------------------------------------------------------------------
# hexagon output

def to_hexagon(prog):
    b = Builder()
    b.put(0, 0, '.')
    b.put(1, 0, '\\')
    b.put(0, 1, '|')
    ey = 2 + prog.ey
    b.col(0, 2, ey - 1)
    b.put(0, ey, '>')
    b.place(prog, 1, 2)
    W = max(x for x, y in b.cells) + 1
    H = max(y for x, y in b.cells) + 1
    n = max(W, (W + H + 1) // 2, 2)
    sys.stderr.write('tile %d x %d\n' % (W, H))
    rows = []
    for i in range(2 * n - 1):
        r = i - (n - 1)
        qmin, qmax = max(-(n - 1), -(n - 1) - r), min(n - 1, n - 1 - r)
        rows.append(' ' * abs(n - 1 - i) + ' '.join(b.cells.get((q, i), '.') for q in range(qmin, qmax + 1)))
    return '\n'.join(rows) + '\n', n


def main():
    args = sys.argv[1:]
    global WMAX
    out = None
    if '-w' in args:
        k = args.index('-w')
        WMAX = int(args[k + 1])
        del args[k:k + 2]
    if '-o' in args:
        k = args.index('-o')
        out = args[k + 1]
        del args[k:k + 2]
    src = open(args[0], encoding='utf-8').read()
    prog = parse_program(src)
    g = Gen(prog)
    tile = g.block(prog)
    text, n = to_hexagon(tile)
    sys.stderr.write('hexagon side %d, %d cells, %d registers + %d temps\n'
                     % (n, 3 * n * (n - 1) + 1, len(g.slots) - g.ntemps, g.ntemps))
    if out:
        with open(out, 'w', newline='\n') as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == '__main__':
    main()
