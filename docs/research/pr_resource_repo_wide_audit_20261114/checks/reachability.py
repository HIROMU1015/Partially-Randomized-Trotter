import ast, re, json, collections, sys
from pathlib import Path
S = Path('.'); newfiles = json.load(open('newfiles.json'))
union = {}
for tag in 'ABCD':
    root = S/'tree'/tag
    for p in root.rglob('*.py'):
        rel = str(p.relative_to(root))
        if 'typings' in rel: continue
        union.setdefault(rel, p)
mods = {}
for rel in union:
    if rel.startswith('src/'): mods[rel[4:-3].replace('/', '.').removesuffix('.__init__')] = rel
    elif rel.startswith('scripts/'): mods[Path(rel).stem] = mods.get(Path(rel).stem, rel) if '/' in rel[8:] else rel
    elif rel.startswith('tests/'): pass
pathre = re.compile(r"(src/[\w/]+\.py|scripts/[\w/]+\.py)")
def edges(rel):
    src = union[rel].read_text(errors='replace'); out = set(); kind = {}
    try: tree = ast.parse(src)
    except SyntaxError: return out, kind
    pkg = rel[4:-3].replace('/', '.').rsplit('.', 1)[0] if rel.startswith('src/') else None
    for n in ast.walk(tree):
        names = []
        if isinstance(n, ast.Import): names = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ''
            if n.level and pkg:
                parts = pkg.split('.'); parts = parts[:len(parts) - (n.level - 1)]; base = '.'.join(parts + ([n.module] if n.module else []))
            names = [base] + [base + '.' + a.name for a in n.names]
        for nm in names:
            while nm:
                if nm in mods: out.add(mods[nm]); kind[mods[nm]] = 'import'; break
                nm = nm.rpartition('.')[0] if '.' in nm else ''
    for m in pathre.findall(src):
        if m in union and m != rel: out.add(m); kind.setdefault(m, 'path-string')
    return out, kind
cur = [p for p in newfiles if 'track_d' not in p and not p.startswith('scripts/run_h6_df19') and p in union]
seen = set(cur); stack = list(cur); via = collections.Counter()
graph = {}
while stack:
    r = stack.pop(); e, k = edges(r); graph[r] = sorted(e)
    for t in e:
        if t not in seen: seen.add(t); stack.append(t); via[k[t]] += 1
tot = len(union); lines = lambda rels: sum(union[r].read_text(errors='replace').count('\n') for r in rels)
src_all = [r for r in union if not r.startswith('tests/')]
print('union files (src+scripts+tests):', tot, 'non-test:', len(src_all), 'lines', lines(src_all))
print('current-line seed files:', len(cur), 'lines', lines(cur))
print('reachable (imports + pinned path strings):', len(seen), 'lines', lines(seen), 'discovered via', dict(via))
legacy = sorted(seen - set(cur))
by = collections.Counter('/'.join(r.split('/')[:3]) if r.startswith('src/') else r.split('/')[0] for r in legacy)
print('legacy reachable by area:', dict(by))
json.dump(dict(seed=cur, reachable=sorted(seen), graph=graph), open('reach.json', 'w'))
indeg = collections.Counter(t for r, e in graph.items() for t in e)
print('most depended-upon:'); [print('  %3d %s' % (c, t)) for t, c in indeg.most_common(40)]
