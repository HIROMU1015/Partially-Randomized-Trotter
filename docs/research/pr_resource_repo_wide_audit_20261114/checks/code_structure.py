import ast, hashlib, collections, sys, re, json
from pathlib import Path
root = Path(sys.argv[1])
files = sorted(p for p in root.rglob('*.py') if 'typings' not in p.parts)
stats = collections.Counter(); names = collections.defaultdict(list); bodies = collections.defaultdict(list)
execs = collections.Counter(); gitshow = collections.Counter(); commits = collections.Counter(); syspath = 0; lines = collections.Counter()
hexre = re.compile(r"['\"]([0-9a-f]{40})['\"]")
for p in files:
    src = p.read_text(errors='replace'); rel = str(p.relative_to(root)); top = rel.split('/')[0] if not rel.startswith('src/') else '/'.join(rel.split('/')[:2])
    lines[top] += src.count('\n'); stats['files:' + top] += 1
    for h in set(hexre.findall(src)): commits[h] += 1
    if 'exec(compile(' in src or 'exec(' in src: execs[top] += 1
    if "'git'" in src and ("'show'" in src or 'cat-file' in src): gitshow[top] += 1
    if 'sys.path.insert' in src or 'sys.path.append' in src or 'sys.path[:0]' in src: syspath += 1
    try: tree = ast.parse(src)
    except SyntaxError: stats['syntax_error'] += 1; continue
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names[n.name].append(rel)
            body = n.body[1:] if n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], 'value', None), ast.Constant) and isinstance(n.body[0].value.value, str) else n.body
            dump = ast.dump(ast.Module(body=body, type_ignores=[]), annotate_fields=False)
            size = sum(1 for _ in ast.walk(n))
            if size >= 25: bodies[hashlib.sha1((ast.dump(n.args) + dump).encode()).hexdigest()].append((rel, n.name, size, (n.end_lineno - n.lineno + 1)))
print('files', len(files), 'lines', sum(lines.values()))
print({k: (stats['files:' + k], v) for k, v in lines.items()})
total_funcs = sum(len(v) for v in names.values()); print('function defs', total_funcs, 'distinct names', len(names))
dup = {h: v for h, v in bodies.items() if len(v) > 1}
dup_lines = sum(sum(x[3] for x in v[1:]) for v in dup.values()); print('identical function bodies (>=25 AST nodes): groups', len(dup), 'redundant copies', sum(len(v) - 1 for v in dup.values()), 'redundant lines', dup_lines)
print('top duplicated groups:')
for h, v in sorted(dup.items(), key=lambda kv: -len(kv[1]))[:25]: print('  x%d %s (%d lines) e.g. %s' % (len(v), v[0][1], v[0][3], v[0][0]))
print('helper names defined in many files:')
for k in ('digest', 'sha', 'sha256', 'read', 'write', 'save', 'git', 'blob', '_blob', 'same', 'functions', 'function', 'fixed_module', 'load', 'require', 'canonical', 'main'):
    print('  %-12s %d' % (k, len(names.get(k, []))))
print('files with exec:', dict(execs), 'files with git show/cat-file:', dict(gitshow), 'files touching sys.path:', syspath)
print('distinct 40-hex literals:', len(commits))
json.dump(dict(dup={h: v for h, v in dup.items()}, commits=commits), open(sys.argv[2], 'w'))
