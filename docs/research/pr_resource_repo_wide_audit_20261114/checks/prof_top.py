import pstats, sys, io
for f in sys.argv[1:]:
    st = pstats.Stats(f); tot = st.total_tt
    rows = []
    for (file, line, name), (cc, nc, tt, ct, callers) in st.stats.items():
        rows.append((ct, tt, nc, name, file.split('/')[-1], line))
    print('==', f.split('/')[-1], 'total %.1fs' % tot)
    keys = ('candidate_id', 'same', 'deepcopy', 'digest', 'dumps', 'parse', 'check_output', 'run', 'structure', 'validate', '_validate', 'identity', 'read', 'compile', 'sha256', 'loads', 'exec')
    seen = set()
    for ct, tt, nc, name, file, line in sorted(rows, reverse=True):
        if name in keys and (name, file) not in seen and ct > 0.03 * tot:
            seen.add((name, file)); print('   cum %6.1fs (%4.0f%%) self %6.1fs calls %8d  %s:%s' % (ct, 100 * ct / tot, tt, nc, file, name))
    print('   top self:', ', '.join('%s:%s %.1fs' % (r[4], r[3], r[1]) for r in sorted(rows, key=lambda r: -r[1])[:7]))
