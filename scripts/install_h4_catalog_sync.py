"""Install only this user's project-local saved-publication synchronization job.

No service/OS configuration or scientific runner is changed. --dry-run does not
write a crontab. Existing unrelated crontab entries are preserved verbatim.
"""
import argparse
from pathlib import Path
import shlex
import subprocess


BEGIN = '# BEGIN PRT H4 C CATALOG SYNC'
END = '# END PRT H4 C CATALOG SYNC'
DEFAULT_PYTHON = '/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python'


def current_crontab(run=subprocess.run):
    result = run(['crontab', '-l'], text=True, capture_output=True)
    if result.returncode == 0:
        return result.stdout
    if result.returncode == 1 and 'no crontab for ' in result.stderr.lower():
        return ''
    raise RuntimeError('Cannot read this user\'s crontab: ' + result.stderr.strip())


def own_block(text):
    lines = text.splitlines(keepends=True)
    begins = [i for i, line in enumerate(lines) if line.rstrip('\r\n') == BEGIN]
    ends = [i for i, line in enumerate(lines) if line.rstrip('\r\n') == END]
    if not begins and not ends:
        return None
    if len(begins) != 1 or len(ends) != 1 or begins[0] >= ends[0]:
        raise ValueError('Malformed or duplicate H4 sync block; crontab left unchanged')
    return begins[0], ends[0] + 1


def job_block(worktree, interpreter=DEFAULT_PYTHON, minutes=10):
    root = Path(worktree)
    if not root.is_absolute() or not Path(interpreter).is_absolute():
        raise ValueError('Explicit absolute worktree and interpreter paths required')
    if type(minutes) is not int or not 1 <= minutes <= 59:
        raise ValueError('Cron interval must be an integer from 1 to 59 minutes')
    # Cron parses percent signs before the shell, including inside shell quotes.
    def quote(value):
        value = str(value)
        if '\n' in value or '\r' in value or '\0' in value:
            raise ValueError('Newlines/NUL are not valid scheduler arguments')
        return shlex.quote(value).replace('%', '\\%')
    wrapper = root / 'scripts/run_h4_catalog_sync.sh'
    cmd = '/bin/bash ' + quote(wrapper) + ' --python ' + quote(interpreter)
    return BEGIN + '\n' + '*/' + str(minutes) + ' * * * * ' + cmd + '\n' + END + '\n'


def replace_job(text, block):
    span = own_block(text)
    if span is not None:
        lines = text.splitlines(keepends=True)
        return ''.join(lines[:span[0]]) + (block or '') + ''.join(lines[span[1]:])
    if block is None:
        return text
    return text + ('\n' if text and not text.endswith('\n') else '') + block


def main(argv=None, run=subprocess.run):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--install', action='store_true')
    action.add_argument('--remove', action='store_true')
    action.add_argument('--show', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worktree', type=Path)
    parser.add_argument('--python', default=DEFAULT_PYTHON)
    parser.add_argument('--minutes', type=int, default=10)
    args = parser.parse_args(argv)
    if not any((args.install, args.remove, args.show, args.dry_run)):
        parser.error('Use --install, --remove, --show or --dry-run')
    old = current_crontab(run)
    if args.show:
        span = own_block(old)
        print(''.join(old.splitlines(keepends=True)[slice(*span)]) if span else 'NOT_INSTALLED', end='' if span else '\n')
        return 0
    if args.remove:
        new = replace_job(old, None)
    else:
        if args.worktree is None:
            parser.error('--worktree ABSOLUTE_PATH is required for installation/preview')
        if not args.worktree.is_absolute():
            parser.error('--worktree must be an explicit absolute path')
        wrapper = args.worktree / 'scripts/run_h4_catalog_sync.sh'
        if not wrapper.is_file():
            parser.error('Project-local synchronization wrapper does not exist: ' + str(wrapper))
        new = replace_job(old, job_block(args.worktree, args.python, args.minutes))
    if args.dry_run:
        print(new, end='')
        return 0
    if new == old:
        print('UNCHANGED')
        return 0
    written = run(['crontab', '-'], input=new, text=True, capture_output=True)
    if written.returncode:
        raise RuntimeError('Cannot update this user\'s crontab: ' + written.stderr.strip())
    print('REMOVED' if args.remove else 'INSTALLED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
