"""Authorization/source gates are tested without any scientific input."""
import json
import os
import subprocess
import pytest
from trottertracks.algorithm_codesign import freeze
from trottertracks.algorithm_codesign.freeze import check_text_path, verify_launch


def test_draft_rejects_before_any_source_or_input_read(tmp_path):
    # Empty plan and malformed domain are intentional: neither is touched
    # until the independent execution authorization is present.
    with pytest.raises(PermissionError, match='NOT_AUTHORIZED'):
        verify_launch(tmp_path, {}, {'science_execution_authorized': False}, b'not JSON', b'not JSON')


@pytest.mark.parametrize('path', ['../source.py', '/outside.py', 'artifacts/input.npz'])
def test_source_inventory_cannot_turn_into_molecular_IO(path):
    with pytest.raises(ValueError, match='text paths'):
        check_text_path(path)


def test_boolean_flip_does_not_authorize_unbound_source(tmp_path):
    with pytest.raises(PermissionError, match='bind'):
        verify_launch(tmp_path, {}, {'science_execution_authorized': True}, b'{}', b'{}')


@pytest.fixture
def publication(tmp_path, monkeypatch):
    """Isolated Git fixture, never a scientific execution authorization.

    The remote satisfies the repository's HIROMU1015-only commit rule;
    no fetch/push/network command is used by this fixture.
    """
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], env=env, stderr=subprocess.DEVNULL).decode().strip()
    git('init', '-q', '--object-format=sha1')
    git('config', 'user.name', 'BF1 Synthetic Test')
    git('config', 'user.email', 'bf1-test@example.invalid')
    git('config', 'core.hooksPath', '/dev/null')
    git('remote', 'add', 'origin', 'https://github.com/HIROMU1015/Partially-Randomized-Trotter.git')
    def put(path, data):
        target = tmp_path/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    def commit(message):
        git('add', '--all')
        git('commit', '-q', '-m', message)
        return git('rev-parse', 'HEAD')
    # Only the domain hash is replaced with a tiny synthetic JSON; the real
    # publication/auth/source verification functions remain under test.
    domain = b'{"synthetic_formula_fixture":true}\n'
    tests = b'{"all_passed":true,"scope":"synthetic_fixture"}\n'
    monkeypatch.setattr(freeze, 'DOMAIN_SHA256', freeze.sha(domain))
    source_path = 'src/trottertracks/algorithm_codesign/synthetic_source.py'
    source = b'"""Synthetic Git source-binding fixture."""\nVALUE = 1\n'
    put(source_path, source)
    plan = dict(execution_id=freeze.EXECUTION_ID, input=freeze.INPUT,
                publication_scheme=freeze.PUBLICATION_SCHEME, environment=freeze.environment(),
                sources=[dict(path=source_path, sha256=freeze.sha(source))])
    put(freeze.PREPARATION_REL+'/source_plan.json', freeze.canonical(plan))
    put(freeze.DOMAIN_REL, domain)
    put(freeze.PREPARATION_REL+'/synthetic_semantic_report.json', tests)
    source_commit = commit('synthetic fixture: source')
    auth = dict(execution_id=freeze.EXECUTION_ID, plan_fingerprint=freeze.sha(freeze.canonical(plan)),
                domain_sha256=freeze.sha(domain), semantic_report_sha256=freeze.sha(tests),
                review_verdict='APPROVED_FOR_ONE_BF1_RUN', source_content_sealed=True,
                run_count=1, mandatory_stop=True, science_retry_authorized=False,
                automatic_next_stage=None, BF2_authorized=False, input=freeze.INPUT,
                publication_scheme=freeze.PUBLICATION_SCHEME, science_execution_authorized=True,
                source_commit=source_commit,
                explicit_user_execution_instruction='SYNTHETIC TEST FIXTURE ONLY; no scientific input or run')
    def authorize():
        put(freeze.AUTHORIZATION_REL, freeze.canonical(auth))
        return commit('synthetic fixture: authorization only')
    def verify():
        return verify_launch(tmp_path, plan, auth, domain, tests)
    return dict(root=tmp_path, git=git, put=put, commit=commit, authorize=authorize,
                verify=verify, auth=auth, source=source_path, source_commit=source_commit,
                plan=plan, domain=domain, tests=tests)


def test_source_plus_authorization_child_has_no_self_reference(publication):
    authorization_commit = publication['authorize']()
    assert authorization_commit != publication['source_commit']
    assert publication['verify']() == dict(source_commit=publication['source_commit'],
        authorization_commit=authorization_commit, publication_scheme=freeze.PUBLICATION_SCHEME)


def test_fixed_authorization_document_is_allowed(publication):
    publication['put'](freeze.AUTHORIZATION_DOC_REL, b'Synthetic authorization receipt only.\n')
    publication['authorize']()
    publication['verify']()


def test_commit_external_authorization_is_not_the_selected_B_scheme(publication):
    publication['put'](freeze.AUTHORIZATION_REL, freeze.canonical(publication['auth']))
    with pytest.raises(PermissionError, match='child'):
        publication['verify']()


@pytest.mark.parametrize('path', ['src/trottertracks/algorithm_codesign/synthetic_source.py', 'unrelated.md'])
def test_authorization_commit_cannot_change_source_or_unrelated_files(publication, path):
    publication['put'](path, b'Unauthorized change in a synthetic Git fixture.\n')
    publication['authorize']()
    with pytest.raises(PermissionError, match='non-authorization'):
        publication['verify']()


def test_grandchild_is_not_silently_treated_as_authorization_only(publication):
    publication['authorize']()
    publication['put'](freeze.AUTHORIZATION_DOC_REL, b'Extra synthetic descendant.\n')
    publication['commit']('synthetic fixture: extra child')
    with pytest.raises(PermissionError, match='single authorization-only child'):
        publication['verify']()


def test_dirty_source_is_rejected_under_clean_authorization_commit(publication):
    publication['authorize']()
    publication['put'](publication['source'], b'VALUE = 2\n')
    with pytest.raises(ValueError, match='Source-content mismatch'):
        publication['verify']()


def test_dirty_or_different_authorization_json_is_rejected(publication):
    publication['authorize']()
    publication['put'](freeze.AUTHORIZATION_REL, freeze.canonical(publication['auth'])+b'\n')
    with pytest.raises(PermissionError, match='committed, clean JSON'):
        publication['verify']()


def test_source_plan_and_report_are_bound_to_the_source_commit(publication):
    publication['plan']['uncommitted_addition'] = 'synthetic tamper'
    publication['auth']['plan_fingerprint'] = freeze.sha(freeze.canonical(publication['plan']))
    publication['authorize']()
    with pytest.raises(ValueError, match='Source plan is not bound'):
        publication['verify']()
