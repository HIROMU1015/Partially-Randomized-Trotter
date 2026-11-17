# RomuTail operational service overlay

`scripts/prepare_romutail_controller_deployment.py` builds a new deployment from the archived
server2 service, imported prior-run manifest, and [LAYOUT.json](LAYOUT.json).
It does not start a controller, worker, queue or scientific leaf.

The original SERVICE/PROBLEM/ROUND files stay byte-identical under
`.rpe-central/deployment/source-server2` and the hash-verified server2 snapshot.
The new operational files change paths, controller ownership, current transport and host participation only.
Scientific conditions, input locators, scope bounds, original computation28d, timed Q3 adapter349,
zero Q3/preparation budgets and `science_approved=false` are preserved.
The nine prior-run roots map to the imported snapshot; unresolved attempts and STOP files remain there.

Fixed mapper/metadata views were placed under `.rpe-central/views/<SHA>`; existing RomuTail Q3 route
and f956 preparation views are reused. The saved snapshot/receipt and DF H4_context are copied with
the source hashes checked. The helper records any unresolved controller-local absolute path as a HOLD.
No server `/home/AbeHiromu` layout is reproduced with symlinks.

Host profiles preserve each host's scientific root, Python, route views and timed adapter manifest.
They combine the original f956 preparation rules with latest a18d rules and their unchanged SOURCE manifests.
The private PR-DF descriptor/config/provider mapping from the preparation subtask is added to RomuTail/WSL.
Host-specific server2/WSL paths remain host-specific; they are not interpreted as controller-local files.
RomuTail owns `distributed.local_worker`. Server1 is OFF, local0-wsl remains explicitly OFF.
Primary priority group is server2/RomuTail/server1; optional WSL has no affinity or discovery waiting time.
Future user-authorized runs still perform ordinary exact REQUEST fingerprint/admission checks.

Completion/controller metadata paths are passed through `candidate_completion.controller_paths`.
The selected integrated source supplies orchestration; fixed scientific providers and their contracts remain separate.
The C publication configuration is copied from layout to the new SERVICE.
The new SERVICE uses `entry=schedule`; the normal schedule CLI delegates mixed/default PF rounds
to the existing ordinary round path. The absolute-kappa route also remains the schedule entry.

After the final integrated commit is fixed, generate a **new** deployment directory:

```bash
python3 -B scripts/prepare_romutail_controller_deployment.py \
  --source-directory '/home/abe/Project/Partially Randomized Trotter/.rpe-central/deployment/source-server2' \
  --snapshot-manifest '/home/abe/Project/Partially Randomized Trotter/.rpe-central/imports/server2-20261118/MANIFEST.json' \
  --layout docs/research/rpe_romutail_controller_20261118/deployment/LAYOUT.json \
  --source FIXED_INTEGRATED_COMMIT \
  --output '/home/abe/Project/Partially Randomized Trotter/.rpe-central/deployment/FIXED_RELEASE_ID'
```

The layout `code_root` must be checked out at that SHA. Preview/PlanOnly/Start should resolve the
generated SERVICE through the tested-release registry. The helper itself does not register or activate it.
Root migration deployment owns release registration, publication synchronization and actual acceptance.

For a controller-only update, the layout may explicitly set `candidate_completion_source` to the
previous fixed preparation-wrapper commit and `worker_script` to the existing fixed worker path.
Omitting these keys uses the selected integrated commit and its worker script, as before.
Both normal and absolute-kappa rounds record the independently resolved wrapper SOURCE; the new
SERVICE records the new controller SOURCE. Native provider and Q3 SOURCE/profile bindings stay fixed.
Generate a new directory and register a new release; never rewrite the already registered service.

The current draft is `.rpe-central/deployment/operational-draft02`.
[DRAFT_DEPLOYMENT.json](DRAFT_DEPLOYMENT.json) has controller-local path HOLD count0.
It is a draft tied to the recorded intermediate commit, not the final registered release.
Its output is not a scientific execution grant.

Synthetic preservation test: [test_deployment.py](test_deployment.py), [TESTS.log](TESTS.log), one PASS.
The test verifies scope/budget/input and archived-byte preservation, zero affinity wait and no extra preparation attempt cap.
The preparation subtask's six saved PR-DF inspection PASS records remain separate from new-run execution eligibility.

Performed by Codex sub-agent `prep_inputs`; not an independent scientific review.
