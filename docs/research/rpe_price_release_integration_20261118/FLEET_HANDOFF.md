# Release assets on four hosts

The existing central `RELEASES.json` selects a release. `RELEASE_COMPONENTS.json` is its public component/runtime mapping; it is not another registry. `SERVICE.json` fixes that manifest and the controller-local `hosts/*.json` layouts. The controller remains RomuTail and the central state remains `.rpe-central/state`.

Each component records its full SOURCE. The optimizer/operation client, fixed scientific Q3, timed adapter, preparation wrapper, price worker, and native provider views may intentionally use different commits. Host layouts resolve those names to existing checkouts and Python interpreters. Public configuration contains no credentials or host-specific absolute paths. Runtime observations preserve the full Python version, including `candidate`/serial; execution still uses the existing REQUEST, runtime, compiler, SOURCE and AST guards.

The new installer does not change Pause/intake. WSL intake stays OFF; server1 remains deferred/offline. Existing runs keep their recorded release and scientific conditions.

Generate metadata after fixing the integrated controller SOURCE:

```bash
python3 scripts/prepare_rpe_release_fleet.py \
  --source "$CONTROLLER_SOURCE" --release "$RELEASE_ID" \
  --service "$PREVIOUS_SERVICE" --price-profiles "$VERIFIED_PRICE_PROFILES" \
  --hosts "$PRIVATE_HOST_DEFAULTS" --output "$NEW_FLEET_DIRECTORY"
```

`PRIVATE_HOST_DEFAULTS` names each host's existing worker repository, target-only object store, new optimizer location, metadata directory and Python interpreter. Optional runtime observations contain metadata read from that host. The generator reuses the previously verified Q3 profile core and only replaces the fixed price-worker rule from its saved verified profile.

Copy `RELEASE_COMPONENTS.json`, the host's generated layout and merged profile to the paths named by `manifest_path`, `layout_path`, and `profile_path`. Obtain the fixed controller objects through the existing repository or an explicitly supplied local Git bundle. Check/install on the host:

```bash
python3 "$AVAILABLE_INTEGRATED_SOURCE/scripts/manage_rpe_release_deployment.py" \
  --manifest "$MANIFEST" --layout "$HOST_LAYOUT" --install
```

The bootstrap/previously installed integrated source supplies the installer; after initial installation the selected optimizer supplies it. The first call installs only absent component roots from the existing object store or declared bundle, and verifies existing roots without replacing them. Repeating it returns `REUSED`. Missing dependencies or saved input files produce a concrete `HOLD`; install dependencies only in the task's dedicated environment. A wrong existing checkout or wrong input hash is preserved and refused. Interrupted checkout creation does not publish a partial root.

Normal common operation first asks the existing central gateway to resolve the release, then runs this inventory locally before forwarding the operation. Worker participation checks the optimizer/transport installation; each assignment also carries `required_release` outside the scientific REQUEST/hash/job ID. The helper checks this release identity, its stage SOURCE, and the existing worker metadata guards before the existing mailbox accepts the assignment. No scientific SOURCE or queue is duplicated.

For a supplied assignment without starting it:

```bash
python3 "$OPTIMIZER_ROOT/scripts/manage_rpe_release_deployment.py" \
  --manifest "$MANIFEST" --layout "$HOST_LAYOUT" \
  --components optimizer worker --profile "$WORKER_PROFILE" \
  --assignment "$ASSIGNMENT"
```

The relay uses `--assignment-stdin` through its existing file-input bridge for Windows/WSL. This avoids the Windows SSH command-line limit and preserves large REQUEST JSON.

Server1 return procedure: use the same selected release metadata; resolve its existing scientific assets from its saved layout, acquire only missing code/inputs into owned paths, run inventory and saved-input checks, then run the dedicated artificial participation/disconnection/recovery check. Confirm its reservation/intake policy before enabling its existing worker. Server1 was not contacted during this integration and is not a completion dependency.

Tests added for missing-only installation/reuse, partial installation cleanup, SOURCE/input rejection, metadata-only optimizer commits, assignment envelope/stage matching, stdin transport, four-host metadata generation and unchanged scientific profile cores. These are technical checks, not scientific validation. New real signal, compile, Q3 and production-search counts remain zero; mandatory STOP remains in effect.

## Installed selection and evidence

The integrated controller SOURCE is `e917165ba1f36b7627614cd25e88b5108d9b20b9`; its release is `romutail-price-fleet-e917165ba`. The frozen service is `.rpe-central/deployment/e917165ba-price-fleet/SERVICE.json`. Its public mapping is [fleet/RELEASE_COMPONENTS.json](fleet/RELEASE_COMPONENTS.json), and the installation/reuse summary is [fleet/DEPLOYMENT_RECEIPTS.json](fleet/DEPLOYMENT_RECEIPTS.json). The existing central registry remains the authority for actual current selection.

RomuTail uses `.rpe-central/releases/e917165ba-price-fleet`. Server2 uses `/home/AbeHiromu/projects/rpe-release-fleet/releases/e917165ba1f3`; WSL uses `/root/projects/Partially-Randomized-Trotter-local0-wsl/.rpe-release-fleet/releases/e917165ba1f3`. The private central layouts are `.rpe-central/fleet/romutail-price-fleet-e917165ba/hosts/*.json`; each records its host-local manifest, layout and profile paths. Installer copies/saved input checks do not enable reception.

All three live hosts passed full component/runtime inventory and immediate repetition with zero additional installations. RomuTail/server2 reused all existing scientific assets. The old WSL13ab partial checkout lacked10 tracked Git blobs/working files. It was preserved; only the missing122,180-byte blob pack was acquired, and a new exact13ab checkout was prepared at `.rpe-release-fleet/releases/13ab1acfaebe`. Only the new WSL placement profile was remapped. Its Q3 profile core, SOURCE, inputs and runtime stayed fixed. The final WSL metadata uses `HOST.complete-provider.json` / `PROFILE.complete-provider.json`; the initial unselected metadata and initial HOLD evidence remain preserved remotely.

The saved real PR-DF REQUEST metadata passed the new release/SOURCE/adapter/runtime checks on all three live hosts without execution. Wrong release, wrong stage SOURCE and wrong checkout were refused. The per-host receipts under `fleet/*-SAVED_BINDING.json` identify that saved REQUEST and the observations. The parent controller task separately ran the dedicated three-host artificial pipeline and rejection checks; this fleet inventory did not start a scientific leaf.

Server1's matching package is retained at `.rpe-central/fleet/romutail-price-fleet-e917165ba/packages/server1.zip`; its private layout remains in the same central fleet directory. Its package hash is recorded in the deployment receipt; the optional missing13ab blob pack is retained beside it. On return, use the same deployment procedure and existing protected roots, resolving any concrete missing asset into a new owned path, then run saved-input and artificial participation checks. No server1 SSH was attempted.

## Registration and switching

Use the existing module, from the selected fixed checkout. Register the candidate after its files are final; use the same evidence path when adopting the immutable entry after the explicit-release Preview and technical checks. One designated controller operator performs the final adoption, with the observed previous current as the compare value:

```bash
PYTHONPATH="$OPTIMIZER_ROOT/src" "$CONTROLLER_PYTHON" -B -m trottertracks.rpe_release_registry \
  --registry "$REGISTRY" --release "$RELEASE_ID" --service "$SERVICE" \
  --evidence "$VALIDATION_RECORD"

# After candidate Preview and the necessary checks, adopt the same entry.
PYTHONPATH="$OPTIMIZER_ROOT/src" "$CONTROLLER_PYTHON" -B -m trottertracks.rpe_release_registry \
  --registry "$REGISTRY" --release "$RELEASE_ID" --service "$SERVICE" \
  --evidence "$VALIDATION_RECORD" --activate --expected-current "$PREVIOUS_CURRENT"
```

The registry refuses a changed current or changed immutable release entry. New Preview/PlanOnly/Start resolve this selected registry entry; Resume/Status/Stop of existing runs use the run's saved controller configuration. Deployment/current changes do not authorize science and do not replace old run settings.

## Stage capability and preservation

The table describes technical placement and saved metadata compatibility. Scientific execution stays at mandatory STOP until a later explicit scope and the existing duplicate preflight. Runtime/package versions and provider SOURCE mappings are in the public manifest; the existing per-request contracts remain authoritative.

| Host | Controller/common operation code | Signal | Compile | Audit | Q3 | Normal intake |
| --- | --- | --- | --- | --- | --- | --- |
| RomuTail | e917 installed; central authority | Ready metadata | Ready metadata | Ready metadata | SOURCE28d, timed349, Python3.11.0rc1 | Existing policy |
| server2 | e917 installed; forwards to RomuTail | Ready metadata | Ready metadata; Qiskit1.3.0 | Ready metadata | SOURCE28d, timed349, Python3.11.0rc1 | Existing policy |
| local0-wsl | e917 installed; forwards to RomuTail | Ready metadata | Ready metadata; Qiskit1.3.0 | Ready metadata | SOURCE28d, timed349, Python3.11.0rc1 | OFF |
| server1 | Same selected package prepared; offline | Deferred | Deferred | Deferred | Deferred | OFF |

Live components and required existing runtime packages passed installation/inventory. Prior unchanged saved-input checks are reused; the changed WSL13ab placement has a dedicated saved price-stage metadata check. Server1 is the only deferred host. No live missing component/runtime remains.

[Preservation receipt](fleet/PRESERVATION.json) confirms the old r4 SERVICE raw SHA remains `1dad63f9d6e89fb931144e01c887265355cf3a0f34a0e4d0c8d7a02c3812022e`; the new frozen SERVICE raw SHA is `b18eb9397530f6aad320ecb87b6fb3f9c870e53e8eb3c46dd6eb8fd5e38c496a` and every fixed setting matches. The actual WSL flag is still false, hash `c534d859465c86dac7fb89dd6759deebd26c4f3c6b1f780ab06dbea26f4dabfd`. Server1 was not contacted. New science and production-search counts are zero.

The changed WSL placement also passed [the two original saved price-stage challenges](fleet/WSL_PRICE_STAGE_READINESS.json): `prep-c8ceb53059f79f4ffe30872b` and `prep-d8fe2ccfec2164d0f3c3657a`. Their original TASKS file hash is `b16ce39ddccf8bba596c209fd7123ca291453a7e5b52589c16d83ea08b5d3347`. The existing checker's metadata-only `check()` function verified SOURCE, fixed input hashes, provider views, runtime and compiler against the final WSL profile. Scientific calls, compile calls and queue operations were all0; this is no new independent price validation or model fit.
