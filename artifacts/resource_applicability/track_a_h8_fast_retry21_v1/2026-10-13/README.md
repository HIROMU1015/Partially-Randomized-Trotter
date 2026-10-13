# H8高速化版・上限21一度再実行の入口

2026-10-13 JST。利用者が1000万instructions・累積compile21/control44を承認。SOURCE `9bfea8bde44cbb2ea9187476ddd496066a8853b3`、branch `track-a-h8-fast-retry21-20261013`。

- [実行契約・条件・予算・STOP](../../../../docs/research/track_a_h8_fast_retry21_v1.md)、[plan](approved_manifest.json)、[auth](approved_authorization.json)、[257source＋newtest](source_closure.json)、[変更3項目・carry・seed/resource保持](prelaunch_review.json)。
- [新23 metadata/mock検査](test_result.txt)。old44/40を繰返さない。real input/trajectory/transpileをtestsに使わず、heavy再失敗時のremaining拒否、子CPU復元を確認。
- [起動状況](launch_status.json)、[launch identity/argv](launch_receipt.json)、[worker claim](worker_claim.json)、[fresh host/FS/quota](fresh_resources.json)、[native318依存subset](fresh_native_profile_subset.json)。旧host/compiler bit同一性は未検証。

run `heavy02`：coordinator410647/CPU31、worker410910/CPU30、pipeline410386。内部thread1・workerAS32/RSS24GiB・既存5秒fail-closed。firstSTOPなしで起動を確認、wrapper完了・実速度倍率は未確認。成功後だけ同じseedで残7を最大3workerへ接続し、完了またはSTOP後終了する。automatic retry/次stage/本格H8なし。

旧failed compile1/control4はcarryへ保持。新20compile/control40と合算して21/44、trajectory4/occurrence15。SOURCE1d064749のsetup01はCPU_UNAVAILABLEで科学worker/one-shot未到達、call0。[setup STOP](setup_stop01.json)、[native exit1](setup_native_exit01.json)を保全し、子exec前のapproved CPU復元だけを修正した。

private log入口 `/home/AbeHiromu/projects/h6-handoff-evidence/20261013/h8-fast-retry21-v1/`。`heavy02/resource_observations.jsonl`、`control02/heavy_coordinator.log`、`control02/heavy_native_exit.json`、成功後`remaining02`、最終`control02/terminal.json`と`remaining02/combined_feasibility_result.json`を参照する。科学入力/SCF/DF/state再生成0、共有環境/venv/H4/他job変更0・GPU0。
