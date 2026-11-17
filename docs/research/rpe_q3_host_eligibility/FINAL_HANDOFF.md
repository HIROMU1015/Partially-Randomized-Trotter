# 実H4：4方式×3計算ホストの準備完了

任意の第4計算host WSLの追加は[optional_wsl/REPORT](optional_wsl/REPORT.md)を参照。以下は固定3台の基準記録。

共通controller/worker実装の固定SHA：`181d14052b0d4db1815bcb9060a2020cafb5f1cf`。
[code hashとSOURCEの区別](FINAL_CODE_SOURCE.json)。

今回の4候補は12セルすべて実入力dry-check PASS。実H4 Q3完了は0件。
[割当policy](HOST_ELIGIBILITY.json)と[全セルの証拠照合](checks/FINAL_TWELVE_CELL_EVIDENCE.json)を参照。
local0はSSH中継・検査指令を担当し、計算hostではない。

| 候補 | 科学job ID | server2 | server1 | local / RomuTail |
|---|---|---|---|---|
| Full PF | 9b9b4b5132ff89768c5c | PASS | PASS | PASS |
| B0I / TRUNCATED_PF | 72864f6cf968d3b514b1 | PASS | PASS | PASS |
| PR-Pauli | 805877c2ab6c3e03f7f1 | PASS | PASS | PASS |
| PR-DF | 48494ae31276a64c46ce | PASS | PASS | PASS |

## 通常入口

新しい出力directoryを指定する。選定用problem/roundは固定339のまま、選択前の保存履歴・現在状態照合を維持する。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/rpe-q3-host-eligibility
/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B \
 scripts/run_rpe_schedule_search.py \
 --problem docs/research/rpe_distributed_h4_plan_20261117/inputs/PROBLEM.json \
 --round docs/research/rpe_distributed_h4_plan_20261117/inputs/FULL_HISTORY_PLAN_ROUND.json \
 --run-dir /home/AbeHiromu/projects/rpe-h4-ready-NEXT \
 --plan-only --prepare-readiness \
 --preparation-cache /home/AbeHiromu/projects/rpe-q3-host-eligibility-checks/cache
```

出力READY/HOST_ELIGIBILITYが候補別の3host許可と正しいprofileを示す。
`worker_profiles/*.json` が共通slot pool用profileで、routeごとに実行環境を自動選択する。
利用者が候補ごとにhost/Pythonを選ぶ必要はない。
`DISTRIBUTED_OPTIONS.json` のexecutor設定とroutes_overrideを、別途採択する新しい科学roundへ結合する。
設定断片は新Q3予算・科学実行許可を発行しない。科学runは今回起動していない。
新REQUESTは全体hash・入力・SOURCE・route・runtimeを再判定し、この12セルのPASSを一律転用しない。

候補の選定・保存履歴取込みに使う旧routeは維持し、選定済みLD4のREQUEST組立でだけ
SOURCE guard付きdescriptor対応route33を採用する。物理schedule/shot/費用/評価契約と科学job IDは不変。
PauliのREQUEST全体hashはroute更新によりed04775d03240eee59d639fd039842652f28eaf4472f59cf4f12f9192a35f693。
他3REQUESTは元339とbyte一致。旧入力・旧ZIP・旧手順・旧spool・既存科学runを上書きしない。

## 実行環境の自動選択

- server2：既存h6環境のPython3.11.0rc1。科学adapterはSOURCE28d。
- server1：公開9dの3.11 launcherを、PF/B0I・Pauli・DFの各固定route profileから自動選択。
  同じPythonを使っても、route view・DP/cache・profileのbindingは各方式で保持する。
- RomuTail：既存.venv311。PF/B0IとPauliはadapter28d、DFは実装e37・配置09efを別manifestで使用。
  DFの内側27e SOURCEはprofile.route_viewsで解決し、サーバー絶対pathの模倣やsymlink追加はしない。

元assignmentの科学SOURCE28dと、新adapter/transportのSOURCEを区別する。
[server1専用profile](SERVER1_PRDF_IMPORT.md)、[RomuTail取込とadapter manifest](ROMUTAIL_PRDF_IMPORT.md)、
[Pauli descriptor routeの条件](pauli_descriptor_route/ROUTE_CHANGE.json)に固定locatorがある。
RomuTail adapter manifestはrunner SHA63cf463024241af20e9c6b3ca55eaf0ce2ebf4a4a7b8c9ca91a8ea74f6cb7fb6と
prefix helper SHA554460755ab6fcd38ad5b9f1a680b543aed7e26906579b26470d73fa659148f2を含む。
implementation=e37f6544dea3e021d3694264f7dae87add7ad30e、deployment=09efb068f242d5d6ff88296df3e50283b2f951bf。
両commitで該当2fileがbyte一致し、元SOURCE28dを新adapterへ偽装していない。

PF/B0I・Pauliの9セルは、実際のexecuteで使う固定total_state_dp ASTをselected Pythonで確認した。
期待/実測SHAは全件3d043e716246e57a8b05784b0d5c0f6777aaf43410fed8c4dd8b68df6dd590b8。
元metadata-only dryでは実行時constructor未到達だったため、3.12のAST型差を見逃していた。
新checkerとworkerは実行時と同じguardを数値計算なしで先に確認し、3.12のeb3150a...を拒否する。
DFの3セルも元runtime_checkのPASS_UNCHANGED_GUARDを保持。SOURCE/AST guardや数値codeを緩和していない。

新transportのworker supervisorは、旧科学rootのworker scriptへ戻らず、導入した新transportを使う。
assignmentに固定したprofile/adapter manifestをWORKER_PROFILEへ保存し、起動・回収・再開でも同じものを使う。
workerを配置する際は共通版の固定commitを別checkoutへ置き、生成profileへ新しいwork_dir/slotsだけを追加する。
科学root、interpreter、route_views、adapter manifest、DP/cache/runtimeチェックはdry-check済み値を維持する。
local0のrelay設定は新transport checkoutとこの生成profileを指す。旧人工runのrelay/config/spoolは再開しない。

## 証拠と検収

- server1 DF：9d022a0f7cc953ba3c1dac0b5e514e295c429fbeの公式PASSを再利用。
- RomuTail DF：e37実装・09ef証拠の実機PASSを再利用。source解決と誤SHA/dirtyの元テストも統合。
- Pauli：固定33e受理経路を再利用し、SOURCE/辞書/target/統計契約の拒否を人工検査。
  server2はruntime guard付きdryを更新。remote2はlocal0 V2で実測PASSし、23返送fileをhash照合。
- PF/B0I：実行時AST guardの見落としに対応し、影響6セルだけ更新。
  server2の2件とremote V3の4件がPASS、35返送fileをhash照合。DF/Pauli6件はV3で再実行していない。

[12セル人工割当](checks/TWELVE_CELL_ARTIFICIAL_ASSIGNMENT.json)では、各host独立の人工queueに4候補を載せ、
通常controllerと実workerクラスの保存・supervisor・adapter引数まで通した。
remote filesystemとruntime metadataは保存証拠のstub、全worker/adapter Popenはmockと明記している。
これは12件の実Q3を起動した記録ではない。科学OUTCOME・採用・本番queue投入は0件。
同jobの他host再取得、同attempt再開の追加起動も0件。正しいprofile/route views/adapter manifest/runtime guardが保持された。

実機の各方式でQ3が完了したとは主張しない。今後、別途採択された探索で実Q3完了件数/時、方式別最安、
計算・待ち・転送時間、prefix cache再利用を記録する。
今回の実H4 Q3、signal生成、価格較正、回路compile、本番queue投入はすべて0件。


## 通常入口の最終確認

[FINAL_NORMAL_ENTRY](checks/FINAL_NORMAL_ENTRY/READY.json)は、V3受領後の通常入口から生成した。
READYまで7.22秒、server2 dry-checkは4件すべて既存結果を再利用し、新しいdry-check呼出し0・科学計算0。
静的な最終policyと通常生成policyは、全12セルの実行profile・現在host fingerprintで一致した。
[関連検査](checks/FINAL_TESTS.json)と[通常入口の段階log](checks/FINAL_NORMAL_ENTRY/PREPARATION_STAGES.jsonl)を保存した。
このCLI時間から以前の利用者実測50分との速度倍率は算出しない。
