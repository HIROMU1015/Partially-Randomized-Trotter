# Track B G9 v2：API境界の最小修正 / source review

## 現在の扱い

**G9 v2 source preparation完了。科学実行は未認可・未実行。**

利用者の「作業を進めて」は、直前に合意した型変換修正・合成を呼ばない境界テスト・source固定/公開までを認可する。
旧G9 v1のmarkerは消費済み、結果は`G9_TECHNICAL_INCONCLUSIVE`、native比較0 rowのまま保持する。
この準備を旧one-shotのretryとして実行しない。新しい実行は別の明示authorizationを要する。

- branch/worktree：`track-b-g9-v2-api-boundary-preparation-20261010`
- 基点（G9 v1結果）：`0b16479b4036b15052ea3286b56b389d59743b8c`
- 旧source S：`d6ecc6bc82c11158a66d175d78007a121d0fac46`
- 新source S：このpreparationを固定して公開したcommit。完全SHAは公開時の報告を正本とし、authorizationには将来承認後に記録する。
- root/Track A/旧G9 worktreeは編集しない。旧G9 runner、contract、結果、marker、STOP、過去の証拠は保持。

## 修正内容

旧runnerは`Fraction(contract['primitive_error'])`を継承numeric APIへ渡し、
`mp.mpf(epsilon)`でTypeErrorとなった。精度値、角度、演算子自体の失敗ではない。

旧runnerは変更せず、別のfuture入口`g9_p5_matched_native_v2.py`を追加。
`acquire_fixed_primitive`はcontractの文字列`1/1000000`をそのまま渡す。
floatへ変換しない。既存sequenceのkey/epsilon validationで使うFractionは維持する。
数式・operator・finite-bit・予算・native会計モジュールは旧sourceのまま。
runnerの残りの比較pipelineはv1から維持し、レビュー用diffを保存した。

新gateは別authorization JSON、直接子commit、同source/contract identity、clean worktree、critical source hashes、過去の保護ledgerを確認する。
pendingのままならgit revision query・runtime検証・marker作成・取得より前に拒否する。
旧G8/G9 v1のscope記録をv2実行承認として受け取らない。

## 科学条件と上限の保持

[static equivalence](../../../artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/static_equivalence_v2.json)で旧contractの46条件を値レベルで一致確認。
管理情報（schema/branch/base/source manifest/protected ledger/result path/authorization）だけを別v2へ束縛した。
旧contractファイルのhashは変えていない。v2 contractは別identityであり、旧contractを上書きしたものではない。

known development `p=(1/5,3/10,1/2),x=5/7,m=5`、同full first operator P5、G8指定3-qubit synthetic provider。
ordinary / partial-return+tail / closed-P3+tail / local full / closed-P5 full / matched CTSのdirect6方式、helper診断5方式。
Re/Im各1/200、22 axes/failure0.05、primitive epsilon10^-6、H160/K256、rho/eta10^-12、同compilerとstrict phaseを維持。
Pauli情報が安く得られるI1 synthetic contextであり、分子/DF/held-out/I0取得優位ではない。

固定inventoryは旧G9の19 keys（18旧sequence再利用、CTS新規1）を同path/hashで参照。
新角度、target、precision、threshold、seed、backendは追加しない。
wall1200/CPU900 s、RSS512 MiB、AS1536 MiB、per-key wall30/CPU20 s、new synthesis1、retry0、output16 MiB、shot cap10^8を維持。
accepted capと総T cap、T intercept+K*T_prep/readout、別CX/1Q/workspaceの扱いも変更していない。

## 境界テストと実行数

新focused tests **19件PASS**。既存G9 semantic23件はsource/hashと保存済みPASS記録を保持し、今回再実行していない。

- real numeric adapterを呼び、pygridsynth backend関数をstubに差し替えて、epsilon/4の引数がstubへ届くところでsentinel停止。
- 旧Fraction failureもoff-domain angle1/3のstub-only fixtureでbackend前に発生すると確認。
- key表現、固定options/精度、strict phaseの保持を確認。
- synthetic authorization/direct-child/clean/hash binding、pending拒否、critical source変更拒否、消費済みsynthetic marker拒否を確認。

実backend0、strict error guard評価0、登録matrix/budget/generator/比較row0、実量子shot/trajectory0、LP/DF/分子/NPZ/GPU0。
テスト内のfake git/authorization/markerは一時directoryだけ。registered result directoryは存在せずmarker未作成。
APIテストのatan1/3はoff-domain fixtureであり、新science inventoryへ加えない。
初回のmock lookupが同名public関数へ解決されてfixture3件errorとなったため、明示module importへ修正した。
その失敗でもbackendは呼ばれておらず、source固定前のtest修正として記録した。

新しい数値結果、native資源の肯定/否定、CTSとのwinner、新規性/主methodの採択はない。

## Source・provenance

- [v2 contract](../../../artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/contract_v2.json)
- [pending authorization](../../../artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/authorization.json)
- [focused tests記録](../../../artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/focused_tests_v2.json)
- [runtime identity確認](../../../artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/runtime_preflight_v2.json)
- [provenance確認](../../../artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/provenance_audit_v2.json)
- [source manifest](../../../artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/source_manifest_v2.json)
- [runner差分](../../../artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/runner_implementation_diff.patch)
- [future runner](../../../scripts/tracks/algorithm_codesign/g9_p5_matched_native_v2.py)
- [authorization gate](../../../src/trottertracks/algorithm_codesign/g9_v2_launch.py)
- [境界tests](../../../tests/tracks/algorithm_codesign/test_g9_v2_api_boundary.py)
- [旧G9失敗結果/GPT handoff](g9_results_and_gpt_handoff_20261010.md)

旧982pathを保護（共通索引9pathは基点全文をprefix保護）。共有src/trotterlibのcode変更0。
新sourceのcritical manifestはauthorizationを除外する。authorizationは将来許可された直接子Aでのみ変更し、S自己参照を避ける。
現在のpending `source_commit=null`は欠落した実行準備ではなく、未承認実行を拒否するための状態。

## 次に必要なauthorization（今回作成しない）

将来、固定source Sのone-shotが明示認可された場合だけ、Sの直接子Aを作る。
変更できるのは以下2pathのみ。

1. `artifacts/track_b_g9_v2_api_boundary_preparation/2026-10-10/authorization.json`
2. 必要なら`docs/tracks/algorithm_codesign/g9_v2_execution_authorization_receipt.md`

authorizationを`status=APPROVED_FOR_ONE_G9_V2_RUN, science_execution_authorized=true, source_commit=S`にし、
新しい実行指示の原文、v2 contract hash、runs1/retries0/mandatory_STOP=trueを束縛する。
HEAD=A、Aの唯一parent=S、remote SHA、clean、authorization-only diff、source/runtime/input identity、v2 marker absentを確認する。

future commandは既存isolated runtimeから
`python -B scripts/tracks/algorithm_codesign/g9_p5_matched_native_v2.py --source-commit <full S>`。
registered取得/比較は一回だけ、marker消費後はpartial failureでもretryしない。
全outcomeでmandatory STOPし、保存値・provenance確認と必要資料の公開だけを行い、研究判断をGPTへ戻す。

**今回の作業はsource preparationの公開で終了。authorization=false、科学実行0、次stageへ自動進行しない。**
