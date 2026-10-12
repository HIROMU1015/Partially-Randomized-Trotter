# N3幅比較v2：GitHub公開と再取得確認

2026-10-12レビュー系列。比較実装へのGOに従う準備を完了し、**明示科学実行認可前でMandatory STOP**。
中心テーマの採択・N1再開・N2再開・新pilotの実行認可を行っていない。

## 固定資料

repositoryは`HIROMU1015/Partially-Randomized-Trotter`、branchは`n3-width-comparison-v2-preparation-20261012`。
payload commitは`9da12e2753405ba0ca029dfa905dd98f2bd27a07`、数値source commitは`3eba0814e690e95993db60809c4c3c96784b07c0`。

- [準備と検証記録（固定payload）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9da12e2753405ba0ca029dfa905dd98f2bd27a07/docs/research/n3_width_comparison_v2_preparation_20261012.md)
- [数学・比較契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9da12e2753405ba0ca029dfa905dd98f2bd27a07/docs/research/n3_width_comparison_v2_contract_20261012.md)
- [事前登録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9da12e2753405ba0ca029dfa905dd98f2bd27a07/docs/research/n3_width_comparison_v2_preregistration_20261012.md)
- [実行計画JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9da12e2753405ba0ca029dfa905dd98f2bd27a07/docs/research/n3_width_comparison_v2_execution_plan_20261012.json)
- [未使用6入力pack](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9da12e2753405ba0ca029dfa905dd98f2bd27a07/artifacts/n3_width_comparison_v2/2026-10-12/preparation/input_pack.json)
- [generatorの固定source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eba0814e690e95993db60809c4c3c96784b07c0/src/trottertracks/representation_exploration/n3_width_generator_v2.py)
- [独立checkerの固定source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3eba0814e690e95993db60809c4c3c96784b07c0/src/trottertracks/representation_exploration/n3_width_check_v2.py)
- [保存合成証拠](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9da12e2753405ba0ca029dfa905dd98f2bd27a07/artifacts/n3_width_comparison_v2/2026-10-12/preparation/synthetic_saved_proofs.json)

input SHA-256：`4c59ba790c7a1faefa6e60dbfbde3d13b718e168bb6491fa71594103b3ac8395`。
plan SHA-256：`dc43a8d781e76ba4c3fc92f528c973f225861bf47160c5866bff2bebd9eb9d53`。
source fingerprint：`9f7687e74fe93f4b238579ee70b846521f4ea296195491c9f22e536a60066d96`。

## 取得検証

GitHubへpush後、別bare cacheへGitHubからfetchし、旧基点からの全変更ファイルと共有checker依存を新しいexportへ展開した。
cacheにはlocal object alternatesがなく、浅い履歴の境界は残るが、今回のpayload treeとsource commitの全対象blobを取得できた。
[blob台帳](../../artifacts/n3_width_comparison_v2/2026-10-12/publication/retrieval_inventory.json)の52 blob、合計3,497,382 bytesがlocalと一致。
source commitの13 blobはplanのSHAとも一致した。

[remote preflight](../../artifacts/n3_width_comparison_v2/2026-10-12/publication/remote_preflight.json)は再取得sourceを使ったidentity/schema/環境/settings検査でPASS。
exportには`.git`がないため、commit/blob関係は別途bare cacheの`git show`で照合した。
新6入力のbound・eigensolver・solver・certificate探索・幅・採否の計算は0。

[再取得した合成証拠の独立検査](../../artifacts/n3_width_comparison_v2/2026-10-12/publication/remote_saved_synthetic_audit.json)は90 certificate、360 PSD checksを保存データのみで確認した。
producer、NumPy、SciPy、CVXPY、SCSをimportせず、証拠の再生成や科学batchの再実行をしていない。
合成32テストの再実行や第三者の独立科学再現、CI実行としては扱わない。

## 次の明示認可の範囲

完了済み[科学レビュー第14節](../../artifacts/n3_width_comparison_v2/2026-10-12/preparation/review_intake.md)は
「新しい科学実行の明示承認が必要な位置で停止する」と指定している。
実装・合成検証・公開だけを理由に同内容の詳細GPTレビューを繰り返さず、次は上記source/input/planに対する一回の実行指示を待つ。

対象は3stress family×full5/6の6H、ν2/3、候補2mode/H、非負density、一外部mode、ε_proj=.01。
7method×kernel/cold×3 timing repetitions、504 mode-method records、managed SCS calls最大432。
2workers/thread1、AS3GiB/worker、aggregate sampled RSS6GiB、output128MiB、wall/CPU capなし。
attempt1/retry0、専用execution branch、fresh run1で実行し、独立保存監査とGitHub公開後にMandatory STOPする。
新結果を受けてGPTが研究上の価値を判断する。新入力・追加scan・実分子・多外部mode・一般二体への拡張は含まない。

旧primary `NO_ADDITIONAL_ACCEPTANCE_OBSERVED`、旧2-base gate、N1停止、N2保留を維持する。
global manifestは既存`relevant_commits[7].description`欠落によりFAILのまま。今回の取得・合成検査PASSと区別する。
