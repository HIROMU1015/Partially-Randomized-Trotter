# B-1a構成精度adapterの準備報告

2026-11-03 JST。保存入力境界、独立Law/Taylor構成、条件付き比較基準、限定precision ladderを実装した。
人工テストのみで検証し、実H4の新演算子構成・power・mean・再解析は実施していない。
詳細は[入力境界](B1_INPUT_BOUNDARIES.md)、[実行計画](B1_SCIENTIFIC_EXECUTION_PLAN.md)、
[次段判断](NEXT_STAGE_DECISION.md)を参照。

## 再利用と最小差分

| 既存資産 | 固定SOURCE・再利用範囲 |
|---|---|
| `precision_adapter.py` | Task A SOURCE4479a1c…のarray検査、exact_real/exact_matrix、binary_power、scalar_context、number/complex_difference、gap_relation、precision_ladder |
| `run_precision_isolation.py` | 同SOURCEの固定input/NPY header照合。実H4のexecute/float_baselineは呼ばない |
| `run_m17_mean_validation.py` | 元m17 SOURCE128e9a5…のliteral_law_mean、fixed_bytes、atomic write_result。grouped_averagesは人工試験のみ |
| `src/trotterlib/rte.py` | carrier4591b8d…のfinite_taylor_operatorをfloat replayで使用。mpmath版のHP演算に適合しないため短いrecurrenceだけ新実装 |
| Task A RESULT `fixed_factors` | 保存高精度mean・収束を再利用。変更しない旧tailを再評価しない |
| 既存人工テスト | Task A18件・元m17 13件を回帰確認。旧ファイルは変更しない |

新規は[b1_adapter.py](b1_adapter.py)と[run_b1_construction.py](run_b1_construction.py)。
旧NumPy関数はbinary64を強制するため、HP event和・Taylor recurrenceには小さな独立実装が必要だった。
LawをTaylor実装に統一する変更はない。新しい汎用管理基盤や承認証明機構は追加していない。

## 実装上の境界

default/`--describe`は保存byte/hash、JSONとNPY headerだけを確認する。
real配列の`np.load`、scalar SOURCE replay、operator/mean評価は明示的`--run`の内側だけ。
`--run`の指定自体は承認証明ではない。操作者が別途の科学実行許可を守る。
未承認の実H4演算がdefaultから起動しないことをmockで確認した。

新θ64を保存済みscalarとは呼ばず、入力値/hexとprovenanceを残す。
保存πを勝手に正規化したり、分布dataclassを通してcanonical weight/π/Bを再生成したりしない。
component確率のp/sum(p)処理は元Aの構成境界に残し、今回は二つの保存Aを固定する。
reference normalizerだけ一回適用し、Lawには追加B除算なし。
HP tailはHPのまま`(R*tail)*F`、q乗、ψ期待値へ渡す。

float replayが保存tailと違った経路だけ新tail固定の比較基準を評価できる。
一致経路はTask Aの保存meanを再利用する。この条件分岐は「同一SOURCEなら原本とbyte一致」との仮定を避けるためである。
両pathのmeanとtailを精度間比較し、必要な場合だけ224 bitへ進み、その後はSTOPする。
matrixの`ctx.norm(matrix)`はentrywise 2-norm（Frobenius）。収束値は厳密boundへ昇格させない。

## 人工検証と未実行事項

[test_b1_construction.py](test_b1_construction.py)は人工2×2行列・人工scalarで、保存/replay区分、
dyadic変換、確率再正規化とsign、event phase/rotation、degree K+1、normalization一回、
非可換なF/R順序とr/q、ψ非正規化、float/HP差、HP round-trip、中断原本、fresh出力、
既存基準再利用/差のあるpathだけ再評価、精度上限、4job spawnを検証する。
realファイルはdefault経路のbyte/JSON/header検査だけに使用する。
新規23件＋Task A18件＋元m17 13件の計54件PASS。
実行結果・環境は[VALIDATION.txt](VALIDATION.txt)。これはLOCALの技術確認であり独立科学レビュー・CI科学証拠ではない。

H4のθ/cos/sin/log/normalizer replay、新A/event/tail/S/power/mean、
保存予算の再計算、SCF/DF/state生成、sampling、compile、B-1b/B-2はすべて未実行。
実行用artifact directoryも新設していない。科学synthesis/validation manifestは変更しない。
2026-10-25共通運用指示§8.1/8.4/10とAGENTSを確認し、専用worktreeの新ファイルと入口索引だけを変更した。
旧SOURCE・RESULT・NPZ・STOP、他branch/worktree/H8研究は保全した。
