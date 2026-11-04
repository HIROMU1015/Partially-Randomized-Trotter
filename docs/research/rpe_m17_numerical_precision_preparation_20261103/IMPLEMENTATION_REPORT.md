# 精度切り分けadapter：実装・人工検証・再利用

2026-11-03 JST。**Task Aの実行準備完了。実H4の新しい数値評価は0。**
親は元m17結果commit `cd749066904d3f29b28b03c19ab69370372b157f`。
branchは`codex/rpe-m17-numerical-precision-isolation-preparation-20261103`。

## コードと再利用

- [precision_adapter.py](precision_adapter.py)：complex128の正確なdyadic変換、mpmath context、
  binary powering、float基準、scalar差・収束・原因候補の整理。
- [run_precision_isolation.py](run_precision_isolation.py)：固定ファイルのbyte/header adapter、
  Task Aの2段階、process並列、途中結果・STOP記録。defaultはmetadataだけを読む。
- [test_precision_isolation.py](test_precision_isolation.py)：人工2x2/scalarとmetadataの18 tests。
- [VALIDATION.json](VALIDATION.json)：実行範囲・人工検査・code hash・人工timingの最小記録。

元runnerの`write_result`を変更なしで再利用し、結果を一時fileからreplaceして保存する。
`fixed_bytes`は固定Git objectのbyte照合、`residual`はfloat基準の元mean再現確認に使う。
元`evaluate_context`全体は呼ばない。これはlaw one-eventと内側r乗を再構成するため、Task Aの固定tail比較に不適合である。
必要な末尾の`R @ tail @ F`、NumPy matrix_power、vdotという演算順だけを短い基準関数として再現する。
新しいsampler・DF engine・state生成・compiler・RPE decoderは実装していない。

高精度matrix backendはLOCALの既存mpmath1.3.0を用いる。
この環境はPython backendで、python-flint/gmpy2は利用できなかった。
既存の[h4_ball_evidence.py](../../../src/trotterlib/h4_ball_evidence.py)にはfloatを有理数にする先例があるが、
flint dependencyと別研究の固定契約を持つためmodule全体を流用しない。
新しいmpmath adapterでもas_integer_ratioによる同じ「binary値を十進数へ置換しない」原則を用いた。
別研究のgrant・STOP・private runtime条件を今回へ移植せず、原SOURCEも変更していない。

mpmath標準の整数powerはbinary方式だが、最後に未使用の平方も計算する。
短いbinary loopでこれを避け、q131072では17回の平方を使う。
このqではNumPy側もbinary poweringのため、**独立した累乗アルゴリズムの検証とは主張しない**。
人工の閉形式・mpmath標準powerとの照合でinterfaceを検査した。

## 入力境界と正確な変換

元operators.npzはhashとheaderを確認する。指示中のA_prob/F/law_one_event等は略称であり、
実keyはA_probability/forward/law_event_mean等。[対応表](NUMERICAL_DISCREPANCY_ANALYSIS.md)を作った。
archiveをrenameしない。
defaultと準備検査では配列payloadを数値decodeしない。science loaderは--runの後だけ呼ばれる。

floatの実部・虚部をそれぞれ正確な整数比へ変換し、context精度53 bit以上でdyadic値を保持する。
負値・隣接float・subnormalも人工テストで確認した。保存前に失われた桁は戻らず、psiの再正規化もしない。
mpmathのclone contextをjobごとに使い、共有global精度を変更しない。
complex64・不正shape・非有限配列は拒否し、黙ってcomplex128へ変換しない。

Task A-stepは新しく保存したS64を固定、Task A-factorは元F/R/tailを固定する。
後者も内側r乗やnormalizationを再計算しない。
coefficient/Taylorとliteral-lawの経路を統一して差を消す処理はない。
B-1/B-2は[段階別計画](PRECISION_ISOLATION_PLAN.md)だけを示し、実行adapterには入れていない。

## 人工テスト

18 tests PASS。検査は主実行者によるもので、独立重要科学レビューではない。

- binary64 dyadicの保持、十進0.1との区別、隣接値/subnormal、低すぎる変換精度。
- 2x2 nilpotentのq131072閉形式、一般の小さいqと標準power、非可換F/Rの正しい積順。
- 保存attenuationを二重に掛けないこと、fixed-step/fixed-factorで入力境界が混ざらないこと。
- 収束判定、224 bit追加条件、閾値近傍の保留、原因候補とu_num/RPE認定の分離。
- default CLIでnp.load/累乗/workerを呼ばないこと。実ファイルはbyte/headerだけ確認。
- 人工spawn workerの完了、中断記録・上書き拒否、基準失敗ならHPを開始しないこと、
  gapが残る条件でだけA-factorへ進むこと。

元の人工13 testsで確認済みのDF context/r32/threshold/平均tail構成を、今回新しく実行し直す対象にしていない。
必要な既存関数とテストのinterfaceを読み、固定tailを入力とするTask Aへ適合させた。
実H4のS再組立て、q乗、mean再現、任意精度の差・収束・原因候補の実判定は未実施。

## 精度と性能

96/160、必要時224 bitの選定理由と収束規則は[固定計画](PRECISION_ISOLATION_PLAN.md)に記した。
高精度結果は計算精度以上の表示桁で保存し、scalar比較には288 bitの独立contextを使う。
出力文字列の桁数や収束差を、外向き丸め・真の平均への誤差上界とは扱わない。
u_num_bound=null、RPE_tolerance_certified=false、scientific_status=UNDETERMINEDを維持する。

人工70x70行列の96 bit積1回は約.354秒だった。分子入力を使わない小測定であり、実H4の速度倍率ではない。
各modeの初期4独立jobをprocess並列化し、利用可能CPUとjob数からworker数を選ぶ。
NumPy基準のjob内BLASは初期4 thread、mpmathの行列積はPython多倍長でありnested BLAS並列ではない。
新しいpackageのinstall、共有環境・他ユーザーjob・GPU設定の変更は不要。
将来の実行前に実際の空き資源を確認する。人工的なCPU/RAM/wall-time上限は置かない。

## 保全と公開

元m17 SOURCE/RESULT/NPZ、B/A/V1/V2/V3/R1、N/α/β/epsilon、元manifestとSTOPを変更していない。
新しい科学artifact directoryは未作成。関係する文書索引だけを追加する。
準備成果を専用branchへcommit/pushしてremote SHAと本文を照合後、mandatory STOP。
次は[NEXT_STAGE_DECISION](NEXT_STAGE_DECISION.md)で固定したTask A範囲へのユーザーの科学実行許可を待つ。
