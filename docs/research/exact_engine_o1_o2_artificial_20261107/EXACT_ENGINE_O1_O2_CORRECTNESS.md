# O1/O2：人工correctness・数学的同値性

2026-11-07 JST / SERVER1。**人工技術検証PASS。科学engineへの組込み・実H4認証は未実施。**

## 再利用と最小差分

設計正本は `fb28b6a287803ce63a51a1513a57828302003090` の [検証計画](../exact_engine_static_analysis_20261107/EXACT_ENGINE_VALIDATION_PLAN.md)。dense oracleは固定 `560c6092bfb5e71af8eedc4a4be5416922764f98` の `src/trottertracks/algorithm_codesign/a1_engine_connection.py`、SHA256 `3fb23b8f6e838441645708bece6b1bc1d5de4baf394190e7853f66090d8a7bd9`。

[oracle.py](oracle.py) はこのファイルをhash照合し、`ZERO,multiply,add,norm_squared,apply_exact,require_probability,born_probability,threshold_count` の元ASTを変更せずcompileする。元関数の本体、引数、validation、loop順序を再記述していない。科学依存・loader・RNG・gate builderをimportせず、標準libraryだけで既存関数を使うための限定loaderである。SOURCEが異なる場合は停止する。旧module・科学runtime・PROFILE_LOCKを変更する仕組みではない。

[prototype.py](prototype.py) の追加はO1/O2だけ。

- O1：matrixをexact Fraction pairとして検査・不変snapshot化し、各rowのexact非零entryを元j昇順で保存。実値で対角性を判定する。対角では一つの積、一般疎では元`multiply/add`の積和。全entry非零のdenseは旧`apply_exact`へfallbackする。
- O2：`(state_dimension, ordered_qubits)`をkeyとして旧式と同じindex blockを遅延生成・LRU再利用。計画はfrozen dataclass＋nested tupleで返す。state計算そのものはO1と共通である。
- stateは(real Fraction, imag Fraction) pair、matrixは正の2冪dimensionの正方形に限定。state dimension/support/重複/範囲・matrix対応を検査する。float/intの振幅やbool/float supportは明示拒否する。旧kernelより入力型を狭くしており、実engine側の互換性は後日確認事項である。
- 入力stateのin-place更新なし。元と同じ`out=list(v)`を残し、各出力は全て元vから計算する。O5のcopy/Born変更、O3のgate照合省略、確率/state cache、float/GPU/intervalは導入しない。

## 数学的説明

complex pairの加減乗算は全てFractionなので丸めがない。O1の各出力rowは元の `sum_j m_ij*v_j` からm_ij=(0,0)の項だけを除き、残る列順を保持する。対角はその特殊例で、零rowは(0,0)になる。極小非零は除かない。dense fallbackは元関数そのものなので同じ結果を返す。

O2は元の `base | sum(((j>>k)&1)<<qubits[k])` をそのまま使う。support bitが零のbaseごとに同じordered local index列が得られ、異なるbaseのblockは交差せず、全amplitudeを一回ずつ覆う。順序を変えたsupportは異なるkeyであり、同じ集合へ丸めない。cacheはこの不変列だけを再利用する。

従って、有効なexact入力に対し各gateの全出力が元と同じ有理数値になる。gate列に関する帰納で全中間stateが同じになり、同じ旧Born比・thresholdから同じp/tと同じ固定wordのbitが従う。これは実装方針の代数的説明であり、Python実装の形式検証・全科学SOURCEの認証完了ではない。

## 人工fixture・テスト範囲

[fixtures.py](fixtures.py) は手書き/固定式の有理数のみを生成する。保存H4 state、実gate matrix、LAW/catalog、旧bit/streamを一切使わない。sin/cos/sqrtやQiskit gate生成もない。Hadamard型は係数±1/2の人工混合、XXPlusYY型は3/5と−4i/5のblock、ordinary closed-control型はlocal LSB側control=1へtarget blockを埋めた人工matrixである。全matrixが物理unitaryであるという前提を置かない。

| 検査 | 範囲と判定 |
|---|---|
| 局所作用 | 1/2/3-qubit matrix、identity/diagonal/tiny/sparse/dense/H型/XX型/control型。4-qubit人工stateの全ordered support permutationと4 state種、計1,008条件 |
| State | sparse、dense complex、非単位norm、zero成分、正負相殺、dyadic、異なる一般分母。float allcloseを使わず全real/imag Fractionの等値 |
| 3方式 | 1,008条件でdense/O1/O1+O2 cold/warmを比較。全て同じ出力 |
| gate列 | 測定前固定の10 fixture×12 gate=120中間更新について3方式を毎gate比較。各段のp/tも明示比較し、固定6 wordからのbit一致 |
| index plan | n=0…4の全ordered support（空support含む）で旧式のindex列・全amplitude一回更新を比較 |
| cache | same-key hit、同集合逆順、dimension違い、容量1/2でeviction/rebuild、計画の外部変異拒否、不正support/dimension拒否 |
| validation | 零/非2冪/非正方matrix、state/matrix/support不一致、重複・負/範囲外・bool support、非Fraction値を拒否。fallbackでvalidationを回避しない |
| phase相当・tiny | complex非対角、符号/虚部を保存、極小非零1/2^120と−i/3^80を保持。実global/controlled回路の生成はしていない |
| Born | 旧norm_squared/born_probabilityを再利用、9/25の既存人工例・非単位norm・scale不変、zero norm拒否 |
| threshold | 旧53-bit exact ceil、p=0/1、k/Mと±1/M²、threshold直前/同値/直後の事前指定規則word、固定word集合のbit一致 |
| 非変異 | 人工matrix/state snapshot不変、PreparedMatrixのdetached snapshotとnested tuple/frozen性 |

元の `test_a1_engine_connection.py` のqubit順・norm・grid・入力拒否例を参照して適用した。同file全体を実行するとQiskit fixture生成/qualification経路を含むため、今回は実行せず、必要な人工算術検査を上記の限定suiteへ追加した。

## 実行結果と技術修正

O1単独は [O1_TESTS.log](O1_TESTS.log) の7 tests PASSを確認してからO2を追加した。初期の[log](O1_INITIAL_TESTS.log)では、比較ループは通ったが末尾の件数assertの期待値1056が実際の1008と違った。期待件数だけを修正し、kernel出力不一致はなかった。人工fixture定義の括弧の構文誤りも実行前に修正した。

O2追加後の [CORRECTNESS_TESTS.log](CORRECTNESS_TESTS.log) は11 tests PASS。benchmark開始前にも同じ最終SOURCEのsuiteを実行し、[raw_run01/TESTS.log](raw_run01/TESTS.log) と [CORRECTNESS.json](raw_run01/CORRECTNESS.json) に **11 tests、failures0、errors0、success=true** を保存した。後者がbenchmarkに対応する最終結果である。

cacheのcold/warm/再生成を含めてO1/O2の状態・p・threshold不一致は0。独立の異なる実装/hostによる再現、CI、形式検証、実H4での確認は行っていない。

## 既存認証との対応と残る仕事

今回が示すのはこの独立prototypeの人工correctnessであり、認証済みengineではない。保存certified matrixを読み込まず、元`b_engine`・local証明・SOURCE hash・PROFILE_LOCKも変更していない。

同じcertified有理数matrix、同じstate、ordered support・gate列・phase・ordinary control・最終norm、旧Born/ceil/word消費を新SOURCEでも保ち、matrix identity照合をそのまま残せるなら、保存局所δ*/E/N2/gridの証拠を再利用できる可能性がある。新SOURCEと入力・環境の対応付け、実gate列との接続、有限実pathでのstate/p/t比較、追加memory、STOP/call accountingの確認は残る。旧profileを緩めて実装を差し替えない。

実H4性能検証には対象SOURCE/input/path/軸・有限call・RNG/build/compileの範囲・fresh出力・STOPを具体化した別明示承認が必要。科学的GO・総PR/RPE費用改善は今回判断しない。公開照合後mandatory STOP。
