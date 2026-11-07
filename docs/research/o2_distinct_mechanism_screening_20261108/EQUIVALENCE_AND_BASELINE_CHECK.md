# 同じ情報・同じ最適化を認めたbaseline照合

2026-11-08。候補の式と入力は[DISTINCT_MECHANISM_CANDIDATES.md](DISTINCT_MECHANISM_CANDIDATES.md)。これは限定技術選別で、学術的新規性・科学的GO・全O2のNO-GOの正式判断ではない。

## 比較の同一条件

同じactual finite Q、phase付きselected full U、同じ状態アクセス、coordinate target/ε/α、全attemptの状態消費、未知費用を残したfull wrapperを比較する。候補に与えたsummary・score・逆回路・途中測定・fusionはbaselineにも認める。未知のground/Born/path signal/true varianceを無料にしない。

「どの量子測定もPOVMで書ける」「どのreturnも広義aggregationで書ける」という抽象的包含だけでは、費用付きの操作的同値を証明したことにならない。式だけの包括的なNO-GOを主張せず、同じ実装可能な情報を与えた際の具体的手順を対応させる。

| 今回の方向 | 強いbaselineで再現する操作 | 残り得る差分 | 判定 |
|---|---|---|---|
| normalizer表を作らない二段推定 | 同じmetadata scoreのrejection、独立不偏normalizer推定、同じA_hat/a weight、全拒否費 | score自体を安く得る新構造は未取得。正確normalizer表の不在だけでは差でない | REDUCES_TO_EXISTING_METHOD |
| finite-state joint sampler/return | 同じ遷移表でbackward path和とconditional IS、同じendpointへReturn/fusion | compactでphase/law/full価格を保持するsummaryの新しい取得量差は未確認 | DISTINCTNESS_UNRESOLVED |
| 新しい座標outcome測定 | 同じHの既知spectral/direct/grouping、又は同じblock encodingからの既知observable測定 | 安価なspectralアクセスの新構成は未確認 | KNOWN_METHOD_APPLICATION |

## 既存O2との重複を最小限に抽出

| 既存機構 | 今回の還元確認に必要な部分 | 今回しないこと |
|---|---|---|
| positive LCU＋early abort | Eは同じ早期flagを与えたLCUと操作・outcome・分岐費が同じ | 旧16 testの再実行、Eのflag/decoder変種実装 |
| M/direct/grouping | 共通observableを直接測る場合は強い対照。literal十分条件の負結果は一般同値性否定でない | Mの別slot、232群の再分割/救済 |
| classical/quantum CV | 平均既知controlでcenter、量子側アクセス・学習費が必要 | 特徴量追加・真signal oracle化 |
| stratified/IS | 元lawへのweight、全positive support、配分・学習費を戻す | 単なるscore再最適化 |
| Multilevel/telescoping | 差分が小さいだけではbit noiseが消えず、coherent access/scaleも有料 | A2の別名実装、Dへの自動移行 |
| Return/Closed/CTS/Parent | 同endpoint/factorizationの集約・短returnを対照にも認める | 旧classのparameter再探索 |
| D2/fusion/processing | 同じliteral inverse、basis/gate fusionを両側に適用 | O1/O3・C-S実験 |

## SOURCEが提供する情報と提供しない情報

以下はbase f085b565内の静的参照。関数の実行・科学資格の再認証ではない。

- [return_aggregation.py](../../../src/trottertracks/algorithm_codesign/return_aggregation.py)：reduce_word、ReturnKernel.word_series/mass/parent、dyadic_bernoulli/index。局所係数とfinite-bit変換を提供し、全reduced-word universeやwhole価格normalizerは取得しない。
- [parent_allocation.py](../../../src/trottertracks/algorithm_codesign/parent_allocation.py)：digital_proposal/normalized_enclosures。明示tableと外部costを消費する。新しい安価なcost取得APIではない。
- [a1_control_variates.py](../../../src/trottertracks/algorithm_codesign/a1_control_variates.py)、[a1_n3_connection.py](../../../src/trottertracks/algorithm_codesign/a1_n3_connection.py)：actual CategoricalLaw、既知質量・conditional取得・stream/stage分離。new score normalizerや新observable oracleを作らない。
- [a2_lcu_access.py](../../../src/trottertracks/algorithm_codesign/a2_lcu_access.py)、[e_label_erasure.py](../../../src/trottertracks/algorithm_codesign/e_label_erasure.py)：有料PREP/SELECT・phase付きcontrol・全attempt decoder。無料のcoherent Q sampler・state逆準備なし。
- [m_label_average.py](../../../src/trottertracks/algorithm_codesign/m_label_average.py)、[m_n0_header_binding.py](../../../src/trottertracks/algorithm_codesign/m_n0_header_binding.py)：適格性・header規約は参照のみ。実labelを新候補へ再分割しない。
- [rte.py](../../../src/trotterlib/rte.py)、[Hadamard wrapper](../../../src/trotterlib/rpe_hadamard_interrogation.py)：RTEEventApplication/event phaseとbit対応。平均channelへの置換やphase除去は認めない。
- [D2 boundary](../../../src/trottertracks/algorithm_codesign/rte_development/boundary.py)、[全attempt accounting](../../../src/trottertracks/algorithm_codesign/rte_development/accounting.py)：literal inverse・費用の既存範囲。同じ簡約はbaselineにも与え、未知full価格を局所加法で補わない。

旧39 stageの再監査・raw再計算・旧testの再実行は不要とした。新frameworkやsource copyを作らない。

## 非同値性の未解決条件

finite-state方向には、既存codeに存在しない「安価な表構築」の可能性だけが残る。必要なのは、具体的な入力family、summaryの内容と更新、phase/conditional-lawの保存、dとbit費の上界、同じstrong baselineがその利得を再現するための取得量との比較。単なるoracle宣言や、同じ等価類を別名で持つだけでは不足する。

仮に古典生成が速くなっても、量子回路/variance/Nが不変なら量子資源削減を示したことにならない。新しいreturnで実際に操作が減る場合も、同じfusionを認めたbaseline後に残るfull費用差が必要である。

一copy POVMの二次moment下界はoracle/query/回路価格の下界ではない。既知direct measurementを実装不能な無料oracleとしてbaselineへ与えて候補を否定するのではなく、候補が要求する同じspectralアクセスで再現されることを確認した。この区別を満たさない候補は「非同値性未確認」に留め、実装しない。

## 固定結果の参照範囲

[O2統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d28164b5d51e0ca5a0ade74ded8ff176b443dab8/docs/research/o2_research_results_integration_20261108/O2_RESEARCH_SUMMARY.md)のJ1/Return/Parent/A1/SCV/A2の障害を再利用した。[M結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e98589ae129d3b81ad428c947d2f4c88623caa9b/docs/research/o2_m_n0_metadata_result_20261108/REPORT.md)は保存linear H4/1Å/STO-3G/legacy DF12/8system/L_D3/m0q1/δ.02/r1K2/atol0/actual Q/selected fullのn=0唯一slotに限定された232単独群である。[E結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f085b56566d60d9cb5df221f352a2b52ac51ea88/docs/research/o2_e_early_abort_technical_validation_20261108/REPORT.md)は人工1/2-system qubitで同じearly LCUとの差0。どちらも全O2のNO-GOにしない。旧status/STOP・M/D/E保留とC1凍結を維持する。
