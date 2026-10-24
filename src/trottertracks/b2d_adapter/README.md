# B2-D最小adapter・人工技術検証報告

2026-10-24 JST。**構造準備のみ。科学検証・回路生成・compile・性能認定は未実施。**

## 起点と成果物

- 設計：PR repository `1e2feb033bd7857482db09dd1f3d5b8692f100bf` の [B2限定設計](../../../docs/research/research_b_b2_limited_design_20261024.md)。
- 新SOURCE：[`adapter.py`](adapter.py)、[`_published.py`](_published.py)、stdlibだけを使うpackage [`__init__.py`](__init__.py)。既存科学module・トップレベル `trotterlib.__init__` をimportしない。
- 人工test：[`tests/management/test_b2d_adapter.py`](../../../tests/management/test_b2d_adapter.py)。
- 将来の[科学検証契約草案](../../../docs/research/research_b_b2d_science_validation_contract_draft_20261024.md)。未承認・grantなし。
- 専用branch：`codex/research-b-b2d-adapter-preparation-20261024`。既存tracked SOURCE、結果、共有概要、manifest、科学契約は変更しない。

## 実装した接口

| API／記述 | 実装した役割 |
|---|---|
| `context_from_preparation`／`DContext` | 既存 `DFPartialS2Preparation` からpreparation／tail identity、ordered block labels、system register、constant／抽出identityを読むmetadata bridge。実際のpreparation生成はしない |
| `published_math_sequences` | 固定版のYP8m8 kernel／processor生成関数を再利用。係数生成・探索・修正なし |
| `DRun` | `G_D(h)=P(τ)K(τ)^n_D P(τ)†` の数学的S2列とappend用block指数記述を返す。`τ=h/n_D` |
| `actual_adjoint` | 実現したD指数記述を逆順・signed duration符号反転して返す。opaque tailの逆は作らない |
| `TailOccurrence` | 既存tail occurrenceのpayload、event phase、normalizationを**同じobjectのまま**保持。新samplingやtail変換をしない |
| `make_b2d_plan` | `h=δ/2` の同方向Gをtailの前後へ置く。preparation／tail／時間／controlのmetadata不一致やmissing metadataを拒否 |
| `fuse_adjacent_d_runs` | 直に隣接し、contextとτが完全一致するD runだけkernel反復数を加算。残差はbarrierとし、P(τ)をP(2τ)へ替えない |
| `PhaseOperation` | constant／tail identityのscalarをphysical総時間に対して一度だけ記録し、globalかrelative ancillaかを区別。複素指数は評価しない |
| `structural_cost` | kernel反復数とD runごとのprocessor pair数を分離。支払い単位は一execution内の連続D run。native価格・shot数を返さない |

返すものはblock label・有理数duration・control／phase metadataであり、量子gateや回路ではない。実block specを使ったgate emission、basisの実現inverse、compiler、physical演算子評価は将来の別承認対象である。opaque payloadの中身と宣言identityの一致も今回のadapterは数値検証しない。payloadはdeep copyせず、呼出し側が元データとそのidentityを保持する。

## 再利用SOURCEと、追加moduleが必要な理由

固定PF repository：`HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae`、commit `20d64c2fea1ed2d2188777df4f3d962ee94df486`。

| 出所 | 再利用・静的照合 |
|---|---|
| `src/trotterlib/product_formula.py`、blob `70d68691a1dcdedea5447aac4b8bbfa51141d4cb` | `morales_2025_yp8m8_kernel_list`、`morales_2025_yp8m8_processor_gamma_list`、`morales_2025_yp8m8_processor_sequence` の3関数をbody変更なしで移植 |
| `src/trotterlib/pf_decomposition.py`、blob `b0985414adfa2323b7884c0d4683821f91eaec09` | `symmetric_s2_sequence`、`inverse_s2_sequence`、`_iter_s2_steps` の3関数をbody変更なしで移植。上記と合わせて6関数のASTを取得した固定SOURCEと直接照合し、testにもhashを固定 |
| `processed_cost.py` | kernelと明示 `processor_pair_count` を分ける接口を参照。H-chain Pauli／RZ layer価格表は移植・呼出しせず、今回のnative価格に転用しない |
| Published Morales／YP8m8 validator | 列消費順と `unitary=gate @ unitary` を静的確認。固有値・overlapの旧保存結果を新adapterの証拠にしない |

現在のPR `pf_decomposition.py` はcompactな対称PF用で、任意の非対称processor列をそのまま表せない。さらに `trotterlib.__init__` はQiskit・化学・計算moduleを広くimportする。固定SOURCEを変更せず科学依存を避けるため、必要なstdlib依存閉包だけを独立namespaceへ移植した。新しいPF生成原理は作っていない。

PR側では `df_partial_randomized_pf.py` のsplit、`df_partial_s2.py` のordered block／phase、`df_partial_s2_repeated.py` の境界・trajectory、`rte.py` のraw／corrected mean、`finite_rte_signal_validation.py` の比較対象、dictionary `circuits.py` のactual adjoint／wrapperを必要範囲で静的確認した。今回はそれらの科学関数を呼んでいない。

## 数学的積とappend順の決定

Upstream processor関数の説明は `P=Q(t)Q(-t)`、`Q(t)=S2(γ10 t)…S2(γ1 t)` である。本adapterではそのlistを**数学的積の左から右**として読み、append時に反転する。数学的 `P K^n P†` のappend順は `P† → K…K → P` となる。

Kernelは対称でもprocessorは一般に非対称なので、listを反転せずappendすることを同じ規約とは扱わない。既存validatorはfull listを直接appendしているため、**本adapterとそのvalidatorの数値的operator同値性は認定しない**。今回のmockは、固定設計の数学的作用順を抽象記述として実装したことを確認した。将来の科学検証では独立参照に対してprocessorの向きとfull operatorを照合する。

Upstreamのfloat係数は元の演算のまま使い、そのfloatを厳密に表す `Fraction` に変換して記述する。印刷小数の無限精度係数へ置換したり、係数和を修復したりしていない。S2 stage間のfloat加算mergeをせず、実現記述のactual inverseを保持する。native合成後の同値性は別問題である。

## 人工testの結果と限界

Focused command：

```bash
/usr/bin/python3.11 -S tests/management/test_b2d_adapter.py -v
```

**15 tests PASS**。使用interpreterはPython 3.11.0rc1。`-S` とimport barrierでnumpy／scipy／Qiskit／OpenFermion／PySCF／Numba／Torch／CuPy／trotterlibの読込みを避け、stdlibの抽象ラベル・`Fraction`・mockだけで実行した。既存science test一式は実行していない。

| 検証対象 | 確認した範囲 |
|---|---|
| Published SOURCE | 6関数のAST一致、出所commit／blob、kernel／processorの順序 |
| 作用順・inverse | 数学的積からappend列への変換、逆順・符号反転、formal label列のinverse相殺 |
| D配置・境界 | 同方向の左右D半時間発展、残差barrier、同一context／τの隣接融合、異なる準備・順序・時間・control等では融合しないこと |
| Signed time・phase | 負時間の符号、scalarの総physical時間、global／relative destination、event phase metadataの同一object保持 |
| Tail／normalization | opaque payloadとnormalizationの保持、同一lawを計算・生成しないこと、metadata不一致の拒否 |
| 費用metadata | 反復kernelとprocessor pairの構造的支払い単位。native gate count／資源予測ではない |

これらは新B2構成のunitary同値性、controlled相対位相の数値的正しさ、非可換性、physical信号改善、高次精度、finite-bit保証、必要shot数、同精度総資源の証拠ではない。labelのformal cancellationを実回路のphase検証に格上げしない。契約草案に科学検証項目を残した。

実行前に利用可能CPU・memoryを確認した。負荷は小さい記述列とstdlib assertionで、科学計算backend・GPU・BLASを使わず、focused fileを単一processで実行した。worker／thread／memory制限は追加せず、共有環境を変更していない。速度倍率や科学実行時の性能を主張しない。

## 次の担当・未承認条件・STOP

1. **研究B／統括／ユーザー**：契約草案のfixture採択、state／sector、時間窓、q／n_D／r_RTE／K、数値精度・許容差、5arm比較budget、科学call数・grant・retry／STOPを判断する。未採択値を今回恣意的に埋めていない。
2. **Codex（別承認後）**：必要な科学評価器の接続・固定小規模operator／signal検証を担当する。今回はdescriptorを回路へ変換するrendererやscience runnerを追加していない。
3. **研究A／C**：normalization・finite-bit・signal半径・range／shot条件と、DF表現・native processor費・controlled位相をそれぞれ連携する。C1圧縮DFの優先順位を維持し、共同実装・合同科学計算は未承認。

変更は新adapter package・人工test・契約草案のみ。既存SOURCE・raw・grant・ledger・marker・科学契約・共有概要・manifest・旧STOPを保持する。科学計算・Hamiltonian生成・operator／signal評価・回路build／compile・保存数値の再解析・重要な科学レビューは未実施。

専用branchのremote SHAと公開内容を確認後、**mandatory STOP**。人工test PASSをB2-Dの科学的有効性・正式機構採択・総資源改善としない。
