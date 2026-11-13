# H4 PR：全13ラウンドの物理誤差とRPE誤差契約

2026-11-13 JST。**保存証拠のSOURCE整理・転記と技術検証のみ完了。全energy RMSE、十分N、共通有効Hamiltonian、実samplerの資格は未取得。** 新しいsignal/reference/normalization/N/RMSEの計算・保存値の再解析は0。Bのdecoder・統計処理は実装していない。

成果物：[物理input](PHYSICAL_INPUTS.json)、[B向け契約](HANDOFF.md)、[有限科学計画](SCIENCE_PLAN.md)、[新8テスト](TESTS.txt)、[既存13回帰](REGRESSION.txt)。旧S0/m1/S1＋S2の結果・packet・SOURCE・STOPは変更していない。

## 固定条件とSOURCE

Linear H4、1.00 Å、STO-3G、保存tol-only 1e−8 DF10、weighted-Hermitian completion、sector(2,2)/dimension36、S2/L_D1/K2/ΔPF=.1、h=.1、M12、P=0。H/state/snapshot、fragment順、scalar/identity一回会計、prior、decoder、law、compiler/profileは元S0のまま。

κはB正本configurationのexact有理数を転記し、表示近似から再生成していない。εE=.00159360010199040 Ha、εqpe=εtrot=.00079680005099520 Ha、minkowski、finite_B_diagnosticを保全。これは誤差契約の**目標**であり、成分の達成証拠ではない。

| 出典 | 固定commit・用途 |
|---|---|
| A全13round結果 | `169d3316dd008eb98dd9be56fe4ecefc5c1a7d60`、[元REPORT](../h4_pr_s1_s2_rmse_results_20261113/REPORT.md)、result/N_HANDOFF/SOURCE_INDEX |
| S1＋S2科学SOURCE | `c76ffe5dbb2a8dd1a05f1a85038f2d48b959f73e`、completion/power-mean。今回は起動せず |
| S0結果／科学SOURCE | `7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75`／`7c3e1e5db0c55dd8249e9030a4c7e9954496fa7e`、m0/m6 |
| m1結果／科学SOURCE | `3e035c0d61386010f59ad7ea792bc7a5cefd18ac`／`ac873e7fcc42e937084fa153476cbd87565702b4` |
| B V2 | `3216d27a69ac8532d2a22e78f5f94491d202d238`、`h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json` |
| 旧mean backend | `e417ffc9f52b13c18f5493193ce510d858664259`、`scripts/h46_long_time_signal.py:evolve_batch`、`h46_new2_core.py:polynomial` |
| finite-K law | `71ba49bb28b295aa28e51101bc3b1bf07d500767:src/trotterlib/rte.py`、finite_rte_distribution / finite_taylor_operator / finite_rte_operator_moments |

canonical configuration IDは `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。新packetはBのN packetの置換ではなく、そのcanonical行を参照するA物理evidenceである。

## 13ラウンドの保存coverage

次表は保存metadataとnominalの転記。各行の**全complex signal・bias・phase・radius・logB/B・u/margin・reference pair**はPHYSICAL_INPUTSの`rounds[m].diagnostics`に元値をそのまま収録した。`diagnostics_source`はresult内selectorまで固定し、`mean_provenance`と`scientific_source`を区別する。source付き値の存在は科学的資格を意味しない。

| m | T | qPF | R_outer | 元診断 | mean / referenceの由来 | nominal各軸 |
|---:|---:|---:|---:|---|---|---:|
| 0 | .1 | 1 | 2 | S0 | P1保存mean・独立pair | 64 |
| 1 | .2 | 2 | 4 | m1 | 89c91c6のrecords[5]・独立pair | 60 |
| 2 | .4 | 4 | 8 | S2 | 高速mean・保存pair、m2のみ逐次照合 | 56 |
| 3 | .8 | 8 | 16 | S1 | 155529eの保存mean・新独立2 action | 52 |
| 4 | 1.6 | 16 | 32 | S1 | 155529eの保存mean・新独立2 action | 47 |
| 5 | 3.2 | 32 | 64 | S2 | 高速mean・保存pair | 43 |
| 6 | 6.4 | 64 | 128 | S0 | P1保存mean・独立pair | 38 |
| 7 | 12.8 | 128 | 255 | S2 | 高速mean・保存pair | 34 |
| 8 | 25.6 | 256 | 510 | S2 | 高速mean・保存pair | 30 |
| 9 | 51.2 | 512 | 1020 | S2 | 高速mean・保存pair | 25 |
| 10 | 102.4 | 1024 | 2040 | S2 | 高速mean・保存pair | 21 |
| 11 | 204.8 | 2048 | 4080 | S2 | 高速mean・保存pair | 17 |
| 12 | 409.6 | 4096 | 8160 | S2 | 高速mean・保存pair | 12 |

m3/m4の`exact_tail_norm_error`は欠測、観測値を0にしていない。元SOURCEのnorm guardを用いた承認済みの条件付きempirical診断のみ。S1のreferenceは旧中心と新2経路を保持、S2 pairの歴史的`reference_state_actions=2`は今回の新call数ではない。

主8高速meanのmaximum normはspectral block境界の観測。m2逐次照合に残る古いdomainラベルは元rawを保全し、SOURCE_INDEXのtechnical_annotationsを新packetへ転記した。逐次経路の実観測範囲は各half-step/event。これらを全中間状態のrigorous誤差証明へ昇格しない。

## 物理量と誤差の意味

| 保存量 | 意味・限界 |
|---|---|
| `reference` | 同じ保存stateに対するfull DF-Hの期待値。厳密groundの単一位相と自動同一視しない |
| `exact_tail` | 同じhead順・S2・scalarのままtailを厳密指数とする内部対照。Full19/Full-H PFやH4 B0/prefix7ではない |
| `outer_PF_or_discard_bias` | exact_tail−reference。このPRは全tailを保持するので、この旧field名の値は分割PF近似の寄与。fragment discardを新たに導入しない |
| `finite_RTE_bias` | corrected−exact_tail。同じ物理候補のfinite-K mean近似の寄与。有限shotのランダム誤差とは異なる |
| `total_signed_bias` | corrected−reference。上記2成分のsignal空間での分解。保存残差も転記し、再評価していない |
| `raw`, `corrected`, B | rawがideal Hadamard平均、corrected=B×rawがnormalization補正したmean。Bは正の全interrogation係数。raw減衰を物理noiseと数えたりB²を二重適用しない |
| phase/radius/disk | 指定referenceに対する局所量。branchを含む最終energy推定量の誤差ではない |
| u/logB margin | R-awareの経験的数値allowance。過去値・SOURCE・normalizer感度を保持。認証区間やrigorous上界ではない |
| nominal N | 元のceil(B²[11+4(M−m)])による経験的配分を**転記**。十分N、actual minimum、全RMSE達成を意味しない |

K2のpaired finite RTEは次数0/2のevent分布を使い、補正meanは3次Taylor polynomialの積になる（`rte.py:finite_taylor_operator`）。回路eventのideal期待値とこのpolynomialの対応には、同じinvolutory tail、signed weights、event順・独立性・phase・scalarの条件が必要。DF fragment全体を無条件にinvolutionとみなすlawではない。実samplerの浮動小数点確率・RNG・跨shot相関や実回路作用の誤差は未評価。

## 全energy RMSEへの接続条件

**局所signal biasから最終energy biasはまだ得られない。** decoderは全roundの位相・branchを逐次使用し、稀なbranch誤りも二乗lossへ効く。局所phase/Tや局所誤差の単純和をenergy RMSEへ代入しない。Bにはraw mean・nominal各軸・全prior/decoder条件を渡し、有限shotを含む全推定量のriskを評価してもらう。

**Minkowskiは選んだL2推定誤差の分解に適用する。** 例えば、同じ確率空間上で定義した実近似推定量Z、基準推定量Zref、固定target Eについて、`||Z−E||₂ ≤ ||Z−Zref||₂ + ||Zref−E||₂`。この不等式自体に成分独立性は不要だが、各項を有限K/PF・QPE予算と同定するには、比較モデル、coupling、target、未解決trialのlossを固定し、各L2量の根拠が要る。別分布の推定量を比較する場合はcouplingを明示する。保存signal差だけではεtrotの達成を認定できない。二乗和への変更もしない。

**共通の有効Hamiltonianは未確定。** exact-tail S2は固定Δ・同じ順序・scalarなので、SOURCE上は一つのunitary stepのqPF乗として表せる。そのunitaryにHermitian logarithmを選べても、energy branchの一意性、target基底分枝、保存stateの固有状態性は別証拠が必要。一方、finite-K corrected stepは一般に非unitaryで、R_outerがroundごとに変わるためtailのshort timeも変わる。13個のscalar meanだけから共通の時間に依存しないHermitian生成子や共通固有位相を認定しない。これは共通モデルの不存在を証明した結論でもない。

**共通有効Hの証明は、固定13 raw meanに対する条件付き統計モデルの実験とは別。** ideal-law・fresh IID・保存数値stateを仮定し、target/lossを明示すれば、その特定decoderのriskを直接評価する道がある。ただし、それを原論文の単一位相モデルの保証や、実sampler・厳密groundへの保証と呼ばない。

**保存stateの証拠水準。** 89c91c6:records[5].diagnosticsの保存solver energy `−2.1663874486347634 Ha`、residual `1.9513566490707353e−15 Ha`、`ground_state_certified=false`を同じbindingで転記した。小さいresidualは近傍の固有値についての数値証拠であり、最低固有値であることの証明にはsector内の順序・gap/包含区間等の追加証拠が要る。今回はground truth proxyとしての新規採用判断もしていない。full-H referenceは保存stateのスペクトル混合を既に含み得る。

**ideal lawと実装law。** raw meanとIID Bernoulliの接続はideal周辺平均のモデル。実samplerのlaw差・跨shot依存、hardware/control誤差が不明なら、その値を0にしない。meanが一致しても独立性がなければBinomial lawとは限らない。認証されたlaw距離や相関の条件と有界lossが得られた場合に初めてrisk摂動を別枠で評価できる。ideal decoder simulationの成功だけでは実装誤差を資格付けしない。

## 技術実装と確認範囲

新規は`h4_pr_physical_inputs.py`と転記CLIだけ。既存SavedArtifacts、MEASUREMENT、B canonical validatorを再利用し、N solver・allowance・geometry・normalizer・decoderを呼び出さない。B validatorは既存af284d8固定helperを通すが、実使用する`_id/_positive/comparison_method/_coverage/_assemble/_validate`のASTが指定B3216と同一であることをテストした。更新されたassess/selectは呼ばない。

新8テストは元値との完全一致、13 identity/law/binding、欠測・SOURCE/候補/policy混入拒否、m3/m4条件付き証拠、未資格null保全、非有限値拒否、科学leaf sentinel、受入validator同一性、上書き拒否を検証。既存completionの13人工・metadata回帰もPASS。技術テストを科学的RMSEの証拠とはしない。

次の最小科学検証は、Bによるこの13 raw mean・既存nominal Nでの**ideal-law全decoder統計**。新A mean/reference/normalizationはそのためには不要。loss/target・数値不確かさの扱い・統計精度をユーザーが判断し、Bの固定SOURCEと有限実行規模を別途承認する必要がある。詳細はSCIENCE_PLAN参照。今回commit/push・remote SHA/tree/主要成果物照合後、mandatory STOP。
