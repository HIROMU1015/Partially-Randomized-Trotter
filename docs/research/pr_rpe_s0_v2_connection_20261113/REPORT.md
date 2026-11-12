# H4 DF10 S0のB V2接続と全13round RMSE検証計画

2026-11-13 JST。**S0の二時点nominal結果を、Bの固定V2生成・packet変換関数で新canonical configurationへ接続した。** 元S0は不変。m0/m6のnominal Nは各軸64/38のまま、他11roundはNOT_EVALUATED、全十分N・global RMSE boundはnull。今回の実分子signal/normalization/N/RMSE再解析・simulation・C/Gはすべて0。

成果：[V2 configuration](metadata01/V2_CONFIGURATION.json)／[V2 N packet](metadata01/V2_N_HANDOFF.json)／[全round不足証拠](metadata01/COVERAGE.json)／[B本体による26軸cell計画](metadata01/B_EVIDENCE_PLAN.json)／[SOURCE索引](metadata01/SOURCE_INDEX.json)／[HANDOFF](HANDOFF.md)／[技術検証](TESTS.txt)。

## 固定条件とSOURCE

Linear H4・隣接1.00Å・STO-3G・保存tol-only DF10・sector(2,2)/dimension36・保存数値state、S2/L_D1/ΔPF=.1/K2/h=.1/M12、P=0。保存Hamiltonian/completion/fragment順/scalar/H-state-snapshot binding、λ_R、prior/decoder、law/compiler/control/full-wrapper/profileを保持する。targetは保存DF-Hの同sector ground energyで、厳密ground同定は未資格。H4の保存PF B0/prefix7をFull-H PFへ読み替えない。今回のpacketはPRだけである。

κはS0 `CONFIGURATION.json:inputs.kappa_paper` の**exact有理数文字列**をそのままBへ渡した。表示値≈24.305274577756407から再構成していない。εE=.00159360010199040 Ha、εqpe=εtrot=.00079680005099520 Ha、minkowski、finite_B_diagnosticもS0のまま。旧α/β・Hoeffding Nを混ぜない。

| 役割 | 固定SOURCEと再利用 |
|---|---|
| S0公開 | `7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75`、旧CONFIGURATION/N_HANDOFF/resultを読取専用で参照 |
| S0科学実行 | `7c3e1e5db0c55dd8249e9030a4c7e9954496fa7e`、runnerと消費SOURCEは変更0・再実行0 |
| B V2 | `af284d853890155bacd19e61850c196bd1d5c660`、controller/integrationはV2基点`6a9d7fd36b687e58aecbed1278c6d7927d6827d6`と同じ。evidence plannerを含む固定関数を再利用 |
| 旧B接続 | `f09af82482aad11460c4bb73d3e13d7ada40be6f`の既存A adapterで、元S0の旧packetを再現・照合。数値leafなし |
| 元P1 | `71ba49bb28b295aa28e51101bc3b1bf07d500767`の保存record・law SOURCE、既存A field inventory/preflight |
| 追加保存mean | `89c91c692e1d35de94d2e7afbe890aabe6d80b7e`（m1）、`155529e20a8fac39e00786c40b57830b71203c4f`（m3/m4） |

B担当branch `codex/r-variable-rpe-cost-pipeline-20261111` のremoteはaf284d853…と一致。別系列のFull PF/Full-H RTE開発をPR S0へ混ぜず、B worktree・controller本体を変更していない。

## V1からV2への接続契約

[最小A adapter](../../../src/trottertracks/pr_rpe_rmse_s0_v2.py)は、既存A `pr_rpe_rmse_pilot`/preflightのGit artifact読取り・field存在判定・SOURCE blob照合を再利用する。B af284d853の`a_spec/generate_from_a_spec/n_from_a`、canonical validation、`pr_rpe_rmse_evidence.plan`を**変更せず**ASTで限定読込みし、価格provider/cost moduleは読まない。Aにidentity hashやplannerを再実装していない。

1. 入力が固定S0公開の元configuration/packetと同じであること、旧B validator、A全schedule、旧B packet再現を照合。
2. BのV2 `a_spec`から新configurationを生成。V1 PRの`A_SOURCE_BOUND_PF_BASELINE`だったsplitの表示は、V2のPR `PREFIX`へ正規化されるため、schema差だけでもない。物理条件は変更しない。
3. 全13roundのphysical candidate、structure/allocation/event、T/qRPE/qPF/δ/R、law/κ/event mapping/SOURCEを旧新で照合し、B `n_from_a`に元A packetを渡す。
4. 旧configuration/candidate ID、新canonical ID、元physical ID、元packet/SOURCEの対応を`connection_provenance`へ保存。元IDを新IDとして再ラベルしない。

旧configuration ID：`6cc89360a60b6f5b6554d044193a28f4c92cafa0208e77efb4d1f2ce9f80181c`。

新V2 configuration ID：**`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`**。

これは固定S0条件を**B自身の公開生成関数で具体化した接続configuration**であり、Bが新しい科学条件・十分Nを採択したとの意味はない。B担当からconfigurationを受け取る場合は`connect(..., canonical_v2=...)`でこれとの完全一致を要求する。

[metadata専用CLI](../../../scripts/connect_pr_rpe_rmse_s0_v2.py)はfresh出力先のみ。科学実行flagやfallbackはない。B plannerは全26cellのN record存在と十分N欠測/未資格を区別し、全体はNOT_EVALUATED、G/winner=null。`qualified_expectation`は価格を一件も渡さないplanner API引数で、新価格policyやC評価の採択ではない。

## 全13round：保存証拠の存在と新評価の欠測

以下は固定CONFIGURATIONのmetadataと保存fieldの存在だけを転記した。新しいnormalization・位相・N計算はない。全roundはS2/prefix1/K2/δ=.1、qRPE=qPFだが意味は異なる（qRPE=2^m、qPF=T/δ）。

| m | T | qPF | R_outer | physical ID | 一致mean / 独立reference pair | 今回のN/RMSE状態 |
|---|---:|---:|---:|---|---|---|
| 0 | .1 | 1 | 2 | `53e00d8b07d1547d19a7` | P1 / 保存あり | S0 nominal64/軸をcopy |
| 1 | .2 | 2 | 4 | `c00e00dc1e9c414fb7d9` | H4 batch / 保存あり | 未評価 |
| 2 | .4 | 4 | 8 | `f2fcd66b1ef9ae2da5df` | 欠測 / 保存あり | 未評価 |
| 3 | .8 | 8 | 16 | `045843b2429bece6f29b` | 旧raw / pair欠測 | 未評価 |
| 4 | 1.6 | 16 | 32 | `3df0cfd3356c358aeda8` | 旧raw / pair欠測 | 未評価 |
| 5 | 3.2 | 32 | 64 | `57d641a831d76360b570` | 欠測 / 保存あり | 未評価 |
| 6 | 6.4 | 64 | 128 | `2a8c6718220eec59d419` | P1 / 保存あり | S0 nominal38/軸をcopy |
| 7 | 12.8 | 128 | 255 | `b42287268e5249dfb4d1` | 欠測 / 保存あり | 未評価 |
| 8 | 25.6 | 256 | 510 | `224aef0a626068b43184` | 欠測 / 保存あり | 未評価 |
| 9 | 51.2 | 512 | 1020 | `06a78c7f9856f373336f` | 欠測 / 保存あり | 未評価 |
| 10 | 102.4 | 1024 | 2040 | `b872ad1347a35c8f4315` | 欠測 / 保存あり | 未評価 |
| 11 | 204.8 | 2048 | 4080 | `d48210cc27c599e268c2` | 欠測 / 保存あり | 未評価 |
| 12 | 409.6 | 4096 | 8160 | `b55c521fdce6a448dbcb` | 欠測 / 保存あり | 未評価 |

m7以降の255/510/…はexactκと既存mapperによる保存scheduleの値。近い256/512/…や旧固定R4のmeanを代用しない。調査は既存inventory/P1/H4 batchと旧raw二archiveに限定した。この範囲外の保存値が新たに見つかれば、同一candidate/H/state/profile/law/SOURCEを確認して取得数を削減できる。

**m1**はH4 batch `run01/result.json:records[5]`。reference/corrected/raw/exact_tail/bias/logB、closure/norm/最大中間norm、二経路referenceのfieldが揃う。旧`CONDITIONAL_ROUND_PASS`は旧高確率policyであり、新RMSE PASSではない。保存allowanceにR-aware policyタグがないため、そのままS0 runnerへ渡せない。将来の保存再解析で既存R-aware式への接続が必要。

**m3/m4**は旧raw `track_a_h46_long_time/2026-10-25/signal_records.json.gz`のID selector。完全なH/state/snapshot binding・compilerとcandidateを照合し、phase_alignment=falseを確認。元finite-RTE `rte.py`はP1と同じblob。保存logB/独立corrected・raw/対照/bias/最大中間norm/closureはあるが、**observed exact_tail_norm_error scalarは保存されていない**。0で補わない。

その元科学SOURCE`e417ffc9f52b13c18f5493193ce510d858664259:scripts/h46_long_time_signal.py:evolve_batch`にはexact-tail norm≤1e−9 gateがあり、成功recordはSOURCE gateの証拠となる。ただし観測scalarの回復や今回のgate再確認ではない。元backendはspectral polynomial powerによるblock適用で、P1のevent逐次適用とはroundoff/中間norm観測箇所が異なる。R-aware allowanceへのbridgeの可否を先に検討し、欠測fieldだけを理由にmean再計算を必須としない。T=.8/1.6は単一referenceがあるが、この調査範囲でH4の独立pairは未発見。H6のT1.6 pairを移植しない。

[COVERAGE](metadata01/COVERAGE.json)はこれらのfield欠測・SOURCE gate・出典・元statusを機械可読で保持する。SOURCE不一致、保存FAIL/UNRESOLVEDは再利用停止理由として保持し、科学retryを起動しない。

## 科学的に十分なNへ進むための不足

原論文v2 Appendix B/Eの`ceil(B_m²[11+4(M−m)])`は経験的shotモデルである。[原論文](https://arxiv.org/pdf/2503.05647v2)のRTE信号モデルと有効Trotter Hamiltonianの前提を、現在のfinite-Kへ無条件に移せない。既存A評価器はこの区別を保持している。

| 論点 | SOURCEで確認できる部分 / 未解決の科学的証拠 |
|---|---|
| finite-K2 | 原paper RTEとのevent単位対応は既存A仕様。有限Kの系統的mean biasは別途全roundで診断・boundが必要。normalizationだけでは位相精度を決めない |
| 保存state/target | 保存DF-Hの数値state・残差という証拠水準を維持。有効PF固有stateや厳密ground/overlap保証、元分子Hの精度保証へ昇格しない |
| Minkowski | 予算の線形和が目標内でも、各成分が同じtarget/確率空間に対する有効なL2誤差上界である証拠が必要。局所disk/βdetをεtrotとしない |
| prior/alias/branch | h・無shift・固定priorとdecoderのbranch規則を保全。E_stateをbranch選択に用いない。有限shotの誤branch・端点・未解決事象を含める |
| 全round統計 | 二点diagnosticは全13roundの統計的energy RMSEではない。既存経験的Cmaxも今回のdecoder/state/lawの検証に代わらない |
| sampling law/実装 | ideal finite meanと、実際の独立sampling・実装誤差を分離。unknown actual/law誤差を0にしない |
| 数値精度 | P1と同じempirical u/margin・reference≤1e−10、norm/closure≤1e−9等のSOURCE gateを保持。実誤差のrigorous上界とは呼ばない。長eventでallowanceや丸めが未解決ならSTOP |

## 次の限定科学検証案（今回はすべて未実行・未承認）

### 段階A：保存一致候補の局所診断

**最小の次承認案はm1/T=.2/qPF2/R4のPR一候補だけ。** 元H4 batchと固定S0 exact scheduleを使い、保存pair/gate/closure/biasを再確認し、R-aware allowance式をSOURCE付きで接続、既存S0 scalar recipeとA RMSE policyでphase/radius・nominal N_X/Yを評価する。新mean・新対照・reference action・finite normalizer・H/state array loadはいずれも0の保存再解析に限定できる。m0/m6は再実行せずcopyする。

`pr_r_dependent_n.analyze`全体は旧Hoeffding solverを含むため起動しない。将来の最小実装はそのallowance式だけを分離し、既存`checked_reference`、S0 `numeric_diagnostics`のscalar gate、`supplied_disk_geometry`、`A.mean_evidence/evaluate`のRMSE経路を接続する。固定m0/m6専用S0 runnerを無理に再利用・変更しない。今回その科学runnerは追加していない。

S0 `numeric_diagnostics`内のgeometry出典は元P1専用である。m1用adapterでは元H4 batch 89c91c692…の実locatorを渡す形でgate/recipeを再利用し、m1の診断をP1由来として記録しない。診断SOURCEの差を科学値の違いと取り違えず、元S0コードを保全する。

m3/m4は追加の保存一致候補だが、SOURCE norm gate/legacy backendのallowance bridgeとpair不足が残るため、m1と同じready状態とは扱わない。bridgeを技術的に具体化した後に、その条件を含む別の保存再解析scopeとして提案する。

### 段階B：残りのmean/reference補完

固定scheduleの欠測meanは**m2,m5,m7–12の8 PR候補**。既存action-only finite meanを使い、各候補でcorrected/rawを独立伝播し、exact-tail対照、normalization精度・scalar一回会計を同時取得する。必要量は8候補分のfinite meanと8対照（backend内で共有可能で、16独立jobの指定ではない）。既存同H/state・同Tのreference pairを8時点とも再利用し、不必要な新referenceを発生させない。

m3/m4の不足pairにはT=.8/1.6の各2経路、**合計4 action**（`expm_multiply`と`expm`）を提案する。保存meanのSOURCE gate/数値allowance bridgeが不十分と判明した場合に限り、**別途明示scopeに含めて**同じm3/m4のmean/対照を各1候補取得する案が残る。0又は2候補という欠測由来の条件分岐であり、FAILした候補のretryや任意の上限ではない。

長時間のm12は固定metadataで33,423,360 events。逐次event backendの計算量を前提に、既存spectral polynomial power/block経路の再利用・候補間のhead共有を検討する。高速経路はlaw/積順/負時間規則/normalization/scalar/数値allowanceの同等性を人工テストと承認された必要最小限の科学照合で確認してから採用する。CPU/RAM/workerの独自上限は作らず、既存環境内で並列化する。新SCF/DF fit/ground-state生成・候補変更は不要。

新mean取得と保存scalar診断は同じ固定scope内でまとめられるが、具体的runner SOURCE/対象/精度を固定してから科学実行を承認してもらう。失敗・入力不一致・不安定丸め・raw/corrected closure破綻・必要証拠不明時はSTOPし、R/κ/M/閾値の変更や科学retryへ進まない。

### 段階C：全RPE RMSE

全roundのraw mean・normalization・局所bias・数値allowanceが揃って初めて、同じ13roundのnominal配分とdecoderを検証できる。保存再解析/新meanだけで、十分Nを認定しない。

- 条件付きideal finite lawの統計pilotなら、各軸のraw meanに対応する±1独立試行又は等価binomial countを用い、**全13round**を固定decoderへ入力する。trajectory samplingとは別の新decoder統計simulationとして承認が必要。今回は実施0。
- `decode_supplied_coordinates`はzero X/Yやbranch tieを未解決（energy=null）とする。これらを落とした成功試行条件付きRMSEは全体RMSEではない。未解決試行のloss/失敗会計と出力範囲、prior/aliasを扱う契約をユーザー判断事項として返す。decoderのclippingやbranch規則を無断追加しない。
- 比較targetは保存DF-H。E_stateをtruth proxyにする場合も、保存数値stateに対する条件付き評価と明記し、ground証明・actual implementation誤差は別に未資格を保持する。
- 統計的推定と厳密/検証付きboundのどちらを目指すかを先に決める。Monte Carlo案の試行数はSOURCEとloss上限・RMSE/MSE解像度・信頼水準から決定し、原paperの反復数や恣意的固定capを移植しない。有限標本の誤差・未解決確率を明記する。旧αtotal=.05を新検証信頼度へ流用しない。
- εqpe/εtrot分離は、有限Kの系統bias、stateと有効発展、全roundの推定biasを含む定義と根拠が必要。統計simulationの点推定だけでMinkowski成分保証や科学的に十分なNを供給しない。必要な数学的成立判断は未確定のまま返す。

## 技術検証と引渡し・STOP

新10件（保存fixture/人工metadata改変）＋既存37件（S0 7・pilot13・RMSE17）の**計47テストPASS**。V1/V2由来、完全物理一致、exactκ/全schedule、law/SOURCE混入拒否、64/38 copy、残11null、十分N/bound null、保存FAIL/欠測scalar保持、scientific leaf未呼出し、元S0/science blob不変を確認した。作者自身の技術検証で、独立科学レビューではない。

**Codex B**はV2 configuration/N packetを固定IDで参照し、未資格のnominal診断として表示する。**ユーザー**は次に段階Aのm1保存再解析だけを承認するか決める。**Codex A**は承認されたSOURCE/有限scopeだけを実行する。段階B/Cは追加承認が必要。今回commit/push・remote SHA/主要blob照合後**mandatory STOP**。新science・C/G・最適化・詳細科学レビューへ進まない。
