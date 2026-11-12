# PR–RPE 二時点RMSE科学pilotの準備（A、2026-11-13 JST）

**技術準備完了、科学実行なし。** B f09最終版のA専用経路を再利用し、1系・固定(M,κ)のPR二時点について、保存証拠・欠測・追加call案を特定する入口と、将来の供給済A結果をBへ渡す処理を追加した。実Hamiltonianのsignal/N/RMSE、保存値の新RMSE再解析、normalization再計数、P2/P3、sampling、C/G、κ/M探索は実施していない。M・κ・RMSE予算は未選択であり、実候補configurationは生成していない。

基点はA e2a472422f0206a2f09fcc8e28d6938ebceb6d0f（preflight）、A 510fbacd01a2d855202de4c045d8ba1972e5ff3d（RMSE model）、B f09af82482aad11460c4bb73d3e13d7ada40be6f。既存A branchを継続し、B/C worktreeを変更していない。旧A実装・旧preflight・P1/科学SOURCE・raw/結果/STOPは不変。

## 候補・λ・保存SOURCEの一致条件

対象候補は保存Linear H4/H6、隣接1.00Å、STO-3G、tol-only 1e−8 DF10/DF19、保存数値state。targetは同sector保存DF-H groundであり、元分子HのDF近似誤差まで保証しない。保存stateは数学的な厳密groundとして認定されていない。 H4 sector(2,2)/dimension36、H6 sector(3,3)/dimension400、保存weighted-Hermitian completion・fragment順・signed weights・scalar・H/state/snapshot bindingを保持する。energy shiftなし・P=0の元条件も、外側設計として明示的に保持する候補である。

[P1_MATCH_METADATA.json](P1_MATCH_METADATA.json)はP1公開71ba49bb28b295aa28e51101bc3b1bf07d500767から候補/SOURCEだけを選択した資料。signal/B/Nは含めず再計算していない。H/state/snapshotの完全hashも同ファイルに保持する。

| 候補系 | 固定候補の参考条件 | 保存P1のRとphysical ID |
|---|---|---|
| H4 DF10 m0、T=.1、qPF1 | S2、δ=.1、PR L_D1/K2、h=.1 | 2: `53e00d8b07d1547d19a7`、4: `03a51c8d13a91a403742`、8: `656998c3354fb8b614e9` |
| H4 DF10 m6、T=6.4、qPF64 | 同上 | 4: `65c51c5cb3e39a2266e3`、128: `2a8c6718220eec59d419`、256: `ae0187ce74cd82207748` |
| H6 DF19 m4、T=1.6、qPF1 | 4th m5_best、δ=1.6、PR L_D7/K2、h=.1 | 11: `5e2581e9c4f25bdf8529`、16: `2d2af6ca830dff14a695`、32: `4269eafc8c0ef8922c23` |
| H6 DF19 m8、T=25.6、qPF16 | 同上、11 signed piece | 16: `53797d9e5a2eeec31a13`、128: `55a8eb0ae50ec3f7c713`、256: `00ab807ee1f600b30440` |

これらは再利用候補であり、今回の正式採択ではない。1系だけを選び、H4はM≥6、H6はM≥8として、Mを**最大round index**と明示する（全0..M、M+1 round）。13roundへ自動設定しない。

λ_Rは`matrix_only_tail_coefficients`が定義する、保存DF tailのfull-Fock I/Z/ZZ展開の**非identity係数の絶対値和**。identity scalarは分離する。sector operator normやDF重みの単純和で代用しない。保存値の所在は以下の通りで、今回この関数を呼んでいない。

- 索引：7a8bc7f132e166390c943e6b32c92cc6abe1baf4、`artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json`。
- H4：`T20_reference_files.H4`、context hash `8396605fa0c5ee34f29b436ee16e0f731cfaa6cd94f3304033106a85e9f5d52f`、`prefixes.1.lambda_R`。
- H6 regular：`T20_reference_files.H6`、context hash `61d87e2fe7e03892f7db6a075a891d9b83af834832b083ec3de3b75e8a7e9d38`、`prefixes.7.lambda_R`。初期prefix5へ流用しない。
- 値・binding・prefix・SOURCEのjoinは固定科学入力の条件。今回の公開metadataにはλの新評価値も、そこから選んだκもない。

κの定義はA/B共通のpaper rotation型 `r*=κ_paper λ_R² T_m²`。既存mapperは `r_target=ceil(r*)`、`R_outer=max(J,ceil(r_target/qPF))`、event数=qPF R_outer。JはH4で1、H6 regularで11。R>Jの保存候補と一致する条件は

`qPF(R−1) < κ_paper λ_R² T_m² ≤ qPF R`。

R=Jなら `0 < κ_paper λ_R² T_m² ≤ qPF J`。二時点の条件を**同じκ・同じregular λ_R**で同時に満たし、元allocation関数のpiece別整数配分まで一致する必要がある。この記号的条件だけを整理し、実λで区間を計算・探索・最適化していない。T/δは既存binary64のexact値から扱い、丸め境界をdecimal表示へsnapしない。q_RPE=2^mとqPFを混同しない。

保存meanの利用条件は、full physical candidate・H/state/snapshot/binding・profile・signed order/allocation・law/SOURCEが一致すること。新しい全schedule/M/κではB IDが変わるため、旧N/旧B IDを付け替えない。同一H/state/Tの保存referenceは、Rが異なっても再利用候補になり得る。上記4時点にはP1の独立reference pair fieldが揃うが、数値gateは未再確認。別Rのmeanは流用しない。

## RMSE条件：理論、仮定、未資格

[原論文v2](https://arxiv.org/pdf/2503.05647v2) Appendix B.2、E.1/E.3を確認した。E1の誤差合成、経験的Nmとκの式を参照するが、現在のfinite-Kに対する厳密十分Nとして移植しない。

| 論点 | 限定pilotで扱えるもの | 残る科学的条件 |
|---|---|---|
| 誤差予算 | 指定されたεqpe²+εtrot²≤εE²と経験的horizonのモデル診断 | 有効energyに対するunbiased統計推定等、二乗和の適用仮定。予算の算術成立だけで保証しない |
| 経験的shot | 各軸 `ceil(B_m²[11+4(M−m)])` のnominal値、X/Y合計2倍 | 論文の経験的校正を有限K/保存stateへ移送する証拠。actual minimum/十分Nとは別 |
| finite-K bias | corrected/raw、full-H reference、exact-tail、数値allowanceによる局所診断 | 原論文のexact unbiased RTEと有限K2は異なる。Bでvarianceを補っても有限K biasやPF biasは消えない |
| state/ground | 保存state条件付きのsignal評価。保存residualとSOURCEを保持 | 厳密ground/overlap、有効PF固有状態との関係、multiphaseや実装/law誤差。未知を0にしない |
| H6 negative time | 元11 pieceの負時間・event allocation・signed propagation、whole B/logB | finite-Kのnorm/位相と、論文の単純exp(1/κ) envelopeの対応。κをABS係数で無断再定義しない |
| 初期H6 | m0..3 p5/R32/q1/δ=T、κ_applied=falseを全scheduleに保持 | regular p7のλやκ envelopeを初期roundへ推論しない。二時点m4/m8は初期4roundを検証しない |
| prior/decoder | 固定prior/h、原decoderと−ET符号を再利用 | alias-free、branch tie/外れ値、全roundの統計RMSE。E_stateでbranchを選ばない |

原論文のrはabstract回転数であり、有限K eventやcompiled RZと無条件に同一視しない。H6では `log B = qPF Σ_j r_j log b_K(λ_R δ w_j/r_j)` という元有限normalizerの会計と、負時間/整数allocationを維持する。Bはwhole interrogationの値で、piece数やX/Y軸数を再度Bへ掛けない。数値allowance/logB margin・小τ精度問題は元gateを保持し、normalizerを便宜的に1へ丸めて成立扱いにしない。

Aの`finite_B_diagnostic`は実finite Bによる経験的plug-in、`paper_kappa_envelope`は別policy条件。どちらを使うか、予算と適用仮定はユーザーが固定する。旧αtotal=.05/β配分や保存Hoeffding NはRMSE policyに継承しない。

二時点で判断できるのは、候補/数値gate・局所bias/phase/radius・normalizationと経験的N計算の利用可否まで。phase/Tの局所値を全energy bias/RMSE boundと呼ばない。全RMSEには全0..Mのmean/normalization、state/target・finite-K移送・prior/alias/decoder・実lawと統計評価の証拠が別途必要。sampling等の検証方式も今回選択・実行していない。

## 実装と実行経路

新module `pr_rpe_rmse_pilot.py`はBの`_assemble/_validate`とintegrationの`a_spec/generate_from_a_spec/n_from_a`等の**固定関数ASTを無変更で利用**する。C class/価格・B cost/select/controller本体は追加・変更しない。旧A preflightはbyte不変。既存CLIへ`--two-time-pilot`を追加し、新scientific runnerは作らなかった。

- `pilot_inventory(root)`：前回field inventoryとP1のlaw_bindingをjoin。P1 SOURCE_INDEXの既存RTE実装hashを照合し、wrapper宣言と実装SOURCEを区別する。actual sampler資格は付与しない。law不明な旧descriptorは保留し、同じ候補の確認済P1 descriptorがあればそれを再利用候補とする。
- `prepare_pilot(B_config,A_schedule,inventory,root=...)`：全schedule/M/κ/元A入力をB固定経路で照合し、1系のPR二時点だけを選択。欠測・SOURCE/law不一致・保存FAIL/数値未解決と追加call案を出す。全round N packetはNOT_EVALUATED/PREFLIGHT_ONLY、nominal/十分N/boundはnull。
- `supplied_disk_geometry`：既存c1 geometryのρ=A−δ、βdet=asin(δ/A)だけを旧Hoeffding Nから分離。明示されたraw単位diskとSOURCEが必要で、未知はnull。局所診断でありRMSE資格やPASSを返さない。旧c1 SOURCEは不変。
- `handoff_supplied_results`：既に評価されたAの二時点recordだけを受け取り、残りをNOT_EVALUATEDに保ち、A b_handoff→B n_from_aへ渡す。旧HP N・十分Nへの昇格・対象外round/PF・別M/κ/SOURCEを拒否する。部分結果も欠測を維持し、数値leafは呼ばない。

B f09では、H4 B0/p7を`A_SOURCE_BOUND_PF_BASELINE`として保持し、`UNQUALIFIED_A_PF_BASELINE_SCOPE`を残す。Full-H PFへrelabelしていない。H6初期p5/R32もphysical rowとlawへ保持され、前回82bcでの技術的表現制約は解消した。どちらも科学的資格の解決を意味しない。PR二時点pilotへPF/純RTEの科学評価は含めない。

## 承認後の有限実行案（未承認・未採択）

[PILOT_INPUT_TEMPLATE.json](PILOT_INPUT_TEMPLATE.json)の選択欄はnull。ユーザーが**1系・M・共通κ、既存外側条件の保持または明示変更、RMSE target/εqpe/εtrot・policyと数値allowance SOURCE**を固定する必要がある。固定時はこの公開成果のcommitと、モデル510/P1科学SOURCE a1a79213a9b5331145e20e8778a3be9d9fc4e497、保存P1/入力hashをSOURCEとして指定する。旧grantや消費済runは再使用せず、今回承認を証明する新しいgrant等も要求しない。

1. 技術preflightでB/A全schedule・二時点candidate・λ value/SOURCEとH/stateを照合し、保存mean/referenceの完全一致を確認する。κ/M未指定のまま進めない。
2. **保存一致時**：対象PR二候補のreference pair gate、closure/exact-tail norm、元R-aware allowance/logB marginを再確認し、局所bias/phase/radius、経験的nominal N_X/Yを再解析する。これも新しい科学実行承認の対象。新mean/referenceは0にできる可能性がある。
3. **欠測時だけ**：不足finite meanは最大2候補、その内部exact-tail対照は最大2。reference pairが未取得の時点だけ2経路（最大4 action）。先に保存診断回収を検討し、SOURCE不一致・保存FAILでは自動fresh計算/retryしない。承認時には上限案でなく実際に許可する対象/call数を固定する。
4. actionには既存`existing_finite_mean`→`h6_df19_rpe_d2.evolve/half_step`、H4 B0だけは`evolve_h4`、referenceは`checked_reference`/必要時`reference_pair`、normalizationは既存rteを使用する。独立raw/correctedを保ち、raw=corrected/Bを事後定義しない。新SCF/DF/eigensolve/state生成なし。
5. A.evaluateへ供給するmeanは承認二時点だけに限定し、他roundは空で返す。旧`analyze/assess_supplied_round` wrapper全体を呼んでHP Nを新RMSE値にしない。元条件のdiskを用いたgeometry・decoderは局所診断に限定し、離れた二時点を連続round decoderへ渡さない。全decoder samplingは含めない。
6. 取得済A recordをhandoff関数へ渡す。全roundの十分N/global boundはnullのまま、未評価roundを残し、保存/数値FAILを報告してSTOP。C/G・R探索・P2/P3へ進まない。

13新規＋旧preflight14＋旧RMSE17の**44人工/mock/保存schemaテストPASS**。[TESTS.txt](TESTS.txt)と[ARTIFICIAL_SUMMARY.json](ARTIFICIAL_SUMMARY.json)に記録。作者の技術確認で、独立科学レビューではない。**ユーザー**が上記条件と有限科学scopeを決定・承認した後だけ、**Codex A**が限定pilotを実行する。**Codex B**には新HANDOFFを渡し、統合/価格はB/Cの責任範囲とする。今回は公開SHA/主要成果物照合後mandatory STOPする。
