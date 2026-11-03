# 研究C：近接weighted L1比較・CDF/XDF共通ideal評価のEvidence Map追補

2026-11-03 JST。固定公開成果の証拠統合。**新科学計算・保存rawの数値再解析・重要科学レビューは0件。**

共通有限政策の16更新endpointを固定H4・r32で下流評価した結果、PR-aware/ENTRYの条件付き総RZ点比は **0.494181551983**、PR側が点値で約50.58%低費用だった。登録近似Welch95%差区間も `NOMINAL_MODEL_PR_LOWER`。新PRの点値は保存XDF8/CDF8の共通ideal費用も下回るが、**PR対XDF/CDFの直接的な統計的優位・同等性・非劣性は未認定**である。

旧[11月3日Evidence Map](research_c_evidence_map_updated_20261103.md)のC01〜C20/N01〜N10/G01/Q01、[旧24 Claim対応](research_c_claim_contribution_alignment_20261102.md)、[貢献候補・論文構成案](research_c_contribution_positioning_20261102.md)の本文・SOURCE・歴史的判定は無変更で保持する。本追補は **C21〜C29/Q02** を追加する。中心claim・独立新規性・論文化GO/NO-GOを採択しない。

## 1. 固定系列・SOURCE・Git依存関係

F16/R32/I61は本追補だけの系列記号で、旧MapのA〜Eや貢献候補A/B/Cとは別。下表の科学SOURCEと結果収録commitを区別する。

| 系列 | 評価・候補 | 科学SOURCE | 結果収録commit | 固定公開証拠 |
|---|---|---|---|---|
| F16 | PR-aware対tail限定weighted entrywise L1二乗・比較II、各16更新 | `70a98adfb701a209db2c65ea60fb789ff62b1fe7` | `2ceb0a8825785ebd4f21241d8ef0f20cb321bf69` | [報告][F-report]・[JSON][F-result]・[SOURCE][F-source]・[契約][F-contract]・[STOP][F-stop] |
| R32 | F16の二endpointを共通r32でPF/RTE・RPE・native32・同J/U強Dへ接続 | `4df8eca29c59ffd55e2dd898f05da16bb246ddbe` | `beb8de5cd8b71f80bb69556f92299435ff431c3e` | [報告][R-report]・[JSON][R-result]・[SOURCE][R-source]・[契約][R-contract]・[STOP][R-stop] |
| I61 | 保存B2 CDF8/B3 XDF8の反実的共通ideal評価 | `2d20ad196a406752de8e4c0ce9516b101e49d927` | `39ee9705b7a459c40c205c542d660b5d2ab90d33` | [報告][I-report]・[JSON][I-result]・[SOURCE][I-source]・[契約][I-contract]・[STOP][I-stop] |

各結果commitの直接の親は表の科学SOURCE。R32はF16結果を起点に準備され、I61は旧Map→[固定公平比較監査][AUDIT]→ideal adapter準備という別系列である。F16とI61を相互の科学SOURCEとして扱わない。依存関係・tree/blobの静的照合は[検証記録](../../artifacts/research_c_near_xdf_evidence_integration/2026-11-03/static_verification.json)に記録する。

統合起点は旧Mapを収録する `ba0f280c8be9e7038c5fa6b5fd390a235b113833`。独立branch `codex/research-c-near-objective-xdf-evidence-integration-20261103` へ文書・索引・静的検証だけを追加し、実験branchのmergeやraw/QPYの複製は行わない。別branchの原本は絶対commit付きURLで参照する。結果収録commitを科学実行SOURCEへ付け替えない。

## 2. 比較に必要な共通条件・候補固有条件

| 条件コード | 固定条件・適用範囲 |
|---|---|
| H4 | Linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、sector36/full256/controlled512、rank8。元g補正κ、nuclear scalar一回、元spin/frame順と相対位相の所有規約を保持 |
| F16 | 通常number CDF8、一般signed実対称J・実直交U、元順LD2。共通tensor-only初期J/U、自己初期mass正規化で初期penalty寄与を一致、γ/β=.1の一設定、PR ε=1e-8・ENTRYの登録η、各16受理U更新・共通有限trial/inner政策。時間・rは最適化自体に非適用 |
| PR32 | 指定F16 endpoint、元順LD2、partial ordered symmetric S2、s1/r32/K0、δ=h=.1、m0..12 |
| D1 | 指定候補のκ＋全8frame ordered symmetric S2、s1、δ=h=.1、m0..12。PR split・r/Kは非適用。R32強Dは同じ候補のJ/U、I61は別の保存DF表現 |
| RPE | CA=.00159 Ha、DF reserve=CA/8=.00019875 Ha、βtotal=π/6、βstat=π/12、各候補failure=.01・軸=.01/26、各DF自身のground-state oracle、状態準備費用除外 |
| P0 | Qiskit1.3.0/opt1/seed17、rz/sx/x/cx、approximation_degree1、qubits_initially_zero=false、保存MPFR160 declared phase、位相fitなし。合法Gaussian/relative fusion、head/X/Y shell込みfenced serial・scheduled serial depth |

| Candidate | Identity・出典 |
|---|---|
| F16-PR | endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a` |
| F16-ENTRY | endpoint SHA256 `345dd11a14a281bf716f42009565b2d1875fdcf9845b928a15c102164208cb3f` |
| 共通初期J/U | SHA256 `78f67ec50f8c286122be795abc77df398058d1a802d1c5a3373582fc80a8b856`。PR追加最適化前のtensor-only CDF8からの共通初期化で、update005/064ではない |
| I61-CDF | `B2_CDF_relative`、元fit SOURCE `da2edb465d5ae6432439fbd1df8e57512ee59c11`、export SHA256 `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08` |
| I61-XDF | `B3_XDF_spin_relative`、元OpenFermion1.6.1 eigh signed λ/G・rank10 snapshotの元順prefix8。CDFのJ/Uと同一視しない |

元H4 input SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`、R32のh/g/scalar packet SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。固定contractのbindingを確認し、新しいHamiltonian/配列decodeは行っていない。各DFのJ/U・達成DF誤差・ground/alias・PF/RTE・law・shots/W・native価格を固有のまま保持する。

## 3. 追加Claimと旧Claimの接続

各行のF16/R32/I61は§1の科学SOURCE・結果commit・固定JSONを指す。先頭 `/` はそのJSON内のPointer。全行でH4条件を適用し、下流値にはRPE/P0も適用する。旧Claimとの接続は証拠の追加であり、元条件の判定の上書きではない。

| ID | 科学的主張・対象／比較 | 条件・固定証拠・JSON Pointer | 資格・未資格 | 旧Claim／研究上の問い |
|---|---|---|---|---|
| C21 | 共通有限政策のF16-PR/ENTRYは各16受理更新・DF予算PASS、J MET/U NOT_MET | F16 [JSON][F-result] 両 `/arms/*/trajectory/accepted_U_updates`、`/arms/*/physical`、`/arms/*/independent_terminal/state` | 数値pilot＋engineering guard。比較IIの有限手順で、比較Iの最適能力・同時一次収束ではない | C01/C10/C11/C15/C16に追加。近接正則化対照の物理的適格性 |
| C22 | F16-PRのexact tail λ .500044243591 HaはENTRY .941312847393 Haより小さい | F16 両 `/arms/*/mass_diagnostics/exact_tail_mass`（`*`はPR/ENTRYに置換） | 一登録強度・一初期化・未収束endpoint。一般的最適性能・目的だけの因果効果なし | C01/C04/C15、貢献Aの直接対照を追加 |
| C23 | ENTRYのB0は小さいが、PRは大きいrow-cancellation gapと小さいλを得た | F16 両 `/arms/*/mass_diagnostics/exact_entry_mass`、`/arms/*/mass_diagnostics/row_cancellation_gap` | 保存scalarの関係。写像後massとentry envelopeの違いに整合する限定観測。軌跡/metric等の交絡を除去していない | C10/C11・旧貢献整理§3.2。符号相殺を目的へ反映する意義の支持材料 |
| C24 | 固定二endpointでPR32の13round潜在適格、ideal shots/W・各native32へ接続 | R32 両 `/arms/*/PR/rounds`、`/arms/*/bindings/ideal_exact_native`、`/arms/*/PR/calibration` | Evidence laneのshots/総費用null。全64固定trajectoryのfull512数値検査PASSは全law上界ではない | C05〜C07/C17。DF→PF/RTE→RPE→価格の候補固有接続 |
| C25 | PR/ENTRY総RZ点比 .494181551983、登録近似Welch95%差区間がPR低費用を支持 | R32 `/comparison/primary` | `NOMINAL_MODEL_PR_LOWER`。固定32/arm、近似IID価格モデル。比CI・validated coverage・actual CAなし | C08/C20に追加する別candidate間の登録主比較。XDF検定とは別 |
| C26 | F16-PRの同じJ/U強Dに対する主RZ点比 .903287050982、nominal95%比区間[.876991946261,.929582155704] | R32 `/arms/PR/statistics/lanes/ideal_exact_native/RZ_PR_D_nominal` | 登録精度PASS・`NOMINAL_MODEL_PR_LOWER`。個別判定で、Welch主比較との同時保証なし | C09/C18/C20。新candidateに対する同J/U対照 |
| C27 | F16-ENTRYの同J/U強Dとの点比は1.69847394866だが登録精度未達 | R32 `/arms/ENTRY/statistics/lanes/ideal_exact_native/RZ_PR_D_nominal` | `UNRESOLVED_PRECISION_TARGET`。保存区間の位置だけで母平均高費用を正式認定しない | C19と同じ精度政策上の境界を保持。PR/ENTRY主比較とは別 |
| C28 | 保存CDF8/XDF8は共通idealで両 `CONDITIONAL_IDEAL_CA_PASS`、二候補比較はXDF lower | I61 D1 [JSON][I-result] `/candidates/B2_CDF_relative`、`/candidates/B3_XDF_spin_relative`、`/comparison` | Native誤差0は反実仮定。DF/PFは保持。厳密CA・全方式最適性なし | 旧強baseline・公平比較監査を共通laneへ接続 |
| C29 | F16-PRの総RZ点94288043542はI61-XDF 100267421030・CDF 102056444569を下回る | R32 `/comparison/primary/total_points/PR` と I61 両 `/candidates/*/cost/totals/rz_count` | 固定候補の条件付き点比較のみ。PRはfinite-bit価格推定、I61は固定価格。直接CI/検定・同等性・非劣性なし | C20/C28を異なる候補identityのまま接続。別DF表現に対する競争力の限定証拠 |
| Q02 | 比較II・近似Welch・個別PR/D・XDF/CDF点比較・未資格laneを分離 | F16 `/comparison`、R32 `/comparison/primary/ratio_interval`・`/actual_CA_achieved`、I61 `/PR_XDF_statistical_superiority` | null/falseと旧判定を保持。新しい科学的保証・正式採択なし | Q01を新3系列へ拡張。支持証拠と不足証拠を区別 |

Wildcard表記の実Pointer・保存scalar・検査結果は[静的照合JSON](../../artifacts/research_c_near_xdf_evidence_integration/2026-11-03/static_verification.json)に明記する。既存32 Claim（旧24＋C14〜C20/Q01）は重複・欠落なく保存する。

## 4. 有限手順endpointと物理予算

以下はF16 JSONの既存値の転記。両armは同じ有限予算・独立終端/物理判定政策を使い、最後の適格受理点を採用した。自己目的値Fのarm間順位は作らない。

| 保存指標 | F16-PR | F16-ENTRY |
|---|---:|---:|
| Exact tail λ / Ha | 0.500044243591 | 0.941312847393 |
| Exact entry envelope B0 / Ha | 1.5591348809 | 1.23977058606 |
| Row-cancellation gap / Ha | 1.05909063731 | 0.298457738667 |
| Sector DF norm / Ha | 4.84367101908e-9 | 1.11037670289e-8 |
| Engineering guard / Ha | 3.2165809758e-9 | 3.31972010087e-9 |
| DF予算CA/8 | PASS | PASS |
| 受理U更新 | 16 | 16 |
| J一次 | MET | MET |
| U一次 | NOT_MET | NOT_MET |

ENTRYの目的は同じtailに限定したweighted entrywise L1**二乗**で、ordered diag5/4・offdiag3/2。Uniform L2やSCDF shift目的、原論文の最強solver全体ではない。共通初期化・自己初期penalty寄与一致・一強度政策・有限手順という比較IIの適用範囲を維持する。両U未達、raw optimizerのprecision loss等も保存されており、全solver正常収束とは表現しない。

## 5. 下流資源・登録統計判断・共通ideal

| 指標（H4/PR32/RPE/P0） | F16-PR | F16-ENTRY |
|---|---:|---:|
| PF/RTE round | 13/13潜在適格 | 13/13潜在適格 |
| Ideal予定shots（X/Y合計） | 14738 | 19612 |
| W | 12539808 | 28556976 |
| Native価格標本数 | 32 | 32 |
| Kernel RZ mean | 7519.09375 | 6681.25 |
| Kernel mean記述SE | 107.32188319090913 | 115.50956530703141 |
| Shell込みideal条件付き総RZ | 94288043542 | 190796364542 |
| 同一J/U強D ideal総RZ | 104383256064 | 112333995286 |

PRのkernel単価は高いがWが小さく、登録会計で総費用が低くなった。単価だけでQPE/RPE負担を判断できない限定例である。J/U、PF/RTE、tail law、attenuation、basis価格が同時に違うため、最終差をλだけの因果効果に分解しない。Wへrや価格校正32を重複乗算せず、同じ価格平均を13roundで共有することを保つ。

主比較の保存PR−ENTRY差は−96508321000、登録近似Welch95%差区間は **[-103702867354.3995, -89313774645.6005]**、`precision_met=true`、`NOMINAL_MODEL_PR_LOWER`。点比.494181551983・点削減約50.58%は保存報告の引用で、比の信頼区間はnull。独立32価格cohort/armを固定した近似IIDモデル内の結果であり、診断PASSはIID・未観測tail・被覆率を保証しない。

個別PR/Dでは、PR armは点比.903287050982・nominal95%比区間[.876991946261,.929582155704]で登録判断PR lower。ENTRY armは点比1.69847394866・保存区間[1.638585159346598,1.758362737970442]だが登録精度未達で `UNRESOLVED_PRECISION_TARGET`。点値や区間がDより上にあることと、登録された母平均の正式判断を区別する。これらは主Welch比較や副指標との同時保証ではない。

| I61・H4/D1/RPE/P0 | 保存CDF8・B2 | 保存XDF8・B3 |
|---|---:|---:|
| Ideal status | CONDITIONAL_IDEAL_CA_PASS | CONDITIONAL_IDEAL_CA_PASS |
| 予定shots（X/Y合計） | 13606 | 13600 |
| W | 9225854 | 9211518 |
| Shell込み条件付き総RZ | 102056444569 | 100267421030 |

I61の登録判定は `XDF_LOWER_CONDITIONAL_INTERROGATION_COST`。Native synthesis信号誤差0は反実仮定であり、PF/DF誤差は保存候補固有のまま。旧numerical laneと各round整数N・総資源が一致したことは、その整数Nが変わらなかった結果で、旧合成誤差が実測0だったという意味ではない。

C29の点比較は、新F16-PRに対する肯定的な追加証拠である。I61は固定決定論価格、R32はfinite-bit sampling lawの価格proxyで、理想lawへの期待価格移送は未資格。PR対XDF/CDFの新差・比・CI・検定は生成していない。旧update064の100330684089.25という別endpoint/独立256価格の結果や、XDFとの近接関係を誤りとして書き換えない。F16-PRは共通tensor-only初期化から16更新した別候補である。

## 6. Evidence laneの資格境界

| 項目 | 維持する証拠 | 未資格・対象外 |
|---|---|---|
| DF/最適化 | 候補hashと保存sector norm＋engineering guard、F16両予算PASS、有限16更新 | 厳密enclosure、同時一次収束、全法最適化、目的だけの因果 |
| PF/RTE/ground | 候補固有13roundのsector数値pilot、own-ground/alias | 全law native synthesis・implemented-law signal距離 |
| Native | PR/ENTRY各32 trajectoryと各強Dのfull512数値correctness、保存shell・P0 | 全law一様誤差・support bound、未観測trajectory、hardware/FT実費 |
| Evidence lane | 不明な誤差をnull、shots/総費用を未資格として保持 | 実装された全方式の同CA保証・actual CA |
| Ideal | Exact native/PRのideal finite-RTE IID仮定、候補固有PF/RTE/DF、finite-bit価格proxy | 誤差0の測定、ideal-law価格への移送資格 |
| Assumed | 登録synthesis/law各≤.001Aという感度仮定 | 実測誤差や追加CIへの格上げ |
| 統計 | C25の登録近似Welch差区間、C26/C27の個別tモデル判定 | PR/ENTRY比CI、PR対XDF/CDF直接検定・同等性/非劣性、validated coverage、一般優位 |

PRの `actual_CA_achieved`、`all_law_synthesis`、`implemented_law_error`、`guaranteed_resource_superiority`、I61の統計優位・同等性・非劣性はnullのまま。RPE failure budgetは各事前候補の範囲であり、候補選択後・全比較同時保証へ拡張しない。

## 7. 貢献候補・論文構成案への追補

既存[貢献候補A/B/Cと一次文献対応](research_c_contribution_positioning_20261102.md)、[旧Map§8](research_c_evidence_map_updated_20261103.md)、[UWC/RC-DF/SCDFと近接目的の比較設計][NEAR-DESIGN]を再利用する。UWCのPR向けtail集中とOO/BLISS、RC-DFの一般CDF/正則化、SCDFのshift/表現制約、PR/RTEの既存費用式は既知という位置付けを保つ。新しい文献網羅調査・独立新規性レビューは実施していない。

既存設計では、UWCのglobal軌道回転/target-sectorを保つshiftと、研究Cのframe別U・一般signed Jによる近似DF/独立物理誤差予算・固定tail辞書を区別した。変換族を全面包含するという主張は維持していない。今回のENTRYは同じtailに限定した近接正則化対照であり、UWC/SCDF全体や原論文の強度調整・solverを再現した比較ではない。肯定的な有限手順結果が追加されても、この方法差と概念的重複の整理は変えない。

| 貢献候補 | 更新した支持材料 | 不足証拠・主張の境界 |
|---|---|---|
| A：PR-tailを意識したDF最適化 | C21〜C23。同じtailの近接weighted L1二乗対照を直接取得。共通有限政策でPR側のexact λが小さい | U一次未達、一初期化・一強度・H4一条件。原論文の最適既存法全体への勝利、一般目的能力・独立新規性は未認定 |
| B：DF表現→最終量子資源への接続 | C24/C25。λ・候補固有PF/RTE・A・shots/W・native単価・shell込み総RZを接続。高い単価でも小さい総費用という限定例 | 接続・RTE理論・shots×価格自体を新理論としない。λ単独因果、全law・実CA・ideal-law価格移送は未資格 |
| C：PR/PF compiled-resource tradeoff | C26〜C29。同J/U強Dへのnominal肯定、別保存XDF/CDFへの共通ideal点比較が追加 | ENTRYの個別精度未達、別表現との直接統計優位・一般性は未確定。実装誤差保証付き同CA順位ではない |

| 既存の論文構成案（未採択） | 追加して書ける限定的な結果 | 残る判断 |
|---|---|---|
| 案A：PR-tail CDF最適化の限定実証 | 近接weighted L1との比較IIを、物理予算と未収束を明示して追加できる | 目的の方法差・新規性の重要性、収束/初期値/強度/他分子をどこまで必要とするか |
| 案B：同一DF上のPR/PF資源比較 | F16-PR/Dの登録nominal判定、ENTRY/Dの精度未達を対比。別DF表現は点比較として別表に置く | 中心claimの適用範囲、モデル資格・保証不足をどの水準で扱うか |
| 案C：mass低減と総資源削減の不一致 | 旧負結果を保全し、今回の肯定例と並べてproxy/単価だけでは決まらない事例群へ更新できる | 機構の因果操作や未使用条件の設計則は未取得。負結果だけの一般否定も採用しない |

これらは支持材料と不足証拠の対応更新で、論文の中心claim・独立新規性・投稿可能性・論文化GO/NO-GOの正式採択ではない。

## 8. 静的検証・保全・次の担当

既存result/SOURCE/契約/STOP形式と `scripts/check_validation_manifest.py` を再利用した。新しい科学runner・統計engineは作らず、既存scalar/判定の照合とJSON Pointer/リンク・差分の技術検査だけを行う。保存rawを統計処理せず、NPZ/QPYを数値decodeしない。検証記録は科学的再現・CI成功・重要科学レビューの記録ではない。

3系列の固定公開15ファイルと補助監査・比較設計、科学SOURCE・結果・起点/監査8 commitのparent/tree、候補SHA256、SOURCE/契約binding、h/g/scalar/位相・compiler政策、旧Claim保持、新規Claim ID、転記・nullを照合した。保存scalar/資格の91項目、SOURCE 108/139/84 files、旧32 Claim、新10 Claim、追加relativeリンク14件がPASS。3実行報告の公開本文は完全一致した。全raw/QPYの再hashや旧worktreeの再棚卸しは行っていない。新成果を同じ統合branchに全て格納したという意味でもない。

既存manifest checkerはclean起点でも `relevant_commits[37]` の旧キー形式でFAIL（`sha/description`欠落、`commit`追加）。過去recordは一括修正しない。今回は現行checkerのschemaに適合する技術統合result-set/索引を追記し、追記範囲を個別検証する。全体checkerをPASSと表示しない。

旧科学SOURCE/raw/契約/grant/ledger/marker/STOP・元dirty checkout・他job・共有venv/cache/OS設定は変更しない。既存claim本文と過去判定を保持し、研究ノートへ追記する。新しいgrant・承認証明や未承認科学実行契約は追加しない。

| 次の担当 | 未実施の研究判断・具体的な境界 |
|---|---|
| ユーザー／研究C | C21〜C29のどれを中心claim候補へ使うか、独立新規性・論文化・研究継続/区切りを判断する。今回は正式レビュー/採択なし |
| ユーザー／研究C | 強めたい主張に合わせ、PR対XDFの直接統計比較、全law/実CA、有限最適化の頑健性、他条件のうち必要な課題を選ぶ。一律の追加計算を要求しない |
| Codex | 本統合と公開照合を完了してmandatory STOP。後続の技術作業は具体的依頼後、新科学計算/保存raw再解析・重要科学レビューはそれぞれ別の明示許可後のみ |

F16 645/最大1629、R32 1231/1231、I61 61/61の既存COMPLETE・FAIL/retry/resume0・mandatory STOPを保存する。未使用callの転用も行わない。Commit/push後にremote SHA/tree、公開本文、主要リンク・JSON参照を照合して**mandatory STOP**する。

## 固定公開参照

[F-report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ceb0a8825785ebd4f21241d8ef0f20cb321bf69/docs/research/research_c_near_objective_practical_comparison_execution_20261103.md
[F-result]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ceb0a8825785ebd4f21241d8ef0f20cb321bf69/artifacts/research_c_near_practical/2026-11-03/run01/result.json
[F-source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ceb0a8825785ebd4f21241d8ef0f20cb321bf69/artifacts/research_c_near_practical/2026-11-03/run01/SOURCE.json
[F-contract]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ceb0a8825785ebd4f21241d8ef0f20cb321bf69/docs/research/research_c_near_objective_practical_comparison_contract_unapproved_20261103.json
[F-stop]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ceb0a8825785ebd4f21241d8ef0f20cb321bf69/artifacts/research_c_near_practical/2026-11-03/run01/STOP.json
[R-report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/docs/research/research_c_near_practical_r32_downstream_execution_20261103.md
[R-result]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/artifacts/research_c_near_practical_r32/2026-11-03/run01/result.json
[R-source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/artifacts/research_c_near_practical_r32/2026-11-03/run01/SOURCE.json
[R-contract]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/docs/research/research_c_near_practical_r32_downstream_contract_20261103.json
[R-stop]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/artifacts/research_c_near_practical_r32/2026-11-03/run01/STOP.json
[I-report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39ee9705b7a459c40c205c542d660b5d2ab90d33/docs/research/research_c_xdf_cdf_common_ideal_lane_execution_20261103.md
[I-result]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39ee9705b7a459c40c205c542d660b5d2ab90d33/artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/run01/result.json
[I-source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39ee9705b7a459c40c205c542d660b5d2ab90d33/artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/run01/SOURCE.json
[I-contract]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39ee9705b7a459c40c205c542d660b5d2ab90d33/docs/research/research_c_xdf_cdf_common_ideal_lane_contract_unapproved_20261103.json
[I-stop]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39ee9705b7a459c40c205c542d660b5d2ab90d33/artifacts/research_c_xdf_cdf_common_ideal_lane/2026-11-03/run01/STOP.json
[AUDIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8375433869999c2e10501b1983c11b2b22c5d423/docs/research/research_c_strong_xdf_cdf_comparison_audit_20261103.md
[NEAR-DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ceb0a8825785ebd4f21241d8ef0f20cb321bf69/docs/research/research_c_uwc_rcdf_near_objective_comparison_design_20261103.md
