# 研究C：2026-11-03成果のEvidence Map追補

2026-11-03 JST。公開済み結果の技術的な証拠統合。新科学計算・数値再解析・重要科学レビューは実施していない。

**固定H4・CDF8・update064/r32では、独立256件のnative価格校正が、同じJ/Uの強決定論S2より条件付き期待総RZが約4.48%少ないという判断を、nominalな擬似乱数IID・近似tモデル内で支持した。** 登録判定は `NOMINAL_MODEL_PR_LOWER`。全law合成誤差・実装lawとの差・actual CA・一般性は未資格である。独立640件のupdate005/r32は `UNRESOLVED_PRECISION_TARGET` のまま。[E報告][E-report]・[E結果][E-result]・[D報告][D-report]

原本の[11月2日Evidence Map](research_c_evidence_map_updated_20261102.md)、[貢献候補・構成案](research_c_contribution_positioning_20261102.md)、[24 Claim対応](research_c_claim_contribution_alignment_20261102.md)、[10月31日統合](research_c_evidence_synthesis_and_decision_readiness_20261031.md)は変更しない。本書は新しい8項目を追加し、旧結果の条件付き位置付けを追記する。中心claim、新規性、論文化GO/NO-GO、研究C継続・終了は採択しない。

## 1. 固定版と依存関係

本書のA〜Eは実験系列の記号であり、旧文書の資料A/Bや貢献候補A/B/Cとは別である。結果収録commitと科学SOURCEを区別する。科学SOURCEのtreeに後日の結果があるとは仮定しない。

| 系列 | 内容・candidate | 科学SOURCE | 結果収録commit | 公開原本 |
|---|---|---|---|---|
| A | update005のr再調整 | `1cdeaa8bd55eb07a1d5b8cf2080259e929d405d1` | `f200e45afecc0a80b9f8a895b68c75ba4dda5dd5` | [報告][A-report]・[JSON][A-result]・[SOURCE][A-source]・[契約][A-contract]・[STOP][A-stop] |
| B | update005開始、追加64 U更新→update064 | `025d495166311a58ec9c24c43f07ed6e40f65f42` | `ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6` | [報告][B-report]・[JSON][B-result]・[SOURCE][B-source]・[契約][B-contract]・[STOP][B-stop] |
| C | update064/r32の下流・同じ064の強D | `928fda0a6df260b66eba1bbea63b51f2734e7b54` | `cca6b487c8e1ed3362f8825e95c937ebbcc87755` | [報告][C-report]・[JSON][C-result]・[SOURCE][C-source]・[契約][C-contract]・[STOP][C-stop] |
| D | update005/r32、独立640価格 | `62fa4c0144b9e36f9d96b7dff67628051946632d` | `459c8472d66689bf149a60fd6cc7311ff5237c65` | [報告][D-report]・[JSON][D-result]・[SOURCE][D-source]・[契約][D-contract]・[STOP][D-stop] |
| E | update064/r32、独立256価格 | `9837eac59b94113bf057fcc311a27b35e60343f5` | `3718d369b3db515b2dd13b2cdbb6c49d57880121` | [報告][E-report]・[JSON][E-result]・[SOURCE][E-source]・[契約][E-contract]・[STOP][E-stop] |

各結果の直接の親は表の科学SOURCE。共通資料起点 `a4d24682e79857f1e97f105cc03448d0e1786ea2` からA/Bが分岐し、A結果からC/Dが分岐、C結果からEが派生した。**B結果はCのGit祖先ではない**。C/EはBのterminalとphysical証拠をpeer commit＋hashで取り込む。Eが再利用した価格処理はDの技術SOURCE `62fa4c0…` であり、Dの640標本結果をEへpoolしていない。

統合branchは共通資料起点から分離した。A〜Eの大量raw/QPYを複製せず、上の絶対commitリンクを使う。固定25ファイルおよび補助JSON2件の公開本文とGit blob、10 commitのSHA/parent/treeを照合済み。SOURCE manifestと契約のhash、固定source filesの対応も一致した。Bは新5ファイルと継承207ファイルを合わせて212件である。[静的照合記録](../../artifacts/research_c_latest_evidence_integration/2026-11-03/static_verification.json)

## 2. 科学条件とcandidate identity

| コード | 固定条件・適用範囲 |
|---|---|
| H4/C8 | 保存直鎖H4、隣接1.00 Å、STO-3G。Nα=Nβ=2、sector36/full256/controlled512。通常number CDF rank8、一般signed実対称J・実直交U。元g補正κ、核scalar一回、spin/frame順を保持 |
| U64 | BのPR-only reduced目的γ=.1、元順LD2。Newton-CG＋2 refinement、資格付き有限Armijoと独立終端・物理判定。時間・rは最適化自体には非適用 |
| PR32 | 指定candidate、元順prefix LD2、partial ordered symmetric S2、s1、δ=h=.1、r32、K0、m0…12。Aだけr16/32/64新評価＋保存r128再利用 |
| D1 | 指定candidateのκ＋8 frame全9因子ordered symmetric S2、s1、δ=h=.1、m0…12。LD/r/Kは非適用。比較相手と同じJ/Uを使う強決定論対照 |
| R13 | CA=.00159 Ha、DF reserve CA/8=.00019875 Ha、βtotal=π/6、βstat=π/12、候補failure=.01、軸=.01/26。各DF自身のground-state oracle、状態準備費用除外 |
| P0/serial | Qiskit1.3.0/opt1/seed17、rz/sx/x/cx、approximation degree1、qubits_initially_zero=false、MPFR160 declared phase、位相fitなし。head相対Gaussian・tail exact basis_key共有、D全sweep相対融合。Fenced serial、ancilla head/X-Y shell込み |

| Candidate | 保存export SHA256 | 参照・所属 |
|---|---|---|
| update005 | `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e` | A/Dのselected。Bの初期点。11月2日の5更新後の候補 |
| update064 | `53d8f5027ce9e61bc72f6943c0268666bfcf87515154850a9a99326687b26ed3` | Bの新trajectoryで64更新後の `terminal.npz`。C/Eの候補 |

update064は旧trajectoryを通算して64とした名前ではない。Bで追加64回受理した終端である。両候補はJ/U、ground、PF/RTE、law、shots、W、強D分母が異なる。以下の表は上記条件と一組で読む。

## 3. 新規Claimの追跡

旧C01〜C13/N01〜N10/G01と衝突しないC14〜C20/Q01を追加する。Q01は資格境界の整理であり、新しい実験結果ではない。SOURCE/結果commitは§1の系列ごとに固定し、JSON参照は各公開 `result.json` のJSON Pointer（先頭 `/`）を示す。

| ID | 支持される主張・candidate・比較 | 条件 | 固定証拠・JSON参照 | 資格・未資格 | 中心RQとの関係 |
|---|---|---|---|---|---|
| C14 | update005固定で登録r16/32/64/128中、r32が条件付き総RZの最小点推定 | H4/C8、Aのr集合、R13/P0 | A SOURCE/結果、[JSON][A-result] `/new`、`/saved_r128`、[保存128添付][A-saved128] | Numerical/conditional、各新rはnative32。母平均順位・集合外最適性・064の最適rは未確定 | DF固定後の実行条件調整にも改善余地 |
| C15 | 同じPR目的で追加64回の資格付きU更新、Fγとexact tail λが低下 | H4/C8/U64、005→064 | B SOURCE/結果、[JSON][B-result] `/trajectory/accepted_U_updates`、`/trajectory/history`、`/terminal_state` | Numerical。JはMET、UはNOT_MET。真の勾配精度・同時収束・最適性なし | 旧有限step停止から前進できた有限trajectoryの証拠 |
| C16 | update064と全受理点は登録物理DF予算を保持 | H4/C8/U64、CA/8 | B SOURCE/結果、[JSON][B-result] `/trajectory/history/64/physical`、[終端物理JSON][B-physical] | Sector数値pilot/engineering guard。厳密DF enclosure・actual CAなし | 下流評価へ接続する候補固有の入力資格 |
| C17 | update064/r32の13roundが潜在適格、仮定laneのshots・W・native32へ接続 | H4/C8/PR32/R13/P0 | C SOURCE/結果、[JSON][C-result] `/PR/rounds`、`/PR/lanes`、`/PR/calibration` | Numerical＋conditional。Evidence laneのshots/費用はnull。全law・実CAなし | λだけでなくPF/RTEと価格まで候補固有に測定 |
| C18 | 同じ064 J/Uの強Dと比較し、初期native32価格でPR/D主RZ点比 .962683333226 | H4/C8、PR32対D1、R13/P0 | C SOURCE/結果、[JSON][C-result] `/D`、`/comparisons` | Conditional point estimate。32件とEの256件は別証拠。全指標支配・厳密CAなし | 同J/Uの公平な強D対照での限定的なcrossover |
| C19 | update005/r32独立640件では、価格差の符号と登録精度が未解決 | H4/C8/PR32対同005のD1、R13/P0 | D SOURCE/結果、[JSON][D-result] `/primary`、`/old_samples_pooled` | Nominal近似t、`UNRESOLVED_PRECISION_TARGET`。同等性・非劣性・厳密被覆率なし | 点順位を母平均勝敗へ格上げできない実例 |
| C20 | update064/r32独立256件が同064強Dより条件付き期待総RZ約4.48%低い判断を支持 | H4/C8/PR32対同064のD1、R13/P0 | E SOURCE/結果、[JSON][E-result] `/primary`、C `/D/ideal_exact_native` | `NOMINAL_MODEL_PR_LOWER`。固定実装lawの擬似乱数IIDモデル＋nominal95%近似t。全law・actual CA・一般性なし | この固定条件で主RZ低費用を支持する最新の主要価格証拠 |
| Q01 | C20のnominal価格判断と、未資格の信号/全law/CA・因果・一般性を分離 | 005/064それぞれの条件 | D/E `/actual_CA_achieved`、`/guaranteed_resource_superiority`、`/primary/ideal_law_cost_transfer_qualified`、C `/PR/lanes/evidence` | null/falseを保持。新しい保証・検定・科学判断を追加しない | 限定肯定結果を正確に利用する境界 |

## 4. r再調整・U更新・下流の保存値

### 4.1 update005の登録r集合（A）

| r | 条件付き総RZ点推定（ideal lane） | 価格cohort |
|---|---:|---|
| 16 | 117578674155 | 新native32 |
| 32 | 106752458375 | 新native32 |
| 64 | 131452851483 | 新native32 |
| 128 | 207919150188 | 旧同005結果を再利用 |

r32/r128の保存比は .5134325447101659（約48.66%減）。これは登録集合の点比較で、理論的最適rや選択後の母平均保証ではない。update064はr32だけを下流評価しており、最適rを認定しない。[A報告][A-report]

### 4.2 追加U更新（B）

| 保存指標 | 初期update005 | 終端update064 |
|---|---:|---:|
| Fγ | .006617509794531903 | .006033479843764215 |
| Exact tail λ [Ha] | .5683498546862645 | .5426906587519222 |
| DF物理予算 | PASS | PASS |
| J一次 | MET | MET |
| U一次 | NOT_MET | NOT_MET |

151点（初期＋150 trial）のうち146点admitted、5点未資格。64回の数値・物理適格U更新を受理し、81回はArmijo/値不成立、4回はreduced-gradient未資格、1回はJ未資格。453 inner solve（refinement302）、独立終端64、物理評価66件が記録されている。raw solver status0は388件、precision loss/status2は64件、maxiter/status1は1件で、全optimizer正常収束とは記さない。[B報告][B-report]

停止は `MAXIMUM_ACCEPTED_UPDATES`。終端stable J∞=4.94210978980265e-8、stable U₂=.6086603836696319でU閾値1e-5未達。終端sector誤差5.673413458637335e-9 Ha、guard3.2244465214668673e-9 Haを別々に保存しCA/8内。tensor Frobenius3.358086260639999e-9、general-K/raw-CAR最大差4.135580766728708e-15。B自体はPF/RTE・native・shots・総資源を評価していない。[B結果][B-result]・[物理結果][B-physical]

### 4.3 update064/r32の初期下流（C、native32）

| 保存指標 | PR064/r32 | 同じ064 J/Uの強D |
|---|---:|---:|
| ideal lane十分shots（X/Y合計） | 15008 | 13706 |
| coherent反復重みW | 13381920 | 9526398 |
| kernel RZ価格 | 標本平均7556.34375 | 固定11026 |
| ideal条件付き総RZ | 101118440083 | 105038112319 |

PR13/13は `POTENTIALLY_FEASIBLE`。m12の保存PF operator誤差 .0909586527274、補正RTE operator誤差 .0431759381054、attenuation .828210321268、ρ .717118668701。PF/RTE誤差は無次元のoperator比較で、DFのHa単位と根拠なく加算しない。Own-DF ground/physical ground、aliasを確認し、保存DF予算を再利用した。[C報告][C-report]

この32件によるPR/D点比は .962683333226。旧005/r32の32件との総RZ点比 .947223526486も保存されているが、直接候補間の統計的優越性は認定しない。Dのkernel価格11026はCで新測定した値で、005からの無条件転用ではない。候補別shots/W/D分母を維持する。Assumed laneのPR circuit sizeはDより大きいので、全6指標での支配とは扱わない。Cの32件をEの256件にpoolしない。

## 5. 独立価格校正：005と064を別々に読む

下表は保存JSONの転記であり、rawからの再集計や新しいCI計算ではない。両者の主estimandは、固定したideal laneのshots/W/shellと同candidateのD分母に、実装finite-bit native価格母平均のモデルを接続した条件付き主RZである。[D結果][D-result]・[E結果][E-result]

| 項目 | D：update005/r32 | E：update064/r32 |
|---|---:|---:|
| 独立新cohort固定N | 640 | 256 |
| 旧標本pool | false | false |
| Kernel RZ mean | 7546.08125 | 7497.4765625 |
| Kernel mean SE | 22.418706153568692 | 34.23387068763307 |
| 固定ideal shots | 15202 | 15008 |
| 固定W | 13979232 | 13381920 |
| 固定shell RZ | 53207 | 52528 |
| 条件付き総RZ点推定 | 105488473691.6 | 100330684089.25 |
| 総RZ SE | 313396294.4605644 | 458114918.8322508 |
| 同candidate強D総RZ | 105532782860 | 105038112319 |
| PR−D点差 | -44309168.3999939 | -4707428229.75 |
| PR/D点比 | .999580138349438 | .955183617395431 |
| Nominal95%比区間 | [.9937486698850807, 1.0054116068137955] | [.9465946344829994, .9637726003078625] |
| 登録半幅目標 | D費用の0.5% | D費用の1% |
| 観測半幅 | 615411095.2039692 | 902170551.8619565（D比約.858898%） |
| precision_met | false | true |
| 登録decision | UNRESOLVED_PRECISION_TARGET | NOMINAL_MODEL_PR_LOWER |

005は区間が1を含み精度目標も未達。母平均差の符号、同等性、非劣性は未確定。064は登録1%精度を満たし区間上端が1未満で、約4.48164%低い点比をnominalモデル内で支持した。005の0.5%目標を064に当てたり、064の緩い目標を005へ事後適用したりしない。

両者とも固定N・中間推論なし、除外0。歪度は005=-.2680774239377625、064=.01246237912359828で警告なし。Delete-one影響screenも警告なしだが、005は半幅1倍、064は半幅.25倍という異なる登録規則。064の最大削除影響/半幅=.0909481768471796はその.25閾値以内。これらはIID性、未観測rare tail、CI被覆率を保証しない。[D契約][D-contract]・[E契約][E-contract]

Eは新cohort8 batch×32、旧800 seed keyとの非重複を保存gateで確認した。非重複keyだけで統計的独立性を証明したとは扱わない。Eのsecondary assumed総RZ 100818650091.84375は記述値で、第二のCIや6指標同時順位を付与しない。

**005と064の二つの区間を比較して、064が005より母平均で統計的に優れているとは言わない。** Candidate、shots/W、強D分母、precision/influence規則が異なり、直接候補間の比較設計ではない。独立校正は価格標本の独立性というモデル上の位置付けであり、外部研究者による独立再現ではない。

## 6. 誤差と証拠laneの境界

| 項目 | 取得・維持した資格 | 取得していない資格 |
|---|---|---|
| DF | 保存sector誤差＋engineering guard、CA/8 PASS | 厳密丸めenclosure、物理入力全体の新保証 |
| PF/RTE/ground | 候補固有sector reference、registered十分条件・ground/aliasの数値pilot | 全law native synthesis、implemented-law signal距離 |
| Native | 対象標本のfull512 raw/P0/actual adjoint/X-Y correctness、保存shell | 全trajectory一様合成誤差、未観測全support |
| Evidence lane | 未知のsynthesis/law誤差をnullに保持、shots/総費用null | actual CA、保証shots・資源優位 |
| Ideal lane | exact native synthesis/ideal finite-RTE IID signalを仮定した既存十分shots/Wと条件付き会計 | 実装lawのsignalがこの仮定を満たす証明、ideal価格母平均への転送 |
| Assumed lane | synthesis/law各.001Aという明示的感度仮定、保存された記述的価格 | その誤差予算の実測・保証、別の統計CI |
| 独立価格CI | 固定Nの擬似乱数IID・近似Student-t、Eのnominal判定 | validated coverage、post-selection/global optimum、6指標同時保証 |

同じ価格cohortを全roundで使うため、価格SEは固定W倍の既存伝播を用いる。13roundを独立価格校正と数えたりsqrt13で縮めたりしない。予定shotsはRPE/Hoeffding十分式の出力で、実量子測定回数の実績や情報理論的普遍下界ではない。有限残差や標本最大値を全law上界へ昇格させず、未知誤差を0で埋めない。

## 7. 旧24 Claimとの対応・歴史の保全

旧各Claimの本文・数値・SOURCE・当時の判定は[原本対応表](research_c_claim_contribution_alignment_20261102.md#4-全24-claimの追跡表)を保持する。以下は上書きではなく、追加証拠によって現在の読み方を限定する追補である。

| 旧ID | 元条件で維持する内容 | 今回の接続・位置付け |
|---|---|---|
| C01 | 同x0 γ0/.1の初期tail低減・DF予算 | C15/C16は追加trajectory。初期fit比較を変更しない |
| C02 | 旧s4/r16の約83%PR内部点改善 | C14/C17/C20は別candidate/条件・価格資格 |
| C03 | 旧軌跡の5受理更新 | C15の追加64更新を別軌跡として追記 |
| C04 | update005のλ・物理DF資格 | C16は新update064。旧分母・旧物理誤差を保全 |
| C05 | 旧新PR/r128のPF増・RTE減・A増 | C17の064/r32を追加。全候補に同じ変化を仮定しない |
| C06 | 旧r128のN/W低減 | C17/C19/C20は候補・r固有の新N/W |
| C07 | 旧r128 native32の価格上昇 | C14はr変更、C19/C20は独立価格。不利だった旧値を残す |
| C08 | update005/r128のPR内部総RZ11.19%減 | C14のr調整、C18/C20の新同J/U対照とは分ける |
| C09 | 同005/r128対強Dで主RZはDが低い | 条件不変。005/r32はC19未解決、064/r32はC20でnominalにPR低い |
| C10 | 初期RC2限定対照・DF予算 | C15はPR-only。収束済み最強RC-DFへの勝利に置換しない |
| C11 | 共通政策RC2/PR両未収束 | 新PR終端もU未達。旧対照の公平性不足は維持 |
| C12 | 別RC2/PR終点の下流/shots/native | その総費用を本書で新算定せず、064へ転用しない |
| C13 | stable RC2の64更新・U一次未達 | 今回BのPR64更新とは別。RC2新終端DF未評価を維持 |
| N01 | tail-P4のsupport減でも価格/総費用増 | C20はP4の復権や疎化因果の証拠ではない |
| N02 | 通常CDF7の主RZ点改善なし | CDF8の今回結果を低rank一般に広げない |
| N03 | 固定複合surrogate未資格 | 新しいsurrogate/ηを採択しない |
| N04 | 第三DFへの登録switch転移不成立 | C20は新DF汎化の検証ではない |
| N05 | 厳密shared-basisは通常低rank CDFと同値 | 数学的同値は不変。新しいDF表現classを主張しない |
| N06 | 人工tail融合でcomplete費用改善なし | 今回compiler/融合規則は維持。一般不可能性へ広げない |
| N07 | 旧J/U・r128のround混合は全D選択 | 064/r32の混合scheduleを新評価したとは扱わない |
| N08 | 調査集合で追加sampling自由度を未特定 | r調整・U更新の肯定結果から新sampler独立性を認定しない |
| N09 | 固定U/tail保存LPの小さいλ改善 | BはUも変える別問題。LP結果・資格は保全 |
| N10 | 当時のnative価格機構準備TECHNICAL_NO_GO_CURRENT_EVIDENCE | 当時未特定だった新cost-aware操作の判断。研究C全体のNO-GOではなく、C14/C20とも両立 |
| G01 | 旧F2/F3-K1各固定trajectoryのArb192 validated CAR | 005/064・r32全lawや13round、CA/CIへ保証を移さない |

旧「主RZでDが低い」は誤りとして消さず、旧candidate/r/価格cohortの結果として残す。最新の中心的な肯定結果は**同064 J/U・r32・nominal価格モデル内**に限って追加する。

## 8. 科学的貢献候補・論文構成案への影響

既存[貢献整理](research_c_contribution_positioning_20261102.md)の一次文献対応（RC-DF 2212.07957v3、SCDF 2403.03502v1、PR/RTE 2503.05647v2、CDF/QFD 2104.08957v1）を再利用する。一般J/U、正則化、inner/outer最適化、RTE normalization、shots×価格、基底融合が既知であるという整理は維持する。今回は網羅的文献再調査や新規性の新しい科学レビューをしていない。

| 貢献候補 | 新たに接続した証拠 | なお不足する証拠・書けない主張 |
|---|---|---|
| A：PR-tailを意識したDF最適化 | C15/C16：同目的で64追加更新、Fγ/λ低下とDF予算保持 | 同時一次収束・真のreduced勾配保証、十分最適化された近接RC-DF対照、独立初期値/分子への頑健性 |
| B：DF→PR資源の接続 | C14：r再調整で条件付き費用が変わる。C17/C18：064固有のPF/RTE・ground・W・native・同J/U強Dへ接続 | 既存RTE理論/会計を新理論としない。λ単独の因果、全law・実CA・一般性は未確定 |
| C：PR/PF compiled-resource tradeoff | C19/C20：005では符号未解決、064では主RZ低費用をnominalに支持。旧r128不利から限定crossoverを観測 | PR一般の優位、最適r、hardware実時間/FT T・Toffoli、全指標支配、直接005対064の統計勝敗は未評価 |

| 既存論文構成案（未採択） | 更新した支持材料・可能な限定記述 | 弱点・不足証拠 |
|---|---|---|
| 案A：PR-tail CDF最適化の限定実証 | C01/C03/C04にC15/C16を接続。「物理予算を保った有限更新でtail proxyを低減し、固定条件の下流へ接続」まで | 独立した方法差・既存正則化対照・同時一次・汎化は未確立。C20だけで目的の因果効果は証明されない |
| 案B：同一DF上のPR/PF compiled-resource比較 | C09の旧不利とC14/C18/C19/C20を区別。「固定064/r32の主RZでは独立価格のnominalモデル内で低費用」まで | 強Dを含む同J/U比較の適用範囲は狭い。価格CIと実装信号/CAの資格は別。比較条件外へ拡張しない |
| 案C：mass低減と総資源削減の不一致 | 旧C05〜C09・N01〜N04を保持し、r調整と064の肯定結果を対比。Proxy単独で価格/総費用を決めない事例群 | 旧負結果だけで全政策を否定する構成にはできない。価格機構の因果操作・未使用条件での設計則は不足 |

この表は支持証拠と弱点の対応であり、案の採択・重要性・投稿可能性の認定ではない。

## 9. 静的検証・保全・残る判断

既存のSOURCE/契約形式、result/STOP、claim対応、価格処理 `c_r32_price_statistics.py` と064 adapter、既存manifest checkerを参照した。科学runnerは起動せず、保存NPZ/QPYの数値decode・fit・operator・PF/RTE・shots・新統計・新順位計算は0。引用scalarは固定JSONへの一致だけを検査した。

今回の確認範囲は、A〜Eの固定公開本文25件と補助JSON2件、commit/parent/tree、SOURCE manifestと契約のbinding、source files A67/B212/C91/D70/E96、2 candidate exportのhash、JSONの既存scalar/qualification、追加リンクと変更差分である。旧raw/QPYの全hash再監査・他担当worktree監査はしていない。統合差分は文書・索引・既存manifestへの技術統合索引・静的確認JSONに限定し、科学SOURCE/raw/契約/grant/ledger/marker/STOPを編集しない。

`scripts/check_validation_manifest.py` はclean起点でも既存 `relevant_commits[37]` のキー形式でFAILする（`sha/description`欠落、`commit`追加）。今回の文書統合による新規不具合ではなく、過去manifestを一括修正せず同じ既知FAILを記録する。今回追加分のJSON、リンク、転記、旧record保全、差分の静的検査は別に実施する。全体checker PASSや科学的再現テストPASSとは表示しない。

| 次の担当 | 具体的な残る判断・境界 |
|---|---|
| ユーザー／研究C | C20の限定的価格結果をどの中心claimに使うか、重要性・独立新規性・論文化・継続/区切りを判断する。本書は採択しない |
| ユーザー／研究C | 強めたい主張に応じ、全law/actual CA、収束・近接既存法対照、初期値/分子移送、直接候補間比較のどれが必要かを選ぶ。全項目を一律必須化しない |
| Codex | 今回の資料統合・静的照合・公開を完了してSTOP。追加技術調査は具体的依頼後、新科学計算/保存raw再解析は別の明示許可後のみ実施する |

今回の範囲に追加の科学実行契約や未完了計算はない。各旧STOPと未使用callを維持し、公開SHA/tree/本文照合後に**mandatory STOP**する。

## 固定公開参照

[A-report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f200e45afecc0a80b9f8a895b68c75ba4dda5dd5/docs/research/research_c_rte_r_retuning_execution_20261103.md
[A-result]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f200e45afecc0a80b9f8a895b68c75ba4dda5dd5/artifacts/research_c_rte_r_retuning/2026-11-03/run01/result.json
[A-source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f200e45afecc0a80b9f8a895b68c75ba4dda5dd5/artifacts/research_c_rte_r_retuning/2026-11-03/technical/SOURCE.json
[A-contract]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f200e45afecc0a80b9f8a895b68c75ba4dda5dd5/docs/research/research_c_rte_r_retuning_contract_unapproved_20261103.json
[A-stop]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f200e45afecc0a80b9f8a895b68c75ba4dda5dd5/artifacts/research_c_rte_r_retuning/2026-11-03/run01/STOP.json
[B-report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6/docs/research/research_c_u_step_control_execution_20261103.md
[B-result]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6/artifacts/research_c_u_step_control/2026-11-03/run01/result.json
[B-source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6/artifacts/research_c_u_step_control/2026-11-03/technical/SOURCE.json
[B-contract]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6/docs/research/research_c_u_step_control_contract_unapproved_20261103.json
[B-stop]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6/artifacts/research_c_u_step_control/2026-11-03/run01/STOP.json
[C-report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cca6b487c8e1ed3362f8825e95c937ebbcc87755/docs/research/research_c_update064_r32_downstream_execution_20261103.md
[C-result]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cca6b487c8e1ed3362f8825e95c937ebbcc87755/artifacts/research_c_update064_r32_downstream/2026-11-03/run01/result.json
[C-source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cca6b487c8e1ed3362f8825e95c937ebbcc87755/artifacts/research_c_update064_r32_downstream/2026-11-03/technical/SOURCE.json
[C-contract]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cca6b487c8e1ed3362f8825e95c937ebbcc87755/docs/research/research_c_update064_r32_downstream_contract_unapproved_20261103.json
[C-stop]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cca6b487c8e1ed3362f8825e95c937ebbcc87755/artifacts/research_c_update064_r32_downstream/2026-11-03/run01/STOP.json
[D-report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/459c8472d66689bf149a60fd6cc7311ff5237c65/docs/research/research_c_r32_price_statistics_execution_20261103.md
[D-result]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/459c8472d66689bf149a60fd6cc7311ff5237c65/artifacts/research_c_r32_price_statistics/2026-11-03/run01/result.json
[D-source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/459c8472d66689bf149a60fd6cc7311ff5237c65/artifacts/research_c_r32_price_statistics/2026-11-03/technical/SOURCE.json
[D-contract]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/459c8472d66689bf149a60fd6cc7311ff5237c65/docs/research/research_c_r32_price_statistics_contract_unapproved_20261103.json
[D-stop]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/459c8472d66689bf149a60fd6cc7311ff5237c65/artifacts/research_c_r32_price_statistics/2026-11-03/run01/STOP.json
[E-report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/docs/research/research_c_update064_r32_price_statistics_execution_20261103.md
[E-result]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/artifacts/research_c_update064_r32_price_statistics/2026-11-03/run01/result.json
[E-source]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/artifacts/research_c_update064_r32_price_statistics/2026-11-03/technical/SOURCE.json
[E-contract]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/docs/research/research_c_update064_r32_price_statistics_contract_unapproved_20261103.json
[E-stop]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3718d369b3db515b2dd13b2cdbb6c49d57880121/artifacts/research_c_update064_r32_price_statistics/2026-11-03/run01/STOP.json
[A-saved128]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f200e45afecc0a80b9f8a895b68c75ba4dda5dd5/artifacts/research_c_rte_r_retuning/2026-11-03/run01/saved_r128.json
[B-physical]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6/artifacts/research_c_u_step_control/2026-11-03/run01/physical/update064.json
