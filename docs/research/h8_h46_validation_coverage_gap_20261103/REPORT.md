# H4/H6/H8の検証範囲とH8追加検証の要否

2026-11-03 JST、LOCAL。既存公開文書・保存metadata・SOURCEの静的調査による比較資料。
新しい科学計算、保存値の数値再解析、sampling、回路生成・compile、科学runnerのimportは行っていない。
本資料は追加検証の必要性に関する提案であり、既存の正式採択、Phase 3、実行契約、STOPを変更しない。

## 1. 判断

**現在のO1を、強い決定論PFとの有限標本・条件付き資源比較として記述するために、新しいH8科学計算は必須ではない。**
H8は固定2精度の別価格確認、強化184対照の5精度比較、追加6精度の境界評価、希少層の正式採択済み感度を既に持つ。
中心メッセージである「比較集合・精度によってPRの競争力が変わる」を、既存H8だけで補強できる。[SYNTHESIS] [OPTIONS] [STRONG] [EPS] [CLOSURE]

ただし、二つの「同程度」は異なる。

| 比較水準 | H8の到達点 | 残る差と必要性 |
|---|---|---|
| A：T=.8、ε=.05/.0005の固定候補・別価格確認 | 主表の固定PR/PFは直接実測。保存mean/paired SE・方向と候補identityは比較できる | 公開詳細completion/seed/provenanceはH4/H6と同水準まで確認できない。G04はmetadata回収の問題。科学再実行は要求しない |
| B：H4/H6長時間研究と同じ広い科学条件 | H8にもT=.8の複数prefix/q/r/K・5精度・強いS2/S4決定論対照・N/B/C/G会計がある | H8の複数T、高次PR、m5を含む同入力の広い比較・固定確認は、調査した公開資料にはない。H4/H6と同じ長時間検証済みとは書けない |

H8長時間・高次PRには、より大きな保存入力で「qの減少」と「signed piece・normalization・shot・whole回路価格の増加」の競合を判別する科学的価値がある。
しかしO1の成立条件ではなく、主張をそこまで拡張するとユーザーが選ぶ場合だけの候補である。
全格子、同じ候補数、40セルの再現を目標にしない。[LONG] [HIGH_CONFIRM]

## 2. 証拠の入口と読取範囲

指定された研究統合の5資料はcommit `2ce2b5f161e94567fec322d694fd8b01d03cd854` を使用した。
H4/H6長時間は `155529e20a8fac39e00786c40b57830b71203c4f`、固定主表・protocol scopeは
`4591b8d75a2a6a55c74a51225f0c61648023c8ee` を使用した。
固定報告、必要なJSON/CSV欄、SOURCE/テストの静的読取を行い、元rawからmean/SE・適格性・費用・順位を再計算していない。
固定資料と読取範囲は [VALIDATION.json](VALIDATION.json)、条件別の対応は [H468_VALIDATION_COVERAGE.csv](H468_VALIDATION_COVERAGE.csv) に記録した。

G01/G02の既存追補も使用した。旧642価格候補、後続追加7物理候補、S2 q32の追加域、確認価格は別集合である。
旧集合33/40と追加後33/40は異なる集合の保存結論で、今回合算・数え直しをしていない。[G_REPORT]

実際のAGENTS.md、PROJECT_MAP、研究概要・現状、正式共通指示2026-10-25の§8.1/§8.4/§10を確認した。
現在の元checkoutの概要・validation台帳は古い系列も含む入口なので、他branchの全成果を包括する最新台帳とは扱わず、指定固定結果を優先した。
独立worktreeで、本ディレクトリだけを追加する。原本・他研究・整理成果物を変更しない。

## 3. A：固定2精度の主表

共通の表示条件はlinear H-chain、隣接1 Å、STO-3G、tol-only DF threshold1e-8、保存rank H4/H6/H8=10/19/28、
T=.8、P=0、通常signed bias、allowance倍率1。PRは二次partial PF＋元の無条件finite RTE。
`δ=T/q`、prefixは保存fragment列のL_Dである。
費用は `G=N_cos C_cos+N_sin C_sin`、Cは測定込みfull Hadamard wrapperのcompiled RZ、Nは解析的十分shot数、
nは古典価格trajectory数。同じtrajectoryのcos/sinをpairedとしてSEを保存する。[MAIN_REPORT] [MAIN]

| 系・ε | 固定PR | L_D / δ | 固定PF | n | 保存方向／低い点費用 |
|---|---|---|---|---:|---|
| H4 .05 | B2_p3_q1_2nd_r4_K2 | 3 / .8 | B0I_p5_q1_2nd | 1108 | NOT_REPRODUCED / PF |
| H4 .0005 | B2_p2_q8_2nd_r2_K4 | 2 / .1 | B0_p7_q8_2nd | 78 | REPRODUCED / PR |
| H6 .05 | B2_p4_q2_2nd_r4_K2 | 4 / .4 | B0I_p7_q1_2nd | 48 | REPRODUCED / PF |
| H6 .0005 | B2_p4_q8_2nd_r4_K2 | 4 / .1 | B0I_p11_q4_4th | 104 | REPRODUCED / PR |
| H8 .05 | B2_p8_q2_2nd_r4_K4 | 8 / .4 | B0I_p11_q1_2nd | 48 | POINT_REPRODUCED / PR |
| H8 .0005 | B2_p7_q16_2nd_r4_K4 | 7 / .05 | B0I_p15_q4_4th | 48 | POINT_REPRODUCED / PR |

この表は保存値の転記。exact G/SEはCSVの固定6行から元JSONの `/rows/0..5` へ辿れる。
主表の全固定PFは直接実測で、全探索domainのPFが実測済みという意味ではない。
H6粗精度のREPRODUCEDはPF低費用という事前方向の再現である。H4粗精度の非再現も正式な採用結果であり、標本追加の理由にしない。
H8の2行はMechanism32内のH8_CONFIRM_1/3と同じcohortの再引用で、追加独立実験ではない。[MAIN_REPORT] [PROTOCOL]

### 入力・科学条件・監査の対応

| 観点 | H4 DF10 | H6 DF19 | H8 DF28 | 比較で維持する限定 |
|---|---|---|---|---|
| H/state/reference | 保存H・state・full-H reference | 保存weighted-Hermitian completion後のHとstate/reference | 保存weighted-Hermitian projection・元DF/state/reference | 同じ分子名から旧入力との同一性を推測しない |
| 生成policy/rank | tol-only、観測rank10。legacy DF12と別 | tol-only、観測rank19 | tol-only、観測rank28、signed generation order/cutoff0 | rankは生成時の固定最終rank指定ではない。全provider/runtime/積分・軌道手順の完全同一性は未認定 |
| state solver | sector36、dense scipy.linalg.eigh、位相固定 | 記録はsector400、eigsh SA/HF初期vector | 記録はsector4900、eigsh SA/HF初期vector | H6/H8の記録設定はk1/tol1e-12/ncv40/maxiter1000。private実体を今回再監査した値ではない |
| 正規化・residual | eigh vector、保存後再正規化なし、gate1e-10 | 記録上一度のbinary64正規化、abs1e-9/rel1e-10 | 同じ記録政策 | 元vectorを再生成・共通手順へ置換しない。residualだけで真のsignal精度を保証しない |
| bias/B/shot | 元通常signed・経験的allowance・N | 同左 | 同左。後述の別非相殺表を混ぜない | normalization Bによるattenuationとheadroomを維持。Nとn、biasと測定誤差を区別 |
| native/compiler | full cos/+Im sine wrapper、symmetric_directional、scalar一回、actual adjoint | 同じ記録条件 | 同じ公開SOURCE/記録条件 | ordinary `diag(I,U)` は意味論。Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/topologyなし。旧compiler全出力のbyte同一性とは別 |
| 探索・確認 | 別固定確認、旧FP/legacyと混合なし | 別固定確認、Q2/Q8の停止・dispatch履歴を保持 | 固定2候補、探索n8/旧n32と混合なし | 確認後候補差替えや価格poolをしない。元H8確認cohort内の完了済み1軌道再利用を新独立標本へ加算しない |
| 価格独立性 | G02は列挙範囲のDOCUMENTED_DISJOINT | G02はRECORDED_SEPARATE | G02はRECORDED_SEPARATE、Mechanism aliasはREUSED_OR_OVERLAPPING | cohortの名称やseed非衝突だけで数学的独立性・全履歴disjointを証明しない |
| 終了証拠 | 公開completionで1186 trajectory/2372軸 | 公開completionで152 trajectory/304軸 | 公開結果96 trajectory/192軸、詳細PR監査はlocal PASS申告まで | PFの公開exit/reap証拠をPR確認へ流用しない。作者の技術検査を外部独立review/CIと呼ばない |

入力・solverの記述は既存の公開例外表からの引用である。[EXCEPTIONS] [METHODS]
Hamiltonian/state/referenceのidentityは公開binding・SOURCE・保存報告の対応を確認した範囲であり、private NPZ/state/reference原bytesを今回読み直した意味ではない。
共通のbias/allowanceという名称だけで、各系列のu・log_B marginの数値や算定規則が完全同一とも認定しない。長時間の共通PF政策と主表の元保存政策は各出典に従う。
H4/H6の公開completionがあることと、H6の最終seed seal・全履歴除外まで回収できていることも別である。[G_REPORT]
したがって「H8だけ科学的に無効」「三系が完全に同一の手順・保証」とするどちらの表現も採らない。

### G04補完の取得状態

調査時点の公開head・GitHub検索・固定mainから、SERVER2指示①に対応する新しいG04補完の固定URLを特定できなかった。
既存統合に記録された詳細候補commit `659e555714c8ccf3a6ec7fd1cb0dbf05cfcc342d` と
`96aa3878a344f479ee064118cb90002a770db627` のGitHub GETは、今回 `422 No commit found` で取得できなかった。
以前の統合の404記録を今回の応答と混同しない。[OPEN]

この結果はprivate原本の不存在やSERVER2作業の失敗を意味しない。現在のサーバーfilesystemには接続していない。
G04補完を取得済みとせず、H8の技術statusは `REPORTED_LOCAL_PASS_NOT_REVERIFIED_FROM_PUBLIC_RECEIPTS` を保持する。
不足は、実行SOURCE/input binding、固定nominee/cohort/index、完了・回収・最終照合、既存aliasを結ぶ公開可能なmetadataである。
まず既存記録の限定回収で扱い、同じ科学条件を再実行して埋めない。原profile/provider bytesまで回収できるかは未解決。[PHASE3_GAPS] [METHODS]

## 4. B：広い科学条件のcoverage

下表は全て保存結果の引用。各行の物理条件は上記のlinear1 Å/STO-3G・保存DF10/19/28で、H4 legacy DF12とは分ける。

| 要素 | H4/H6長時間 | H8の取得済み範囲 | 実際の残り |
|---|---|---|---|
| 時間・精度 | T=.8/1.2/1.6/2、5精度 | T=.8で5精度。さらに.04/.025/.015/.000875/.00075/.000625 | H8の長時間移送。精度数をそろえる作業は不要 |
| 決定論PF | S2/Yoshida4/m5、Full/B0/B0I | 元Full S2/global Yoshida4、強化B0/B0I S2/S4、184対照 | H8 m5を含む同条件強対照の取得は公開範囲で未確認。S4対照が未実施とはしない |
| PR | 3公式、複数prefix、R/Kの選択 | 元prefix7/14/21、q1/2/4/8、r1/2/4、K2/4。固定主表はp8/q2とp7/q16、r4/K4 | H8高次PR、広いR/K、長時間での固定確認。全格子は不要 |
| 適格性・shot | 11632構成/58160精度endpoint（両系合計） | 元92、D0の5精度化、追加152 signal、強化184の保存適格性・shot | 同じ件数ではなく、新しい時間/公式で何を判別するかが必要 |
| compiled価格 | 642取得候補、全登録域の直接compileではない | 元mapの実測、強化対照のselected-axis実測とmodel、後続DBAの2構成4軸 | 未測定modelの全置換を自動要求しない。新しい資源方向を主張する候補には直接価格が必要 |
| 探索・nominee・確認 | 固定59比較、58方向再現。39候補価格を共有 | 元union6は先に固定、精度別minは確認後選択。主表2点は別固定確認 | 元union6の事後minと主表n48を同じ選抜保証へ統合しない |
| N/B/C/G競合 | 公式別・同q/prefix・r/K・固定δの保存説明 | B/信号/shotとwhole費用、P感度、主表同cohortのmechanism因子、強化対照・希少層感度 | 新しいH8長時間/高次PRの同条件説明。新数値分解は現在不要 |

出典：[LONG] [MAP] [D0] [PHASE_A] [STRONG] [EPS] [DBA] [MAIN_REPORT]

元H8 mapは探索8本と固定union6各32本を分離した。D0・強化184・追加精度はその保存価格を共有するので、精度点数を独立実験数にしない。
強化184の通常signed・allowance1では、保存削減率は精度.05/.01/.005/.001/.0005順に
−8.0230/+33.2373/+31.4549/+22.9613/−50.8125%。これはPR union6/n32と当該価格版での結果である。[STRONG]

一方、固定主表n48は別のp8/q2・p7/q16候補で、同じ精度でも両点PR低費用を保存している。
候補・q・価格cohort・選抜対象が違うので矛盾ではなく、強化184の負方向を新主表で消したり、古いunion6で新主表を否定したりしない。
後続DBAで得たB0 p15/q8 S2・B0I p16/q4 S4の実測も既存証拠だが、今回新しい184全体の再会計へ組み込んでいない。[H8_CONFIRM] [DBA]

H2/H4/H6共通588決定論候補と118標本の高次PR確認はH8を含まない。
H8に高次決定論対照があることから、高次PRまで検証済みとは推論しない。[PF_COMMON] [HIGH_CONFIRM]

## 5. 希少eventと未保証事項

H4/H6/H8とも、有限価格標本と保存SEは全supportの母平均順位・母分散を認証しない。
H4主表のS2_MULTI/S4未観測、H6 K2の観測範囲、H8 n48のTaylor2/4未観測を、同じ希少event保証があるという意味にはしない。
K2でsupport外の次数と、K4でsupport内だが未観測の次数を区別する。[MAIN_REPORT]

H8の正式終了範囲は、固定p8/q2・p7/q16、r4/K4、T=.8、通常signed/allowance1、P0の
S0各48本とS2_SINGLE条件付きpilot各43本による**層別有限標本評価と残余費用の条件付き感度**である。
保存境界の残余混合平均/S0は約1276倍/888倍。ただし既知層の経験平均を固定した境界で、母平均保証でも実際の残余費用上界でもない。
S2_MULTI/S4_PRESENTの平均・分散は未知、0補完なし。48+43を無条件91本へpoolしない。[RARE] [CLOSURE]

C-1のbinary64整数比条件付きsamplingとcanonical層確率のengineering一致も、費用加重bias上界ではない。
追加singleだけで主要不確かさが解消する根拠はなく、multi/S4の有限pilotだけでも全母平均保証にはならない。
現在のO1に残余samplingを課すことは、正式終了方針に反する。[RARE] [CLOSURE]

新しい長時間・高次結果を仮に得ても、実signal誤差の厳密certificate、全lawのcompiled作用精度、未観測tail、
選抜後の母平均最良、状態準備、routing/noise/FT、一般分子への移送、最終energy/QPE/RPE総費用は未保証のまま残る。
H4/H6/H8を統一した分子サイズscaling実験にしない。

## 6. 不足の分類と実務上の扱い

CSVの `gap_class` は今回の不足整理であり、Phase 3のA–Eや正式採択statusを置換しない。
`original_status` と固定locatorを別欄で保持する。

| 分類 | 主な該当 | 現O1への対応 |
|---|---|---|
| EXISTING_EVIDENCE_AVAILABLE | 固定6比較、H8 map/5精度/強化184/追加精度/希少感度 | 固定条件と価格版を明示して引用 |
| METADATA_RECOVERY_NEEDED | H8 G04、H6/H8の限定seed/cohort監査 | 監査表現を強める際の限定回収。新計算ではない |
| SCIENTIFIC_REANALYSIS_NEEDED | 新しい統一費用比・後続価格を入れた再ランキングを要求する場合 | 現O1に不要。求める場合だけ保存値解析の別明示承認 |
| NEW_SCIENTIFIC_COMPUTATION_NEEDED | H8長時間・高次PRという新しい結果を主張する場合 | 条件付き将来候補。現在は未固定・未承認・未実行 |
| NOT_REQUIRED_FOR_O1 | 同数の全格子、残余層sampling、H8 B3/K6、統一size scaling・最終energy保証 | 必須作業へ追加しない |
| UNRESOLVED | 原profile/provider bytes・旧compiler全出力の回収可能性と必要性 | 不存在・完全一致を推測しない。現在の限定点主張を維持 |

## 7. 最小の次段階とSTOP

具体案は [H8_MINIMAL_NEXT_PLAN.md](H8_MINIMAL_NEXT_PLAN.md)、論文配置は [PAPER_IMPLICATIONS.md](PAPER_IMPLICATIONS.md)。
推奨は既存証拠でO1を構成し、H8長時間・高次PRは将来拡張として残すこと。
拡張を選ぶなら、全格子ではなく「同じ保存H8入力の、より長い時間でS2と高次PRが強いPFに対してどの費用競合を示すか」を判別する限定比較を候補にする。
新しいq/T/ε/標本数/seed/実行回数は今回固定していない。

技術検査は文書・CSV・保存token・固定Git参照の整合性だけで、科学的PASSや独立reviewではない。
既存checkerのCSV/Git/literal検査を再利用し、専用の小さい検査器だけを追加した。
科学pipelineを新設・起動せず、保護SOURCE/input/raw/QPY/grant/ledger/marker/STOPを変更しない。
元checkoutのHEADは照合時に不変だったが、共有作業中のstatus一覧の変動を観測した。本taskの書込先は独立worktreeの本ディレクトリだけで、元checkoutの変動は巻き戻していない。原本全rawのbyte監査や共有環境の凍結を行ったとは主張しない。

**ユーザー**：現在の有限・条件付きO1にH8を配置するか、H8長時間・高次PRの主張拡張を採るかを判断する。
**SERVER2担当Codex（次の指示がある場合）**：G04の既存公開可能metadataを回収・提示する。
**LOCAL Codex（次の指示がある場合）**：公開補完の固定SHA・本文を照合し、監査欄だけを追補する。
新しい科学解析・signal/sampling/compileはそれぞれ許可された固定範囲の明示承認が必要。重要な別レビューも科学実行とは独立した許可を要する。
**本資料のcommit/push・remote SHA/公開本文照合後、mandatory STOP。**

[CLOSURE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md
[D0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_h8_d0_saved_value_result_20261015.md
[DBA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_h8_dba_20261018.md
[EPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_h8_epsilon_boundary_run01_results_20261018.md
[EXCEPTIONS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/resource_applicability/track_a_h468_paper_display_methods/2026-10-19/input_exceptions.md
[G_REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/REPORT.md
[H8_CONFIRM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6eea5387a6e84d1e39a9adb3b61975acc74e200/docs/research/track_a_h8_new_pr_confirmation_results_20261019.md
[HIGH_CONFIRM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/docs/research/track_a_h246_reconstructed_confirmation_results_20261022.md
[LONG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[MAIN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/main_table_proposed.json
[MAIN_REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_h468_paper_evidence_integration_20261019.md
[MAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/docs/research/track_a_h8_resource_map_result_v1.md
[METHODS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_h468_paper_display_methods_20261019.md
[OPEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ce2b5f161e94567fec322d694fd8b01d03cd854/docs/research/pr_research_paper_evidence_synthesis_20261103/STATUS_AND_OPEN_QUESTIONS.md
[OPTIONS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ce2b5f161e94567fec322d694fd8b01d03cd854/docs/research/pr_research_paper_evidence_synthesis_20261103/PAPER_SCOPE_OPTIONS.md
[PF_COMMON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/docs/research/track_a_h246_pf_common_final_20261022.md
[PHASE3_GAPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/235e56ee7d5cac4682775a62bc2a8262f15047dd/docs/research/pr_cross_host_coverage_audit_20261102/coverage/GAP_CANDIDATES.csv
[PHASE_A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_h8_phase_a_run01_results_20261016.md
[PROTOCOL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/protocol_scope_status.csv
[RARE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/145b438c05b1888ab479081ef72462b93429f89a/docs/research/h8_rare_rte_saved_integration_results_20261021.md
[STRONG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md
[SYNTHESIS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ce2b5f161e94567fec322d694fd8b01d03cd854/docs/research/pr_research_paper_evidence_synthesis_20261103/RESEARCH_SYNTHESIS.md
