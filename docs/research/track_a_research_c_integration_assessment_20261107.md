# Track Aの現在地と研究Cとの関係：原稿構成の静的調査

2026-11-07 JST。担当：Codex主エージェント。静的資料調査と構成上の提案であり、重要科学レビュー、新規性判定、論文化GO/NO-GOではない。

**推奨は、両研究の独立性を維持し、必要な場合だけTrack AのDiscussion／Limitationsに表現依存性を1段落述べる案B。言及が不要なら案Aでよい。研究Cの数値表やDF最適化をTrack Aの主結果へ加える必要はない。主図変更不要。**

Track Aには通し原稿と完成した図・補足があるが、PM原稿、geometry原稿、H6原稿、後続のH4/H6/H8掲載packet、高次PF・長時間結果を一つに統合した正式最終稿は、今回の関連資料範囲では確認できなかった。まずこの版と採用範囲の区別が必要であり、研究Cによって原稿の不足を埋めるという関係にはない。

## §1 調査目的・資料一覧

### 1.1 範囲と資料の優先順位

Track Aの目的・成果・現存原稿を固定資料から確認し、別研究Cとの接続を比較した。新科学計算、保存数値の再解析・再集計・再ランキング、科学コードの実行・import、回路生成・compile、正式科学レビューは行っていない。数値は既存報告の引用に限る。以下の表は研究間の数値順位を新しく計算したものではない。

実際に読んだ正式運用原本は `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research_project_instructions_unified_2026-10-25.md`。`/home/AbeHiromu/projects/AGENTS.md`、作業元の`AGENTS.md`、`PROJECT_MAP.md`、`研究概要・現状.md`、`VALIDATION_STATUS.md`、`artifacts/validation_manifest.json`の関連部分も確認した。科学SOURCE・契約・raw・結果・grant・ledger・marker・STOPを保持する。サブエージェントは使用していない。

作業branchは `codex/track-a-research-c-integration-assessment-20261107`、独立worktreeは同名末尾の専用ディレクトリ。基点はGitHub mainと一致した **`546bca7ed2cece59b714b986a264d7b0b0871a01`**（以下S）。これは閲覧・集約snapshotであり、個々の科学SOURCEではない。変更対象は本報告書一つだけである。

[PROJECT_MAP][MAP]と[統一索引][INDEX]は、旧概要を記載時点の履歴とする。9月20日の概要・中央台帳だけで、10–11月の全系列の現在地を決めない。後続の固定報告、明示された採用記録、科学SOURCEの順に対応付ける。mainへの保存は研究方針の採択や科学再現を意味しない。

### 1.2 固定資料と役割

| ID | 固定commit／文書path | 役割・採用状態 |
|---|---|---|
| S | `546bca7ed2cece59b714b986a264d7b0b0871a01` | 今回の基点。後述のmain内文書はこの固定版 |
| A0 | S: `docs/research/track_a_shared_research_status.md` | 10月14日の正式情報ノート。RQとR09採用記録。H8進捗欄は後続結果で更新して読む |
| A1 | S: `docs/research/track_a_post_pm2_claim_evidence_map.md`、`track_a_post_pm2_manuscript_design.md` | PM主張C1–C6と図・章の設計。設計時の「図未生成」は後続原稿と区別 |
| M1 | `8e0814e70c14ecf526444fac8a2142799610dc96`: `docs/pr2_matched_accuracy_m1_b1_result_validation.md` | H4開発210候補の完了報告 |
| M2 | `054bb7b2b2ca788b47ebdc5c14c115f24c8b6379`: `docs/pr2_matched_accuracy_m2_transfer_result_validation.md` | 固定5構成の1.30 Å移送完了報告 |
| PM | `5a1adffad780f0ec4272f5e8bb94713f9ff0f2bc`: `docs/pr2_pm2_precision_resource_result_validation.md` | PM-2確定結果。PM-0/1の一次報告とSOURCEはA1のE2/E4/E5から接続 |
| G | `7c4483378047b57f514c47dad98905e3f2a9a756`: `docs/research/track_a_h4_geometry_results_20261013.md` | 6距離の旧4方式結果 |
| I | `98cdb35cd08469714f20d6cc3b982d9c80908f46`: `docs/research/track_a_h4_b0i_map_results_20261014.md` | B0+I実測追加。R09が条件依存性を中心に採用 |
| V2 | S: `docs/manuscripts/track_a_resource_study_v0_2.md`と同`supplement_v0_2.md`、`claim_audit_v0_2.md` | PMのtechnical resource/application note草稿。元は未commit原稿を保存収録したもの |
| V4 | S: `docs/manuscripts/track_a_resource_study_geometry_native_v0_4.md` | geometry/B0+I通し草稿。後続R09採用と、草稿内のレビュー前表記を区別 |
| H6V | `653d64bedba7516d4c17f2c7457cefbc80f877a2`: `docs/manuscripts/track_a_h6_resource_study_v0_2.md` | H6別条件の草稿。強化対照の後続結果を全て反映した最終稿ではない |
| A6 | S: `docs/research/track_a_h468_paper_evidence_integration_20261019.md` | H4 DF10/H6/H8固定独立確認の採用。科学freezeではない |
| DISP | S: `docs/research/track_a_h468_paper_display_methods_20261019.md`と対応artifact packet | 主表6条件、Figure 2A/2B、英語caption/Methodsの配置固定。最終原稿採用判断は残る |
| PF | `155529e20a8fac39e00786c40b57830b71203c4f`: `docs/research/track_a_h246_pf_common_final_20261022.md` | 共通588決定論候補の後続比較。PRは保存探索標本 |
| L | 同`155529…`: `REPORT.md` | H4/H6長時間B2、PF次数・q・prefix・r/K拡張。主表変更・主要claim採択は別 |
| B3 | `ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed`: `artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/REPORT.md` | B3の後続負結果。別SOURCEとSTOP |
| LAND | S: `docs/research/pr_repository_scientific_landscape_20261101.md` §2–3 | 上記の時点差を整理した既存総括。今回はTrack A部分のみ再利用 |
| P1 | `0d5ca357ccfd8fcc850bef800c5bcadfde2a153e`: `docs/research/track_a_b2_p1_shots_20261107/REPORT.md` | 直近の旧B2からの全18round条件付きshot拡張。価格は未取得 |
| P1D | `2583ce4957ed44e914ce6571e3e676830439dfa7`: `docs/research/track_a_b2_p1_strong_pf_20261107/REPORT.md` | 強PF 8 arm、3条件付きPASS／5数値STOP |
| C1 | `417fa459b823509be5c2010c5836e58955088282`: `docs/research/research_c_integrated_h4_100_results_20261106.md` | 旧辞書、PR-aware対ENTRY・公平retuning・同J/U D |
| C2 | `3015d4fb342d4d98b58e41d57553e75539a07655`: `docs/research/research_c_h4_relocated_downstream_results_20261106.md` | 線形Z再配置後の順位・同J/U D・未解決統計診断 |
| C3 | `77ded5c12db21572933653cb26758436da3d9ecd`: `docs/research/research_c_baseline_comparability_20261106.md` | RC-DF/SCDF/UWCの既存調査。再調査せず引用 |
| C4 | `91417fb3ef75e7eec0849ad39f0649d170754664`: `docs/research/research_c_h4_strong_d_relocation_results_20261107.md` | 保存CDF8/XDF8＋共通LD2再配置強Dの確定比較 |

C1–C4、M1、PM-2、V2/V4、DISP、L、B3、P1D、H6VはGitHub接続でも固定本文を取得した。ローカルGit objectとのblob照合対象と公開後の確認は§12に記す。既存先行研究報告を引用することと、そこで引用された全外部論文を今回独立に再調査したことは区別する。IDEに開いていた「新A／PR Reference／Track C2」は今回の研究Cの指定ではなく、統合対象に加えていない。

## §2 Track Aの研究目的

### 2.1 中心に据える1–3文

**固定DF Hamiltonianと保存参照状態について、残差切断・全決定論・部分ランダム化・二体random-dominantを同じ有限時間複素信号精度で比較し、bias、RTE normalization、測定shot数、実compile回路費用の競合が資源順位と適用域をどう変えるかを調べる研究である。中心は、PRが有利になる条件と、対照・精度・geometry・PF政策でその利点が変わる境界を、固定候補内の資源地図として示すことにある。未知系の最適selector、新PRアルゴリズム、一般的優位、最終energy/RPE資源の認証を中心主張にしない。** [A0][A0]、[V2 Abstract/§1][V2]、[LAND §2–3][LAND]

9月の出発点には全RPE roundの期待総compiled cost最小化がある。しかし、現存するPM・geometry・H6原稿の主要な実証taskは有限Tの`z(T)=〈ψ|exp(−i H_DF T)|ψ〉`である。この二つを同一の達成済み成果として書かない。

### 2.2 定義・変数・評価量

| 項目 | 現存原稿に沿う定義・役割 |
|---|---|
| 固定表現 | `H = scalar + H_one + Σ_l H_l`。DF rankとfragment順は各研究系列内で固定。rankとprefix長`L_D`を区別 |
| 決定論側 | one-body correctionと先頭`L_D`個のDF fragment。scalar/identityはその系列の規約で一回保持 |
| ランダム側 | 残りの二体fragmentを回転基底I/Z/ZZへ展開したtail。identityを分けたcanonical finite paired-Taylor RTE。qDRIFTとはしない |
| B0/B0+I | B0はtail discard。B0+Iは既知tail identityのみ戻して非identity tailをdiscard。線形Z再配置とは別操作 |
| B1/B2/B3 | B1＝全DFの決定論PF、B2＝中間prefix＋tail RTE、B3＝二体prefix0だがone-bodyは保持。B3を全H純RTEと呼ばない |
| 比較変数 | 主に残差処理、`L_D,q,δ=T/q,r,K`、signal精度ε。後続ではPF公式、T、強化control/合成・対照集合も比較。J/Uの再fitを中心変数にしない |
| PF次数 | PM原稿は二次を固定。H6草稿ではPR S2対full S2/global Yoshida4。後続ではPR自身のS2/Y4/m5も比較。高次化のbias改善を総資源改善と同一視しない |
| 適格性 | 例：PMでは各軸の`h_a=ε/√2−b_a>0`。`N_a=ceil(2B² log(2/α_a)/h_a²)`。後続native系列の経験的u/B_upperは別契約 |
| 主指標 | `G_RZ=Σ_a N_a C̄_a,RZ`、状態準備P=0。Nは解析的十分shot、C̄は古典trajectoryからのcompiled費用。実量子shotsの総費用実測ではない |
| 補助指標 | CX count、RZ/CX depth、total depth、size、仮想共通P感度。RZをT/Toffoli数・物理runtimeへ読み替えない |

理論・数値・回路の接続は「既知PR/PF/RTEとshot規則 → 候補固有の信号bias/B/適格性 → full測定wrapperのcompiled費用 → shot加重資源」である。既存PRに対する追加価値の主張は、実装条件を明記した比較・強い対照・固定移送・独立確認・資源因子の説明に限定される。PR原理やcost×shots原理の発明としない。C_use等の経験的係数、engineering SE、有限標本の観測は数学的保証と区別する。[A1][CLAIMS]、[V2 §2/6][V2]

## §3 Track Aの科学的な現在地

### 3.1 完了した結果と原稿への採用

下表の「完了」は元報告の固定scopeの実行・文書化が完了したという意味であり、外部独立再現、immutable CI、母平均winner、論文完成を意味しない。

| 結果・条件 | 固定結果／科学SOURCE | 保存された成果・限界 | 原稿との関係 |
|---|---|---|---|
| M1 H4 1.00 Å/STO-3G/legacy DF12/T=.8/S2、210候補、q1/2/4/8、random各32 | 結果`8e0814…` [M1][M1]。SOURCE `33f436bb3a7d5b9cefa23604bb22c8d1fb17cd62` | B2 L_D3/q1/r4/K2のG=130,774,896.65625 RZ、6指標ParetoはB2二件。r/Kの厳密winner未確定 | V2主結果の基礎。旧selectorのprimary regret0とPareto欠落1件を区別 |
| M2 H4 1.30 Å、同basis/rank/T、事前固定5構成のみ | [M2][M2]。SOURCE `2978e2fea672b7a1ff20cac74269ec9a610159dc`、launch `40a11d02a67954175686b2e532b83c7953ec8316` | `TRANSFER_SUPPORTED`、B2二件がpoint Pareto。元ε=.05で支持。移送先method再最適化ではない | V2 §5・図3に採用 |
| PM-0 帰属・same-R | [既存報告][PM0]、入力snapshot `b6e65c6123475add5e620ec1064f361378bead95`。SOURCE/保存表はA1 E4 | 保存値の事後解析。q効果の因果分離や未知系selector完成ではない | V2 §4.2図4・補足S3 |
| PM-1 近接discard L_D4/5×q1/2/4/8 | [一次報告][PM1]、結果入力収録`194cc604b90c56a0e7e949b91b064a4bcfc846da`。SOURCE `fd7552edc0334ccf57ecf501a128c85c8d22822a` | 8 signal/16 wrapper完了。最良登録B0=229,718,060。追加後も上記B2点値が43.07%低い（既存報告の削減率） | V2 §3・図1。全prefix/強合成/高次対照一般への優位ではない |
| PM-2 development218＋使用済みM2固定5、ε=.005–.1の302点 | 結果`5a1adffa…` [PM][PM2]。SOURCE `324435d77b6642dbd44e8d1f178420daf62e77ed` | 保存値POSTHOC。開発ではB2のq4→2→1、移送5では厳しいε側30点がB3、残り272点B2。同じ32標本を共有 | V2本文・4主図・補足。302独立実験としない |
| H4 geometry追加6距離 .70/.80/.90/1.10/1.40/1.60 Å、DF12/T=.8/S2 | 結果`7c448337…` [G][GEOM]。科学SOURCE `d0726e946f89f8c30a08d383fb8ff48b363629ec`、dataset `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525` | 元4方式では6×302全点B2点最小 | V3の結果。追加B0+I後の結論へ無条件拡張しない |
| 同geometry＋B0+I 120構成/240軸 | 結果`98cdb35c…` [I][B0I]。SOURCE `924b779c2bc6f6e844d6283c355f91c73aed6445` | 234新compile＋照合済み6再利用。拡張集合はB0+I397/B2 1415点。finite-domainの点値で勝率ではない | V4主図1–4/S1。A0のR09は条件依存性を主・件数を補助として採用 |
| H6 1Å/STO-3G/tol-only DF19/T=.8、PR S2対full S2/S4 | 探索`df99d6ee28942df8d12aac3c4db7cd202df55d77`、確認`321cd8c3066194fe98741cbebbcdba1515d1deaa`。科学SOURCE `5a903aea2492c2d211c1e15e7737df0d50694c12`、確認execution `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7` [A0][A0] | 探索8と独立32を分離。希少S2条件付き8本の補助費用は局所scope。後の強いB0I対照まで全てPR優位とはしない | H6Vに採用済み。後続強化対照・独立Pは別結果 |
| H4 DF10/H6 DF19/H8 DF28、1Å/T=.8、ε=.05/.0005、固定6確認 | [A6][A6]。結果H4 `821cc0f6ced57168293ea166644024867b064ffb`／H6 `3cccfcb2051dacf29e91473e2e1152acef1574bd`／H8 `e6eea5387a6e84d1e39a9adb3b61975acc74e200`。SOURCEは同報告の`3c664ca4`/`bf0c26da`/`745a6745`と原manifest | H4粗：PF低費用・方向NOT_REPRODUCED、細：PR。H6粗：PF方向REPRODUCED、細：PR。H8両点PR。n=1108/78、48/104、48/48。母平均保証なし、H8公開詳細監査に制限 | 主表6条件の掲載配置はDISPで固定。V2/V4本文へ全統合済みとはしない |
| H2/H4/H6共通PF588候補、rank3/10/19、S2/Y4/m5、q≤8、T=.8、5ε | [PF][PF]、固定結果を含む`155529…`。H2 SOURCE `1b35f9ac2efab4c15bc00ee2d3c81643aca422f9`、H4 `e2179f68f6979f3ad26392bff877fb422ddfb744`、H6 `bdf6772aade7063c65d69f24dfdc5e303a2819a2`（中断前取得SOURCE別記） | H2全5点PF、H4/H6粗い3精度PF・厳しい2精度PR。PR側は保存探索標本 | 後続の強対照拡張。PM旧原稿の比較集合を置換しない |
| H4/H6長時間B2：rank10/19、T=.8/1.2/1.6/2、5ε、S2/Y4/m5 | [L][LONG]結果`155529…`。signal SOURCE `e417ffc9f52b13c18f5493193ce510d858664259`、native `27ebabdd2943a4de787ec8b976c38e06fda5a497`、拡張/継続は同provenance | 11,632 signal構成に対し価格642候補。取得域33/40でB2低点費用、その33方向は別標本でも再現。高次対S2の一例は確認で逆転 | 完了した追加研究。元報告はTrack A主表・Figure2不変更、主要claim採択をユーザーへ戻している |
| 同長時間B3、L_D0・one-body保持 | [B3][B3]結果`ebe9e2aa…`。signal SOURCE `e47696e86c2f98f52f390ed084bb1bfeff4bd197`、native `638e0a989b4471a075549ccb58617b1030e6748f`、確認/復旧SOURCEは同scientific_provenance | B2/PF envelopeより低い点費用0/40。有限取得域の負結果、全域最適性ではない | 追加研究の結果・図は完成。主原稿への採用確定とは別 |

独立確認6条件の短縮SOURCEの完全SHAは、H4 `3c664ca4a9f003880b9f6d56f2a0e55ed5b56603`、H6 `bf0c26dae4477de5fe7264a7c91e077661ef9b38`、H8 `745a674546f2cb6258526fbfde1d9a00601689f8`。Figure2のrender SOURCEは `3de17acd912a28a1df9905b2f3c3e2e3e15aa0dc`。結果commit・render SOURCE・科学SOURCEを区別する。

H4/H6/H8は同じtol-onlyという表記でもinput生成・solver・normalization・state政策が完全同一とは確認されていない。単一size-scaling曲線へpoolしない。探索と確認、H8主表と同一cohortを使う機構図、legacy DF12とDF10を二重計上しない。[DISP][DISP]

### 3.2 AX系列・準備・保留・STOPの境界

| 系列 | 今回確認できる現在地 | Track A中心との関係 |
|---|---|---|
| AX-0/1/1b | 契約整理・保存値モデル監査は完了。RQ-R主、RQ-P1補助、RQ-P2未達・完成必須外という既存採用記録 [A0][A0] | 資源比較の目的と情報分離を定める。FEW改善だけを中心主張にしない |
| AX-2A／AX-2B初期H4 pilot | native/control/sector/有限RTE接続の技術基盤。H4 v5は限定技術pilotとして受入、総uや測定込み優位の認定ではなかった [AX][AX] | 後続H4/H6/H8へ接続した基盤。AXという系列名全体を「準備のみ」または「科学全完了」とまとめない |
| 10月14日のH8 RUNNING表記 | その時点の観測。後続独立確認・主表をA6/DISPが収録 | 現在live稼働の根拠にせず、旧「最終結果未取得」を最新結果へ残さない |
| H8希少層の後続 | 既存LAND §2.3が10/21の限定scope採択・追加sampling不要を記録 [LAND][LAND] | 全law保証は残らないが、希少層が全て未取得という旧記述も採用しない |
| 全RPEモデル | 9月後続opt2に全18roundのproxy集計あり。約4.858%のpartial点利益は局所5%等の不確かさと重なるという既存総括 [LAND §3][LAND] | 「総費用の集計は一度もない」は誤り。公平な最終energy資源の認証完了とも言わない |
| 11/7旧B2 P1接続 | [P1][P1]：DF12、元H状態、L_D3/S2/r4/K2/δ=.02/m0..17、ε_E=.00015936001019904 Ha。branch/energy前段と条件付き7,805,784 shots。SOURCE `05015ae5ceeacc747f88d848cce8c3558a4881da`。価格・総RPE費用null | 完了した条件付きshot拡張。旧T=.8のcompiled価格を無条件乗算しない。PM論文の主結果に未統合 |
| 同強PF接続 | [P1D][P1D]：SOURCE `a7940b58c6cb939d77e9f8e44d214b3787b7e101`、Full S2 s1/s2・Y4 s1は全18round条件付きPASS、5 armは固定数値照合でSTOP | PF位相cap超過による物理的失格ではない。強PF branch/shot/価格/総費用未完了 |
| 強PF数値診断 | 固定`e217d145bd99c2b34302017390ecb57987c06b90`の[既存技術報告][DIAG]：人工例による増幅機構、実H4主因未確定、修復なし、3 PASS/5 STOP不変 | 技術調査完了を再資格PASSにしない。今回は同報告を静的参照しただけ |
| 旧失敗・中断・STOP | 旧S0入力不一致、H4/H6旧pilot/MemoryError、export不一致等をA0/LANDが保全。後続成功は別identity | 本調査はどのSTOPも解除しない。別研究Cの1.15 Å STOPも別に保持 |

AX後続、PF次数、長時間、RPE接続は、Track Aの問いを補強・一般化する仕事である。ただしPM事例研究を構成するために全てが必須だったわけではなく、後続の全成果を一つの論文へ入れる採択も確認していない。Track Aを単に「H4 B2全勝研究」と要約すると、強化対照と負結果を落としてしまう。

### 3.3 古い記述との相違

1. [9月概要][OLD]の「最終cost保留」は当時の段階。後続モデル集計・有限T資源地図・全18round条件付きshotを別に記す。ただし最終energy資源の未認証は維持。
2. M1報告のq=8→可変qの大きな改善を、qのみの因果効果としない。PM-0とV2のsame-domain/selector訂正を優先する。
3. 「H4全点B2」は元登録集合の結果。B0+I拡張後は397/1415であり、旧集合の主張を拡張集合へ移さない。
4. V4の「H8 feasibilityのみ／レビュー待ち」は草稿作成時点。A0のR09、A6/DISPの後続採用を併記し、本文が自動更新済みとはしない。
5. 古いDF screening失効・grouped-UWC prose-onlyは中央台帳のまま保全。後続Track A検証やC3で取得済みの著者UWCコードと混同しない。

## §4 Track Aの論文構成

### 4.1 原稿の版・完成状態

| 原稿・packet | 実在する完成部分 | 正式採用・未完成の区別 |
|---|---|---|
| PM v0.1 → V2 v0.2 | 通し日本語本文、主図4、補足S1–S7、補足図S1、claim audit。technical resource/application note | V2の当時の編集判断は`CLOSE_AS_TECHNICAL_RESOURCE_NOTE`。著者・投稿先未定。mainの収録表はV2原稿を**UNCOMMITTED原本**、関連HEAD `b2e1bf65…`として記録する。V2のbytesがその古いHEADでcommitされていたとは言わず、公開固定版Sを参照する [保存来歴][COLLECT] |
| geometry v0.3 → V4 v0.4 | 通し草稿、B0+I実測の主図1–4・補助図S1、PNG/SVG/PDF、provenance bundle | V4表題のレビュー前labelは履歴。R09が候補B中心を採用した記録はA0にあるが、R09後の最終文面へ全面改稿済みとは確認できない |
| H6V v0.2 | 要旨、§1–7、保存結果表、希少次数と再現性・限界 | H4とは別条件のcase study。後続B0I/独立P/長時間まで反映した最終稿とはしない |
| A6/DISP | 固定独立確認主表6条件、Figure2A/2B、英語caption/Methods、入力例外・H8証拠限界 | ユーザーが配置を固定。最終evidence freeze・最終原稿採用・詳細科学レビューは未開始と明記 |
| PF・長時間L/B3 | 確定報告、比較表、既存図 | 研究結果と表示資産はある。PM/V4の章へ統合済みという証拠はない |

したがって「原稿なし」でも「統合論文が完成済み」でもない。複数の本文草稿と掲載用資産があり、**どの証拠集合で一つの原稿を完成させるかが未固定**という状態である。第1研究の高次PF論文やPauli論文草稿を、この不足を埋める正式Track A最終稿へ読み替えていない。

### 4.2 実際の章構成

| 論文要素 | PM V2の現存構成 | geometry V4／後続packetとの相違 |
|---|---|---|
| 仮題 | 「DF-prefix時間発展における精度条件付き資源比較：残差切断・ランダム補完・固定構成移送の事例研究」 | V4は「既知scalarを保持する対照との比較による部分ランダム化時間発展の有限signal資源」 |
| Abstract | 固定H4 task、近接discard後43.07%、302点の設定変化、元精度移送と別精度不適格 | V4は6距離・B0I397/B2 1415・対照依存性 |
| Introduction / Related Work | §1、§1.1、§1.2。既知PR/DF/測定費用を認め、限定resource map・stress test・適用条件の3寄与 | V4 §1は既存v0.3の枠組みを継承。独立Related Work節はない |
| Method | §2.1固定H/state/B0–3、§2.2 bias/B/shot、§2.3費用/SE、§2.4候補域・事前/事後 | V4 §2対象・compiler、§3 B0I/制御位相、§4精度/shot/SE |
| Numerical / Resource設計 | 1Åのdevelopment218、1.30Å固定5、T=.8、S2、同じ保存32価格標本 | V4は6geometry×302精度、ordinary/structured。A6/DISPは別native/symmetric-directionalと独立確認 |
| Results | §3 Development、§4 Precision/resource competition、§5 Frozen transfer | V4 §5元集合/追加実測、§6因子と再選択、§7選択後SE・希少次数、§8別H4 anchor/H6/H8 |
| Discussion / Limitations | §6、§6.1不確かさ・一般化、§6.2二次/表現/合成/P scope | V4は§7–9に分散。独立したDiscussion見出しを勝手に新設しない |
| Conclusion | §7、有限taskでの条件付き資源比較 | V4 §10は採否前草稿。後続採用情報は外側のA0/A6/DISPで確認 |
| Supplementary | S1比較台帳、S2追加metrics/SE、S3 selector/欠測、S4 P、S5証拠来歴、S6再表示、S7編集採用とSTOP | V4は補助図S1とbundleを持つが、V2 S1–S7と共通の完成補足巻とは確認できない |

H6Vは要旨、§1研究の問い、§2固定対象/費用、§3探索と確認、§4費用と測定、§5希少次数、§6 H4/size拡張、§7結論/限界、データ/再現性という別構成である。

### 4.3 Figure・Tableと主張・SOURCEの対応

番号は各原稿・packet内の番号であり、**統合後の共通番号は未確定**。同じ「図2」を同じ図と扱わない。既存図の内容・存在・captionを確認し、描画・数値更新はしていない。

| Figure / Table | 示す結果・対応主張 | 科学SOURCE・保存結果 | 本文での役割・状態 |
|---|---|---|---|
| V2 図1 `figure_1_development_cost_components` | ε=.05の4方式、shots/単価/workを分離、近接discard後のB2点利益 | M1/PM1/PM2、結果`5a1adffa…`。描画資産は`artifacts/resource_applicability/track_a_manuscript_v0_1/2026-10-05/` | §3主図。生成済み、単なる設計案ではない |
| V2 図2 `figure_2_development_precision` | 218候補・302表示点、B2内q4→2→1 | PM2 SOURCE `324435d7…`、同結果。保存値の事後感度 | §4.1主図。独立302試験や厳密切替ではない |
| V2 図3 `figure_3_frozen_transfer_precision` | 固定5構成の適格性と費用順位、移送支持と別精度の限界 | M2 SOURCE `2978e2fe…`＋PM2保存解析 | §5主図。元blind transferと使用済みデータ感度を分離 |
| V2 図4 `figure_4_same_R_competition` | 同L_D3/K2/T=.8/R=8でbias改善と費用増の競合 | PM0 `same_R_candidate_comparison.csv`、A1 E4 | §4.2主図。q単独の一般法則ではない |
| V2 図S1／補足S4 | 仮想共通P感度、候補集合差 | PM2 `P_envelope.csv`、PM0 | 補足。実準備回路ではない |
| V4 図1 `figure_1_measured_method_map` | B0I397/B2 1415のgeometry/precision領域 | G/I、SOURCE `d0726e94…`/`924b779c…`、`track_a_h4_b0i_manuscript_integration/2026-10-14/figures/` | §5主図。R09の条件依存性を支える |
| V4 図2 `figure_2_measured_cost_ratio` | 再選択したB0I/B2費用比 | 同G/I。独立確認ではなく保存32標本 | §5主図。最良deterministic family比とB0I単独比を区別 |
| V4 図3 `figure_3_shot_circuit_factors` | shots比・有効単価比・総費用比 | 同G/I保存会計 | §6.1主図。compilerだけの因果効果ではない |
| V4 図4 `figure_4_paired_sample_comparison` | 点差とpaired ±2SE | 同G/I、H4の同じ32標本 | §7主図。formal CI・母平均winnerではない |
| V4 図S1 `figure_S1_matched_phase_bias_shots` | 同prefix/qのB0→B0I、位相/bias/shot | 同I、120 matched構成の保存資料 | §6.2補助図。family再選択効果と分ける |
| DISP 主表1・6比較 | H4 DF10/H6/H8独立確認の点方向 | A6の各独立P SOURCE、nを維持 | 掲載配置固定。V2のdevelopment結果とは別 |
| DISP Figure2A `figure2A.png` | legacy/探索の軸費用・shot・総費用因子 | 原図commit `eb33de67475e7d33eed56801b74a2a277147e4be`、render SOURCE `3de17acd`、後続因子完成`1ff6ac0b6ffed6231dac5f655275b620598d0201` | legacy/探索の機構説明。新H4 DF10/H6 Pへ因子を転用しない |
| DISP Figure2B `figure2B.png` | H8独立n48の2比較の分解 | 同原図、H8 P `e6eea538…` | 主表の同じ標本。追加独立証拠ではない |
| L `cost_vs_time`/`selected_delta_vs_time`/`selected_formula_grid`/`fixed_delta_time_dependence` | 時間・次数・刻みと費用の依存 | Lのscientific_provenanceと取得済み候補 | 生成済み追加研究図。最終本文の番号未採択 |
| B3 `cost_vs_time`/`selected_q_delta`/`R_outer_B_N_C_G`/`expanded_best_method_map` | B3の負結果、政策とnormalization/費用 | B3の別SOURCE・保存報告 | 生成済み追加研究図。主原稿への採用未確認 |

DISP原図pathは `artifacts/resource_applicability/track_a_mechanism_figure2_a82_readiness/2026-10-19/figures/`。公開captionが旧A82注記を描画時点の履歴と説明しており、元画像の上書きを要求していない。[掲載caption][CAPTION]

### 4.4 ストーリーの一貫性と残る編集問題

V2は「同task・同実装classでのresource map → 近接discard → 要求精度 → 固定移送」という一貫した順序を持つ。V4はbaselineを強化したときのgeometry/精度依存性を前へ出す。A6/DISPと長時間結果は、この主題を支える別証拠だが、状態・DF・compiler/control・標本来歴を追加説明する必要がある。

すでに複雑な点は、DF12/DF10の区別、B0/B0I、ordinary/symmetric-directional、探索/確認、posthoc/blind、希少層、古いdraftと後続採用のずれである。研究CのJ/U・再配置・13round・CA資格まで入れる前に、Track A内でどの原稿系列を採るかを決める方が構成上の優先事項となる。これは科学的採否を新しく決定したものではない。

## §5 研究Cの確定成果

### 5.1 科学的な問いとSOURCE

研究Cは、一般対称signed core `J_t`と実直交frame `U_t`の近似DF表現を、tensor fidelityとtail massに関係する正則化目的で調整し、物理DF予算・PF/RTE・RPE・native価格へ接続した研究である。固定frame mask LD2、number-CDF8、J/U有限更新recipeが中心で、Track Aの固定表現内の構成探索とは変数が異なる。

| 固定資料 | 科学SOURCE・来歴 |
|---|---|
| C1旧辞書 | `artifacts/research_c_integrated/2026-11-06-picklefix/technical/SOURCE.json`、SHA256 `8113f23e66c5eca3c09a8ed1c91fc58c01eeaa91d0c07850b685a20c816539eb`。17本の保存fitと1本の新SOURCEを組み合わせた来歴を維持 |
| C2再配置 | 科学SOURCE `cfc64b70861a6482dddd042da8a110abebd6ad7a`、`artifacts/research_c_h4_handoff/2026-11-06/technical/SOURCE.json`。元a670契約と検証済み288標本の再利用＋512新標本という引継ぎ |
| C3静的baseline調査 | 基点`a67036a6baafc00deaa38a1b0b801d313d519ad5`。既存目的・API・論文式の対応調査。新baseline科学実行ではない |
| C4強D | 科学SOURCE `e6652ab841aae5ae700286bf1ab249ad4fb85461`、68-file SOURCE manifest。保存CDF8/XDF8、固定LD2再配置・強Dを一回実行 |

共通H4条件は直鎖1.00 Å/STO-3G、4 spatial/8 spin、Nα=Nβ=2/sector36、own-DF ground oracle、準備費用除外。下流はpartial ordered S2/s1/h=.1/K0、m0..12、CA=.00159 Ha、DF reserve CA/8=.00019875 Ha、β_total=π/6、β_stat=π/12、failure=.01/候補・axis .01/26、P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx、位相保持・fenced serial・shell込み。比較対象の各近似DF Hamiltonianは候補ごとに異なる。この条件表だけでTrack AのHamiltonian/stateと同一とは認定しない。[C1–C4][C1]

### 5.2 固定報告から引用できる限定結果

| 段階 | 保存された主結果 | 同時に保持する限定 |
|---|---|---|
| C1旧辞書 | C0/64更新でPR-aware対ENTRYは共通r32で51.08%、公平retuning後45.47%低い条件付き総RZ点値。3start×2予算の公平retuning6比較で同方向 | PR/ENTRYは**DF目的arm**。両側とも下流PRを利用。36 checkpointはDF予算PASS/J一次MET/U一次NOT_MET。大域最適・同時一次収束の比較ではない |
| 同J/U D | C0 retuned PR/D点比.891018、個別名目95%比区間[.874508,.907528] | 同じ候補固有J/UのD。全baseline同時優位・強CDF/XDF一般への比較ではない |
| C2線形Z再配置後 | P/r32=92,464,828,249対E1/r32=86,352,772,151 RZ。公平retuning P/r16=87,514,417,699対E2/r16=75,621,395,471 RZ。主A/Bとも`NOMINAL_MODEL_ENTRY_LOWER` | 旧45.47%を新辞書へ移せない。再配置後目的で再fitした比較やZだけの因果実験ではない |
| C2同J/U D | ENTRY E2/r16対同J/U D=105,217,244,042 RZ、PR/D点比.718716748、個別名目モデルでPR低値 | P/r16は影響度診断で`UNRESOLVED_DISTRIBUTION_DIAGNOSTIC`。best-policy副比較も未解決。主97.5%差判定と個別95%診断は別 |
| C4保存強D | ENTRY E2＋PR/r16の保存75,621,395,471に対し、再配置CDF8＋強D=108,561,283,488、再配置XDF8＋強D=101,780,388,260 RZ。登録判定`E2_POINT_BENEFIT_REMAINS` | 固定2候補・LD2・D S2/s1のideal lane点比較。新しい比・CI・p値は作らない。全DF/PFの最良性ではない |

C4のDはideal laneと数値合成誤差laneで整数shots・費用が一致したが、合成誤差は非零である。Dの数値laneは`NUMERICAL_PILOT_CA_PASS`、E2＋PRは全law/implemented-law/価格移送の証拠不足で数値lane未資格。**DのlaneをPRへ移送せず、ideal laneでの条件付き点比較だけを保持する。** actual chemical accuracy、厳密丸めenclosure、実機・FT・準備込み優位、母平均保証は未資格。[C4][C4]

C1の1.15 Åは`INDEPENDENT_TERMINAL_VALUE_GRADIENT_MISMATCH`でSTOPし、移送結果はない。H4 1.00 Åの結果を一般geometry・H6へ拡張しない。今回その原因調査・修復・再実行はしていない。

### 5.3 目的関数・再配置・既存baselineの関係

以下は[C3 §2–5][C3]の既存整理の引用であり、今回の新しい数理導出ではない。

- 旧PR-aware目的はtensor fitと回転Z/ZZ atomの平滑mass二乗。旧ENTRYは別の固定entry重みで、同じ目的関数ではない。
- 線形Z再配置はtailのZを既存一体headへ移し、対応scalarを一回加え、元frame identity/ZZを保持する**全Fock空間のHamiltonian恒等変換**。同じHでもPF split・誤差・shots・価格が保存されるとは限らない。
- 再配置後の非平滑`M_ZZ²`は、tail mask、ordered diag重み1/4・off重み1/2のweighted entrywise L1和の二乗であり、RC-DF本文のweighted L1二乗族に含まれる。保存ENTRYのdiag5/4・off3/2とは異なり、一つのstrength倍率で一致しない。平滑化、head mask、scale、solver、物理精度選定まで原法と同一ではない。
- SCDFのsector symmetry shiftは全Fock Hを同じにする操作ではなく、target sectorの補償付き同値を扱う。UWCはPauliの可変head/tail、OO/BLISS等を扱い、固定frame LD2のJ/U fitとは異なる。

この既存対応を、研究C独立新規性のGO/NO-GOへ格上げしない。PRが条件付きで有益なことと、PR-aware DFの目的関数自体が新規・優位であることは別の問いである。再配置後にENTRY由来が低費用でもPR対Dの点利益は残り得る、という固定結果がその区別を具体化している。

## §6 Track Aと研究Cの比較

### 6.1 比較表

| 比較軸 | Track A | 研究C | 重複・接続・独立の判定 |
|---|---|---|---|
| 科学的な中心課題 | 固定DF内で構成・精度・測定込み実装資源・適用域を比較 | DF表現J/U・正則化・線形Z再配置とPR下流資源 | 資源設計で接続するが主課題は独立 |
| 新規性を主張する対象 | 限定resource map、baseline stress、適用条件・独立確認の実証 | 目的/表現設計の追加価値が検討対象。正式独立新規性未判定 | 同じ新アルゴリズムclaimへまとめない |
| Hamiltonian分割 | 固定fragment prefixとtail、B0/B0I/B1/B2/B3 | 元順LD2、tail Zをheadへ移したsplit、同J/U D | head/tail概念は共通、同じL_Dでも定義差 |
| DF表現 | legacy DF12、後続tol-only10/19/28等を系列別固定 | number-CDF8の一般signed J/U、保存CDF8/XDF8も対照 | Cは表現自体を変える独立自由度 |
| 最適化変数 | method/prefix/q/r/K/PF次数/T等 | J/U・strength/start/有限更新、下流rの公平retuning | 相補的。共同最適化は今回の既存主張でない |
| PF次数・実行政策 | PM S2、後続full/partial S2/Y4/m5、可変q/T | 主H4はordered partial S2/s1/h=.1/K0、13round。D全frame S2/s1 | 政策の固定条件が異なる |
| PR/RTE・RPE | finite-RTEのbias/B/shot、主原稿は有限T。RPEは別拡張 | PF/RTE・13round RPEの条件付き会計 | 同原理・共通部品あり、完成資格は移せない |
| 物理精度の扱い | 主に保存H_DFへのsignal精度ε。元化学HとのDF差・ground certificateは別 | 元物理Hへのsector norm＋guardをDF予算に配分、own-DF ground仮定 | εとCA[Ha]、状態・targetが異なる |
| 回路資源指標 | full cos/sin wrapperのΣN C̄、6指標、P=0 | W×mean kernel＋shell、6指標、P=0 | RZ等は重なるが会計単位は別 |
| Native compiler | Qiskit1.3/opt1/seed17/rz,sx,x,cx。ordinary/structuredとsymmetric-directionalを区別 | 同名P0設定、StrongBuilder/relative Gaussian、phase保持、fenced serial | 基盤再利用。profile名だけで同価格としない |
| 対象分子・geometry | H4 1.00/1.30 Å、追加6距離、別H6/H8、限定H2等 | 今回確定成果はH4 1.00 Å。1.15 Å移送STOP | H4名の一致は同一入力証明でない |
| 主要な数値結果 | B2点利益例、強対照で逆転、精度・固定移送の適用域、高次/長時間の条件依存 | 旧PR-aware低費用→再配置後ENTRY低費用、ENTRY＋PR対固定強D点利益 | 同方向の一般優位へ合算しない |
| Baseline | B0/B0I/full S2/Y4/m5、random-dominant | ENTRY objective、各候補同J/U D、保存CDF8/XDF8＋再配置D | 強対照という問題意識は共通、対照集合は独立 |
| 一般化・移送検証 | 元精度1.00→1.30 Å固定5支持、使用済み精度感度、別系確認 | 1.15 Åの移送完了なし | CをTrack A移送の追試にしない |
| 理論的な貢献 | 既知手法・十分shot式の接続と限定実証。新一般最適性定理でない | 表現変換・目的対応の整理、有限recipeの観測。既存L1族との関係あり | 理論を共有しても独立新規性を同一視しない |
| 論文としての主張 | 固定taskで精度と測定込み資源の競合・適用条件を説明 | 表現依存性とDF目的/下流資源の関係。論文化の最終claimは未決定 | 当面独立が明確 |

### 6.2 D1–D5の確認

**D1：PR資源評価の再利用。** 原理だけでなく部品も共通である。C4固定版の`c_pr_aware_df_stage_b.py`は既存`c1_stage2_native`/`c1_stage3_connection`/`c1_stage3_reference`を呼び、native経路は`StrongBuilder`・既存basis provider・`trotterlib.rte.event_unitary`を使用する。`c1_stage3_connection.configure_rte`は`trotterlib.rte.make_rte_config`等を利用し、`c1_stage3_accuracy`は既存RPE horizonとHoeffding helperへ接続する。再配置adapterは既存native modelとsplitを構成し直す。[C4固定SOURCE][CSOURCE]

したがって研究Cの資源会計全体をTrack Aと独立に新発明したと扱うのは不適切である。一方、C固有のgeneral-J codec、元g補正、物理DF admission、独立CAR参照、candidate固有ground、13round/shell/fenced価格・統計資格を追加している。**再利用した実装基盤と、同じ実験条件・同じ科学的保証は別**である。今回このコードは静的に読み、関数を呼んでいない。

**D2：DF最適化の独立性。** V2/V4は固定DFという範囲を明示して成立する設計である。J/U fit・再配置・solver比較はその比較を再現する必須変数ではない。DFが資源へ影響するという限界・別研究の接続としては有用だが、中心Methodへ導入する必要はない。

**D3：強baseline。** C4は「保存DFにも同じLD2再配置を与えた固定強Dに対し、ENTRY＋PRのideal条件付き点利益が残る」という別条件の補助事例になる。Track AのDF12/DF10・finite-T・保存state・full wrapperと同一対照ではないため、既存比較の分母を強化した結果として置き換えられない。掲載にはgeneral CDF/XDF、元入力、DF誤差、own-ground、13round、LD2、lane、点値統計の説明が必要となる。

**D4：ストーリー。** 全面統合は「固定表現のPR構成評価」から「表現最適化・再配置・PR政策の共同設計」へ問いを広げる。共同設計の因果効果や最適性は既存報告の結論ではなく、節を増やすだけでは根拠が揃わない。

**D5：数値。** H4/STO-3G/1Å、同compiler名、P=0という共通項だけでは比較できない。Track AのDF12/10とCのCDF8、保存stateとown-DF ground、ε_signalとCA、T=.8とm0..12、K2/4等とK0、full wrapperとserial kernel/shellの差を保持する。本報告は両研究のRZ値を一つの新価格表や順位へまとめていない。

## §7 統合4案の比較

分量はV2の§1.1・§6.2・補足S1–S7、V4の§8–9を確認したうえでの**編集案**であり、ページ数・所要工数の実測ではない。どの案も今回実装・原稿編集していない。

| 評価項目 | 案A：完全分離 | 案B：短い言及 | 案C：補足限定 | 案D：全面統合 |
|---|---|---|---|---|
| Track A中心主張 | そのまま維持 | 固定DFの限界を述べる範囲で維持 | 条件を切り離せば維持可能 | 中心課題が拡張・再定義される |
| ストーリーの明確さ | 高い | 高い。関連の境界も明示できる | 本文は維持、補足の説明負担が増える | 表現・政策・保証の複数軸が混在 |
| 本文追加量 | 0 | V2 §6.2に1段落を基本。必要時§1.1に相互参照1文 | 本文に補足への導線1文。補足に定義・条件・結果/資格の2–3小節案 | Intro/Method/Results/Discussion/Conclusionの広範改訂。分量未見積り |
| 数式・定義追加 | 0 | 0。定量価格を書かない | 再配置・J/U・M_ZZ、CAとsignal精度・laneの定義が必要 | 上記にoptimizer/solver/tuning・比較設計も必要 |
| Figure・Table追加 | 0 | 0 | 図0でも可。固定結果の引用表1と条件表が候補。図は必須でない | 表現/最適化/資源比較を示す新図表構成が必要。数未固定 |
| 既存SOURCE再利用 | 部品・引用のみ | 既存報告の出典を参照 | C1–C4の固定報告と保存表を出典付き引用 | 一部可。ただし共通SOURCEに統一済みとはできない |
| 追加科学検証 | 分離の判断自体には不要 | 定性的言及には不要 | 既存限定結果をそのまま引用するだけなら不要。数値結合/新claimなら別許可 | 既存結果だけで共同設計・同条件優位は言えない。追加検証の要否・条件を先に判断 |
| 新規性・先行研究負担 | 研究ごとに扱える | 既知表現最適化との関係を必要最小限 | RC-DF L1族/SCDF/UWCとの区別を補足で説明 | Track A全体の新規性を組み直す負担が大きい |
| C成果の扱い | 独立した条件・負結果を保てる | 価格結果を省き、独立課題として扱える | 資格と反転を全て書けば可。成功だけ抽出しない | PR利益と目的関数利益が混同されやすい |
| 将来の自由度 | 高い | 高い | 補足のclaimが両研究を部分的に結ぶ | 両研究の方向と完成条件が結び付く |

案Dを今回の既定案にしない。案Cも、本文が短くなることだけで説明コストが小さくなるわけではない。表現別価格表だけを補足に置いても、本文の主張を広げたように読まれるなら案B/Aを優先する。

## §8 最小限追加する場合の具体的な影響

### 8.1 案Bの変更位置案

| 現存箇所 | 追加する場合の内容 | 追加の必要性・説明負担 |
|---|---|---|
| V2 Abstract／§1.2 Contributions | 変更しない | 研究Cの結果はTrack Aの既存寄与を成立させる必須証拠でない |
| V2 §1.1 Related work | 既存RC-DF等の段落がある。独立研究Cへの相互参照を必要時1文だけ | DF最適化文献を新たな中心課題へせず、既存外部調査を再実施しない |
| V2 §2 Methods | 変更しない | J/U fit・線形Z定義・13roundを導入する必要なし |
| V2 §3–5 Results | 変更しない。研究C価格表を追加しない | 同一task比較と誤解させない |
| **V2 §6.2 Scope choices** | **既存の固定DF/RC-DF範囲外という段落に、表現・分割が資源順位へ影響し得ること、J/U最適化と再配置は別研究であることを1段落** | 数値なし・最適性/新規性claimなし。最も自然な位置 |
| V2 §7 Conclusion | 変更不要 | 固定DF内の既存結論は研究Cなしで述べられる |
| V4を採る場合 | §8の異系列比較・外挿限界の後、または§9に同趣旨を1段落 | 新Discussion節を存在したことにしない。B0IとZ再配置を区別 |
| H6Vを採る場合 | §7の限界、必要時§6の異系列区分へ短く言及 | H4 CDF8の数値をH6の補強結果にしない |
| DISP Methods/caption／主図 | 原則変更不要 | 6確認条件とFigure2のcohortを維持 |

1段落で扱う論点は「本稿はDF表現を固定した比較である。表現や一体項への再配置を変える課題は、その表現内のPR利用の利益と別に評価する必要があり、別研究として扱う」で十分である。これは追加内容の範囲案であり、原稿へ書き込んだ本文ではない。独立研究の引用方法・公開名はユーザーが原稿採用時に決めればよい。

**主図変更不要。** B2の既存主図へ研究Cの点を重ねず、既存図番号・保存値・captionは今回も今後の案Bでもそのまま使える構成である。

### 8.2 案Cを選ぶ場合の補足案

V2補足S1–S7の後に**新しい独立付録を設ける案**である。番号は未採択。V4には同じS1–S7巻が確認できないため、共通補足が既にあるとはしない。

1. **定義と比較単位**：number CDFのJ/U、元g補正κ/scalar、LD2、線形Z→head、元identity/ZZ保持、全H保存とPF split変更の区別。B0Iとの違い。
2. **条件と引用結果**：H4/sector/rank/ground/CA/13round/P0/fenced serial/selection32/final64。同一J/U Dと保存CDF8/XDF8 Dを分離。旧辞書→再配置後の反転も併記し、C4点値だけを成功例として抜き出さない。
3. **資格・先行対応・限界**：M_ZZ²とweighted L1²族、有限更新/U未収束、P/r16統計診断、PRの数値lane未資格、D pilotとactual CAの区別、一般化未達。

既存C1–C4の限定結果表をそのまま出典付きで引用するだけなら新計算は不要である。ただしTrack Aの主表と値を結合する、共通物理H/状態へ揃える、共通費用比・新CIを作る、同条件baselineと主張する場合は新しい再解析・計算範囲の明示許可が必要となる。今回は全て未実施。

この付録の価値は「固定表現での結論の射程を示す別事例」にはある。既にV2 §6.2が表現を範囲外と宣言しているため、Track A完成の必須補足ではない。読者が価格だけを一般的PR/DF優位と受け取るなら追加しない方がよい。

## §9 Track Aのストーリーへの影響

### 9.1 研究管理上の6問

| 問い | 既存資料からの構成上の整理 |
|---|---|
| Track AはCなしで独立に構成できるか | できる。実在V2/V4/H6Vと図・補足がCの最新結果以前から独立taskを定義している。投稿可否・新規性の正式判定とは別 |
| Cは異なる科学問題か | J/U・目的・線形Z再配置・物理DF admissionを扱う点で異なる。PR/RPE会計は共通基盤 |
| C結果は補強か別論点か | 固定強Dに対するPRの条件付き点利益という関連例にはなるが、Track A自身の対照を同条件で強化した証拠ではない。数値を主結果に入れると別論点が増える |
| 主張を変えず取り入れられる要素 | 表現依存性、PR利用の利益とDF目的利益の区別、共通コードを再利用する出典・再現性の説明 |
| C独立研究で残る判断 | 再配置後に何を主課題とするか、既知weighted L1族との残る差、有限solver/strengthの意味、PR未資格lane/統計診断・移送未達をどう主張へ反映するか。今回は解決・GO判断しない |
| 共通利用すべき基盤 | canonical RTE、PF schedule/relative Gaussian、phase/scalar、CAR/sector参照、RPE radius/shot、6指標・shell会計、selection/finalとSOURCE/input来歴。ただし研究別adapter/契約/結果/STOPは独立 |

### 9.2 全面統合で起きる構成上の問題

Track Aには「どの残差処理・PF政策が同じsignal taskで安いか」という横軸がある。Cを全面統合すると「どの近似Hamiltonian表現と目的・solverが良いか」という軸と、energy目標・13round・数値laneの軸が加わる。読み手はDF fitの性能、再配置の効果、PRの効果、compiler/shotの効果を分けて理解する必要がある。

特に、旧PR-aware低費用をAbstractへ残しながら再配置後ENTRY低費用を補足へ隠す構成は、現存結果を正しく表さない。逆にENTRY＋PR対強DだけをTrack Aの新主結果へ入れると、PR-aware目的の研究課題が消える。既存原稿の明確さを維持する目的には案Dは適さない、という構成上の提案である。Cの科学的価値や研究終了の判断ではない。

## §10 推奨案

**研究は独立のまま、原稿上は案Bを任意の最小接続とする。相互参照が読者に不要なら案Aのまま完成させる。案Cは表現依存性の補助事例を読者へ示す必要が具体化したときだけ再検討し、案Dは現段階の推奨としない。**

理由は、Track Aの既存主張を支える結果・図・補足がすでにあり、CのJ/U最適化や価格比較がその成立に必須ではないこと、数値を追加すると異なるtarget・誤差・政策・laneの説明が増えること、Cでは目的armの順位反転とPR利用利益を別々に保持すべきことである。

これは今回の静的調査に基づく**原稿構成上の推奨**である。研究Cの独立新規性・論文化GO/NO-GO・継続/終了、Track Aの最終主要claimの科学的採否は判定していない。推奨を原稿へ反映する正式採択・編集も今回の成果ではない。

## §11 残る判断事項

### 11.1 最終報告Q1–Q12への対応

| 質問 | 回答 |
|---|---|
| Q1 中心目的・主張 | 固定DFの有限時間signal taskでbias/B/shot/compiled費用を接続し、PRの資源競争力と適用境界を調べる。一般PR優位・DF目的最適化が中心ではない（§2） |
| Q2 完了と保留 | M1/M2/PM、geometry/B0I、独立確認、高次PF/長時間の限定結果は完了。全law/母平均・最終energy資源・直近強PFの5数値STOP等は未解決。追加研究と主結果採用を分ける（§3） |
| Q3 原稿・図・補足 | 複数通し草稿、PM4主図と補足、V4の4主図/S1、掲載主表/Figure2/Methodsはある。後続全成果を統合した正式最終稿は未確認（§4） |
| Q4 科学的重複 | head/tail、PF/finite RTE、B/shot/資源接続、native/phase/compilerと共通コード（§6） |
| Q5 C固有 | J/U・正則化・有限最適化、線形Z再配置、物理DF予算、旧/新辞書順位、同J/U Dと保存強D比較（§5） |
| Q6 補強か追加課題か | 別条件でのPR点利益と表現依存性の関連例。Track A主結果を直接強化する同条件対照ではなく、価格を載せれば別課題が増える（§6/9） |
| Q7 全面統合の問題 | 固定表現構成比較から表現/solver/PR共同設計へ中心が拡張し、反転・誤差lane・公平条件の説明が増える（§9） |
| Q8 最小追加 | V2 §6.2を基本に1段落。必要時§1.1に相互参照1文。V4なら§8/9。Methods/Results/Conclusionの変更不要（§8） |
| Q9 補足の価値 | 任意の表現依存性事例としてはあるが完成必須でない。定義・条件・全限定を説明できる場合のみ。価格表だけの追加は避ける（§8） |
| Q10 明確さを維持する案 | A/B。関連を明示したいときB、不要ならA。Cは条件付き、D非推奨（§7/10） |
| Q11 Cなしで完成可能か | 構成上は可能。既存草稿と証拠の採用範囲を整える仕事が残る。査読誌採択可能性を認定した意味ではない（§4/10） |
| Q12 次の担当・承認 | 先にユーザー/GPTがTrack A原稿系列・後続結果の掲載範囲・A/B/Cを選ぶ。採択後、Codexが別依頼で資料/原稿を整理。静的編集のみなら新科学実行許可は不要。重要レビュー・新再解析/計算を望む場合だけ別々の明示許可（下表） |

### 11.2 次段階の担当（未実行）

| 担当 | 次に判断・実施すること | 承認・実施状態 |
|---|---|---|
| **ユーザー／GPT** | PM technical note、geometry中心、H4/H6/H8掲載構成のどれを最終原稿の基礎とするか。高次/長時間/RPE拡張をどこまで載せるか。CはA分離かB短い言及か | 今回の報告を使う次の研究方針判断。Codexが代行して採択していない |
| **Codex** | 上記採択後、指定原稿の版・主張・図表・引用を既存証拠に沿って整理する | 別の具体的な原稿編集依頼後。今回本文変更は禁止されており未実施 |
| **ユーザー** | 必要と判断した重要科学レビューの対象・開始を許可する | 本静的報告や案Bを読むための必須工程にはしない。今回開始していない |
| **ユーザー／Codex** | 新しい数値結合/再解析・baseline/fit/PF/RPE/nativeを望む場合、対象SOURCE・条件・scopeを指定してユーザーが許可し、Codexが実施 | 今回は不要・未許可・未実行。既存STOP/one-shotの条件を変更するならその具体変更の許可が別途必要 |

Codexの自動的な後続作業なし。結果取得後の研究方針はユーザー／GPTへ戻す。

## §12 SOURCE保全・MANDATORY STOP

### 12.1 実施した確認と非実施事項

実施内容はMarkdown・SOURCEテキスト・管理metadataの読解、固定Git object/GitHub本文の取得、出典・版・path・blobの技術照合、本報告書の差分・リンク・構造確認、commit/pushと公開確認である。科学結果の独立検算、保存標本の再集計、数理的正しさの独立審査をしたとは主張しない。

公開前の文書確認では、主要固定資料13件（C1–C4、M1、PM-2、V2/V4、DISP、L、B3、P1D、H6V）のGitHub blobとローカル固定objectが全件一致した。固定参照32件のGit object、12節・Q1–Q12、V2/V4/補足が参照する既存画像の実在とMarkdown差分の書式を確認した。科学テスト・画像再生成は行っていない。

SCF/積分/Hamiltonian/state生成、DF fit/rank探索/JU評価、保存NPZ/HDF5/QPYのdecode、PF/RTE/RPE計算、sampling、回路build/compile、価格やCIの新計算、図表生成・変更、研究原稿変更、重要レビューは全て0。H6 ENTRY技術問題、第1研究高次PF論文、Track B共設計を再調査・統合していない。

新規tracked fileは本Markdownのみ。基点からのtree差分をこの一ファイルに限定することで、基点内の既存SOURCE・raw・結果・契約・STOP・grant・ledger・marker・原稿・図はそのまま保持する。別worktreeは参照のみで、process/cache/環境・共有設定を変更しない。他チャットのHEAD変動は凍結条件にせず、引用は上記固定commitに限定する。今回新しいSOURCE lock・実行契約・承認証明用仕組みは作っていない。

### 12.2 公開照合と停止

本報告書を専用branchへcommit・pushし、公開branchのcommit SHA、commitのtree SHA、報告書のblob/本文とローカルの一致、親commitから一Markdownのみ追加されたことを確認する。検査の実績SHA/tree/本文一致は最終チャットで報告する（自己参照するcommit SHAを本文へ埋めるための再commitは行わない）。GitHub認証に問題があれば共有認証設定を変更せず既存GitHub接続を使う。

**公開照合後MANDATORY STOP。** 原稿編集・統合実装・新計算・STOP解除・重要科学レビュー・研究Cの独立新規性/論文化GO/NO-GO・他研究拡張へ自動進行しない。

[MAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/PROJECT_MAP.md
[INDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/RESEARCH_RESULTS_INDEX.md
[OLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/研究概要・現状.md
[A0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/track_a_shared_research_status.md
[CLAIMS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/track_a_post_pm2_claim_evidence_map.md
[M1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8e0814e70c14ecf526444fac8a2142799610dc96/docs/pr2_matched_accuracy_m1_b1_result_validation.md
[M2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/pr2_matched_accuracy_m2_transfer_result_validation.md
[PM0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/pr2_post_m2_evidence_attribution.md
[PM1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/pr2_pm1_discard_result_validation.md
[PM2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5a1adffad780f0ec4272f5e8bb94713f9ff0f2bc/docs/pr2_pm2_precision_resource_result_validation.md
[GEOM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/docs/research/track_a_h4_geometry_results_20261013.md
[B0I]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/docs/research/track_a_h4_b0i_map_results_20261014.md
[V2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/manuscripts/track_a_resource_study_v0_2.md
[V4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/manuscripts/track_a_resource_study_geometry_native_v0_4.md
[H6V]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/653d64bedba7516d4c17f2c7457cefbc80f877a2/docs/manuscripts/track_a_h6_resource_study_v0_2.md
[COLLECT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/main_local_results_collection_20261103_files.tsv
[A6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/track_a_h468_paper_evidence_integration_20261019.md
[DISP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/track_a_h468_paper_display_methods_20261019.md
[CAPTION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/artifacts/resource_applicability/track_a_h468_paper_display_methods/2026-10-19/figure2_caption.md
[PF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/docs/research/track_a_h246_pf_common_final_20261022.md
[LONG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[B3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/REPORT.md
[LAND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/pr_repository_scientific_landscape_20261101.md
[AX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md
[P1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d5ca357ccfd8fcc850bef800c5bcadfde2a153e/docs/research/track_a_b2_p1_shots_20261107/REPORT.md
[P1D]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2583ce4957ed44e914ce6571e3e676830439dfa7/docs/research/track_a_b2_p1_strong_pf_20261107/REPORT.md
[DIAG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e217d145bd99c2b34302017390ecb57987c06b90/docs/research/track_a_b2_p1_strong_pf_numerical_diagnosis_20261107/REPORT.md
[C1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/docs/research/research_c_integrated_h4_100_results_20261106.md
[C2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/docs/research/research_c_h4_relocated_downstream_results_20261106.md
[C3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/77ded5c12db21572933653cb26758436da3d9ecd/docs/research/research_c_baseline_comparability_20261106.md
[C4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/91417fb3ef75e7eec0849ad39f0649d170754664/docs/research/research_c_h4_strong_d_relocation_results_20261107.md
[CSOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/91417fb3ef75e7eec0849ad39f0649d170754664/src/trottertracks
