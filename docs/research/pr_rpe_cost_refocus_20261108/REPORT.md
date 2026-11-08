# PR＋RPEコスト削減への再集中：横断調査

2026-11-08 JST／SERVER 2。基点：7819602bdb7a5a0e6d4b43098144f681790f0651。
専用branch：codex/pr-rpe-cost-refocus-survey-20261108。

**PR内部のnative総資源削減、ReferenceのCX／depth削減、有限trajectoryのT削減は既存成果である。一方、新Aの全18round総T削減は未資格。次はm17保証を延長せず、既存担当の費用改善課題へ判断を戻す。** 本資料は静的統合であり、科学レビュー、研究採択、新しい科学的優越の認定ではない。

## 1. 調査範囲と証拠の読み方

入口は[3環境統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/REPORT.md)・[再利用地図](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/DUPLICATION_AND_REUSE_MAP.md)・[既存台帳](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/IMPROVEMENT_VERIFICATION_LEDGER.csv)と、[O1〜O6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/RESEARCH_OPPORTUNITIES.md)。旧台帳を再構築せず、その後の費用判断に必要な差分を調べた。default mainは照会時546bca7ed2cece59b714b986a264d7b0b0871a01。関連remote branchは397名称をページ末尾まで取得し、主要な結果／後続設計branchのHEADをGitHubから固定した。本文はその固定snapshotのREPORT・SOURCE記録・結果・STOPと、必要箇所のSOURCE静的読解に基づく。

branch収集は2026-11-08 08:57:50 UTC時点のsnapshot、主要後続7 branchの再照会は09:17:19 UTC（JST 18:17:19）で、上記採用SHAからの変更なし。以降の別チャットの進行を凍結するものではない。

別チャット本文、他SERVERの非公開runtime原本、進行中workerは参照・操作していない。全397 branchの全文や全rawの再監査ではない。確認できない実行状態・非公開資料はUNKNOWN／資料未取得であり、未実施とは断定しない。原著文献の新たな網羅調査も行わない。先行法の位置付けは固定文献調査と各原報告の限定確認を再利用した。

適用AGENTS.md、PROJECT_MAP.md、研究概要、正式共通指示2026-10-25版に従う。今回の追加は本ディレクトリの3文書のみ。旧索引・科学status・SOURCE・raw・契約・grant・ledger・marker・STOPを変更しない。

- [費用・条件・SOURCE対応表](COST_IMPROVEMENT_MATRIX.md)
- [重複分類・候補・最小scope](OVERLAP_AND_NEXT_SCOPE.md)

数値は原報告／保存結果の転記で、新しい削減率・N・価格・順位を計算していない。十分shot数は真の必要最小値と区別する。価格MCのSE、RPE失敗確率、operator保証、実機性能は別の量である。

## 2. 実際に削減されたもの

| 系列 | 既存成果の引用 | 強baselineと資格の限界 |
|---|---|---|
| Track C：共同isometry | 同一basis runの必要列を共同保持し、旧singleton比で全18round RZ 3.355%減、CX 1.789%減。serial depthは0.696%増 | RZ目的の同じ回路policy。Fullはなお安い。[原報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2745d57bea5c28604739bf9c01338e14e649c467/docs/research/track_c_joint_isometry_20261108/REPORT.md) |
| Track C：control／対角融合 | 全18roundで旧joint PR比RZ18.196%、CX45.118%減。ただし大部分は既知の単一Pauli control。強いterm-control対照からのPR run融合追加はRZ約0.135%、CX約1.67% | Fullにも同じcontrol／network改善を与えている。18.196%をPR固有の新機構と呼ばない。[原報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/docs/research/track_c_diagonal_fusion_20261108/REPORT.md) |
| Track B→C：Closed P3 | 同じ有限P3でnormalization／rangeを減らし、固定policyのcanonical比全18round RZ1.965196%減。trial合計17186→17028 | strong Fullの総RZ5.660906974e11に対しClosedは1.394938574e12。[採用FAIR結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e38af3b12200c48c4f1b793638d223baff24e76/docs/research/track_c_closed_p3_20261108/REPORT.md) |
| Track C：その後の共同選択 | h／resolution／αを両P3に同等に再最適化。Closedは旧固定Closed比RZ6.03267655%、CX5.23495599%減。同じ再最適化canonical比RZ2.11497621%減 | 最終Closed 1.303898609e12 RZ、Full 5.647934021e11。両P3は同じ18round構成を選択し、law固有の構成選択差は観測しない。[extended正本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md) |
| 新A Reference | 強4-CX B′に対し全18round期待CX削減≥4.681782%、native期待DAG depth削減≥40.995952%という条件付き上下界 | RZは≥0.150597%増、qubit9→17、shots8598／8606。exact期待depth・wall time・総Tではない。[B′](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7819602bdb7a5a0e6d4b43098144f681790f0651/docs/research/pr_reference_bprime_20261107/REPORT.md)・[depth補強](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7819602bdb7a5a0e6d4b43098144f681790f0651/docs/research/pr_reference_depth_v5_reinforcement_20261107/REPORT.md) |
| 新A CT-4B | m7/q128/r1の保存nonrare／rare各1 trajectoryで、1-shot T+T†がB′比1.865933%／1.881674%減 | 一様ε=1e-11、保存199 shots/axis条件。全support平均・全18round総Tへ移さない。[原報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8012c3e9547690f0d722edac9d0b9a9391ba2f8e/docs/research/pr_reference_ct4b_20261107/REPORT.md) |
| Track D Reference | 13-round主scenario AでCX11.1671%、total depth54.6549%、RZ1.1200%減の条件付き点推定 | 総資源約99.4〜99.5%がq≥64 proxy。保証付き母平均優位・actual成功は未資格。[原報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8828b298975c88a2c0d1fd57ff6ed905233840f5/docs/research/track_d_bc_total_20261108/REPORT.md) |

これらの改善率は異なる対照・段階・条件に対する値である。連続した改良系列でも価格holdout、α、N、回路policyが異なるため、率を足した「統合改善率」は作らない。

**さらに強い対照の更新を見落とさない。** Track C後続の[DF truncation比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/docs/research/track_c_df_truncation_20261108/REPORT.md)は同じ元H ground／18round精度条件のH2/H4/H6を扱う。H4のTruncated PFは総RZ3.337772238e11で、上表のFullとPRより安い。相殺を分離した保守的評価BでもH4は3.864575491e11で両対照より安い。[P1/P2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/92495a04d52c65142ed16b3abe977345065f9745/docs/research/track_c_df_cancellation_validation_20261108/REPORT.md)
H2では改良PRが登録Full／Truncatedより安い保存結果があるが、H4の勝利へ一般化しない。H6ではcombined評価のTruncatedが最小、保守的BではPRより高い。従って「PRはどの系でもFullに勝てない」とも、「PRの一般優位を確立した」とも整理しない。有限menu・保存Hermitian-H・理想native・元H ground供給に限定された結果である。

## 3. 費用を支配するもの

同じ条件で得た分解だけを引用する。

- [Track C bottleneck](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10bf5a4813559cea807fcd8c50f3b966407747d1/docs/research/track_c_bottleneck_20261108/REPORT.md)の改善前policyではRTE basisがRTE RZの約91.44%。改善後holdoutではRTEが総RZの約70.6%、m17/m16が約62.7%/21.9%。これは当該段階の値で、後続Closedや新Aへ固定比率として移さない。
- [最新Closed共同選択](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md)では、旧固定ClosedからD総RZは3.373971816e11→3.439699811e11と増え、RTEは1.050211269e12→9.599285356e11と減った。m17のmicrostep数25%減に対しshotsは[253,253]→[303,302]と増える。局所単価とshotの交換を全体で評価した成果である。
- [新A CT-4B](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8012c3e9547690f0d722edac9d0b9a9391ba2f8e/docs/research/pr_reference_ct4b_20261107/REPORT.md)の有限例ではD未融合部分が約8.2百万Tを占め、D境界補正・中央回転の差がReference CSWAPの17,472 Tを上回った。これはm7の2例の内訳であり、m17期待Tの支配要因の測定ではない。
- [Track A C0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/docs/research/track_a_o1_m1_c0_parallel_science_20261106/REPORT.md)では極小の長round半径によるshot増が安いPR単価を打ち消した。tail massや1-shot価格だけの最小化は不十分だった。

## 4. 他Trackとの重複と未資格

**同じReference機構の再実装は不要。** 新A、Track D、O2 vacuum-referenceは既知SE型のsplit／reference／CSWAPを使う系列である。新AはDF12/LD3/h=.02/K2/compiler A・18round、Track Dはnumber-CDF8/LD2/h=.1/K0/P0・13round、O2には人工／legacy DF12準備の別scopeがある。SOURCE・law・phase・N・strong baselineの一致を確認せず、価格や保証を交換しない。[Dの整理](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2131917656446bd22f294946d7c1064a47332e4c/docs/research/track_d_o5_algorithm_design_20261108/REPORT.md)・[O2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0edc0fa5ad4656fdd492ca6c96ade4f77cadf412/docs/research/o2_vacuum_reference_cswap_20261108/REPORT.md)

**Closed P3接続とh/resolution/α共同選択は実施済み。** Track B調査の「H4で未確立」は同調査の収集範囲の記述で、その後／別branchのTrack C全18round結果と区別する。G9 Closed P5の数値をK2=P3へ使うこともできない。[Bの資産](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a8299889de93c201bf21e727415d629f5e4b9f48/docs/research/track_b_existing_results_survey_20261108/REPORT.md)・[C後続](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md)

**新たなcontrol／fusion案も既存担当を優先。** Track Dには方式selector、近傍cut、head–tail接口C1の設計・限定検証資料がある。C1はSOURCE上の具体的な未接続箇所を示すが、P0後の増分はUNKNOWN。Track Cで既に融合された境界を同じ未接続箇所と呼ばない。[C1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62c7b578b8c0bb3d0079434daf637c587f53c24c/docs/research/track_d_o5_c1_head_tail_static_20261108/REPORT.md)

**表現改善の成果はあるが入力promiseに注意する。** 研究CのENTRY／PR-aware DF＋Z再配置の13-round比較はown-DF ground。保存点値の差と統計資格を保ち、元H groundの新A／Track Cへ直接移送しない。CDF0対ENTRYはUNRESOLVED_PRECISION_TARGETのまま。[三者整理](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/709813805d41e470676e311cd63a740af23bf33c/docs/research/research_c_three_method_z_relocated_comparison_20261108/REPORT.md)

O2の同価格IS、Parent、層別CV、単純LCU、共通basis周辺化等は既存の負結果・既知法への還元・未取得アクセスを持つ。同じ機構を新Bとして再命名しない。[O2統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d28164b5d51e0ca5a0ade74ded8ff176b443dab8/docs/research/o2_research_results_integration_20261108/O2_RESEARCH_SUMMARY.md)・[非同値性選別](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1bf88b251fd09caa0a227972ee0f1b56c5888e6a/docs/research/o2_distinct_mechanism_screening_20261108/REPORT.md)・[共通basis](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ffcdd1fae89b53b4ce1203be4f9797994e292b9e/docs/research/o2_common_basis_static_validation_20261108/REPORT.md)

新A [D1/D2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7819602bdb7a5a0e6d4b43098144f681790f0651/docs/research/pr_reference_inverse_d_numeric_20261108/REPORT.md)は保存poolの範囲でinverse追加改善なし、D guard改善はT増を伴い、非中央残余はなお負。これは加法十分証明の不足であり、実際のRPE失敗や新規合成必須の証明ではない。C2、V0/V1、S1/S2、旧C-N/C-S/C-P、P-D、BF/BM、PR-3、C1凍結等の負結果・STOPも維持する。

## 5. 次の研究判断と停止

以下はCodexの提案であり、正式採択・実行開始ではない。

1. **ユーザー／GPT**：次の主資源をnative RZ/CX/depthのどれにするか、最終総Tへ進む前にどのnative改善証拠を必要とするかを選ぶ。native勝利からT勝利は導けない。
2. **Track D担当を優先**：既存C1案の一種類のhead–tail接口・最大8人工wrapperによるP0増分判別が、具体性と追加作業量の点で最小の次候補。強対照にも同じ融合を与え、増分0でも終了する。既存scopeの採択と実行許可はユーザー判断で、新Aへ移管しない。
3. **Track C／D担当の調整候補**：最新の最適化済みnative pipelineにReferenceを接続する場合だけ、元のlaw・uncontrolled factorization・vacuum位相・actual inverse・同じ強Full/Truncated対照への機会を先に静的固定する。既存gadgetの再実装はしない。この組合せの比較は確認資料では未検証であり、全branchで不存在を証明した意味ではない。
4. **Track A／表現担当を優先**：表現・partitionを選び直す場合、既存own-DF結果と元H groundの課題差を先に解消する。一般的な再fit／rank gridは推薦しない。H4ではTruncated PFを強対照から落とさない。

詳細な資源因果経路、再利用SOURCE、最小検証、承認／STOP条件は[次scope表](OVERLAP_AND_NEXT_SCOPE.md)に記載した。裏付けのない削減率やT価格で順位を作らない。

**保留を明記する：** D3 inverse defect、basis全体actual-adjoint化、composite word新guard、新Clifford+T word、m17 strict誤差会計追加改善、43-key runtime全面資格は、一旦保留。再開には総費用目的への必要性の判断と別の明示許可が必要。

今回、新しい科学数値計算、保存数値の科学的再解析、sampling、回路build/compile/simulation、guard、合成、shot/RPE再計算、価格取得、総費用最適化、重要科学レビューは全て0。研究moduleもimportしていない。文書とGit metadataの技術照合だけを行った。

**3文書をcommit/pushしremote SHA・tree・本文を照合後、mandatory STOP。次担当はGPT／ユーザー。Codexによる後続科学作業なし。**
