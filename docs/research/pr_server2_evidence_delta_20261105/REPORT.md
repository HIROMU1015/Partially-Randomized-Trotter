# SERVER 2：PRリソース評価・成果証拠差分（2026-11-05）

**判定：MATERIAL_NEW_EVIDENCE。** 固定全体統合後のH8長時間・高次PR探索と独立価格確認が未反映で、既存の限定主張を補強・拡張する重要証拠がある。既存主要結論を覆す同一条件の不整合は、今回の確認範囲では見つからなかった。重要度はMEDIUM以下で、HIGHはない。研究全体の完成・正式受理・中心claimの採択は認定しない。

[差分一覧 DELTA.tsv](DELTA.tsv)／[固定出典・匿名metadata EVIDENCE.json](EVIDENCE.json)。LOCALが既存正本へ追記するための引継ぎであり、全研究の新統合版ではない。

## 1. 担当環境・固定比較点・範囲

担当はSERVER 2。既存Phase 1のSERVER_2記録と、その登録rootに対応する既存サーバー差分TSVを照合した。内部host名やIPから推測していない。H8 Stage A/BとStage Cの独立作業cloneも自環境に存在し、それぞれの固定履歴を直接読めた。Stage Cは依頼で指定された後続参照として照合した。作業cloneの存在だけから科学実行元hostを新認定せず、他環境の同じ引用を別実験に数えない。

- SERVER 2 Phase 1：`8834de0cf938b13490c08e05006e3c566fed7c7a`。
- Phase 2：`604f0235840feb18d368f99ea0828f370ce70fc8`。CANONICAL_CATALOG、RECORD_LINKS、UNRESOLVED、RELATIONSHIPS、SERIES_INDEX、REFERENCE_INDEX、SOURCE_COMPONENTS、INPUT_FILESを再利用。
- Phase 3：`235e56ee7d5cac4682775a62bc2a8262f15047dd`。既存21 claim／44 coverage行と証拠索引を再利用。
- 既存全体統合：`2ce2b5f161e94567fec322d694fd8b01d03cd854`、[RESEARCH_SYNTHESIS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ce2b5f161e94567fec322d694fd8b01d03cd854/docs/research/pr_research_paper_evidence_synthesis_20261103/RESEARCH_SYNTHESIS.md)と[CLAIM_EVIDENCE_MAP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ce2b5f161e94567fec322d694fd8b01d03cd854/docs/research/pr_research_paper_evidence_synthesis_20261103/CLAIM_EVIDENCE_MAP.csv)、残存課題・範囲候補。
- 観測した公開main：`546bca7ed2cece59b714b986a264d7b0b0871a01`。mainのファイル収録と、研究統合の証拠採用は別である。LOCALの未公開途中成果は仮定しない。

正式共通運用指示2026-10-25（特に§8.1・§8.4・§10）、親・対象AGENTS、mainのPROJECT_MAPと研究概要を確認した。本人所有の既知root、固定Git object、既存索引に限定し、他サーバー接続・広域raw探索・inventory再生成は行わなかった。

## 2. LOCALへ追加する重要証拠

### S2-D01/D02：固定H8長時間・高次PR

linear H8／隣接1.00 Å／STO-3G／保存tol-only1e-8 DF28、16 system qubit・sector4900、元state・fragment順0..27、T=1.6、複素signal精度ε=.0005、P=0。Qiskit1.3.0、rz/sx/x/cx、opt1、transpiler seed17、backend/target/topology指定なし。full Hadamard wrapperのcompiled RZと解析shot NによるGを扱う。energy精度・全RPE費用とは異なる。

探索は11月3日開始、11月4日23:06 JSTに最終代表検査を完了、結果carrier `7fb1d125e05238a7d932fc89b8dab6ac6a7f8c84`。保存報告はStage A 69候補、Stage B 19価格候補／51 trajectory／102軸とする。探索の復旧・同seed継続・cache再利用・位相会計修正は追加独立標本ではない。旧HOLD・中間未確定版を保持し、最終完了版と時点を区別する。[探索結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7fb1d125e05238a7d932fc89b8dab6ac6a7f8c84/docs/research/h8_long_time_execution_20261103/EXPLORATION_RESULTS_20261104.md)

Stage Cは11月5日00:31:24.562287〜01:22:15.014179 JST、SOURCE `82fdbdd098e9d3edae01a930437e3d76c1f8bf36`、結果carrier `eb07156a37b1938ed3aed95c13abbb545466c573`。固定PRはm5／prefix8／q1／R_outer32／K4、candidate `7eb0a3d691b94089969b`、確認cohort n9。固定PFはB0I／m5／prefix15／q1、candidate `1425ae6ed83fd9098b39`で、探索時の保存直接価格を再利用し、Stage CでPFを再sampling・compileしていない。[確認結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb07156a37b1938ed3aed95c13abbb545466c573/docs/research/h8_stage_c_execution_20261105/REPORT.md)／[保存JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb07156a37b1938ed3aed95c13abbb545466c573/artifacts/resource_applicability/track_a_h8_long_time/2026-11-05/confirmation.json)

| 保存値の引用 | 値 |
|---|---:|
| 確認PR平均G | 67335317264200 RZ |
| 固定PF G | 89291219516568 RZ |
| 確認PR/PF（報告表示） | 0.754109056061 |
| 観測平均削減率 | 24.58909439% |
| paired G sample SE | 103180211675.02368 RZ |
| 事前固定の記述的SE目標 | 121655623062.28098 RZ |

比較方向と記述的SE目標は保存結果で達成。新cohortは探索とpoolせず、cos/sinは同一trajectoryのpaired価格。新seed namespaceと旧seed非衝突は元実行報告の記録を引用しており、今回はprivate seed registryを再監査していない。95%区間・母平均優位・全候補最適性は認定しない。未観測order2/4の費用とpopulation tail boundはnullのまま。希少費用・母集団保証の追加検証を要求しない。

Signal SOURCE `d1272016592ff06c0567862f770ed796fd61b366`、native SOURCE `df93fbb88761b61c2691bf01f761ed20a7987d7f`、位相補助SOURCE `18ce41942051065fa097644a8a937bd11adba708`、Stage C SOURCEと結果carrierを別に保持した。公開JSONが記録するStage A/B圧縮入力2件とSOURCE3件のhashは一致。圧縮科学結果はdecodeせずbytesで照合した。

**LOCAL最小対応：** A07/A08/A09の関連追補としてH8 T=1.6の固定比較・別cohortを追加し、長時間範囲の説明（A11関連）を拡張する候補とする。既存A11のH4/H6 33/40や、旧H8 T=.8の主表を更新・合算しない。旧domainでの主張を否定する結果ではない。重要度MEDIUM、反映のための科学計算は不要。

### S2-D03/D04：既存だが全体統合で参照されない限定機構・matched分割

H6 P1/P2は11月1日の固定成果。P1はPauli918／T=.8／S2／q8／K2、Hybrid prefix78・r16/32/64と純RTE r64について、外側splitの差を解像し、可換保持内部の理想PF差と区別した。小finite/grid差のUNRESOLVEDを保持する。P2は保存index0000のFullを再利用し、R/E/Emptyの固定program価格と非加法性を診断した。全law・母平均・一意な因果寄与率ではない。[P1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bfb4ca75437a9b3605a8f57b5f82100615dd034f/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/bias/REPORT.md)／[P2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c9073274daee69c9b69d8db3dc332bf6132c2259/artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/cost/REPORT.md)

P3は11月2日のlinear H4／1Å／STO-3G／tol-onlyDF10、共通Gaussian T_eff保持・V_square残差、Pauli184対DF rotated Z/ZZ360・10 basis、T=.8／S2／q8／r4／K4／P=0の別case study。Stage 4の5精度・条件付き会計と、finite-bitのengineering診断がある。純RTE対DF B3や最新最良候補の完全matched順位ではなく、表現単独の一般因果効果でもない。Stage 4のfinite-bit gap=0／compiled rho=0等の仮定と、後続finite-bit工程が実装全域保証を閉じないことを併記する。[Stage 4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a1d3f32aba6416fd1e200cd66a97d1a825b5dd73/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage4/run01/REPORT.md)／[finite-bit結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/426ccda5df044d2e94d4a18d98cde75a96ab1e6b/artifacts/df_pauli_matched_partition_validation/2026-11-02/finite_bit/run01/REPORT.md)

全体統合の本文・claim mapにはこれらの固定結果参照がない。[既存Pauli追補](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f6eb34dad982f8f753a618c609b528fb3c20d142/docs/research/pauli_h2_h4_h6_paper_evidence_addendum_20261102.md)が既に局所結果・未保証事項・原稿修正候補を整理しているため、再統合せず再利用できる。P05/F01の説明範囲を補うMEDIUMの統合漏れで、14/15や既存DFの勝敗を変更しない。[全law転送調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0e1c7aae386986f6c1bb9483548671655447f516/docs/research/h6_whole_law_transfer_feasibility_20261102.md)は静的補足（LOW）で、保証完成ではない。反映に新科学計算は不要。

## 3. 既存統合に反映済みの成果・保全差分

H4/H6長時間B2、G01/G02の7候補・cohort対応、Pauli最新14/15とH6 r32追加、Track B C1の分子logical path完了とnative全task未保証は既存統合に反映済み。別branch・報告・main収録版は新実験として加算しない。旧DF12とtol-only DF10/19/28、H6の保存weighted completion、Pauli prefixとDF prefixを同じ入力・分割とみなさない。

**G03／B3（S2-D06）：** 保存最終commit `f794a2aa4c76bc024540c808bf9400d9f523bb76`を自環境で取得できた。公開`ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed`に対し、主要8表・provenanceのblobが一致（EVIDENCE.json）。全対象treeでは6 path差を確認した。読んだJSON metadata差は保全場所の匿名化表記とpublication statusで、主要負結果は変わらない。圧縮結果2件のbytes差はdecode・再集計しておらず、全raw同一性は認定しない。G03の原statusを変更せず、主要表一致の限定追補だけ可能。LOW。

**G04／旧H8 n48（S2-D07）：** 指定2commit `659e555714c8ccf3a6ec7fd1cb0dbf05cfcc342d`／`96aa3878a344f479ee064118cb90002a770db627`は、既存inventoryが明示した自環境rootの一つで存在した。公開mainにはcomparisons.csvとpaired_metrics.csvがあり、保存された匿名化版と同blobだった。これらの限定公開表はLOCALがmainの固定URLから再利用できる。

旧SERVER_2-REC-007120はPhase 2のTMP-103047、REF-00150へ対応し、原UNRESOLVEDのsource/input/cohort/custody不足は保持する。旧主表とMechanismは同cohortのaliasで、新しいn48実験ではない。詳細SOURCE/input/native hash inventoryには既存の非公開制約があるため、その原本・一覧・private保存先を転載しない。今回の所在地確認だけで完全公開audit・全履歴独立性・raw再現性を閉じない。LOCALは「SERVER 2で現存未確認」という説明を今回の限定custody観測で追補できる。LOW。

H8 rareの既採択scopeとSTOP、RPEの条件付き成功・費用不確かさ、Track Bの一時凍結を維持する。依頼のH4 RPE `d4c016960ff4b1ca16cc73770147cde326989427`／backend契約`655bf6cb84ab161706baf843a8ae255da39bf7a6`はLOCAL側の後続参照とし、今回のSERVER 2 H8条件へ結び付けたり、再計算・再検証したりしていない。

## 4. LOCAL向け引継ぎ・取得限界

1. 最新の直接結果はH8 Stage C `eb07156a37b1938ed3aed95c13abbb545466c573`。探索carrierは`7fb1d125e05238a7d932fc89b8dab6ac6a7f8c84`。新結果2段階は同じ科学scopeの探索／確認で、PF価格は共有する。
2. A07/A08/A09・長時間範囲の追補候補、P05/F01の限定機構・P3参照、G03/G04の限定custody説明を更新する。既存claim ID・Phase3原status・統合正本をSERVER側で編集していない。
3. 科学的に重要な未保証は、既存と同様に全law実装精度、母平均、一般因果帰属、最終energy資源等。今回の限定主張を未完成へ戻す理由にせず、今回の引継ぎに新計算は不要。希少費用・状態準備の追加検証へ進まない。
4. 今回未取得／未照合：新H8 private raw・全seed registry・QPYの全量独立byte監査、旧H8の詳細private inventoryの公開、B3圧縮rawの意味論的全比較。公開projection・固定SOURCE/input対応と元監査記録を越える認定を避け、大容量decodeや非公開制約の解除を行わなかった。
5. LOCALの並行未公開成果・他サーバー原本・全科学SOURCEの詳細レビューは対象外。新しい母平均保証・完全外部再現に必要な追加証拠の不足を、本報告の完了阻害やHIGH判定へ転用しない。

## 5. 技術照合・公開・終了

既存inventory/metadata parser、Phase 2/3の固定参照・selector方針、保存値のliteral転記を再利用。H8報告用コードは新平均・SE等を計算するため静的確認に留め、実行・importしなかった。新規は本3成果物と、標準ライブラリによる限定CSV/JSON読取・hash/文字列比較だけ。科学module・simulator・compilerは起動していない。

GitHubで取得した既存統合本文、H8探索報告・確認JSON、H6 P1/P2、P3 Stage 4・finite-bit、旧H8のmain公開2表の計9資料は自環境固定Git bytesと一致。新規3ファイルのschema・14固定出典・literal転記・匿名化・差分範囲の技術検査はPASS。これは作者自身の技術照合であり、別担当者の独立レビュー・新科学性能PASSではない。

作業は独立cloneの専用branch `codex/pr-server2-evidence-delta-20261105`に限定。公開は本ディレクトリの3ファイルだけ。元SOURCE・結果・grant・ledger・marker・STOP・固定契約・既存統合・共有環境に書き込んでいない。出力cloneは既存object store依存で、独立backupとは呼ばない。

**commit/push後、remote SHAと公開3ファイルのbytes/blobを照合しmandatory STOP。** 最終SHAとURLは最終チャットで渡す。次の担当は**LOCAL Codex**：本差分を既存統合に反映する。SERVER 2 Codexの追加科学実行・詳細科学レビュー・論文執筆・別研究再開は行わない。
