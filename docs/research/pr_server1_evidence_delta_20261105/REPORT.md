# SERVER 1 成果・証拠差分照合（2026-11-05 JST）

**判定：NO_MATERIAL_DIFFERENCE。確認した自環境の範囲では、PRリソース評価の主要な比較方向を変え得る未統合の重要証拠、同一条件の重大な不整合は見つからなかった。** ただし、既存の境界3候補は主要統合への直接参照が欠けており、A11への **MEDIUMの補足2件** がある。これは「全成果が統合済み」という判定ではない。

対象はSERVER 1の保存証拠と登録済み自環境のみ。[既存SERVER 1 inventory][P1]で環境区分を確認した。inventoryに載る保存物がすべてこのhostで実行されたという認定や、旧SERVER_A/Bとのhost全体の対応付けはしない。既存全体統合・claim map・研究概要・manifestはread-onlyとした。

比較基準は指定されたPhase 1/2/3と、全体統合 `2ce2b5f161e94567fec322d694fd8b01d03cd854`。観測した公開mainは `546bca7ed2cece59b714b986a264d7b0b0871a01` で、差分branchの基点もこのSHA。後続のmain収録は文書・SOURCEの保存であり、新実験や科学的採択と数えない。全差分は [DELTA.tsv](DELTA.tsv)、固定参照・blob・SHA256・selectorは [EVIDENCE_REFS.tsv](EVIDENCE_REFS.tsv)、技術確認は [TECHNICAL_CHECKS.json](TECHNICAL_CHECKS.json) に記録した。

## LOCALへ渡す差分

|対象|既存統合との差|分類・重要度|最小対応|
|---|---|---|---|
|追加7候補|G01/G02とA11に反映済み|ALREADY_INTEGRATED / NONE|追加不要。境界3候補と混同しない|
|H4境界 R16・R128|固定R64との保存比較がA11主要mapに直接未掲載|MISSING_FROM_INTEGRATION / MEDIUM|保存HIGHER診断とR感度の限定を短く追記|
|H6境界 R11|固定R16との保存比較がA11主要mapに直接未掲載|MISSING_FROM_INTEGRATION / MEDIUM|点値とINCONCLUSIVEを併記|
|H8 q16・別精度への再利用|元epsilon .001 signal記録の不足・STOPは維持|PROVENANCE_GAP / LOW|この拡張に言及する場合だけ既存制約を引用|
|原本保存・後続案内|99原本の公開保存と案内branch追加|NO_MATERIAL_CHANGE / LOWまたはNONE|保存版を利用。独立証拠へ加算しない|

新しい科学結果の実行・追加は今回0。11月3日以降の自環境の新規科学HEADは確認しなかった。境界比較は2026年10月27日の既存結果であり、今回発見した新実験ではない。[既存境界文書追補][ADD]にも掲載済みだが、指定された [全体統合][SYN] と [claim map][MAP] への直接参照を確認できなかった。既存Phase 1の259表locators、Phase 2の系列・参照索引、Phase 3のEVIDENCE_INDEXにも当該境界結果の直接locatorは見つからず、全大容量inventoryの再生成・全行監査は行っていない。この限定した検索結果を全private資料の不存在へ拡張しない。

## 境界比較の条件・保存判定

H4/H6 linear、隣接1.00 Å、STO-3G、tol-only threshold 1e-8、保存DF rank10/19。H6は保存weighted-Hermitian completion。元Hamiltonian/state/reference/scalar/fragment順を保持した保存比較で、T=2、q=1、delta=T/q=2、m5_bestの11 signed piece、ABS_TIME_PER_OUTER。H4はL_D=3/K=2、H6はL_D=7/K=6。symmetric-directional/actual adjoint、cosine/+Im sine測定を含むfull Hadamard wrapper、P0、Qiskit1.3.0/NumPy1.26.4、rz/sx/x/cx、opt1、transpiler seed17、topologyなし。精度は **epsilon_signal=.0005** の有限時間signalで、energy精度ではない。経験的allowanceを使う有限価格標本のcompiled RZ会計である。[原公開REPORT][BR]、[保存表][RESULT]

- H4 `f85eb7b56ba5131d1eb1`（R16）と `3a55c1dac3947ae23c76`（R128）は、旧7候補の `45f9fc0a53c1a37c7345`（R64）を保存対照として再利用。確認cohortは `BOUNDARY_20261027_CONFIRMATION`、保存nは745/1185、対照は `SEVEN_20261027_CONFIRMATION` のn800。CSV physical line31/26/56の保存diagnosticは両方 **HIGHER**。この条件内でRを減らしても増やしても総Gが悪化する例として追記できる。R64の全R最適性は意味しない。R32のline36は探索n9のみで、独立確認済みのC候補へ加えない。
- H6 `0e0e9fb6da33cb6ab2ab`（R11、確認n229）は保存R16 `efbb07c848b303a6abab`（旧確認n33）との比較。line21/46とsummaryに **point_lower=True / diagnostic=INCONCLUSIVE** が保存されている。「R11で確定的改善」と書かない。元11-piece allocationの実装上の下限と、一般的な最適Rを分ける。

これらの判定は原fieldの転記・一致確認のみ。新しいG、比率、平均、SE、順位は計算していない。±2 combined-SEは元の正規近似感度診断であり、厳密CI・母平均保証ではない。探索と確認をpoolせず、cos/sinのpairや複数精度を独立標本に数えない。analytic shot Nと価格標本n、compiled価格とmodel価格を区別する。

旧長時間集合の33/40、7候補後の別集合の33/40、境界T2限定集合は別の保存版である。今回40セル全体を更新していない。A11の既存結論「比較方向は条件・対照・取得集合に依存する」を反転させる追加証拠ではないため、HIGHにしない。[G01][G01]、[G02][G02]、[既存境界追補][ADD]

## SOURCE・公開projection・STOP

境界実行SOURCE `ad1286021bdc0e5a0b1760064cefb3c1168b24eb`、会計SOURCE `0f2d167f9c928ba94bfad1a167bf4d3f5964ad1a` は [公開summary][SUMMARY] に記録された識別子。public result carrier `202d25eb34be365aaa6638c6353e25f51acebe42`、文書追補carrier `fbf701356cbc4da6ca8e3b7ef53533e1763243f9` と区別する。完全SOURCE/private rawの再監査・再公開は行っていない。結果projectionがprivate原本の全保証を持つとは扱わない。summaryの `AUDITED_FIXED_SCOPE_COMPLETE_MANDATORY_STOP`、`mandatory_STOP=true`、`workers_joined=true` は保存値として確認した。正確な開始・完了時刻は公開summaryから取得できず、commit日時を実行日時へ代入しない。

H8の不足は別条件：linear1Å/STO-3G/DF28/T=.8/S2/L_D7/q16/r4/K4/delta=.05/P0のepsilon .001 signal原記録。既存epsilon .0005確認からNやGを逆算しない。`SOURCE_OR_DATA_MISSING_STOP` を維持し、既にある.0005の有限確認を未実施へ戻さない。[保存Phase A監査][AUDIT]

依頼で指定されたH8 Stage C `eb07156a37b1938ed3aed95c13abbb545466c573`（T1.6）、H4 RPE条件付き保証 `d4c016960ff4b1ca16cc73770147cde326989427`、測定backend契約 `655bf6cb84ab161706baf843a8ae255da39bf7a6` はLOCAL側の後続参照として区別した。自環境の境界3候補と同一実験へjoinせず、数値・保証を再検証していない。H8 .8の欠測をT1.6 Stage Cの不成立理由にしない。

## 自環境delta・既存資産の再利用

正式共通運用指示2026-10-25の全文、特に§8.1/§8.4/§10、PROJECT_MAP、研究概要全文、関連正本・状態表・manifestを確認した。既存inventory、11月3日main収録差分/guard、11月4日post-collection差分の登録対象を再利用した。

- 元rootの49 local branch headsは既存guard・11月2日inventoryと一致。別の既存Phase 1 cloneの2 headsも保持されている。新たなroot branchは文書案内用1件で、並行中のdirty文書に触れていない。
- 既存61件の比較可能な未コミットmetadata/source文書はGit blob一致。過去のWP11 dirty SOURCE binding12件は履歴として保持。baseline blobがUNKNOWNの既存大きなmetadata CSV1件は、今回も変更判定不能と明記する。科学結果の新規性をそこから推定しない。
- 既存99原本の公開manifestに対し、観測mainのcollected_path/blob/modeは99件すべて一致。[公開manifest][ARCH]。内容収録は全branchの履歴merge・active SOURCE採用・外部再現を意味しない。
- 既存台帳の保全場所64件はstat上すべて現存。既知の旧checkout10箇所は今回の位置では取得できず、残存ref・固定公開資料・保全場所の確認と区別した。未確認の移転先を探索せず、raw全体のhash/復元認定はしない。今回branch/worktree/archiveの削除・移動は0。

Phase 2のCANONICAL_CATALOG/RECORD_LINKS/UNRESOLVED/RELATIONSHIPSは既存REPORTと後続coverage/G01/G02の固定参照を通して利用し、再展開・再統合していない。SERIES_INDEX/REFERENCE_INDEX/SOURCE_COMPONENTS/INPUT_FILES、21 claim・44 coverage行、既存claim mapを読取した。新canonical ID、全体claim map、実験数は作成していない。

既存 `metadata_integration.py`、`verify_outputs.py`、`check_coverage.py`、`validate_synthesis.py` の固定Git参照、CSV schema、literal照合、公開参照検査の方式を静的に再利用した。今回の差分列と旧private guardはschemaが異なるため、stdlibだけの小さな一時adapterで登録済み対象のref/blob/stat照合と差分表を作成した。既存大容量inventory generator・科学runnerは実行/importせず、新しい科学用実装・testは追加しなかった。

原SOURCE/result/grant/ledger/marker/STOP、元index/config/HEAD、既存研究契約は変更していない。共有設定・他ユーザー・他hostへの操作なし。計算資源の人工上限、環境変更、新規依存導入なし。公開差分はこのディレクトリの4資料だけで、private保存先・内部host/IP・seed/event・raw/QPY・入力配列は含まない。

## LOCAL向け引継ぎと未取得範囲

1. 追加参照の最小単位は結果 `202d25eb34be365aaa6638c6353e25f51acebe42` のREPORT/RESULTS/EXECUTION_SUMMARYと文書追補 `fbf701356cbc4da6ca8e3b7ef53533e1763243f9`。固定blob/SHA/selectorはEVIDENCE_REFS.tsvを使う。
2. A11.BASEと全体統合の長時間節に、H4 R感度とH6 INCONCLUSIVEを各一文追加する案。掲載配置・中心claimの採択はLOCAL/ユーザーへ戻す。原Phase3、旧33/40、G01/G02を上書きしない。
3. SOURCEと公開carrier、固定確認対照の共有、探索のみのR32、有限標本/経験的allowance、H8欠測STOPを必要な箇所に残す。
4. 現在の主要主張を変え得る追加不足は今回確認されなかった。上記最小追記に新しい科学計算は **不要**。希少event母集団保証・状態準備コストは再開しない。
5. 未取得はH8の元epsilon .001 signal記録、正確な境界実行開始/完了時刻、完全private SOURCE/rawの再監査、baseline未知metadataの過去bytes、旧checkout10箇所の現在位置。SERVER2のG03/G04原本や新しいLOCAL成果の独立検証は担当外で、他環境へ接続していない。これらを未実施の科学実験や全研究の未完成と読み替えない。

本結果はSERVER 1の差分確認であり、PR研究全体の科学的受理・必要十分性・最終総コスト優位を認定しない。公開branchのremote SHAと4資料のbytes/blobを照合後 **mandatory STOP**。LOCALへ引継ぎ、新しい科学計算・詳細科学レビュー・全研究再統合・論文執筆・旧研究再開へ自動移行しない。

[P1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2de84ed7af00e73339886429aabab0bf8180e9f0/docs/research/pr_cross_host_coverage_audit_20261102/inventory/SERVER_1/REPORT.md
[P2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/604f0235840feb18d368f99ea0828f370ce70fc8/docs/research/pr_cross_host_coverage_audit_20261102/integration/REPORT.md
[P3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/235e56ee7d5cac4682775a62bc2a8262f15047dd/docs/research/pr_cross_host_coverage_audit_20261102/coverage/COVERAGE_MATRIX.csv
[SYN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ce2b5f161e94567fec322d694fd8b01d03cd854/docs/research/pr_research_paper_evidence_synthesis_20261103/RESEARCH_SYNTHESIS.md
[MAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ce2b5f161e94567fec322d694fd8b01d03cd854/docs/research/pr_research_paper_evidence_synthesis_20261103/CLAIM_EVIDENCE_MAP.csv
[G01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/G01_CANDIDATE_BINDINGS.csv
[G02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdc7fe58e7e887786322c6204db3654111873958/docs/research/pr_cross_host_coverage_audit_20261102/g01_g02_reconciliation/G02_COHORT_BINDINGS.csv
[BR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/202d25eb34be365aaa6638c6353e25f51acebe42/analysis/pr_boundary_execution_20261027/REPORT.md
[RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/202d25eb34be365aaa6638c6353e25f51acebe42/analysis/pr_boundary_execution_20261027/RESULTS.csv
[SUMMARY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/202d25eb34be365aaa6638c6353e25f51acebe42/analysis/pr_boundary_execution_20261027/EXECUTION_SUMMARY.json
[ADD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fbf701356cbc4da6ca8e3b7ef53533e1763243f9/docs/data_collection/2026-10-27_pr_boundary_addendum/REPORT.md
[AUDIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a4373f51d7acb5d70fda894b4ad66f8bc781177d/analysis/pr_boundary_phase_a_20261027/H8_Q16_REUSE_AUDIT.md
[ARCH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/server1_onsite_collection_20261103_files.tsv
