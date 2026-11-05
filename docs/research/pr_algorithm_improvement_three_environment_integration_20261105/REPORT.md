# PRアルゴリズム改善研究：LOCAL・SERVER 1・SERVER 2固定証拠の最終統合

作成・検査：2026-11-06 JST。資料cutoffは指定された2026-11-05固定監査版。
独立branch：`local/pr-improvement-three-environment-integration-20261105`。
これは既存証拠の静的統合であり、新しい科学的採択・詳細レビュー・数値評価ではない。

## 成果物と到達点

| 成果物 | 内容 |
|---|---|
| [IMPROVEMENT_VERIFICATION_LEDGER.csv](IMPROVEMENT_VERIFICATION_LEDGER.csv) | 129行・48列。70旧研究索引と59の段階・関連資料。全236入力観測の元ID/statusを追跡 |
| [IMPROVEMENT_GAP_MATRIX.md](IMPROVEMENT_GAP_MATRIX.md) | RQ-01〜07と追加PF/BCH/FR/研究C/RPE/access系列、解消したGapと残る範囲 |
| [DUPLICATION_AND_REUSE_MAP.md](DUPLICATION_AND_REUSE_MAP.md) | 8種類の関係、修正鎖、input/cohort対応、機能/入出力/精度/制約、全143資産群・179固定版 |
| [REMAINING_VALIDATION_MAP.md](REMAINING_VALIDATION_MAP.md) | 追加検証が必要になる条件、既存資産、作業・許可の境界。優先順位・必須backlogではない |
| 本REPORT | 固定入力、照合粒度、確認済み範囲、UNKNOWN、保全・公開確認、STOP |

129行は科学実験数ではない。112＋72＋52を足して実験数へ換算していない。
LOCAL旧112行は歴史的snapshotとして維持し、追加17行と3環境の参照・SOURCE・条件・更新関係を統合した。
既存coverage Phase 1〜3の119,304観測行・claim別coverageは再生成・再統合・変更していない。
その固定PROTOCOLが除外したアルゴリズム改善領域を別の粒度で補う。

## 1. 正式指示・固定入力・資料取得

`PROJECT_MAP.md`、適用`AGENTS.md`、利用可能な正式`research_project_instructions_unified_2026-10-25.md`を実際に確認した。
正式共通指示§8.1/8.2/8.4/10/10.3に従い、許可範囲・SOURCE/結果保全・STOP・共有環境非干渉・既存資産の再利用を維持した。
入口の研究概要・現状、normative研究文書、VALIDATION_STATUS/manifest、研究統一索引、11/1科学landscape/gap、旧cross-host coverageの区分はLOCAL調査から引き継ぎ、今回必要な原固定資料へ戻った。
新しいformal研究判断は行っていないため、既存研究正本・manifest・旧索引への変更を加えていない。

| 環境 | 固定入力commit | 入力directory |
|---|---|---|
| LOCAL | `ba14f71128362d197bfa47c9d0effacd0d7174d9` | [固定暫定5成果物](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/ba14f71128362d197bfa47c9d0effacd0d7174d9/docs/research/pr_improvement_verification_inventory_local_20261105) |
| SERVER 1 | `b6a6fa81b98307beae1d89eef34eeab0c8805b6e` | [固定改善専用監査10ファイル](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/b6a6fa81b98307beae1d89eef34eeab0c8805b6e/docs/research/pr_algorithm_improvement_server1_audit_20261105) |
| SERVER 2 | `0df7646826b3f863b19e69b204140059f9d09112` | [固定改善専用監査4ファイル](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/0df7646826b3f863b19e69b204140059f9d09112/docs/research/pr_server2_improvement_evidence_20261105) |

3commitをbranch/main HEADで代用せず、全19ファイルをGitHubから新たに取得した。
取得本文を改行保持で保存し、対応する固定Git objectとbytes/Git blobを照合した。全件一致。
次表は監査metadataであり、科学rawの再集計ではない。

| 環境／固定入力ファイル | bytes | Git blob |
|---|---:|---|
| LOCAL / [IMPROVEMENT_VERIFICATION_LEDGER.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba14f71128362d197bfa47c9d0effacd0d7174d9/docs/research/pr_improvement_verification_inventory_local_20261105/IMPROVEMENT_VERIFICATION_LEDGER.csv) | 306083 | `41706adf5a7ef2543949b50a309a6c3552d0b541` |
| LOCAL / [IMPROVEMENT_GAP_MATRIX.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba14f71128362d197bfa47c9d0effacd0d7174d9/docs/research/pr_improvement_verification_inventory_local_20261105/IMPROVEMENT_GAP_MATRIX.md) | 10080 | `361b02d1482fedf29932e5b3a4e08cca03c55eb5` |
| LOCAL / [DUPLICATION_AND_REUSE_MAP.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba14f71128362d197bfa47c9d0effacd0d7174d9/docs/research/pr_improvement_verification_inventory_local_20261105/DUPLICATION_AND_REUSE_MAP.md) | 11245 | `9fdf0491c458fa9d1c21b8bf41afe75f613c0b7f` |
| LOCAL / [REMAINING_VALIDATION_MAP.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba14f71128362d197bfa47c9d0effacd0d7174d9/docs/research/pr_improvement_verification_inventory_local_20261105/REMAINING_VALIDATION_MAP.md) | 8563 | `cc4f09627c9cf8202856de4694cb9ccb2de88a69` |
| LOCAL / [REPORT.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba14f71128362d197bfa47c9d0effacd0d7174d9/docs/research/pr_improvement_verification_inventory_local_20261105/REPORT.md) | 15720 | `488cb9d6487d56094c98da1aaf07534f65ad4edf` |
| SERVER1 / [SERVER1_IMPROVEMENT_EVIDENCE.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6a6fa81b98307beae1d89eef34eeab0c8805b6e/docs/research/pr_algorithm_improvement_server1_audit_20261105/SERVER1_IMPROVEMENT_EVIDENCE.csv) | 163311 | `f5857a964f331dd3d30105119616398203dfdb62` |
| SERVER1 / [SERVER1_EVIDENCE_GAPS.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6a6fa81b98307beae1d89eef34eeab0c8805b6e/docs/research/pr_algorithm_improvement_server1_audit_20261105/SERVER1_EVIDENCE_GAPS.md) | 6066 | `30d1b976459df8959d4cc9904fb6decfcc4e8956` |
| SERVER1 / [SERVER1_REUSE_MAP.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6a6fa81b98307beae1d89eef34eeab0c8805b6e/docs/research/pr_algorithm_improvement_server1_audit_20261105/SERVER1_REUSE_MAP.md) | 8263 | `12c0e1d0bb7be38c224c5f779ac39f894e153080` |
| SERVER1 / [REPORT.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6a6fa81b98307beae1d89eef34eeab0c8805b6e/docs/research/pr_algorithm_improvement_server1_audit_20261105/REPORT.md) | 13323 | `b871ee1f0b16617eb1637912d4be9f98a8424ff4` |
| SERVER1 / [REFERENCE_INDEX.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6a6fa81b98307beae1d89eef34eeab0c8805b6e/docs/research/pr_algorithm_improvement_server1_audit_20261105/REFERENCE_INDEX.csv) | 559624 | `05f2110e5b448db96751e05259a2e17e9ac05ece` |
| SERVER1 / [CODE_ASSETS.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6a6fa81b98307beae1d89eef34eeab0c8805b6e/docs/research/pr_algorithm_improvement_server1_audit_20261105/CODE_ASSETS.csv) | 57532 | `17c1ae26311880c2aa6445127719473fd47bb01a` |
| SERVER1 / [INPUT_COHORT_BINDINGS.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6a6fa81b98307beae1d89eef34eeab0c8805b6e/docs/research/pr_algorithm_improvement_server1_audit_20261105/INPUT_COHORT_BINDINGS.csv) | 6133 | `67081729eec01a1f44d48024f57d66717478665b` |
| SERVER1 / [FIXED_REF_RETENTION.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6a6fa81b98307beae1d89eef34eeab0c8805b6e/docs/research/pr_algorithm_improvement_server1_audit_20261105/FIXED_REF_RETENTION.csv) | 28383 | `70c8818841c0e310abb34dee5e2be0aeab99ac0a` |
| SERVER1 / [GITHUB_REFERENCE_CHECKS.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6a6fa81b98307beae1d89eef34eeab0c8805b6e/docs/research/pr_algorithm_improvement_server1_audit_20261105/GITHUB_REFERENCE_CHECKS.json) | 1874 | `14c83087cde4751f255088ec27fed6a95077b294` |
| SERVER1 / [AUDIT_METADATA.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6a6fa81b98307beae1d89eef34eeab0c8805b6e/docs/research/pr_algorithm_improvement_server1_audit_20261105/AUDIT_METADATA.json) | 5275 | `5957a146c2a8375e4dbd2062bb9b6ac7cd40064f` |
| SERVER2 / [SERVER2_IMPROVEMENT_EVIDENCE.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0df7646826b3f863b19e69b204140059f9d09112/docs/research/pr_server2_improvement_evidence_20261105/SERVER2_IMPROVEMENT_EVIDENCE.csv) | 194888 | `369fb9eed80d025ec4456d68556f0d4846ed9f72` |
| SERVER2 / [SERVER2_EVIDENCE_GAPS.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0df7646826b3f863b19e69b204140059f9d09112/docs/research/pr_server2_improvement_evidence_20261105/SERVER2_EVIDENCE_GAPS.md) | 5760 | `77f1b5774634c80be612b7814271c39a569f8acf` |
| SERVER2 / [SERVER2_REUSE_MAP.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0df7646826b3f863b19e69b204140059f9d09112/docs/research/pr_server2_improvement_evidence_20261105/SERVER2_REUSE_MAP.md) | 34593 | `bc76c58b2bd1084c9a4d52d2585059f9f48a15e7` |
| SERVER2 / [REPORT.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0df7646826b3f863b19e69b204140059f9d09112/docs/research/pr_server2_improvement_evidence_20261105/REPORT.md) | 13877 | `d4cf854109ecc3726f6096918d24eb1acd394e54` |

各audit入力の取得完了と、各環境のprivate raw原本の現存確認は別である。SERVER 1/2へSSHしていない。
SERVER 1の原稿custody観測はS1-DOC-01〜03へ継承し、LOCAL自身の原本監査としない。
指定19ファイルに取得失敗はない。全原資料の再download/全private rawの確認を行ったとは主張しない。

### 原証拠へ戻った範囲

Phase 2C/I/B/F1/F2/PS48と正式一時終了、Track B G6〜G10、C1修正/回復/凍結、BF1中断/保存復旧、sampling survey、UWC探索、研究B/研究C旧summary/小stepの固定文書・metadataを読んだ。
後続RPE・研究C005/064/U64/固定word・Pauli-JW/Track AはLOCAL固定台帳の後続証拠を引き継ぎ、SERVER 2の旧変種summaryを最新結果の代わりにしなかった。
主要段階のSOURCE/result commit/path/元status/STOPは台帳へ収録した。実行SOURCE、implementation snapshot、result/report carrier、authorization-only commitを別の役割として残した。

SERVER 1 `REFERENCE_INDEX.csv`全1,280参照、統合台帳内の477参照binding、全179資産snapshot/path/blobをGit object metadataで照合した。
これらの重なりを除いた1,394 commit/path/blobがすべてblobとして一致した。
これはローカルに取得済みの固定objectに対する参照検査で、1,394資料すべての全文を新たに精読またはGitHubから再取得したという意味ではない。
公開済み圧縮case raw等は固定blobを照合するだけで、展開・数値再解析していない。元S1/S2の原証拠inspection深度は固定入力に残す。
別repositoryの高次PF等に関するLOCAL継承参照は今回再取得しておらず、実験独立性の統合根拠に使っていない。

## 2. 統合方法と台帳の読み方

旧70研究索引と、具体的なrun/段階・baseline・資料保全記録を分けた。旧catalog ID/参照/継承条件によるfamily対応は、原実験を同定して一つへ潰す処理ではない。
同path・同Git blobの固定資料は同じ保存参照として関連付けた。異なる文書やrunは元selector/役割を保持し、同じ数値・研究名・Hamiltonianだけでは統合しない。
SERVER 1/2の124行の対応は、family索引80、同固定資料再引用25、同固定stageの別文書1、追加記録/参照18。
8種類の重複・独立性・修正・後続関係は再利用地図で示した。cohort独立性が不明なものはUNKNOWNのまま。

台帳の最初の40列はLOCALから継承したschemaで、旧112行の全field値を変更せず保持した。
したがって、その中の当時のhost未取得・SOURCE不明・read depth記述は歴史的な原観測である。
**現在の取得状態と追加SOURCE/条件/更新は次の8列と本報告を併読する。** 古いUNKNOWNを現時点の一律結論へ戻さない。

| 追加JSON列 | 内容と役割 |
|---|---|
| `origin_observations` | 環境、入力commit/path、元ID/selector、元status/未検証/独立性、S1 reference IDs |
| `evidence_bindings` | repository/固定commit/path/blob/selector/役割。S1は主結果を収録し、全参照は固定REFERENCE_INDEXの元IDへ戻る |
| `identity_assessments` | 同資料・同stage・family関連・追加記録の根拠。run/cohort独立性を自動認定しない |
| `source_bindings` | 各環境が記録したSOURCE commitと役割。implementation snapshotをexecution SOURCEにしない |
| `input_cohort_bindings` | S1公開6 bindingとS2公開入力識別子/試行・cohort/SOURCE部品。未記録値は未記録のまま |
| `precision_and_interface_notes` | 元compiler/precisionと、構成/実装/人工/物理correctness/native/shot/独立移送の層別記録 |
| `current_scope_update` | 後続結果・追加資料による現在の限定更新 |
| `unresolved_identity` | private raw/全seed/host/全closure等、今回閉じていない同一性 |

RFC4180形式のCSV。JSON列は元情報の対応を保持するための通常の来歴記録で、承認証明用システムではない。
追加17行も準備・gap・文書保全を含み、新実験と見なさない。全236観測が一つずつ対応し、元statusを変えていない。
旧source不明のfieldに新carrier commitを代入せず、既知になった範囲を別bindingで記録した。

## 3. 最新到達点・元暫定版からの主な更新

| 系列 | 固定証拠が示す状態 | 依然として支持しない主張 |
|---|---|---|
| F1 | metadata失敗を残し、修正SOURCE8192530b…/結果b0dad8cc…の登録correctness PASS。旧/新13 input hash一致 | 未資格score/selectorや同精度総費用優越 |
| F2 | 統計資格STOPを保存。旧SOURCE9b1e1031…のcompiled operator3 FAILと修正SOURCE9468c708…/結果dc7378ab…の数値・1-shot資源PASSを分離。旧/新7 input hash一致 | N/加重費用/γ資格・統計認証の完了。修正結果のN=nullを0にしない |
| Gaussian singleton | PS48の局所改善と悪化・一律RZ/CX削減不支持、選択規則未確立、正式一時終了を保持 | 全law期待利益・一般selector・Gaussian積の独立新規性 |
| G6〜G10 | 構成/finite-bit、G7/G8、G9v2/G10v3完了まで後続を反映。独立人工受入/実API512/限定pilotを分離 | 全物理provider/native/fullwrapperや一般IID/全task利益。G8/G10の旧key/case再利用を独立再現としない |
| H4 C1 | partition/state/hash修正、5A ordinary/fullとP5R各1024 logical pathを保持し、正式一時凍結 | 全324label strictnative/共通N/Dhalf/prep/全wrapper、全CPU境界の同等性 |
| BF/BM/非prefix | BF1中断と保存復旧を分離。BF-A限定負結果・BM既知compact/BCH対応。非prefix具体差は未確立 | 元BF1 runの科学完了、F固有decision gain、非prefix一般no-go |
| sampling freedom | 10/30 surveyのNO_DISTINCT_FREEDOMと既存13人工test記録を追加 | 実H4 samplerの新実行、全sampling lawの不可能性証明 |
| UWC/旧DF | 指定JSON/sweepのFOUND_UNVERIFIEDを追加。元run provenance未結合、旧DF修正入力NOT_FOUND_IN_SEARCH_SCOPEと失効維持 | 旧UWC/DF科学結果の復活・新検証 |
| 研究B/C/RPE | 研究B別系列・C旧summary/一small U-stepを追加。LOCALの後続005/064/r32・固定word、条件付き18round/energy接続を維持 | 旧条件から他条件への一律優越。自己目的/nominalと厳密CA/全lawの同一視。round別表現/split新方式と最終総費用の完了 |

上表のPASS/FAIL/負結果/STOPは原報告の保存状態を引用したもので、今回の新科学判定ではない。
F1/F2/PS48は4mode/2粒子/occupied(0,1)/LD1/残差2density-squareの合成fixtureで、分子geometry/AO basis/分子DF rankは非該当。
F1はT1/5 q2 δ1/10 r2 h1/20 K2、F2/PS48はT1/5 q1 δ1/5 r2 h1/10 K0。compilerと高精度参照条件はGap/再利用表・元記録に保持した。
C1はlinear H4 1Å/STO3G/legacyDF12/generation-prefix LD3、13basis/324label、T4/5 q1 r4 m5 h1/5、TAIL_ONLY_PLUS_SCALAR。
他の研究のgeometry/basis/rank/state/split/窓は台帳の元条件へ戻り、同名系列だけで移送しない。

## 4. コード・検証資産の静的再利用

S1 CODE_ASSETS全93版（main carrier83、過去版10）、S2 CSV49版と再利用map75参照、LOCALの固定SOURCEリンクを照合した。
合計179 snapshot/path/blobを、同path同blobだけで143群へまとめた。全snapshotと出所IDを再利用表に保持し、全143 Python blobのASTを読んだ。
機能分類には同名C1の区別も反映し、representation macrocycle C1とTrack B H4 whole-path C1を結合していない。
関数/module/runner/testの所在、入出力、元domain/precision、過去検証と未資格部分、流用条件、必要となり得る拡張をprofileごとに整理した。

静的コード存在は新条件でのtest PASSや科学的利益でない。過去テストの件数も実験数へ換算していない。
既存serializer/reference/phase保持compile/return kernel/DP/配分/metadata読取を再利用候補とし、不要な再実装範囲を明示した。
既存研究runnerは科学処理へ接続するため起動せず、CSV/JSON/AST/Git object照合に限った一時的なschema対応処理を用いた。研究用helper/新algorithmは作っていない。

## 5. 残るUNKNOWN・未取得原本

- 全系列を跨ぐ単一execution SOURCE closure、全seed/実行host/各private rawと完全な独立性は未確認。公開資料の同blobは実行cohort独立性でない。
- UWC旧元runのSOURCE/command/当時環境/H-state-groupingへの結合、旧DF corrected input/anchorは未解消。限定探索の未発見をrepository全体不存在としない。
- G6 512のreport-level `artificial_only=True`とtrial-level false/PINNED_G7_F3/core loadの食い違いは元metadataのまま。実API記録を消さず、metadata矛盾も修正していない。
- C1旧P5 worker停止の直接原因・worker内処理済trial数、全label strictnative・共通認証N・全wrapper/準備/readout・公平なCPU境界は未解消。
- F2統計資格/γ/N加重、singleton未使用入力selector、全law/母平均・厳密CA・全実装資源、round別表現/split、connectivity条件の改善は部分検証または選定資料内で証拠未確認。
- formal指示そのもの、各private environment/raw/approval原本は公開成果物へ転載しない。既に公開された旧契約の存在を新承認証明の要件へ拡張しない。

`未実装`、`実装あり未実行`、`実行未完了`、`技術失敗`、`限定負結果`、`既知同値`、`資料未結合`、`厳密保証未成立`、`正式保留`を別状態として残した。
不足を推測で補完せず、追加検証は必要な主張をユーザーが別途選んだ場合の条件付き提案に限定する。

## 6. 技術検査・保全・公開照合

| 検査 | 結果・確認範囲 |
|---|---|
| 固定入力取得 | 指定19ファイルのGitHub本文と固定Git blob/bytes一致 |
| CSV構造 | 129行/48列、ID一意、8 JSON列のparse、CSV read/write roundtrip一致 |
| 原観測保全 | LOCAL112行の元40field値は全件不変。236入力観測の元ID/statusに欠落・二重対応なし |
| 参照 | S1全1,280参照、統合477 binding、179コード版を含む1,394固定commit/path/blobが一致。selector/役割保持 |
| input/cohort | 6 bindingの元SOURCE/closure/run/input字段一致。F1旧新13、F2旧新7、PS48の4 input metadataを文字列照合 |
| 資産 | S1全93資産ID、S2/LOCAL固定参照を179版/143群へ対応。143 ASTを静的に読取、未import・未実行 |
| 文書整合・公開対象 | 5成果物間のlink/主要ID、禁止公開文字列、変更範囲を検査。公開対象は新directoryの5ファイルだけ |
| Git保全 | LOCAL固定commitをbaseとする隔離checkout/新branch。旧LOCAL5ファイル、coverage、研究SOURCE/input/raw/result/grant/ledger/marker/STOP/archiveに差分なし |
| 公開 | 対象remoteはHIROMU1015/Partially-Randomized-Trotter。通常pushのみ。公開commitはこのREPORTを含むGit snapshotで特定し、remote ref SHAと5ファイルの本文/bytes/blobを再取得照合して終了 |

固定Git検査は過剰な環境同一性の要求でなく、今回指定された文書/来歴同一性の照合である。科学数値・順位・性能を再計算していない。
隔離checkoutは共有object/alternates方式で、独立したbackupと主張しない。元作業領域・dirty/untracked・既存branchを変更していない。
公開後の最終SHAと取得照合の結果は最終チャットで示す。このファイルに自己commit SHAを埋め込むための再commitは行わない。

## 7. 未実施作業とmandatory STOP

新科学計算、数値再解析・再ランキング、Hamiltonian/state生成、simulation/sampling/RNG replay、回路build/compile、科学runner/testの起動、旧run retry/resume、新algorithm/研究helper実装、詳細科学レビュー、新保証/新規性/研究方向の採択を行っていない。
旧契約やSTOPの解除、private原本の転載、共有環境設定・他ユーザーのjob/fileへの操作も行っていない。

**mandatory STOP：3環境統合・5成果物の文書検査・専用branch公開・GitHub取得照合が完了した時点で停止する。**
Codexが自動開始する次研究作業はない。今回の完了に追加科学検証は不要。
次の担当はユーザー。必要な場合に限り、残検証地図のどの主張・限定資料照合を別依頼にするか選ぶ。具体的研究方向の優先順位は本統合では決めない。
