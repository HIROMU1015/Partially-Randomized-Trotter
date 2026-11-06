# 新A：PR × reference干渉 — 18round CX/RZ価格上界・位相誤差・強baselineの独立科学レビュー

- **レビュー日**：2026-11-07（JST）
- **レビュアー**：Claude（Cowork）。依頼書「Claude向け個別科学レビュー依頼書（2026-11-07）」の承認範囲内で実施
- **対象**：`HIROMU1015/Partially-Randomized-Trotter` 固定commit `da9fceb944dae27f397e48734be30c6864e15f44`（2026-11-07 00:45:46 +0900、"research: qualify fixed new-A long-round CX/RZ bounds and record STOP"）
- **実施した作業**：固定SOURCE・保存成果物・関連固定REPORT・Qiskit 1.3.0一次SOURCE・指定arXiv文献の読取、保存値どうしの照合（hash照合、保存フラグの参照）、紙上導出と手計算による検算
- **実施していない作業**：新たな数値再解析・再集計・最適化・simulation・回路build/compile・sampling・合成・価格取得。科学SOURCEのimport／実行、GitHubへの書込、既存runの再開、追加実験の起動。既存SOURCE・raw・成果物・STOPの変更

本文中の証拠区分は次の記号で示す。

| 記号 | 意味 |
|---|---|
| **[S]** | 固定commitのSOURCE・保存成果物（JSON/TSV/archive）またはQiskit 1.3.0 SOURCEをレビュアーが直接読んで確認 |
| **[C]** | Codex報告（REPORT/JSONの記述）の引用。レビュアーは独立に再確認していない |
| **[D]** | レビュアーによる紙上導出・手計算の検算。shell（jq・sha256sum）は保存記録の表示、件数の確認、hash照合にだけ使い、価格・誤差・shotの再計算には使っていない |
| **[M]** | 結論が依存するモデル仮定 |
| **[N]** | 未取得・未資格・本レビューで確認できなかった事項 |

---

## 0. 総括（結論の先出し）

| 問い | 判断 | 確信度 |
|---|---|---|
| **A. 価格保証**：固定compiler A・登録実装について、全18roundの期待総CX/RZで「reference上界 < baseline下界」が導けるか | **導ける（条件付き）。** 有限context、chunk境界のfence、D部分のCX式、basis-run期待値式のいずれにも反例・致命的欠落は見つからなかった。ただし中心補題（境界fence補題とcontext網羅）はREPORT本文に証明として書かれておらず、コード内のassertと保存データで担保されている。総費用の大半を占めるr≥4・q≥512のroundには直接のholdoutがなく、構造論証だけが根拠である | 中〜高 |
| **B. 公平性**：14-CX baselineに対する優越を研究claimとして採用できるか | **「登録14-CX Simon–Love実装に対して」という限定付きでのみ採用可。** 登録controlled-RZZはQiskit汎用`.control()`の出力（CCX・CRZ・CCX）で、14 CX・24 RZになる。4-CX分解（parity–CRZ–uncompute）は紙上で厳密に正しく、SE-QPE本文にも同形のkernelがある。保存event記録では、RTE回転のほぼすべてがZZ型だった。保存pathの価格差は構造式で厳密に再現できる。紙上導出では、4-CX baseline（以下B′）に対してもCX差の符号は保たれる見込みだが、event比例部分は約1/6に縮む。報告の約19%はB′へ移送できず、大幅に小さくなる可能性が高い。RZ差（約4%）の大半は登録loweringに由来し、B′比では実質的な意味を失う見込みである | CXの符号：中／CXの大きさ：未定（縮小の見込み高）／RZ：低 |
| **C. 成功保証**：誤差・丸め・shotの接続は正しいか | **固定Hermitian行列・理想native gate・IEEE binary64・finite-bit lawの範囲で整合。** 二重計上・欠落は見つからなかった。真空位相の丸め上界（最終round 4.538×10⁻⁴ rad）は過度に保守的だが、モデル内では妥当。ハードコード定数のうち336・49・24は、紙上で上界として正しいことを確認した。20も、読んだ範囲のbuilderでは安全な上界である | 中〜高 |
| **D. 研究判断** | **限定継続を推奨。** 承認を得たうえで、B′の最小技術検証と18round厳密再会計だけを行う。結果が事前登録した閾値を下回れば一時保留とする。主要claimの対外表明はB′の結果が出るまで保留する。新C、旧C/C2、新BのSTOPは変更しない | — |

**価格証明そのものに致命的不備は見つからなかった。** 結論を弱める主因は証明ではなく比較対象（baseline）の強さと指標の選び方にある。補強が必要な箇所は §3.8、§5.6、§8 に列挙した。

---

## 1. レビューの対象・現在の状態・固定条件・取得資料

### 1.1 対象と状態

- 判定 `CONDITIONAL_ALL18_CX_RZ_REDUCTION`、`STOP.json` は `MANDATORY_STOP`（stage1 COMPLETE、stage2 CX_TRANSFER_QUALIFIED、stage3 ALL18_ERROR_QUALIFIED_IN_FIXED_ARITHMETIC_MODEL）**[S]**
- `SOURCE.json` の `review_status` は "author verification only; no independent or formal scientific review" **[S]**。本レビューは、この状態に対する初回の独立科学レビューである。Codexの「PASS」「certificate」「QUALIFIED」という語は第三者保証とみなさない。本文では、独立に読み直して確かめた部分と、Codex報告に依拠した部分を証拠区分で分けている
- 基準commitは `79cba2f`（旧新A pilot）。新branchは `codex/pr-reference-transfer-qualification-20261106` **[S]**

### 1.2 固定条件（監査対象であり、真と仮定しない）

Linear H4（隣接1.00 Å）、STO-3G、保存DF rank12、Partial S2・LD3・h=0.02・finite-RTE K=2。18round：q=2^m（m=0..17）、r∈{1,2,4,8,16,32}（保存schedule）。energy精度1.5936×10⁻⁴ Ha、全失敗確率0.05、位相予算0.4 rad。各shotで元Hの厳密groundを供給し、その準備費は除外する。ordinary `diag(I,V)` 信号、canonical finite-bit law、fresh-shot RPE。compiler AはQiskit 1.3.0、`rz/sx/x/cx`、全結合、近似なし。登録baselineはSimon–Love型（basis非制御、中央対角のみ制御、D境界融合）。referenceは2-register F/G配置、真空reference、trajectory別位相補正、追加8 qubit、16 CSWAP、reset。いずれも **[S]**（SCOPE.md・REPORT.md・SOURCEで確認）。

### 1.3 資料の取得経路

- `gh`（GitHub API）は、このセッションでは当該リポジトリへのアクセスが有効化されておらずHTTP 403で失敗した。代替の正規経路として、公開リポジトリを **git over HTTPS（読取専用、blobless・sparse取得）** で固定commitから取得した。GitHubへの書込は行っていない。
- 取得したSOURCE 13本のSHA-256は、`SOURCE.json` の `new_source_sha256`／`inherited_unchanged_source_sha256` と一致した **[S]**。4つのarchive（`local_catalogue_boundary` `17381211…`、`local_catalogue_initial` `ba7f64a9…`、`native_circuits` `87b5374b…`、`raw_prices` `d0d306e2…`）も `publication_archives` のhashと一致した **[S]**。`EXECUTED_STAGE2_SOURCE.py` は最終 `pr_reference_transfer_qualification.py` とbyte一致した。`EXECUTED_CATALOGUE_SOURCE.py` と最終版の差分は、境界観測（`pre_last_cancel`）の追加だけである **[S]**。
- Qiskit 1.3.0 SOURCEは公式リポジトリのtag `1.3.0`（commit `896d1c8b`）から該当ファイルだけを読取取得した。
- arXiv 3本は取得用エージェント（web-fetch）経由で読んだ。HTML本文の取得は途中で打ち切られ、後半は要約で返された（確認深度は下表）。

### 1.4 資料一覧と確認深度

| 資料 | 確認深度 | 備考 |
|---|---|---|
| REPORT.md（da9fceb） | 全文確認 | |
| SCOPE.md | 全文確認 | |
| 成果物README | 全文確認 | |
| SOURCE.json / STOP.json / INPUTS.json / EXECUTION.json / ADAPTER_CHECKS.json / LOCAL_CATALOGUE_SCOPE.json / STAGE2_PREDECLARED.json | 全文確認 | hash照合を含む |
| STAGE4_ROUNDS.tsv / PAIRED_NATIVE_PRICES.tsv | 全文確認 | |
| STAGE2_CX_CERTIFICATE.json | 全文確認 | 128 checkを全件表示。64 contextは集約フラグで確認 |
| RZ_PATH_BOUND_CHECKS.json | 関連箇所確認 | `all_passed` と行の一部（32/128） |
| STAGE3_ALL_SUPPORT.json | 関連箇所確認 | 要約フィールド、m=0とm=17の行 |
| STAGE3_LOCAL_INTERVALS.json | 関連箇所確認 | 件数（445）、族別最大上界、解析check件数（2600）。個別区間の再評価はしていない |
| STAGE4_RPE.json | 関連箇所確認 | totals・フラグ・m=8とm=17の行。有理数価格の10進変換・再集計はしていない |
| STAGE1_STRUCTURE.json / ENVIRONMENT.json | 関連箇所確認 | |
| `local_catalogue_boundary.tar.xz` | 関連箇所確認 | 全12 contextの `post_unroll`、fenceフラグ、chunk別CX/RZ、CRZZ・CSWAPのlowering |
| `raw_prices.tar.xz` | 関連箇所確認 | key構成とevent角度 |
| `local_catalogue_initial.tar.xz` | 今回は不要 | 最終版はboundary版を採用 |
| `native_circuits.tar.xz`（QPY） | 今回は不要 | QPYの読込は回路処理にあたるため開いていない |
| `pr_reference_transfer_{qualification,catalogue,cost_bounds,error,accounting}.py` | 全文確認 | |
| `pr_reference_interference_pilot.py`、`track_c_compiler_a.py`、`track_c_h4_finite_bits.py` | 全文確認 | |
| `track_c_compiler_a_18round.py`、`df_rte_qiskit.py`、`df_partial_s2.py`、`df_partial_s2_repeated.py`、`rte.py`、`df_trotter/ops.py` | 関連箇所確認 | 区間行列、RTE builder、block生成、event角・位相 |
| `track_c_h4.py`、`track_c_h4_price.py` | 未取得（未精読） | `d_part`・`conditional_flags`・`diagonal_primitives`。価格samplerとq=1のD部分に関係するが、結論には直接効かない |
| 旧新A pilot REPORT（79cba2f） | 全文確認 | |
| 新A/B静的調査 REPORT（2397c5a） | 関連箇所確認 | SE-QPE適用条件、fresh-shot単一block |
| 18round固定保存行列保証 A_REPORT（720416c） | 全文確認 | |
| compiler A correctness REPORT（f45e9b0）／18round compiler構造上界 REPORT（3258a89） | 関連箇所確認 | pass列、context、位相の扱い |
| Qiskit 1.3.0：`inverse_cancellation.rs`/`.py`、`basis_search.rs`、`circuit_data.rs`（`set_global_phase`）、`add_control.py`、`rzz.py`、`xx_plus_yy.py`、`x.py`（CCX）、`swap.py`（CSWAP）、`rz.py`（CRZ）、`p.py` | 関連箇所確認（InverseCancellationとbasis_searchは全文） | BasisTranslatorの `compose_transforms.rs`、`mod.rs`、UnrollCustomDefinitionsは未精読 |
| SE-QPE（arXiv:2604.14921v2） | 関連箇所確認（部分） | abstractと§I–IVの一部は逐語、以降は取得ツールの要約 |
| Simon–Love（arXiv:2511.13855v1） | 全文確認（取得エージェント経由） | |
| PR原著（arXiv:2503.05647v2） | 関連箇所確認（部分） | abstractと§I–IVは逐語、§V以降・付録Eは要約のみ |

---

## 2. 重要結果の表（証拠区分つき）

| 項目 | 値・内容 | 区分 | レビュー所見 |
|---|---|---|---|
| 判定 | `CONDITIONAL_ALL18_CX_RZ_REDUCTION` | [S] | 登録実装・固定算術モデルに限定された判定として妥当 |
| 総CX（期待値） | baseline [593,162,239,027.1154, …1156]、reference ≤ 480,289,613,434.8142 | [S]（再集計なし） | 報告値の比は約0.81（紙上概算） |
| 総RZ（期待値） | baseline [3,735,644,723,605.280, …281]、reference ≤ 3,577,340,249,201.294 | [S]（再集計なし） | 報告値の比は約0.96。§4・§6のとおり意味は限定的 |
| 十分shot数 | 8,598 / 8,606（最終round 626 / 630 per axis） | [S]＋[D] | 式への当てはめを手計算で検算し、整合した（§5.4） |
| 追加reset・CSWAP | 68,848 = 8×8,606、137,696 = 16×8,606 | [S]＋[D] | 整合 |
| qubit数 | 9 / 17 | [S] | |
| depth・size・T | 全18roundで未資格・未取得 | [S] | 補完されていないことを確認 |
| 局所CX | XX+YY 2、RZZ 2、CRZ 2、登録CRZZ 14、CZ 1、CSWAP 8 | [S] | 全12 contextで不変（保存catalogue） |
| 局所RZ | XX+YY 18、RZZ 1、CRZ 2、登録CRZZ 24、CZ 4、CSWAP 11、Z 1、H 2、Sdg 1 | [S] | RZ数にはClifford・T型・任意角が混在する（§6.2） |
| D部分CX | baseline 1090q+72、reference 984q+272（q≥2）、q=1は1184 | [S]＋[D] | 保存chunk数から紙上で再導出し、一致した（§3.4） |
| D部分RZ | baseline 6180q+664、reference 6168q+1520（q≥2） | [D] | 保存chunk数から本レビューで導出。REPORTに記載はない |
| basis-run期待値 | Σ_b c_b[p_b+(r(1+2p₂)−1)p_b(1−p_b)]（occurrenceあたり） | [S]＋[D] | 指示関数分解で厳密に導出できた（§3.4） |
| 境界fence | baselineの対象chunkに露出X/CX prefixなし | [S] | 全8 baseline contextの保存フラグで0件を確認 |
| holdout照合 | 旧48＋新16組×2軸＝128件でbaseline一致・reference包含 | [S] | r≥4・q≥512は未検査（§3.5） |
| reference実価格 < 上界 | q256・r2で8〜22 CX少ない | [S] | r≥2でRZZ間のCX相殺が実際に起きている |
| 最終round位相誤差上界 | 4.5377757×10⁻⁴ rad（naive 4.5372585×10⁻⁴、正規化 5.172×10⁻⁸） | [S]＋[D] | N=124,780,568、γ_N·M を手計算で検算し、一致（§5.3） |
| 最終round native joint operator誤差 | baseline 5.876×10⁻⁷、reference 5.816×10⁻⁷ | [S] | |
| 最終round total raw disk | baseline 3.678×10⁻⁶、reference 4.574×10⁻⁴ | [S] | referenceは位相丸め項が支配的 |
| 代替4-CX分解 | 正しい（理論上4 CX） | [C]→[D] | 紙上で厳密に証明した（§4.2）。native価格・総価格は [N] |
| RTE回転のsupport型 | 新16 pathで単一Z回転はpathあたり0〜2件、残りはZZ型 | [S] | P_ZZ≈1を示唆。登録CRZZの影響が大きいことの根拠（§4.3） |
| 保存path価格差の構造式による再現 | q128・q256のCX/RZ差が構造式と一致 | [D] | 構造式とD部分RZ式の独立検算（§4.3） |
| 元の連続積分Hの包含、hardware、noise | 未資格 | [S][N] | REPORTも主張していない |

---

## 3. 価格構造の保証の独立検算（重点A）

### 3.1 前提：compiler Aのpass列とQiskit 1.3.0での意味論

**pass列 [S]**（`track_c_compiler_a.py` L37–42）：RejectOpaqueUnitary → InverseCancellation（CX, X, H, Z, (S,Sdg), (SX,SXdg), (T,Tdg)）→ UnrollCustomDefinitions → RejectOpaqueUnitary → BasisTranslator → InverseCancellation（CX, X）。各passは1回だけ実行され、反復・固定点探索はない。commutation系、1-qubit合成、角度cutoffもない。

**InverseCancellationの実装 [S]**（Qiskit 1.3.0 `crates/accelerate/src/inverse_cancellation.rs` L42–107、L160–186）：

1. `op_counts` はpassの冒頭で1回だけ数える。
2. 自己逆gateは**与えられた順**（最終passではCX→X）に処理する。gateごとに、その時点のDAGで `collect_runs({name})` を取り直す。run内で同じqargsが連続する部分を区切り、偶数個なら全部、奇数個なら先頭以外を削除する。
3. runは「後続nodeがちょうど1つ」の間だけ延びる（rustworkxの `collect_runs`）。そのため2つのCXが相殺されるのは、**両方のwireで直接隣接している場合に限られる**。
4. 逆対（S/Sdg等）は自己逆gateの後に処理する（初段passのみ）。

帰結として、最終passのCX相殺は「最終pass直前のDAG」での隣接だけで決まる。CX削除の後でX runを取り直すため、CX削除が新たに作ったX–X隣接は同じpassで消えうる。ただしCXに戻る処理はない。

**BasisTranslatorの規則選択 [S]**（`basis_search.rs` L34–179）：target basisに接続したダミーnodeから等価性グラフ上をDijkstra探索し、source basisの全gateが発見された時点で停止する。各gateに採用される規則は**発見時点のpredecessor**であり、edge costは発見済みgateのcostだけで決まる。したがってsource basis（＝context）が変えるのは停止時刻だけで、発見済みgateの規則は変わらない、と読める。後段の `compose_transforms.rs` は未精読 **[N]**。経験的にも、3258a89 REPORTは「全contextを通じ、各入力variantに対するnative列は1通り」と記録している **[C]**。本件の保存catalogueでも、chunk別CX/RZは全12 contextで同一だった **[S]**。

### 3.2 A-1：有限contextは網羅的か

**catalogueの作り方 [S]**：12 requestはsampleから拾ったものではなく、フラグの組合せで設計されている（baseline：controlled-ZZ/CCXの有無 × product有無 × Y＝8、reference：product有無 × Y＝4）。各requestは、そのarmに現れる全chunk種をbarrierで区切って並べ、context witness（baselineはCRZ、referenceはCSWAP）とsymbolic Pを加えてcompileしている。

**全supportを覆う論証（紙上の列挙）[D]**：SOURCEを読むと、Unroll後のgate名集合は次のように決まる。

- **baseline**で常に現れる族：`p`（冒頭のscalar P）、`xx_plus_yy`・`rz`（frameとD）、`rzz`（D block 1–3）、`cx`（directional right）、`h`（wrapper）、`crz`。`crz` は、q≥2ならboundary controlに必ず現れる。q=1でも、各occurrenceに回転が1つ以上あり、単一Zなら `crz`、ZZなら `crzz` がUnrollで `ccx, crz, ccx` に展開されるため、必ず現れる。任意で現れる族は `ccx`（ZZ回転が1つ以上）、`cz`（productが1つ以上）、`sdg`（Y軸）で、組合せは8通り。
- **reference**で常に現れる族：`cswap`、`h`、`p`、`rz`、`rzz`、`xx_plus_yy`。任意で現れる族は `z`（product）と `sdg` で、組合せは4通り。
- 保存された `post_unroll` 集合はこの列挙と一致する。例：`Simon_Love_ccx1_rare1_Y1` は [ccx, crz, cx, cz, h, p, rz, rzz, sdg, xx_plus_yy]、`Simon_Love_ccx0_rare0_Y0` は [crz, cx, h, p, rz, rzz, xx_plus_yy] **[S]**。
- ordered event、負時間・符号、r・qの変化は族名の集合を変えず、多重度だけを変える。XX+YYの実逆（θ→−θ）はcatalogueに両方含まれている **[S]**。compiler Aには角度に依存する簡約がないので、CX数は角度に依らない。

**標本で観測したcontextとの区別**：新64 whole compileのcontextがすべて12集合に含まれたという検査 **[S]** は、標本に対する整合確認にすぎない。全supportの根拠は上の列挙である。REPORT L52の記述は短く、列挙はSOURCE（`catalog_one` のfilter）に暗黙に埋め込まれているだけなので、文書化を推奨する。

**残る仮定 [M]**：(a) Qiskit 1.3.0の変換が (name, qubit数, パラメータ構造) の関数であること（basis_searchのコードと保存catalogueで支持される）。(b) 実行中に `SessionEquivalenceLibrary` へ等価則が追加されていないこと。(c) 同名の独自gateがないこと。

**判断**：網羅性は構造的に成立している。確信度は中〜高。

### 3.3 A-2：chunk境界のゼロcancellation論証は正当か

**補題（本レビューの定式化、[D]。根拠はQiskit 1.3.0の実装 [S]）**：最終pass直前のDAGが、各wire上でchunk C₁,…,C_K の順に並んでいるとする。どのchunkにも「chunk内で自分の全wireの先頭にあるCX」（露出CX prefix）がなければ、whole回路で最終passが除去するCXは、各chunkを単独で処理した場合に除去されるCXの和集合に一致する。

*証明の要旨*：CX runは削除前に一度だけ収集される。chunkを跨ぐrunが生じるには、C_j のCXの直後に、両wireで C_k（k>j）のCXが続く必要がある。そのCXは C_k 内で両wireの先頭にあることになり、露出prefixに該当する。露出prefixがなければ全runはchunk内に収まり、単独処理と同一になる。CX削除後のX処理ではchunkを跨ぐX–X相殺が起こりうる（例：C_j の末尾が X(a)、C_k ＝ [RZ(b), CX(a,b), CX(a,b), X(a)]）。しかしpassはCXに戻らず、RZは最終passで削除されないので、CX数とRZ数は影響を受けない。∎

- SOURCEが課す条件（baseline対象chunkの `exposed_cancelable_prefix` が空）は、CXの加法性に**十分**である。保存フラグでは、全8 baseline contextで対象chunk（basis、basis_inverse、baseline_D、crz、crzz、cz、h、sdg）の露出prefixは0件だった **[S]**。
- **SOURCEで明示検査されていない追加条件**：初段（source-level）のInverseCancellationが、whole回路でchunk境界を跨いで相殺しないこと（catalogueではbarrierがこれを防いでいる）。baselineで初段のリストに該当するtop-level gateは、directional right内部のCX、wrapperのH・Sdgだけである。directional rightのCXは対象wire側の前後にframe gateがあり、chunk境界で隣接しない。H・Sdgはancilla上でPまたはCXに隣接する。Z、X、S、SX、Tはbaselineのtop-levelに現れない（CZは対象外、CRZ/CRZZ/CCXは初段の後でUnrollされる）。以上から、跨ぎ相殺は起きないと判断する **[D]**。この条件も補題として明記すべきである。
- **複数passの再適用**：compiler Aにはない。**二段階の消去**：同一pass内で起こるのはCX後のX–Xだけで、指標に影響しない。**barrierの有無**：補題はbarrierを必要としない。**commutation**：なし。**node ordering**：`collect_runs` は隣接関係だけに依存し、位相的順序には依存しない。
- **日常のQiskit compileとの差**：`transpile` の最適化レベル1以上は、1-qubit合成、可換相殺などを含み、両armのRZと一部CXを大きく変えうる。価格保証はcompiler Aに固有である **[M]**。
- referenceではRZZとCSWAPに露出CX prefix/suffixがある **[S]**ため、跨ぎ相殺が実際に起きる（q256・r2で実価格が上界より8〜22 CX少ない **[S]**）。相殺はgateを減らす方向にしか働かず、contextも覆われているので、上界の向きは正しい。

**判断**：固定pass列について、CXの加法性の論証として**十分**である。追加で必要なのは前提条件の明文化（初段passの境界条件、chunk種の完全列挙、Qiskit版の固定）であり、新たなprefix/suffix条件は不要である。確信度は中〜高。

### 3.4 A-3：D部分のCX式とRTE basis-runの期待値

**D部分 [S]＋[D]**：保存chunk CX（left_all 528、left_no0 456、right_all 634、right_no0 546、boundary_control 88。half block 72/144/164/148、full block 0 72、CSWAP 8）から次を得る。

```
D_B(q) = 528 + 634 + (q−1)(456 + 546 + 88) = 1090q + 72
D_R(q) = 4·72 + (q−2)·72 + 2q(144+164+148) + 16·8 = 984q + 272   (q = 2^m ≥ 2)
D_R(1) = 2·528 + 128 = 1184
```

REPORTの式と一致した。1 stepあたりの差106 CXは、block 1–3のSimon–Love CX共役（546−456＝90）と、boundary controlの8 CRZ（88−72＝16）から成る。構造はboundary-optimized builder（`df_partial_s2_repeated.py` L591–622）と `symmetric_fused`（pilot L198–213）で確認した **[S]**。

**basis-runの期待値（紙上導出 [D]、前提はSOURCEで確認 [S]）**：

- 前提：1 occurrenceはr eventから成る。各eventは独立に確率p₂でorder 2になり、そのときproduct 2個の後に回転1個を適用する。成分はすべてfinite-bit lawのpからiidに引く（`events_for` と、A_REPORTの平均因数分解の根拠）。builderは、同じoccurrence内で連続する同一basisの適用に1組のframeを共有させ、末尾で閉じる（`df_rte_qiskit.py` L278–338）。identity成分はない（support {1,2}）。occurrenceの間にはD blockが入るので、occurrenceを跨ぐ共有はない。
- 抽選列の長さをLとすると、basis bのrun数は Σ_{j=1}^{L} 1[b_j=b かつ (j=1 または b_{j−1}≠b)] で、その条件付き期待値は p_b+(L−1)p_b(1−p_b)。L = r + 2N₂、N₂ ~ Bin(r, p₂) で、成分抽選と独立。期待値はLについて線形なので、E[L] = r(1+2p₂) を代入すると **p_b + (r(1+2p₂)−1)p_b(1−p_b)** が得られる。これをq倍して c_b で重み付けする。
- 厳密である理由：(i) 非加法なframe融合を、run開始の指示関数で線形化している。成分ごとのframe費用を足し合わせる近似ではない。(ii) 1 runの費用は中身に依らず c_b（frameに露出CXがない）。(iii) canonical lawの下で抽選はiidで、order抽選とも独立。**期待値の線形性と、非加法な融合の確率依存は混同されていない。**
- 回転とproductの寄与（rot_B＝Σp_i(2 or 14)、rot_R＝Σp_i(0 or 2)、eventあたり期待CZ数 2p₂E|S|）はpath価格関数と同じ規則である **[S]**。
- 有理確率は `integer_cdf` によるfloor-CDFのexact値で、量子shot側のlawと同じである **[S]**。
- 注意 **[M]**：量は「finite-bit lawに対する1 shot期待価格 × 整数十分shot」の和で、期待総費用である。ほぼ確実な上界ではない。このshot数では揺らぎは無視できるが、claimでは「期待総費用」と明記すべきである。

**RZのP個数 [S]＋[D]**：order 2のevent位相は (−1)^{order/2}＝−1、つまりπ（`rte.py` L1649）。product符号は0またはπを足す。controlled builderは累積位相が `!= 0.0` のときだけPを付ける（`df_rte_qiskit.py` L246–249、L348–352、L483–490）。order 2を含むoccurrenceでは累積位相 ≥ π > 0、含まなければちょうど0.0になる。したがって E[#P] = q[1−(1−p₂)^r] は正しい。

### 3.5 A-4：上界・下界の向き

- 両armは同じwrapper（H、core、[Sdg]、H、measure）、同じfinite-bit law（roundごとに同じbits）、同じ十分shot規則、同じ費用単位（compiler A後のnative CX/RZ）を共有している。resetとCSWAPは別記されている **[S]**。
- baselineの「下界」は、構造式による厳密期待値を外向き区間にしたものである。全pathで構造的厳密性（§3.2〜3.4）が成り立つ限り下界として有効で、本レビューはその論証を支持する。
- referenceの上界は、全chunk種を数えていること、跨ぎ相殺と初段のZ相殺がgateを減らす方向にしか働かないこと、contextが覆われていることから有効である **[D]**。
- rare pathは厳密確率で含まれている（sampling、0補完ともになし）**[S]**。
- MC価格と構造の矛盾はない。128件でbaselineは一致し、referenceは包含された。r=2では上界との間に余裕がある **[S]**。
- **適用範囲の偏り [N]**：holdoutは (q,r) ∈ {(1,1),(2,1),(4,1),(8,1),(32,1),(128,1),(256,2)} だけである。総費用を支配するm=10–17（r=4〜32、q≥1024）にはwhole compileの照合がなく、補題（§3.3）と列挙（§3.2）だけが根拠である。
- **標本と構造の役割分担（重点D-1）**：一般性の根拠は構造（context列挙、fence補題、厳密な期待値式）である。旧48件＋新16件のpaired標本は、その整合を確認する役割に限られる。新Stage 2〜4はMC平均やMC信頼区間を使っていない。旧pilot（A3）の95%診断上限は、A-4へ進むかどうかの判断にだけ使われた。本件の保証には使われておらず、全support保証への格上げはない **[S]**。

### 3.6 A-5：RZのlocal min/max

- 最終passはCXとXしか消さないので、RZ数は変わらない **[S]**。
- 初段passはreferenceのZ productを消しうるので、referenceのRZは上界として扱うのが正しい。baselineのtop-level H/Sdgはwrapperにしかなく、RZは変わらない **[D]**。
- chunk別RZは全contextで同一（保存値でmin＝max）**[S]**。min/maxの仕組みは保守的だが、本件では実質的に効いていない。
- **P→RZ**：catalogueで、P(α) は単一の RZ(α) に変換され、symbolic parameterが保たれることを確認した **[S]**。P(α) = e^{iα/2}RZ(α) なので、α/2はjoint全体のglobal phase（観測不能）に入り、相対位相αは厳密に保たれる。**RZ価格としては1、相対位相の誤差は0** **[D]**。
- **D部分RZ（本レビューの導出）[D]**：baseline 6180q+664、reference 6168q+1520（q≥2）で、差は12q−856。stepあたりの12は、right_no0の P(2φ) 3個、boundary controlのCRZとRZの差8、boundary controlのP 1個から成る。定数項はreference側の16 CSWAP×11 RZ＝176などである。q≤64ではD部分のRZはreferenceの方が多い。

### 3.7 結論（A）

1. **固定compiler Aによる登録実装の価格保証**：成立する（条件付き）。条件は、補題と列挙の明文化、Qiskit 1.3.0環境、期待値としての解釈である。
2. **任意のより強いcompiler・より強い直接制御回路に対する優位**：**未資格**。1.からは導けない。

### 3.8 補強が望ましい点（A）

1. 境界fence補題（初段passの条件を含む）をQiskit 1.3.0のコード参照つきでREPORTに書く。
2. contextの列挙（常在族と任意族）を本文に書く。
3. r≥4の長r領域に経験的証拠を加える（例：登録scheduleのr=32 cellで短いqの少数holdout）。新規compileなので**要承認**。
4. claimの文言を「期待総費用」と明記する。

---

## 4. controlled-ZZの14 CXと代替4 CX：数理・公平性・結論への影響（重点B）

### 4.1 登録loweringの由来 [S]

- RTEの中央回転は `RZGate(2·angle)` または `RZZGate(2·angle)` で、制御版は `gate.control(1)` で作られる（`df_rte_qiskit.py` L493–501）。
- Qiskit 1.3.0では、`RZGate.control` は標準のCRZGateを返す。`RZZGate.control`（`rzz.py` L125–156）は汎用の `add_control.control()` に委ねる（`add_control.py` L68–250）。RZZを [cx, rz, cx] に展開し、各cxを2制御のmcx（＝CCX）に、rzを `mcrz(use_basis_gates=False)`（＝CRZ）に置き換える。結果は **CCX(c,a,b)・CRZ(c;b)・CCX(c,a,b)** で、追加のglobal phaseはない。
- compiler Aでのloweringは保存catalogueで **14 CX、24 RZ、4 SX**。24 RZの内訳は、任意角（±θ/2）2、T型（±π/4）14、Clifford（π/2、H由来）8。露出CX prefixは0、suffixは1 **[S]**。
- 対照：CRZは2 CX・2 RZ、referenceのRZZは2 CX・1 RZ **[S]**。

### 4.2 4-CX parity–CRZ–uncomputeの紙上検証 [D]

記号：制御qubit c、ZZ supportを (a,b)。RZ(θ)=exp(−iθZ/2)、RZZ(θ)=exp(−iθZ_aZ_b/2)。CRZ(θ)=|0⟩⟨0|_c⊗I+|1⟩⟨1|_c⊗RZ_b(θ)（QiskitのCRZGate。control 0側に余分な位相はない）。目標は C-RZZ(θ)=|0⟩⟨0|_c⊗I+|1⟩⟨1|_c⊗RZZ_ab(θ) で、Qiskitの `RZZGate(θ).control()` および区間評価のsource行列（`track_c_compiler_a_18round.py` L159–164）と同じものである。

**主張**：任意の実数θ（負を含む）について、global phaseの差もなく

```
CX_{a→b} · CRZ_{c→b}(θ) · CX_{a→b}  =  C-RZZ(θ)
```

**証明**：基底 |c,a,b⟩ について、CX_{a→b} で |c,a,a⊕b⟩ になる。CRZは c=0 なら1倍、c=1 なら e^{−iθ(−1)^{a⊕b}/2} 倍する。最後のCX_{a→b}で |c,a,b⟩ に戻る。Z_aZ_b|a,b⟩=(−1)^{a⊕b}|a,b⟩ なので、両分枝で C-RZZ(θ) と一致する。同値な言い方をすれば、CX_{a→b}(Z_b)CX_{a→b}=Z_aZ_b であり、CX_{a→b} は |k⟩⟨k|_c と可換である。∎

- **control 0分枝**：CX·CX=I で、残留位相はない。**control 1分枝**：RZZ(θ)。相対位相は厳密に一致する。
- **負角・実逆**：C-RZZ(θ)† = CX·CRZ(−θ)·CX（3 gateを逆順にしてθの符号を反転。CXは自己逆）。
- **gateの向き**：parityのCXは a→b、CRZは c→b。CX_{b→a} と CRZ_{c→a} の組でも同じく成立する。
- **compiler Aでのnative列**（CRZの定義 `rz.py` L256–282 による）：CX_{a→b}, RZ_b(θ/2), CX_{c→b}, RZ_b(−θ/2), CX_{c→b}, CX_{a→b}。**4 CX、2 RZ（ともに任意角）、SXなし。** 内側の CX_{c→b} と CX_{a→b} はqargsが異なるので相殺しない。
- **解析誤差**：θ/2は2進スケーリングで正確に表せ、恒等式は厳密に成り立つので、native誤差は0（REPORTのCRZ/RZZ解析checkと同じ論理）。
- 実装上の注意：汎用合成を避けるため、カスタムgateの `.control()` ではなく、source上で明示的に cx・crz・cx として出す必要がある。
- 外側のCXに制御が不要なのは、C(UVU†)=U·C(V)·U†（Uがtarget側だけに作用する場合）という標準的な事実による。登録loweringの2つのCCXは冗長である。
- 文献：SE-QPE §IIIに同形のkernel e^{−iθ|1⟩⟨1|_a Z_iZ_j} = CX_{i→j}R_z^{(j)}(θ)CX_{a→j}R_z^{(j)}(−θ)CX_{a→j}CX_{i→j} がある（取得ツールが逐語部分として報告）**[C]**。
- 4 CX未満が可能かどうかは本レビューでは確定していない。結論には不要である。

### 4.3 結論への影響（SOURCEで裏付けられる範囲の構造式）[D]

両armのframe（basis-run）費用は同じ c_b で一致し（`cost_bounds.py` L46のassert）、差では消える。P_ZZ をfinite-bit lawの下でのZZ-support成分の確率合計、E|S| を成分supportの期待値とすると、1 shotの期待CX差は次のとおり。

```
登録baseline:  Δ_CX  = (D_B − D_R) + q·r·[2 + 10·P_ZZ + 2·p₂·E|S|]
B′（4-CX）:    Δ′_CX = (D_B − D_R) + q·r·[2 + 2·p₂·E|S|]
D_B − D_R = 106q − 200（q≥2）、−22（q=1）
Δ_CX − Δ′_CX = 10·P_ZZ·q·r
```

- **符号**：q≥2では Δ′_CX>0（各項が正）。q=1では負で、referenceの方が約20 CX多い。これは保存値（q=1でreferenceが登録baselineより+10 CX）とも整合する。総計は大きいqのroundが支配するので、**B′に対してもCX差の符号は保たれる見込み**である。
- **前提（要技術検証）**：B′のCRZZ′は、referenceのRZZと同じく両端に CX_{a→b} が露出する。同じbasis run内で同じ対のZZ回転が連続すると、内側のparity CX対が両armで相殺される。productによる遮断（同じsystem wire上のZとCZ）も両armで同じである。CRZの末尾は CX_{c→b} で、parity CXとは一致しない。したがってpathごとの相殺集合は両armで一致し、Δ′はpathごとに厳密になると見込まれる。ただしこれは未検証である。
- **大きさ**：finite-bit lawでのP_ZZの厳密値は保存成果物に記録されておらず、本レビューでも算出していない（未承認計算）**[N]**。ただし保存されたevent記録を読むと、新16 pathで回転の大半はZZ型である。各pathの単一Z回転は、q4・q32で0件、q128で128中0〜1件、q256・r2で512中0〜2件だった **[S]**。保存pathの価格差は、この件数を構造式に入れると**手計算で厳密に再現できた** **[D]**。
  - q128・r1・非rare・j0（ZZ 127/128）：CX差 13,368＋2·128＋10·127＝14,894（保存値 155,116−140,222＝14,894）。RZ差 (12·128−856)＋128＋22·127＝3,602（保存値 920,503−916,901＝3,602）。
  - q256・r2・非rare・j0（ZZ 511/512）：RZ差 2,216＋512＋22·511＝13,970（保存値 1,912,703−1,898,733＝13,970）。CXは上界との差 320,980−287,910＝33,070 が 26,936＋1,024＋10·511 と一致した。
  - CXとRZという独立な2指標から同じZZ件数が読め、§3の構造式とD部分RZ式（12q−856）の検算にもなっている。
  - 以上から、**P_ZZは1に近い**ことが強く示唆される。そうであれば、event比例部分は登録比較の約12 CX/eventからB′では2 CX/eventへ、**約1/6に縮む**。例として P_ZZ＝1 と置いた紙上の式では、r=32のroundのstepあたりの差は、登録比較で約106+12r≈490 CX、B′比較で約106+2r≈170 CX となる（18round総計の価格は作っていない）。**報告の約19%というCX削減率はB′へ移送できず、B′比の削減率は大幅に小さくなる可能性が高い。**

**RZ [D]**：

```
登録baseline:  Δ_RZ  = (12q − 856) + q·r·[1 + 22·P_ZZ + 6·p₂·E|S|] + q·[1 − (1−p₂)^r]
B′:            Δ′_RZ = (12q − 856) + q·r·[1 + 6·p₂·E|S|]         + q·[1 − (1−p₂)^r]
Δ_RZ − Δ′_RZ = 22·P_ZZ·q·r
```

- B′では、r=1のroundでおおむね q≥128（m≥7）、r≥2の全round（q≥256）で Δ′_RZ>0 になる。m≤6ではreferenceの方がRZが多い。保存値でも、q≤8ではreferenceのRZが登録baselineより多い。
- 総計の符号は正になる見込みだが、相対的な余裕は小さい。P_ZZ≈1なら、event比例部分は登録比較の約23 RZ/eventからB′では1 RZ/eventへ縮む。P_ZZ＝1と置いた紙上の式では、r=32のstepあたりの差は登録比較で約12+23r≈748、B′比較で約12+r≈44 RZになる。これは両armに共通するframeのRZ（1 frameあたり296〜512、frame対で592〜1,024、保存値）や、D部分の約6,170 RZ/stepに比べて小さい。**報告の約4%というRZ差の大半は登録CRZZのloweringに由来すると判断する。RZ削減claimはB′に対して未資格で、余裕もごく小さいと見込まれる。**
- 登録CRZZのRZ超過23は、任意角1、T型14、Clifford 8 から成る。FT的な指標で見ると、ZZ event 1つあたりの差は、登録比較で「任意角1＋T 14」、B′比較で「任意角1＋T 0」になる。

### 4.4 reference側にも同種の最適化機会はあるか

- **CSWAP**：compiler AのloweringはQiskitの定義（CX・CCX・CX）による8 CX・11 RZ。SE-QPEはCSWAPを7 CXで見積もっている（取得ツール報告）**[C]**。7-CX Fredkinを使えば1 shotあたり16 CXの削減で、影響は無視できるほど小さいが、対称性の観点では適用すべきである。
- **RZZ**：2 CXで最小。同じ対の連続による相殺は実compileで既に起きているが、上界には算入していない。
- **D blockの対角層**：phase polynomial（Gray code）合成でCXを減らせる可能性は、**両arm**に共通する。Simon–LoveのCX共役がparity網に吸収されれば、D部分の差106/stepが変わる可能性もある **[N]**。
- **方針の提案**：既知最良の局所分解を対称に適用する（baselineにはB′、referenceには7-CX CSWAP）。compiler Aはそれ以外変更しない。そのうえで、登録版と対称最良版の両方の比較を報告する。強いbaselineを使うのは比較の原則として正しいが、referenceにも同等の局所改善を与えないと片側だけが有利になる。

### 4.5 baselineは両armに適用可能な最善の既知対照に近いか

| 構成要素 | 登録baseline | 既知の対照・文献 | 評価 |
|---|---|---|---|
| D部分（対称Trotter） | Simon–Love型の方向制御。block 1–3で+90 CX/step（1回転あたり2 CX未満、初段passの相殺込み）[S] | Simon–Love（arXiv:2511.13855v1）"Halving the Cost of Controlled Time-Evolution"：U₁は非制御、U₁′はCNOT共役で方向制御し、control 0で恒等にする。指標は任意角回転数 [C] | 既知最良級 |
| boundary block 0 | frame 1組＋8 CRZ（88 CX）[S] | — | この構造ではほぼ最小 |
| RTE中央制御（basis非制御） | 自然な構造。方向制御はRTEが回文でないため適用できない [D] | PR原著（arXiv:2503.05647v2）はcontrolled evolutionを前提にHadamard testを費用化（読んだ範囲にCX水準の制御最適化はない）[C] | 構造は妥当 |
| RTEの制御ZZ回転 | Qiskit汎用合成で14 CX・24 RZ [S] | SE-QPE（arXiv:2604.14921v2）は4-CX kernelで正準QPEを費用化 [C] | **既知最良ではない** |
| basis共有・D境界融合 | 両armに適用 [S] | — | 対称 |

文献比較は指定3本に限られ、網羅的ではない（§9）。

### 4.6 代替baseline（B′）を技術検証する最小案（Codex向け・要承認）

- **対象gate族**：RTEのZZ-support回転の制御版だけ（`_append_central_application` のcontrolled分岐で|S|=2の場合）。新しいbaseline builder（B′）で `RZZGate(2·angle).control(1)` を明示的な `cx(a,b); crz(2·angle)(c,b); cx(a,b)` に置き換える。D、frame、product、P、wrapper、law、shot規則は変えない。
- **手順**：
  1. **correctness**：符号2×order 2×r 6の角度で3-qubit恒等式を確認する。加えて、ADAPTER_CHECKSのr2と同型の1 occurrence joint operator照合を、B′と ordinary diag(I,V) の間で行う。
  2. **局所catalogue**：B′のchunkを追加する。B′ではbaselineのpost-unroll集合から `ccx` が消え、contextは rare×Y の4通りになる。4 CX・2 RZと露出 CX_{a→b} prefix/suffixを記録する。
  3. **相殺モデル**：B′はfence補題を単独では満たさない。既存の登録path（新16＋旧48）×2軸でB′をcompileし、「B′とreferenceの相殺集合がpathごとに一致する」ことを照合する。128件すべてで Δ′(path) が式どおりなら採用する。合わなければ、B′の下界として「構造式 − 2×（同一対ZZ回転の隣接数）」をeventから厳密に数える。
  4. **誤差**：B′のCRZZ′は解析的に厳密なので、native誤差は登録baseline以下で、shotは8,598以下になる。整合確認のためだけに再計算する。
  5. **18roundの厳密再会計**：同じfinite-bit有理式で Δ′_CX、Δ′_RZ の総計を出し、roundごとの P_ZZ と p₂E|S| も報告する。
  6. （任意）referenceを7-CX CSWAPにした対称版も併記する。
- **取得が必要な代表context・境界**：B′の4 contextと、CRZZ′の前後に来る境界（basis frame、CRZ、CZ、P、同じ対／異なる対のCRZZ′、D chunk）。
- **結果によって変わる判断**：(a) 全18round総計で reference CX上界 < B′ CX下界 なら、強baselineに対するCX削減claimとして残し、大きさを報告する。(b) そうでなければ、登録実装を超えるCX claimは撤回する。(c) RZは§6.2の指標整理と組み合わせて判断する。

### 4.7 判断（B）

- **「登録14-CX baselineに対するCX/RZ優越」**：§3の条件つきで**採用可**。これは登録実装についての事実記述である。
- **「PR×referenceが直接制御よりCX/RZを削減する」という研究上の主要claim**：B′の結果が出るまで**保留**。CXは、D部分の106 CX/stepとeventあたり2 CXの制御overheadにより、符号が残る見込みが中程度にある。ただし報告の約19%の相当部分は登録CRZZのlowering（eventあたり+10 CX）に由来し、B′比の大きさは未定で、大幅縮小の見込みが高い。RZ削減は、B′比では実質的に残らない見込みである。
- **価格証明そのもの**：致命的不備はない。問題は比較対象（baseline）の選択にある。

---

## 5. 真空位相・浮動小数点丸め・全support誤差・raw信号・RPE shotの連結監査（重点C）

### 5.1 C-1：vacuum位相追跡の正しさ

**gadget [S]**（pilot L120–130）：CSWAP(a; j, n+j)×8、lane 0..n−1にF†、lane n..2n−1にG、ancillaにP(φ)、最後に再びCSWAP×8。初期状態は、target laneにψ、reference laneに|0…0⟩、ancillaに|+⟩（wrapperのH）。

**導出 [D]**：

- branch 0：(F†ψ)⊗(G|0⟩) = (F†ψ)⊗e^{iφ_G}|0⟩
- branch 1（swap→作用→swap戻し）：e^{iφ}(Gψ)⊗(F†|0⟩) = e^{iφ}(Gψ)⊗e^{−iφ_F}|0⟩
- coherence：⟨branch0|branch1⟩ = e^{i(φ−φ_F−φ_G)}⟨ψ|FG|ψ⟩

φ=φ_F+φ_G とすれば、これは ⟨ψ|V|ψ⟩（V=FG。`g.compose(f)` はGを先に作用させる）になり、ordinary diag(I,V) のHadamard test信号と一致する。この恒等式は任意のψ（固有状態でなくてよい）と任意のF・Gで成り立ち、条件は「|0…0⟩がFとGの厳密な固有ベクトルであること」だけである。pilot REPORTの式 z_ref = e^{iϑ}⟨ψ|FG|ψ⟩·conj(⟨0|FG|0⟩) とも一致する（⟨0|FG|0⟩ = e^{i(φ_F+φ_G)}）。

- **固有ベクトル条件**：`vacuum_phase` はF・Gのgate族を {rz, rzz, xx_plus_yy, z, p, id} に限定し、それ以外では例外を出す（pilot L95–103）**[S]**。各族は|0…0⟩を位相倍するだけである（RZ: e^{−iθ/2}、RZZ: e^{−iθ/2}、XX+YYは|00⟩に自明、Z: +1、P: 1）。したがって φ = global_phase − Σθ/2 は正しい。
- **符号付きRTE event・scalar・identity**：event位相（±1→0/π）とproduct符号は、非制御回路のglobal phaseに加算される（`df_rte_qiskit.py` L354）**[S]**。φは `qc.global_phase` を通じてこれらを含む。scalar（定数＋抽出identity）は集約global phaseとして入る **[S]**。
- **負時間・実adjoint・F/Gの順序**：φはF（正時間）から計算し、gadgetはF†を作用させる。上の導出と整合する。実adjointは `QuantumCircuit.inverse()`。G＝前半q/2 step、F＝後半q/2 stepなので、FG＝q stepの順序積になる（`assemble` L176–177）**[S]**。数値の整合はpilotのq2/q8照合で確認されている **[C]**。
- **二重計上・欠落**：φはancillaで1回だけ補正している。独立二分枝PRは用いていない。E₀・gap・ground vectorは位相計算に使われていない（`state_or_E0_used_for_phase=false`）**[S]**。

**判断**：正しい。確信度は高。

### 5.2 C-2：局所誤差から全supportへ

- **局所 [S]**：445個の異なるlowering（XX+YY 416、CRZZ 24、CSWAP・CZ・H・Sdg・Z 各1）を192-bit外向き区間で評価している。Frobeniusノルムの上界はoperatorノルムを上から押さえる。各局所gateに1つのscalar gaugeを使い、pivotで選んだgaugeに対して上界は厳密である（`error.py` L29–41、`track_c_compiler_a_18round.py` L153–214）。RZ・RZZ・CRZ・Pは解析的に厳密で、全contextで2,600件の解析checkがある **[S]**。族別の最大局所上界は CRZZ 1.15×10⁻¹⁵、XX+YY 1.07×10⁻¹⁵ **[S]**。
- **全体 [D]**：whole回路の各source gateは、catalogueと同じnative列に置き換わる。最終passと初段passによる削除はすべて厳密な恒等式（CX·CX=I、X·X=I、Z·Z=I等）である。したがってcompile後のjoint operatorは、局所native operatorの積に厳密に等しい。telescoping ‖ΠA_i − e^{iΣγ_i}ΠB_i‖ ≤ Σ‖A_i − e^{iγ_i}B_i‖ はunitarityから成り立つ。tensor拡張でoperatorノルムは変わらない。局所gaugeの積はjoint全体の単一位相になり、分枝別のfitはしていない。評価は2^17の全空間で、粒子数sectorへの射影はない。
- **多重度 [S]＋[D]**：B = (q+1)e_D0 + 2qΣe_Dk + 3qr·e_tail + qr·e_crzz + 4qr·e_cz + (2e_h+e_sdg)、C = (q+2)e_D0 + 2qΣe_Dk + 3qr·e_tail + 4qr·e_z + 16e_cswap + (2e_h+e_sdg)（`error.py` L176–179）。eventあたりの最悪ケース（basis run ≤3、回転1、CZ/Z ≤4）と、block 0の出現回数は構造と一致する。
- **symbolic CRZZ templateの適用範囲**：区間上界が確立しているのは代入した24角度（r 6通り×order 2通り×符号2通り）であり、**全binary64角ではない**。これで十分であるためには、代入角が実際の角度とbit単位で一致する必要がある。
  - r=1,2：保存raw eventの角度（0.01166487938612472、0.003888449899598402、0.0058326381039702805、0.0019442322990180078）は、保存catalogueのCRZZ lowering中の半角として**逐語で現れる** **[S]**。
  - r≥4：実eventの保存がない。実角は `math.atan(λ_tail·(0.02/r)/(order+1))`（`rte.py` L1514–1515、L1653）、代入角は `math.atan(λ_prep·0.02/r/(order+1))` である。r=2^k での除算は厳密なので、λ_prep（`exact_rte_lambda_r`）とλ_tail（`tail.lambda_r`）が同じbinary64値なら両者は一致する **[D]**。この同一性は未確認 **[N]**。
  - 影響：局所上界は角度の滑らかな関数で、ulp程度の差は無視できる。これは形式上の隙間にとどまる。B′を採用すればCRZZ′は解析的に厳密になり、この隙間は消える。

**判断**：全17 qubit joint operatorへの適用は正しい。確信度は中〜高。

### 5.3 C-3：vacuum phaseの浮動小数点上界

**(1) 加算モデルとγ_N·M [D]**

`vacuum_phase` は phase←float(gp) から始め、rz/rzzごとに phase←phase−θ/2 を逐次計算する。θ/2は2進スケーリングで厳密、gp∈[0, τ_f)。厳密な部分和をT_i、計算値をT̂_i、E_i=T̂_i−T_i とする。標準モデル fl(x⊖y)=(x⊖y)/(1+δ)、|δ|≤u から |E_i−E_{i−1}| ≤ u|T̂_i| ≤ u(M+|E_i|)（M ≥ max_i|T_i|）。帰納法により

```
|E_N| ≤ u·M·Σ_{k=1}^{N}(1−u)^{−k} ≤ N·u·M/(1−N·u) = γ_N·M     （N·u < 1）
```

成立条件は、実際の加算順に沿った逐次加算であること、Mがその順での厳密部分和を押さえること、Nが丸め回数以上であること。いずれも満たされている（下記）。**符号相殺は、MがΣ|x_i| ではなく部分和を押さえることで正しく扱われている。** 結果がsubnormalになる減算は厳密で、overflowは起こりえない（|T|≲2×10⁴）。

**(2) Nの紙上導出 [S]＋[D]**

保存Stage 1構造（D frame 8 RZ、one-body対角 8 RZ、two-body対角 8 RZ＋28 RZZ、RTE frame ≤8 RZはassert済み）から、境界融合込みで1 stepあたり次のとおり。block 1–3を2回（frame 2×8＋対角36＝52、×6で312）、融合block 0を1回（16+8＝24）で336。RTEは eventあたり basis run ≤3×16＋回転1＝49 で49r。各halfの非融合block 0の余り24を加えて、**N=(336+49r)n+24** は上界として正しい。m=17では (336+1,568)·65,536+24＝124,780,568 で、保存値と一致する。ただし336・49・24は `error.py` L120にハードコードされている。assertされているのは「frameあたりRZ≤8」と「block 0の2倍化」だけなので、文書化と堅牢化の課題が残る。

**(3) M [S]＋[D]**

M = τ + n(drift + r·atan(τ_r)) + 2·basis_abs + local_diag。driftはstepあたりのD対角の正味角（frameとその逆は厳密に相殺する）、r·atan(τ_r) はstepあたりの回転半角、2·basis_abs＋local_diagは途中のblockを押さえる。厳密部分和の上界として妥当である。

**(4) 手計算の検算 [D]**

N·u = 124,780,568×2⁻⁵³ ≈ 1.3854×10⁻⁸。γ_N·M ≈ 1.3854×10⁻⁸×16,375.94 ≈ 2.2687×10⁻⁴（片側）。両側と最後の加算1回で ≈4.5373×10⁻⁴ となり、保存値 4.5372585×10⁻⁴ と一致した。

**(5) 保守性**

上界はNに線形な最悪評価で、現実の丸め誤差は √N·u·M 程度（10⁻⁸のオーダー）と見込まれる。その代償は、最終roundでreferenceのshotが626→630（各軸+4、総計+8、約0.09%）になることで、結論への影響はない。

**(6) global phase正規化 [S]＋[D]**

- Qiskit 1.3.0の `CircuitData::set_global_phase` は `angle.rem_euclid(2·f64::PI)` を使う（`circuit_data.rs` L848–866）。Pythonの `qc.global_phase += x` は、binary64の加算1回の後に `rem_euclid` を呼ぶ。Rustの `f64::rem_euclid` は r=self % rhs（IEEEのfmodで厳密）で、r<0 のとき r+|rhs|（丸め1回）。理想の mod 2π との差は、周期誤差 k|τ_f−2π|（k ≤ |x|/τ+1）と高々2回の丸めである。したがって `γ₄(|x|+2τ)+(|x|/τ+2)|τ−2π|` は保守的な上界である（`error.py` L124–127）。
- **回数**：片側あたり、|x|≤2τ の正規化20n回、|x|≤3rπ の正規化n回、集約scalar 1回（`error.py` L131）。builderを読んだ範囲では（`build_additive_circuits` の内部は未精読）、1 stepあたり `_append_block` によるglobal phase更新が7回（`ops.py` L317–321）、RTE occurrenceのcomposeが1回（compose時にglobal phaseを加算。`df_partial_s2_repeated.py` L429–435）である。これとは別に、RTE builder自身の更新が1回あり（eventあたりπ＋product 2個のπで |x|≤3rπ。`df_rte_qiskit.py` L354）、端の効果と集約1回（L575–589）が加わる。**20nは安全な上界である。** blockのglobal phaseが τ 未満であることもassertされている（`error.py` L115–116）**[S]**。
- **scalar集約**：n=2^k なので −0.02·n·(c+e) = n×float(−0.02(c+e)) は厳密で、assertされている（L130）。正規化は1回である。
- **2πの表現**：τ_f=float(2π) は 2π より約2.45×10⁻¹⁶ 小さく、周期ごとに計上されている。

**(7) 何が保証されているか [M]**

保証されているのは「IEEE-754 binary64 round-to-nearestの演算モデルの下での上界」であり、**実際のQiskit wheel・OS・libmの動作まで実証した上界ではない**。ただし依存範囲は狭い。真空位相の和は加減算だけでlibmを使わない。正規化はfmod（IEEEでは厳密）と加算だけである。libmの `atan` は入力角を決めるだけで、和の上界には影響しない。実証に格上げするには、次のいずれかが必要である。

- (a) 実行platformとABIの記録（x86-64・SSE2・glibc版。ENVIRONMENTには一部しかない）
- (b) Qiskit 1.3.0 wheelの `rem_euclid` がfast-mathなしのfmodであることの確認
- (c) 保存sample回路について、φを厳密有理数で事後再計算して照合すること（**要承認**）
- (d) 補正正しい和（`math.fsum`）や閉形式のφを採用し、この項自体を無視できる大きさにすること（**SOURCE変更、要承認**）

**(8) 過小計上・二重計上の点検 [D]**

- P角の誤差δは、|e^{iδ}−1| ≤ |δ|（|z|≤1）でraw diskへ1回だけ加算されている（referenceのみ）。
- compiler誤差は、新しいnative joint operator誤差の2√2倍で、旧compiler diskを置換している。
- 旧event phase disk（RTE builder内のπの浮動小数点累積）は、制御SOURCEのP角と非制御F/Gのgpに同じ浮動小数点値として現れ、両armに等しく効く。これは1回だけ継承されている。正規化項はgp更新の加算とrem_euclidだけを扱うので、event phase diskと重複しない。
- 欠落は見つからなかった。

**判断**：上界は妥当で保守的。確信度は中〜高。

### 5.4 C-4：raw信号・shot・aliasへの接続

- **単位**：Hadamard testのraw coherence z（|z|≤1、RTE正規化前）**[S]**。
- **2√2 [D]**：joint operatorで ‖U−U′‖≤ε（global phaseを除いて）なら、±1観測量について |Δ⟨X⟩|, |Δ⟨Y⟩| ≤ 2ε、よって |Δz| ≤ 2√2ε。native joint誤差に対しては正しい。旧SOURCE/finite-RTE operator上界（V単体の上界なら |Δz|≤ε で足りる）に対しては保守的である。
- **旧compiler diskの置換**：妥当である。新native誤差は同じ理想source operatorに対する誤差で、両armのsource operatorは同じfloatパラメータの下で ordinary diag(I,V) と厳密に一致する（SL liftのCX共役による符号反転と2φは厳密。referenceは§5.3で押さえた真空位相の算術を除いて厳密）。
- **diskから角度へ**：arcsin(D/ρ_nom) をatan2で評価、ρ=ρ_nom−D、β=0.4−PF−arcsin(D/ρ_nom)（`error.py` L182–188）**[S]**。いずれも区間演算で外向きに評価されている。
- **shot [D]**：±1結果のHoeffding不等式 P(|X̂−X|≥ε) ≤ 2exp(−Nε²/2) ≤ α=1/720 から N ≥ 2ln(1440)/ε²。|ΔX|,|ΔY| ≤ ε なら |Δz| ≤ √2ε = ρ sinβ で、統計的な角度誤差は β 以下になる。36軸-roundでunion boundを取れば 36/720=0.05。
- **最終roundの手計算検算 [D]**：baselineは ε=0.5726873·sin(0.3859870)/√2≈0.152453（保存値0.15245342）で 2ln1440/ε²≈625.8→626。referenceは ε≈0.152036 で ≈629.2→630。いずれも保存値と一致した。β差 ≈7.9×10⁻⁴ は arcsin(4.574×10⁻⁴/0.57269)−arcsin(3.68×10⁻⁶/0.57269) と整合する。
- **alias**：720416cの結果を継承している（Gershgorinで |spec|≤3.0553 Ha、δ₀‖H‖+0.4<π、同じq scheduleと0.4 rad cap）**[C]**。両armとも各roundの位相誤差を0.4 rad以下に抑えているので、armによらず適用できる。
- finite-bit proposal diskとPF phaseは1回だけ継承されている **[S]**。
- 十分shotは「十分数」（上界）であって最小数ではない。

**判断**：整合している。確信度は中〜高。

### 5.5 C-5：成功保証の範囲

- **主張されている範囲**：保存Hermitian行列モデル、各shotでの厳密ground供給、理想native gate（RZ/SX/X/CX）、理想reset・測定、固定finite-bit law、IEEE binary64モデル、192-bit区間評価の下で、全roundの位相誤差が同時に0.4 rad以下となる確率が0.95以上であること。そこから既存Track CのRPE論証により、energy精度が得られる。
- **主張されていない範囲（主張すべきでもない）**：無条件の95%成功、native noise、hardware、元の連続積分Hの包含、T合成誤差（合成なし）。REPORTの記述はこれと整合している **[S]**。
- **形式証明との距離**：計算機支援（mpmathの区間、厳密有理数）に、assertとして埋め込まれた非形式の構造補題を加えたものである。形式検証、第三者CI、独立な再実行はない。

### 5.6 判断（C）と補強点

成功保証は固定モデル内で整合しており、確信度は中〜高である。補強が望ましい点は次のとおり。

1. ハードコード定数（336・49・24・20）を、builderから導出した値とassertに置き換える。
2. λ_prep と λ_tail の同一性をassertする（またはB′でCRZZ依存をなくす）。
3. 真空位相計算の厳密化・補償和化を検討する（要承認）。
4. 実行platformのIEEE関連情報を記録する。

---

## 6. 反証・競合する説明・強baseline・結果の解釈範囲

### 6.1 CX差の内訳：構造的な効果と実装上のartifact [D]

1 shot（q≥2）の期待CX差は、次の成分に分解できる。

| 成分 | 大きさ | 性格 |
|---|---|---|
| D部分 | 106 CX/step（SLのCX共役90＋制御boundaryの対角16） | **真の構造効果**。control-freeにより対称Trotter部分の制御が不要になる |
| event：制御overhead | 2 CX/event（CRZ対RZ、またはparity型CRZZ対RZZ） | **真の構造効果** |
| event：登録CRZZの汎用合成 | 10·P_ZZ CX/event | **実装artifact**。B′で消える |
| product | 2p₂E\|S\| CX/event | 真の効果だが、p₂が微小なので無視できる |
| 定数 | −200 CX/shot（16 CSWAP＝128ほか） | reference側の固定費 |

P_ZZ≈1（§4.3）、r=32 の場合、stepあたり約490 CXの差のうち約320 CXが登録CRZZに由来する（紙上の式。18round総計ではない）。**報告されたCX削減の相当部分は、比較対象の実装の弱さで説明できる**。これが最も有力な競合説明である。一方、D部分とeventあたり2 CXの差は、control-freeという性質に根ざした効果として残る。

### 6.2 RZという指標の意味

- compiler A（1-qubit合成なし）のRZ数には、Clifford型（π/2の倍数。H、S、Sdg、SX由来）、T型（±π/4。CCX由来）、任意角の3種類が混在する。例：XX+YYは18 RZで、定義上の任意角は最大4（RZ(±β)、RY(−θ/2)×2）、残りは主にCliffordの分解に由来する **[S]＋[D]**。
- NISQの超伝導機では、RZは通常virtual（frame変更）でほぼ無償である。FTでは、任意角回転の合成とT数が費用を支配し、Clifford型RZは安価である。
- したがって約4%の「RZ削減」は、その大半が登録CRZZ内のT型（CCX）とClifford型（H）のRZに由来するとみられ、hardware上でもFT上でも意味のある資源差に対応しない **[D]**。
- 意味のある比較には、**任意角回転数とT型回転数を分けて数える**ことが必要である。構造的には、eventあたりの任意角回転は、登録baselineとB′で2、referenceで1になる（制御Pauli回転は任意角回転2個を要する、というSimon–Loveの指摘 **[C]** と整合）。これはcontrol-freeの真の効果である。ただし両armに共通するframeの任意角回転で希釈される。

### 6.3 その他の解釈範囲

- **期待値**：総費用は期待総費用であり、ほぼ確実な上界ではない。
- **compiler固有**：compiler Aに固有で、1-qubit合成や可換相殺を含む通常の最適化compileでは結果が変わりうる。
- **全結合の仮定**：referenceのancillaは16 qubitとCSWAPで相互作用し、baselineのancillaは8 system qubitと相互作用する。routingとfan-out（SE-QPEはcat変種に言及 **[C]**）の費用は未計上である。
- **qubit数**：17対9（+8）。reset 68,848回が必要で、shotごとに新しいreference準備ができることを前提とする。
- **depth**：holdoutでは約55%の削減が観測されている（q128：314,444→140,631、q256・r2：654,113→291,569）**[S]**。F†とGのlaneが並列に動くので構造的にも妥当だが、18roundでは**未資格**である。qubitとdepthのtrade-offなので、space-time比較にはbaselineにも公平なqubit予算を与える必要がある（例：control fan-outによる並列化）。
- **小さいq**：qが小さいとreferenceの方が不利になる。CXはq=1で、RZはB′比でq≤64、保存値でq≤8。利益はq·rに対して漸近的に現れる。
- **古典費用**：shotごとの真空位相計算はO(N)（m=17で片側約10⁸演算）。量子資源ではないが、shotごとの回路生成と同程度の古典費用がかかる。

### 6.4 反証・失敗シナリオ（結論を覆しうるもの）

| シナリオ | 本レビューの評価 | 影響 |
|---|---|---|
| 12 context外のpost-unroll集合が一部trajectoryで現れる | 列挙は完全で、Qiskit 1.3.0固定なら低リスク | baseline厳密性が崩れる |
| baselineでchunkを跨ぐsource-level相殺がある | 読取上は該当なし（§3.3） | 同上 |
| rare pathでbuilderが余分なgateを出す | identity成分なし、零角skipは決定的、Pはorder 2のときだけ（§3.4） | 低リスク |
| **B′比較で差が大きく縮む** | **高い見込み**（P_ZZ≈1） | 主要claimの大きさが変わる |
| B′比でRZの相対差がほぼ0 | 高い見込み | RZ claimが実質的意味を失う |
| 最適化compilerを使う | 両armのRZを大きく変えうる | claimはcompiler A固有 |

### 6.5 現時点で解釈可能な範囲（一文）

> 固定compiler A（Qiskit 1.3.0、rz/sx/x/cx、全結合、1-qubit最適化なし）、固定保存H4行列モデル、finite-bit law、登録14-CX Simon–Love実装の下で、全18roundの**期待**総CXと**期待**総RZについて、reference上界がbaseline下界を下回る。

これを超える一般化（より強いbaseline、FT資源、depth、実機、新規性）には、§8の追加検証が必要である。

---

## 7. 科学的判断と確信度

### 7.1 claimの区分

| # | claim | 区分 | 確信度 | 根拠・条件 |
|---|---|---|---|---|
| 1 | 登録実装に対する全18round期待総CXの不等式（reference上界 < baseline下界） | **採用可（条件付き）** | 中〜高 | §3。条件：fence補題とcontext列挙の明文化、Qiskit 1.3.0固定、期待値としての解釈 |
| 2 | 登録実装に対する全18round期待総RZの不等式 | **採用可（条件付き、事実記述として）** | 数式は中〜高、科学的意味は低 | §3.6、§6.2 |
| 3 | 十分shot 8,598／8,606と全失敗確率0.05（固定モデル） | **採用可（条件付き）** | 中〜高 | §5 |
| 4 | 最終round真空位相誤差上界4.538×10⁻⁴ rad | **採用可（IEEEモデル下）** | 中〜高 | §5.3 |
| 5 | 4-CX parity–CRZ–uncomputeが制御RZZと厳密に一致 | **採用可（紙上証明）** | 高 | §4.2 |
| 6 | 4-CX baseline B′に対するCX削減（符号） | **未資格**（紙上導出は支持的） | 中 | §4.3。pathごとの相殺一致が未検証 |
| 7 | B′に対するCX削減率（大きさ） | **未資格** | — | P_ZZ≈1が示唆され、大幅縮小の見込み |
| 8 | B′に対するRZ削減 | **未資格**（実質的意味なしの見込み） | 低 | §4.3、§6.2 |
| 9 | depth・size・Tの優越 | **未資格・未取得** | — | depthはholdoutで大幅減を観測（§6.3） |
| 10 | 全資源のPareto優越・実機での有利・世界的新規性 | **不支持（現段階）** | 高 | §6、§7.2 |
| 11 | PR固有のpathwise位相追跡とfinite-RTE接続の独立貢献 | **条件付き・限定的** | 低〜中 | §7.2 |

### 7.2 先行研究との対応と論文化上の位置づけ（重点D-3）

- **SE-QPE**（Rowe, Gaggioli, Szulakowska, Muñoz Ramo, Manrique（Quantinuum）、arXiv:2604.14921v2 "Split-Evolution Quantum Phase Estimation for Particle-Conserving Hamiltonians"）**[C]**。target registerと真空reference registerのCSWAP干渉でcontrolled evolutionを置き換え、真空位相を古典的に補正する。正準QPEとの同値には U_A と U_B の共通固有基底が要る。正準QPE比で約33% CX、25% Tの削減、CX層depth比3/Nを主張し、2次Trotterで評価している。読めた範囲では、RTE・qDRIFT・PR・RPEは扱っていない。
- **Simon–Love**（arXiv:2511.13855v1 "Halving the Cost of Controlled Time-Evolution"）**[C]**。対称Trotterの方向制御で、制御時間発展の任意角回転数を半減する。登録baselineはこれをD部分に適用している。
- **PR原著**（Günther, Witteveen, Schmidhuber, Miller, Christandl, Harrow、arXiv:2503.05647v2 "Phase estimation with partially randomized time evolution"）**[C]**。決定的S2とRTEを組み合わせ、単一ancillaのHadamard test＋RPEで使う。費用はToffoliと回転数で数える。読めた範囲にcontrol-freeの工夫はない。
- **新Aの性格**：主に「**既知gadget（SE-QPE）のPR・fresh-shot RPEへの適用検証**」である。PR固有の要素として次が挙げられる。
  1. 非可換で共通固有基底を持たないrandomなF/Gに対する、単一blockのpathwise信号恒等式。SE-QPEの代数（単一blockの展開）からほぼ直接に従い、2397c5aも同趣旨を記録している。新規性は低い。
  2. 符号付きRTE eventを含むtrajectory別の真空位相追跡。必要な技術的作業だが、平明である。
  3. PR下でのSL制御との構造的費用比較（D部分の106 CX/stepなど）。情報価値はあるが、大きさはbaselineの強さに依存する。
  4. 全support・compiler水準・有限精度まで含む厳密な会計。方法論として丁寧だが、それ自体は科学的新規性のclaimになりにくい。
- **論文化上の評価**：現時点では、独立した貢献としては弱い〜中程度である。B′、FT的指標（任意角回転・T）、depth・space-timeの整理を経た後で、PRの資源評価論文の一節か、短い技術noteとして位置づけるのが現実的である。
- **文献比較の非網羅性**：比較したのは指定3本だけで、SE-QPEとPR原著はHTMLの一部しか読めていない。SE-QPEの序論は他のcontrol-free手法に言及している（文献本体は未取得）。参照固有状態を使うcontrol-free Hadamard testの先行例は存在する可能性が高いが、本レビューでは確認していない **[N]**。

### 7.3 研究判断（重点D-5）

**推奨：限定継続。** §8のV1（B′の最小技術検証と18round厳密再会計）とV2（保証の補強）だけを、ユーザー承認のうえで行う。その間、主要claimの対外表明は保留する。

- **継続を支持する理由**：D部分（106 CX/step）とeventあたりの制御overheadの削減は、control-freeという性質に根ざした真の効果である。holdoutで観測されたdepth削減も大きい。B′の確認は既存の構造式の小さな変更で済み、判断を決める力が大きい。
- **縮小・保留を支持する理由**：報告された大きさのかなりの部分が登録CRZZの弱さで説明できる。CX/RZ（compiler A）はFT資源の指標として弱い。qubit数は+89%になる。SE-QPEに対する新規性は限定的である。
- **良い結果を前提にしない判断規則**：V1の結果がユーザーの事前登録閾値を下回る場合は**一時保留**とする。CX削減がB′比で消える場合は、登録実装を超えるCX claimを撤回する。新C、旧C/C2、新BのSTOPは変更しない。

---

## 8. 次の選択肢・Codexへの最小追加検証・STOP条件・承認事項

### 8.1 選択肢

| 選択肢 | 内容 | 選ぶ条件 |
|---|---|---|
| **限定継続（推奨）** | V1とV2だけを承認して実施し、結果を見てから次を判断する | 現時点の推奨 |
| 一時保留 | 先に指標の優先順位（FT資源、depth・space-time、CX）をユーザーが決める | CX/RZ（compiler A）が意思決定上重要でない場合、またはV1が閾値未満の場合 |
| 停止 | 新Aの資源比較を終える | B′比でCX差が消え、depth・space-timeにも関心がない場合 |

**優先順位の根拠（重点D-4）**：第一は公平なbaseline（B′）である。結論を最も大きく変えうるのはbaselineの強さで、P_ZZ≈1の下ではevent比例部分が約1/6になるからである。丸め上界の技術監査の優先度は低い。上界はモデル内で妥当で、結論への影響は最終roundの各軸+4 shot程度にとどまる。文書化（V2）は安価なので並行してよい。**結論を変えうる追加取得**は、B′の18round総計と相殺一致の確認、FT的指標（任意角回転・T型）の比較、の2つである。

### 8.2 Codexへの最小追加検証案（**すべてユーザー承認後のみ。本レビューから自動実行しない**）

**V1：B′の技術検証と18round厳密再会計（優先1）**

- **目的**：強いbaselineに対して主要claimが成り立つか、成り立つならどの大きさかを決める。
- **対象gate族**：RTEのZZ-support制御回転だけ（`RZZGate(2·angle).control(1)` を、source上の `cx(a,b); crz(2·angle)(c,b); cx(a,b)` に置き換える）。D、frame、product、P、wrapper、law、bits、schedule、shot規則は変えない。既存SOURCEは変更せず、新規builderとして追加する。
- **入力**：固定commit `da9fceb`、共通入力pickle（SHA-256 `68269aa0…`）、保存されたfinite-bit設定。
- **作業**：
  1. 3-qubit恒等式の照合（符号2×order 2×r 6の24角）
  2. 1 occurrenceのjoint operatorを ordinary diag(I,V) と照合（ADAPTER_CHECKSのr2と同型）
  3. B′の局所catalogue（rare×Yの4 context、最終cancellation前の境界観測つき）
  4. 登録済み64 path×2軸でB′をwhole compileし、pathごとに Δ′(path) と構造式を照合する（相殺一致の確認）
  5. B′のnative誤差上界とshotの再計算（CRZZ′は解析的に厳密なので、shotは8,598以下になる見込み）
  6. 同じ有理式による18round期待総CX/RZの厳密再会計（任意で7-CX CSWAP版referenceも）。roundごとのP_ZZとp₂E|S|も記録する
- **事前登録（結果を見る前に固定する）**：「実質的なCX削減」とみなす閾値θ_CX（ユーザーが決める）と、下記の判断規則。
- **STOP条件**：
  - S1：1・2の不一致 → STOP（UNDETERMINED）
  - S2：4でB′の実価格が構造式・相殺一致モデルと合わず、証明可能な下界も作れない → STOP（UNDETERMINED、外挿しない）
  - S3：18round総計で reference CX上界 ≥ B′ CX下界 → 登録実装を超えるCX claimを撤回してSTOP
  - S4：0 < 削減率 < θ_CX → 「限定的」と分類し、一時保留を推奨してSTOP
  - S5：上記以外 → 結果を報告してmandatory STOP（depth・Tへ自動的に進まない）
- **結果で変わる判断**：主要claimの採否、V3〜V5へ進むかどうか。

**V2：保証の補強（優先1、V1と並行可）**

- (a) REPORTの補遺として、fence補題（初段passの境界条件を含む）とcontextの列挙を、Qiskit 1.3.0のコード参照つきで書く（文書作業）。
- (b) 新しい検証moduleで、代表stepのbuilder出力からrz/rzz数（N）とglobal phase更新回数を導き、ハードコード定数（336・49・24・20）以下であることをassertする。小さな回路の構築を伴うため**要承認**。
- (c) λ_prep と λ_tail がbit単位で一致することをassertする（B′を採用すればCRZZ依存は消える）。
- (d) 実行platform（CPU、glibc、Python build、wheel）のIEEE関連情報を記録する。
- **STOP条件**：assertが1つでも破れれば、該当する上界をUNQUALIFIEDに戻し、修正の承認を待つ。

**V3：指標の整理（優先2）**

- 保存catalogueのnative RZを角度で分類する（Clifford型＝π/2の倍数、T型＝±π/4の奇数倍、任意角）。新規compileはしない。そのうえで同じ構造式で、登録baseline、B′、referenceの期待任意角回転数とT型数を出す。再解析にあたるので**要承認**。
- **STOP条件**：任意角回転の削減が事前閾値θ_rot未満なら、RZ系のclaimを外す。

**V4：長r領域の経験的確認（優先3・任意）**

- r=32 eventを含む単一occurrence（RPE条件ではない）を両armとB′でwhole compileし、構造式の厳密性を長r領域で確かめる。不一致ならSTOP。

**V5：depth・space-time（優先3・任意。V1の後）**

- depthの利益を追う場合だけ行う。baselineにも公平なqubit予算（fan-out等）を与えてから、critical pathを資格化する。

**V6：真空位相計算の厳密化（任意）**

- `math.fsum` や閉形式で φ を計算し、4.5×10⁻⁴ radの項を実質0にする。効果は小さい（最終roundで各軸−4 shot程度）。SOURCE変更にあたるので**要承認**。

### 8.3 ユーザー承認が必要な作業

V1〜V6のすべて。加えて、新しいnative価格の取得、compile、simulation、sampling、合成（T）、SOURCEの追加・変更、既存STOPの解除、新しい物理条件・PF・lawへの拡張。

### 8.4 役割とSTOPの維持

- **Claude**：本レビューの作成までを担う。研究の正式採択、契約変更、計算開始は代行しない。
- **Codex**：ユーザーが明示承認した項目だけを担当する。本レビューから自動的には起動しない。
- **ユーザー**：レビューの採否、研究方向、承認、閾値（θ_CX、θ_rot）を決める。
- 最新新A・旧新A・Track C/C2・新B・新C・P-D・BF/BM・PR-3等のSOURCE、結果、STOPは変更していない。本レビューはgrant・承認証明・ledger・markerを作成・変更していない。

---

## 9. 結論に影響する未取得資料・実施できなかった確認・一次資料

### 9.1 未取得・未確認事項と影響

| 事項 | 影響 | 取得方法 |
|---|---|---|
| finite-bit lawでのP_ZZの厳密値 | B′比の大きさ（§4.3） | V1（要承認） |
| B′のnative価格・whole compile | 主要claimの採否 | V1（要承認） |
| r≥4・q≥512のwhole compile照合 | 長r領域の経験的裏付け | V4（要承認） |
| Qiskit 1.3.0の `compose_transforms.rs`・BasisTranslator `mod.rs`・UnrollCustomDefinitionsの精読 | context不変性の完全なコード論証（現状はbasis_searchと経験的不変性に依拠） | 静的読取（承認範囲内で後日可能） |
| `track_c_h4.py`・`track_c_h4_price.py` | q=1のD部分と価格sampler（結論への影響は小） | 静的読取 |
| λ_prep と λ_tail の同一性 | CRZZ角の代入の形式的完全性（影響は微小） | V2(c) |
| SE-QPE・PR原著の全文（今回は部分のみ） | 文献比較の精度 | PDF全文の取得 |
| 他のcontrol-free Hadamard test文献 | 新規性評価 | 文献調査 |
| depth・size・T、topology、noise | 全資源評価 | V3・V5ほか |

### 9.2 実施できなかった確認（依頼の制約による）

新たな数値再解析、18round総計の再集計、回路build/compile/simulation、B′の価格取得、区間の再評価、真空位相の事後厳密計算、QPYの読込は、いずれも未承認のため実施していない。本文の「検算」は、保存値と紙上の式を突き合わせる手計算に限っている。

### 9.3 一次資料URL・source箇所

固定commitのbase：`https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da9fceb944dae27f397e48734be30c6864e15f44/`（以下、相対path）

- REPORT：`docs/research/pr_reference_transfer_qualification_20261106/REPORT.md`（L52 context、L56 fence、L58 局所CX・D式、L62 basis-run式、L74 4-CX記述、L90–98 位相丸め）
- SCOPE：`docs/research/pr_reference_transfer_qualification_20261106/SCOPE.md`
- 成果物：`artifacts/pr_reference_transfer_qualification/2026-11-06/`（README.md、STAGE4_RPE.json、STAGE4_ROUNDS.tsv、STAGE2_CX_CERTIFICATE.json、RZ_PATH_BOUND_CHECKS.json、STAGE3_ALL_SUPPORT.json、STAGE3_LOCAL_INTERVALS.json、PAIRED_NATIVE_PRICES.tsv、STAGE2_PREDECLARED.json、SOURCE.json、STOP.json、local_catalogue_boundary.tar.xz、raw_prices.tar.xz）
- `src/trottertracks/pr_reference_transfer_cost_bounds.py`：L13–33 `catalog_bounds`（fence assert L26、族別CX L28）、L36–50 `parameters`（frame対の一致assert L46）、L53–56 `d_cost`、L59–74 `path_prices`、L85–101 `expected`
- `src/trottertracks/pr_reference_transfer_catalogue.py`：L18–30 `fence`、L83–129 `catalog_one`（境界観測 L101–109）
- `src/trottertracks/pr_reference_transfer_error.py`：L29–41 `interval_one`、L44–87 `actual_records`（CRZZ template L62–84）、L93–96 `gamma`、L99–134 `phase_bound`（N L120、rem_bound L124–127、正規化 L131）、L176–179 B/C、L180–188 disk・半径・β
- `src/trottertracks/pr_reference_transfer_accounting.py`：L28–60 `expected_rz_bounds`、L69–107 `validate_rz_paths`、L110–183 `account`
- `src/trottertracks/pr_reference_transfer_qualification.py`：L143–153 `events_for`、L156–180 `assemble`、L226–254 `sample`
- `src/trottertracks/pr_reference_interference_pilot.py`：L95–103 `vacuum_phase`、L120–130 `reference_core`、L139–144 `wrapper`、L170–189 `directional_right`、L198–213 `symmetric_fused`
- `src/trottertracks/track_c_compiler_a.py`：L30–46 `compile_reference`（pass列 L39–42）
- `src/trottertracks/track_c_compiler_a_18round.py`：L153–214 区間行列
- `src/trotterlib/df_rte_qiskit.py`：L246–249・L348–356 位相、L278–338 basis共有、L472–502 中央適用（CRZZ生成 L499）
- `src/trotterlib/rte.py`：L637–643 `unsigned_rotation_angle`、L1514–1515 τ、L1594–1654 `_make_event`（位相 L1649、角 L1653）
- `src/trotterlib/df_partial_s2_repeated.py`：L429–435 `_compose`、L575–589 集約位相、L591–622 boundary-optimized
- `src/trotterlib/df_trotter/ops.py`：L295–321 `append_diagonal_primitives`

関連固定REPORT：
- 旧新A pilot：`https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/79cba2fc4882bc42282f8feee7257a69f7628072/docs/research/pr_reference_interference_pilot_20261106/REPORT.md`
- 静的調査：`https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2397c5ac511148423ae89f25f74f1de967d1d39a/docs/research/pr_reference_signal_static_investigation_20261106/REPORT.md`
- 固定保存行列保証：`https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/docs/research/track_c_guarantee_cn_price_20261106/A_REPORT.md`
- compiler A correctness：`https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f45e9b07bf730f7272933b28055914cfe76fdaab/docs/research/track_c_compiler_a_correctness_20261106/REPORT.md`
- 18round compiler構造上界：`https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3258a89f225d9530fba38e9d2343bc0165dd6d7d/docs/research/track_c_compiler_a_18round_qualification_20261106/REPORT.md`

Qiskit 1.3.0（tag `1.3.0`、commit `896d1c8b`）：
- `https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/inverse_cancellation.rs`（L42–107、L160–186）
- `https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/basis/basis_translator/basis_search.rs`（L34–179）
- `https://github.com/Qiskit/qiskit/blob/1.3.0/crates/circuit/src/circuit_data.rs`（L848–866）
- `https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/circuit/add_control.py`（L68–250）
- `https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/circuit/library/standard_gates/rzz.py`（L99–156）、`.../rz.py`（CRZ定義 L256–282）、`.../x.py`（CCX定義）、`.../swap.py`（CSWAP定義）、`.../xx_plus_yy.py`（定義）

文献：
- SE-QPE：https://arxiv.org/abs/2604.14921v2
- Simon–Love：https://arxiv.org/abs/2511.13855v1
- PR原著：https://arxiv.org/abs/2503.05647v2

---

## 付録：主な記号

| 記号 | 意味 |
|---|---|
| q, r | roundのPR step数（q=2^m）、occurrenceあたりのRTE event数 |
| n | 片側（FまたはG）のstep数 = max(1, q/2) |
| p₂ | eventがorder 2になる確率（finite-bit law） |
| P_ZZ, P_Z | 回転成分がZZ-support、Z-supportである確率（P_Z = 1−P_ZZ） |
| E\|S\| | 成分supportの期待サイズ（= 1+P_ZZ） |
| c_b | basis bのframe対（frameと逆）のCX数 |
| D_B, D_R | D部分（決定的block＋CSWAP）の1 shot費用（baseline／reference） |
| B′ | 制御RZZだけを4-CX分解に置き換えた代替baseline |
| φ_F, φ_G | F、Gの真空位相（global_phase − Σθ/2） |
| u, γ_N | 2⁻⁵³、Nu/(1−Nu) |
| ρ, β, ε | raw半径下界、残余統計位相、軸あたりの許容誤差 ε=ρ sinβ/√2 |
