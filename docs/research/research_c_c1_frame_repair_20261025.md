# C1 Stage 1：raw CAR frame縮約の修正・人工検証（2026-10-25）

**raw CAR側の人工並列実行で大きな不一致を再現し、その縮約経路を明示的な行列積へ置き換えた。人工・既存回帰63 test＋29 subtestsは全PASS、別プロセス8回・64 frame比較も全PASS。** 今回は技術修正と人工検証だけを実施した。保存H4の数値decode／fit export再解析／operator・norm・energy・residual／実fit／回路build・compileは0。旧科学的STOPは保全し、H4の同値性PASSへ読み替えない。

[人工検証summary](../../artifacts/research_c_c1_frame_repair/2026-10-25/technical_summary.json)、[SOURCE](../../artifacts/research_c_c1_frame_repair/2026-10-25/SOURCE.json)、[未承認の引継ぎ](research_c_c1_frame_repair_handoff_unapproved_20261025.json)。

## 固定点と今回の作業範囲

起点は、[前回のframe0同値性FAIL・部分証拠・STOP](research_c_c1_stage1_compat_execution_20261025.md)を公開した `f65bbba4d1c54658dfffee11a960370089789030`。ユーザーは「修正を行って　中断された場合は修正を行って再実行してよい」、続いて「続きを行って」と指示した。直前に提案した人工再現・原因切り分け・最小修正・回帰を実施し、技術的な中断は修正可能と解釈した。分子の新規科学実行／保存値の数値再解析を追加承認したものとは解釈せず、旧one-fit枠・grant・STOPを再利用していない。

| 固定点 | 値 |
|---|---|
| Repository | `HIROMU1015/Partially-Randomized-Trotter` |
| Branch / worktree | `codex/research-c-c1-frame-repair-20261025` / `/tmp/prt-c1-frame-repair-20261025` |
| 新実装SOURCE commit | `2ab9977994afa31cdfffb8629ce57c6efa83a842` |
| 新SOURCE manifest | `artifacts/research_c_c1_frame_repair/2026-10-25/SOURCE.json`、29ファイル |
| manifest SHA256 | `a0434c395c99d41114dfe1cb7984c41453227b991006b2ac02230e62b035c9e3` |
| raw kernel識別子 | `car_density_batched_matmul_v2` |
| 引継ぎ状態 | `execution_authorized=false, grant_issued=false, dispatch_enabled=false` |

今回の入力は人工の4空間／8 spin軌道と8 frameで、分子geometry／AO basis／分子DF rankは非該当。科学的なP-H4条件は変更していない。保存P-H4（直鎖H4・1 Å・STO-3G、Nα=Nβ=2／sector36、必要base／frameはfull256）、通常C-DF number／元g由来κ、max_vecs8／tol1e-8／L-BFGS-B、保存rank10 XDF、物理DF誤差とadapter同値性の別判定は前契約のまま。L_D／delta窓はStage 1非該当。Stage 2/3へ進まない。

## 切り分けで確認できたことと限界

4軌道全体を混ぜる実／複素rotationと、符号・非対角成分を持つ一般Jを人工生成した。旧testは2軌道blockだけを混ぜるケースや少数frameが中心だった。4軌道を混ぜるだけでは旧raw／Kが一致するケースもあり、係数・spin順序が普遍的に誤っていると断定できなかった。

8 frameのraw／Kを、新しいPythonプロセス内で並列に初回評価すると、旧rawのoptimized einsum縮約に大きな不一致が現れた。[旧経路／修正経路の比較原本](../../artifacts/research_c_c1_frame_repair/2026-10-25/artificial_parallel_audit.json)。保存された旧経路trialではframe1のraw対直接和max-entryが約0.4800、frame3が約1.1434となる一方、K候補対同じ直接和は丸め誤差の範囲だった。

同じ人工式を直列で評価した場合や、candidate backendを主threadで先にimportしてから並列化した場合は不一致が現れないことがあった。[backend先読込の負の対照4プロセス](../../artifacts/research_c_c1_frame_repair/2026-10-25/preloaded_backend_audit.json)では、旧raw対直接和の最大値は3.552713678800501e-15だった。この対照も削除せず公開する。

したがって、人工データ上で**起動・読込順序に敏感な旧raw縮約経路の不具合**は確認したが、NumPy／OpenBLAS等の内部原因を同定したとは主張しない。また、旧H4の2.8302 HartreeのFAILをこの不具合だけへ因果的に帰属させるには、修正後SOURCEでの別承認付き再検証が必要である。今回H4行列・J/Uを再計算してその結論を先取りしていない。

## 最小修正と科学条件の維持

変更は [c1_stage1.raw_frame](../../src/trottertracks/c1_stage1.py) の3つのoptimized einsum縮約を、同じ数式の明示的な行列積へ接続する部分である。[差分](../../artifacts/research_c_c1_frame_repair/2026-10-25/implementation.patch)。

元のCAR密度とframe演算子は、引き続き

`rho_i = sum_(p,q,spin) U[p,i] conjugate(U[q,i]) a†_(p,spin) a_(q,spin)`

`H_frame = 1/2 sum_(i,j) J[i,j] rho_i rho_j`

で構築する。密度係数を行列としてCAR bilinear配列へ掛け、Jを密度配列へ掛け、最後にbatchの行列積をframe内で加算する。新しいCAR代数・fitter・数値ライブラリ・Gaussian合成器は作っていない。既存のCAR bit作用、一般K adapter、OpenFermion候補、κ／scalar、保存XDF評価、solver、依存版は再利用した。

一般K／I/Z/ZZ係数、回転・spin・bit convention、frame順序／数、物理Hamiltonian、fit、XDF baselineは変更していない。rawにK係数や候補の密度を流用せず、独立な組立てを保持した。射影・再直交化・対称化・係数pruningは加えず、dtypeはcomplex128のまま。加算・縮約の順序は変わるため、旧経路とのbitwise一致は要求しない。

数値allowance `binary64_operation_scale_guard8_v1` の式・guard8・norm係数32・演算長・precision・閾値は変更なし。新経路の主要な和の長さは密度n²、J縮約n、行列積d、最後のframe内加算nで、既存raw演算長 `8d+40n²+32` の工学的見積り内に収まる構成を維持した。これは厳密誤差証明への格上げではない。人工testのabsolute tolerance3e-13はテスト用であり、分子契約のallowanceを置き換えない。

frame workerは従来の `min(R, available affinity)`、BLAS threadは `floor(available/workers)` のまま。直列化・一律のthread数制限で問題を隠す修正はしていない。今回の8 frameは8 worker／BLAS4 thread。科学callの定義と成功経路 `83+8R`／最大147にも変更はない。評価結果にraw kernel版を記録し、旧結果との区別を可能にした。科学runnerの入口とapproval判定は変更していないため、旧grantが新SOURCEを承認することにはならない。

## 検証結果

[全コマンド・JUnit・log](../../artifacts/research_c_c1_frame_repair/2026-10-25/tests/)。localの人工検証をGitへ固定した証拠であり、immutable CI／外部再現／分子資格と区別する。

| 検証 | 結果 |
|---|---|
| 修正用新test | 4 PASS。4軌道全体を混ぜる実／複素rotation・一般J・非連続配列、直接CAR和／K候補との一致、fresh-process並列、8 frame full256／sector36のend-to-end |
| 既存Stage1／execution | 27 PASS。κ・scalar・符号・spin／bit順序、一般K／独立reference、mock fit/export、call／approval／STOP境界 |
| 既存ffsim互換性／fit環境 | 19 PASS。実ffsim人工初期化はoptimizerをmockで遮断。実分子fitなし |
| 既存一般K adapter | 13 PASS＋29 subtests PASS |
| 合計 | **63 test＋29 subtests PASS、fail/error/skip0** |
| 別プロセス並列監査 | 修正経路は8新規プロセス×8 frame＝64比較、全PASS。raw対直列のmax-entryは0、raw対直接和は最大3.552713678800501e-15、raw対Kは最大4.440892098500626e-15 |
| end-to-endの計上 | 人工8 frameで全37 gateがPASS、物理誤差の非gate診断2件も記録。評価call139＋fit/load相当8＝契約成功経路147を確認 |

直接CAR参照は、空間p/q・spinを明示したloopで各密度を作り、16個のJ重み付き密度積を足す。新kernelのreshape／einsumやK展開は共有しない。K側は従来の独立OpenFermion density／I/Z/ZZ経路である。並列旧経路FAILを毎回再現するflakyなassertはunit testへ加えず、人工監査の観測記録として保全した。unit testは修正後の参照一致をassertする。

専用fit/eval venvをread-onlyで再利用し、cache／tempは今回のworktree内へ分離した。[fit runtime](../../artifacts/research_c_c1_frame_repair/2026-10-25/fit_runtime.json)、[eval runtime／BLAS](../../artifacts/research_c_c1_frame_repair/2026-10-25/eval_runtime.json)。CPython3.11.1、NumPy1.26.4、SciPy1.14.1、ffsim0.0.83、JAX0.4.38等のlockは同じ、ffsim Python102ファイルは一致、pip check両方PASS。NumPy／SciPy／ffsim本体や既存venvは編集していない。venvへのtask内symlinkはGit対象外とした。

資源は32 CPU、MemAvailable約52.4 GiB、memory pressure0。[probe](../../artifacts/research_c_c1_frame_repair/2026-10-25/resource_preflight.json)。人工行列を既存BLAS・frame並列経路で処理し、恣意的なCPU／RAM／worker／wall capは追加していない。速度比較は実施しておらずspeedupは主張しない。

## 保全、残件、次の担当

[保全記録](../../artifacts/research_c_c1_frame_repair/2026-10-25/preservation.json)に、旧artifact1,388ファイルのGit blob identity、元workspaceのbyte／diff hash、旧SOURCE25ファイル中の変更が `c1_stage1.py` だけであることを記録した。現行manifestへの新entry追記を除き、旧科学／技術証拠・source manifest・結果・grant・ledger・marker・STOPは変更しない。今回のSOURCE／test／証拠もhashで固定する。

**人工範囲での修正・回帰検証は完了したが、Stage 1の科学的資格は未完了。** 旧H4 frame0のFAIL、未測定のCDF／XDF物理sector normと未検証のsector adapter同値性は残る。今回の技術PASSを使って旧科学的STOPを解除しない。

- **ユーザー／統括（次の判断）**：保存済みの収束J/U exportを使う評価だけの準備・再検証を進めるか判断する。追加fitが必要と確定したわけではない。
- **Codex（その依頼後の準備案）**：旧fit source／input／export hashと新評価SOURCEを別々に保持した評価専用接続・call契約を準備し、人工確認する。現行のone-fit runnerをそのまま起動すると新fitになるため、保存export再利用の入口として使わない。今回その新接続は実装していない。
- **ユーザー／統括（科学実行前）**：新評価SOURCE・契約・固定保存export・fresh outputへの明示承認を行う。旧grant／run／STOPは再利用しない。Stage 2/3は引き続き未承認。

修正SOURCEと人工証拠をcommit／pushし、remote SHA／SOURCE／成果物を照合後、**mandatory STOPする。** 分子科学実行・保存数値再解析へ自動進行しない。
