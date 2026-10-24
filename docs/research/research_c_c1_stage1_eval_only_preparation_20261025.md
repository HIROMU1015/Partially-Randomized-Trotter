# C1 Stage 1：保存fit評価専用経路の技術準備（2026-10-25）

**新規fitを呼ばない評価専用entryを実装した。人工・既存回帰82 test＋29 subtestsは全PASS。固定exportのbyte hash・metadata・Git来歴と新評価SOURCEのpreflightもPASSした。** 今回の保存H4 J/U・元入力の数値decode、分子operator／norm／residual／energy、分子fit／optimizer実行、実回路build／compileは0。科学契約は未承認で、mandatory STOPする。前回のH4 frame0 FAILは未再検証のまま保全する。

[新SOURCE](../../artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/SOURCE.json)、[未承認科学契約](research_c_c1_stage1_eval_only_contract_unapproved_20261025.json)、[人工検証summary](../../artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/technical_summary.json)、[preflight証拠](../../artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/provenance_preflight.json)。

## 固定したidentityと承認範囲

起点 `d28c52bcb4cdb470edb2c89676a0add06a8899b6`。[前回の科学的STOP](research_c_c1_stage1_compat_execution_20261025.md)、[raw CAR修正](research_c_c1_frame_repair_20261025.md)と固定実装・executor・testsを確認した。ユーザーは評価専用の技術実装・人工検証・未承認契約作成を承認し、保存J/Uの数値再解析・H4評価・新fitを承認していない。旧grant／run／ledger／STOPを新しい実行へ流用しない。

| identity | 固定値 |
|---|---|
| Repository / branch | `HIROMU1015/Partially-Randomized-Trotter` / `codex/research-c-c1-eval-only-preparation-20261025` |
| 独立worktree | `/tmp/prt-c1-eval-only-preparation-20261025` |
| 新評価entry実装commit | **`88169af81cbc1140c4e51cfc1acf42a66db340b0`** |
| 新評価SOURCE manifest SHA256 | **`5be0fea39b167845343f94985699f31a0a1bb07d085ea2ed9312a1af9b8dede5`**、32ファイル |
| 新評価専用契約 SHA256 | **`f4d3aa81b3a4fd2272144f0c45fd82bb2aa748f4fc59a6f5875e5dd3e0949dfc`** |
| 再利用する数値評価library | `2ab9977994afa31cdfffb8629ce57c6efa83a842` の `c1_stage1.py` とbyte同一 |
| raw kernel | `car_density_batched_matmul_v2` |
| 旧fit公開SOURCE | `da2edb465d5ae6432439fbd1df8e57512ee59c11` |
| 旧fit SOURCE manifest SHA256 | `e7bf161d2ab45a78d772c960ab5c2c41d3ed145bfd0a1193ee280536c1547240` |
| 旧fit／STOPの証拠commit | `f65bbba4d1c54658dfffee11a960370089789030` |
| 保存export | `artifacts/research_c_c1_stage1_compat_science/2026-10-25/run01/fit_export.npz` |
| 保存export SHA256 | `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08` |
| 保存export metadata SHA256 | `945cbc2ee31930c5367c37300e2e9d59e2c4696e63907a2973a4cb9029311e5f` |
| 元input SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |

旧fitのSOURCE・input・exportと、新評価entry／修正libraryのidentityを混同しない。metadataの旧 `source_manifest_sha256` を新manifestへ書き換えて受理する処理はない。保存fitの収束・実outer rank8は既存JSON宣言として検査するだけで、tensor残差やoptimizer値を新たに評価していない。

## 実装と再利用

新設 [c1_stage1_eval_only.py](../../src/trottertracks/c1_stage1_eval_only.py) と [run_research_c_c1_stage1_eval_only.py](../../scripts/run_research_c_c1_stage1_eval_only.py) は、評価専用のpreflight／承認／I/O glueである。旧one-fit runnerは再起動せず、fit worker・`fit_once`・optimizerを呼ぶ分岐やfit Pythonの引数を持たない。新scopeは `C1_STAGE1_P_H4_SAVED_FIT_EVAL_ONLY_V1`。

以下を既存処理のまま再利用した。

- 旧executorのbyte hash／JSON、runtime identity、load journal、入力whitelist、export schema／array hash検証。
- 修正済み `c1_stage1.evaluate` とraw CAR、一般K／I/Z/ZZ、独立OF、保存XDF sector36 basis作用、SVD norm、固定allowance、frame並列。
- 既存のoperator保存sinkの形式。行列を保存するための追加構築や追加normは行わない。

数値library・一般K adapter・CAR reference・旧executor・旧one-fit runnerは無変更。新fitter・CAR代数・Gaussian合成器・数値libraryは実装していない。

既存 `load_export` はそのまま使い、**元入力と旧fit SOURCE**を渡して保存配列を検証する。その後、新評価SOURCEへ独立にbindingした `evaluate` を呼ぶ。将来の実行出力には `fit_SOURCE.json` と `evaluation_SOURCE.json` を別保存し、両hashとinput／export hashを `provenance.json` へ記録する。旧metadataの歴史的fit call6計上単位は来歴として残すが、新runの科学callへ加算しない。

## 数値decodeしないpreflightと、実行時の確認

既定CLIは読取り専用preflightであり、output・approvalを作らない。固定export bytes／metadata／旧SOURCE／旧契約／旧approval／fit claim／optimizer status／旧STOP／fit runtimeの9ファイルをhashと証拠commitへ照合する。旧SOURCE自体が旧公開commitに存在すること、元実装commitの各ファイルhashと一致すること、新SOURCE32ファイルが新実装commitと一致することも確認する。

metadataのschema、ffsim／compatibility版、number／κ role、旧fit条件、成功status、rank8、frame／group順序、配列shape／dtype／hash宣言を検査する。宣言されたschemaの確認と、実配列の検査は区別している。**NPZ内の数値はpreflightで開かない。** 旧／新manifestの置換、hash drift、歴史的commit不一致、誤ったmetadataはdecode前に拒否する。

[stdlib-only監査](../../artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/provenance_preflight.json)ではNumPy／SciPy／ffsim／JAX／OpenFermion／PySCF／Qiskitのimportを禁止した状態で、実際の固定export・元input bytesによるpreflightがPASSした。これらの数値libraryは一つもimportされず、科学call0・array decode0である。CLIの[preflight結果](../../artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/preflight.json)も保存した。

将来の科学実行では、別の新承認を検証後に限り、入力whitelistを1回、保存exportを1回loadする。既存loaderによりJ/U／κ／c／保存x/jacの全6 memberのschema・hash・finite値を確認する。x/jacは整合性検証用で、optimizer・objective・tensor residualを実行／再計算するために使わない。元fitが読んだh/g/c/MOのmember hashと今回入力を照合し、cの所有を検査する。旧export／metadataを更新・再exportしない。

新承認は、公開評価commit、新SOURCE・契約hash、旧fit source／manifest、保存export hash、元input hash、fresh output、新fit0、allowance版と採択を束縛する。旧fit承認は新scopeで拒否し、すでに存在するoutputはresumeせず拒否する。dispatch直前にもpreflightを再実施する。CLIに `--worker-fit` 等の旧fit modeはない。

## 科学条件・検証順序・call

物理条件は保存P-H4、直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2／sector36、必要base／各frameはfull256。通常C-DF number表現、元gから補正したκ保持、元の1 frameを一般K 1 frameとして保持する。保存収束fitは実rank8、歴史的条件max_vecs8／tol1e-8／L-BFGS-B。対照は同一inputの保存実rank10 X-DF、再分解なし。L_D／delta窓はStage 1非該当。

評価の順序は既存 `evaluate` のままである。

1. κ／元g／保存XDF κと、全J／Uの構造gate。
2. 既存のphysical／XDF sector行列構築（XDF basis作用36を含む）。ここではDF誤差normを計算しない。
3. full256のbase同値性、次にframe0–7の同値性。最初のFAILでSTOPする。
4. 全base／frameがPASSした場合だけ、full Nα／Nβ保存、sector restriction、4種sector Hermiticity、sector36 adapter同値性へ進む。
5. その後、C-DFとX-DFの物理sector spectral errorを別々に測定し、mandatory STOP。

したがって、base／frame FAIL後にsector同値性やCDF／XDF物理誤差を追加計算しない。物理sector行列の構築自体は以前と同じ順序でfull gateより前に行われるため、失敗時に一部行列が残ることはある。その存在を未測定normの代わりには使わない。

allowanceは **`binary64_operation_scale_guard8_v1` の既存JSON・数値係数をそのまま固定**した。`u=2^-53`、`gamma_m=m*u/(1-m*u)`、spectral allowance `8*(gamma_mL*B_L+gamma_mR*B_R+32*d*u*Frobenius(delta))`、structuralはsqrt(d)倍、既存演算長・scaleを保持する。厳密certificateへ格上げせず、事後調整しない。物理DF誤差の絶対閾値やCDF≤XDF条件は設けない。Stage 2適格性は別判断である。

| 今回の評価専用call | 成功経路／上限（R8） |
|---|---:|
| 元input load／保存export load | 各1、計2 |
| adapter import | 1 |
| physical build／XDF build | 各1 |
| XDF basis matvec | 36 |
| base build／frame build | 2／16 |
| full sum／sector restriction | 各2 |
| spectral norm | R+4＝12 |
| structural Frobenius | 2R+11＝27 |
| allowance | 3R+15＝39 |
| **評価小計／load込み合計** | **139／141** |
| **新規fit／optimizer／tensor reconstruction・residual／energy** | **すべて0** |

成功経路の合計は **`77+8R`、固定R8で最大141**。旧成功経路 `83+8R` から、fit phaseの6計上単位だけを除いた。評価内容の追加や省略ではない。内部配列hash／allowance内部workは従来と同じ技術的分類で、別の科学jobへ数え直さない。失敗時は実際のprefixを記録する。

frame pairは先に全16 build jobを台帳登録して並列submitする既存方式を保持する。最初のFAIL後に後続gateは実施しないが、投入済みworkのcleanupを待つ場合がある。frame0 FAILの人工caseはload込み101単位。旧fit込みの107を今回の実績へ流用しない。成功時のoperator保存は24ファイルで、追加計算なし。

## 人工テスト・環境・保全

[新test](../../tests/management/test_research_c_c1_stage1_eval_only.py)、[コマンド・JUnit・log](../../artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/tests/)。local人工検証をcommitへ固定した証拠であり、immutable CIや分子資格とは区別する。

| 検証 | 結果 |
|---|---|
| 評価専用の新規test | **19 PASS**。metadata／byte／履歴drift、旧承認拒否、旧／新SOURCE独立性、人工rank8 full256／sector36、141計上、base／frame FAIL後のsector禁止、割込みSTOP、fresh output／resume拒否、dispatch直前再照合、CLIにfit modeなし |
| 既存評価・raw frame／executor | 31 PASS |
| 既存ffsim互換性・fit環境回帰 | 19 PASS。mock fitと、人工初期化のoptimizer入口遮断を含む既存技術test。評価専用entryはffsimをimportせず、fit／optimizer呼出しも禁止mockで未呼出しを確認 |
| 既存一般K adapter | 13 PASS＋29 subtests PASS |
| 合計 | **82 test＋29 subtests PASS、最終fail/error/skip0** |

初期のtest harnessでは、Git用mockが依存libraryの `uname` 呼出しまで拒否した点と、FAIL prefixの期待値へ旧fit6単位を残した点を修正した。初期失敗log／JUnitはlossless gzipとhashで保全した。科学処理のretryや閾値変更ではない。

新entryの人工成功caseは元のJ/Uからの全37 gateを通過し、非gateの物理DF誤差2件も記録した。故意にbaseまたはframeを壊すcaseは対応full gateでSTOPし、sector比較・物理DF誤差normへ進まない。旧fit worker／one-fit executor／`fit_once`／SciPy optimizerへの呼出しは禁止mockで検出する。

既存の専用eval venvと同版依存lockをread-onlyで再利用した。[runtime／BLAS](../../artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/eval_runtime.json)、[pip check](../../artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/eval-pip-check.log)。CPython3.11.1、NumPy1.26.4、SciPy1.14.1、OpenFermion1.6.1等41 pinsを照合、pip check PASS。新評価entryにfit venv／ffsim／JAXは不要。旧fit環境の人工回帰は別processで実施し、共有環境は変更していない。

資源は32 CPU、MemAvailable約52.5 GiB、memory pressure0、GPU driver利用不可。[資源記録](../../artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/resource_preflight.json)。既存のCAR batch matmul／cached bilinear／frame workerを維持し、R8では8 workerとBLAS4 thread。CPU／RAM／worker／phase・total wall capを新設せず、速度改善率は測定・主張しない。

[保全記録](../../artifacts/research_c_c1_stage1_eval_only_technical/2026-10-25/preservation.json)に、旧artifact1,413ファイルのGit blob、元workspaceのbyte／diff hash、数値library・旧executor・one-fit runnerの無変更を記録した。現行manifestへの新entry追記以外、旧科学結果／SOURCE／grant／ledger／marker／STOPは変更しない。保存fitと元inputはbyte hashとmetadataだけで確認した。

## 完了と次の担当

評価専用の技術実装・人工検証・未承認契約作成は完了。技術blockerは残っていないが、実H4の同値性・物理誤差は今回未検証であり、前回frame0 FAILは解消済みとしない。

- **ユーザー／研究C（次の判断）**：公開評価SOURCE、固定保存export、元input、契約hash、allowance、fresh output、fit0／最大141／retry・resume0を確認し、評価専用科学実行を別途明示承認するか決める。
- **Codex（その承認後のみ）**：新しい専用approvalを作成し、preflight／依存／資源を再照合して、新entryで保存fitを1回評価する。科学FAIL／契約不一致は証拠保全してSTOPし、refitや自動resumeを行わない。
- **ユーザー／統括（Stage 1評価後の別判断）**：全gateと別測定CDF／XDF誤差を確認し、Stage 2適格性を判断する。Stage 2/3は未承認。

commit／push後のremote SHA・SOURCE・成果物照合まで実施し、**mandatory STOPする。実H4評価へ自動進行しない。**
