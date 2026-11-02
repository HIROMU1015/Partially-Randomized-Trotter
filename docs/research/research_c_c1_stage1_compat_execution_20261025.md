# C1 Stage 1互換性修正版：fit収束後、frame同値性FAILでSTOP（2026-10-25）

**通常C-DFの新しい1 fitは収束したが、full256の最初のframe同値性gateでFAILした。状態は `SCIENTIFIC_STOP_EQUALITY`。Stage 1は未完了であり、再実行・追加計算・修正を行わずmandatory STOPした。** C-DFとX-DFの物理sector誤差、sector36の一般K同値性は未測定／未検証のままである。

[実行summary・全gate状態](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/execution_summary.json)、[実行原本run01](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/run01/)、[STOP](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/run01/STOP.json)、[専用approval](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/approval_run01.json)。公開するのはlocal実行の原本であり、immutable CI・外部再現の証拠とは区別する。

## 承認、固定条件、preflight

前回の[互換性修正報告](research_c_c1_stage1_compatibility_20261025.md)で次の作業を「新SOURCE・契約・入力・fresh outputへの新しい1 fitのStage 1科学実行」と明示した後、ユーザーが **「次の作業に進んで大丈夫」** と承認した。この文脈をapprovalへ保存し、新scope／固定hash／新outputにbindingした。Codexが独自に科学条件や実行権限を発行したものではない。旧run01/run02の消費済みfit枠・grant・ledger・STOPは再利用していない。

| 項目 | 固定・実行条件 |
|---|---|
| Repository | `HIROMU1015/Partially-Randomized-Trotter` |
| 実行時公開SOURCE | `da2edb465d5ae6432439fbd1df8e57512ee59c11` |
| 実装SOURCE | `303f73275f133bfed0a0cd93a06f4a46f8ffd7be`。新しいfitter／Gaussian合成器はなし |
| Branch / worktree | `codex/research-c-c1-stage1-compat-science-20261025` / `/tmp/prt-c1-stage1-compat-science-20261025` |
| 契約 | [互換性修正版の契約原本](research_c_c1_stage1_compatibility_contract_unapproved_20261025.json)、SHA256 `e89f8bb323efaf5cfd72a2f8c35b5462fc0ff456b194e4dcd2f0bbc08c0310e8` |
| SOURCE manifest | [原本](../../artifacts/research_c_c1_stage1_compatibility/2026-10-25/SOURCE.json)、SHA256 `e7bf161d2ab45a78d772c960ab5c2c41d3ed145bfd0a1193ee280536c1547240` |
| input | 保存P-H4、commit `86b8b002704aced8ca239414080f53793de81658` の固定snapshot。SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| 分子 | 直鎖H4、隣接1.00 Å、STO-3G、charge0／spin0、保存実canonical RHF MO、4空間／8 spin軌道 |
| 比較範囲 | Nα=Nβ=2、sector36。adapter base／各frameはfull256でも検査 |
| C-DF | ffsim v0.0.83通常C-DF、number表現、元gから補正したκ保持、一般Kは元の1 frameを保持 |
| fit | max_vecs8、tol1e-8、L-BFGS-B、options={}、1 fit。**今回の実Rは8** |
| baseline | 同じsnapshotの保存実rank10 X-DF、再分解なし |
| allowance | `binary64_operation_scale_guard8_v1`、限定pilot用の採択を保持。式・係数・precision・thresholdの変更なし、厳密certificateではない |
| 物理誤差 | CDF／XDFのsector operator normを別々に報告する予定。絶対DF閾値なし、adapter同値性と別判定 |
| L_D / delta窓 | Stage 1では非該当。回路／PF／RTE／samplingは対象外 |
| fresh output ID | `C1_STAGE1_COMPAT_V2_20261025_run01`、新family `artifacts/research_c_c1_stage1_compat_science/2026-10-25/run01` |

契約原本の未承認表示は改変せず、今回の承認を別記録にした。[preflight](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/preflight.json)、[approval preflight](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/approval_preflight.json)、[fit環境](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/fit_preflight.json)、[eval環境](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/eval_preflight.json)はPASS。SOURCE25ファイル／実装commit、input bytes、契約hash、依存pins、ffsim Python102ファイル、JAX x64を再確認した。

前回作成した専用venvをread-onlyで再利用し、cacheは今回のtask内へ分離した。CPython3.11.1、ffsim0.0.83、SciPy1.14.1、NumPy1.26.4、JAX/jaxlib0.4.38、CPU backendは固定どおり。実行直前に人工初期化・venv起動・既存fit環境test19件もPASSした。[JUnit](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/artificial_preflight.xml)。SOURCE・アルゴリズム・実行pipelineの変更は0。

## 取得した結果と全gate

[ffsim optimizer原本](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/run01/raw_status.json)、[export metadata](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/run01/fit_export.json)、[登録gate原本](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/run01/result.json)。

fitはsuccess=true、status0、`CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`、nit68、nfev74、njev74。rank8のJ/U、元g由来κ、scalar、optimizer x/jacをexportした。目的関数は `2.1839632052963425e-9`、tensor residualはmaxabs `1.0550030930367438e-5`、Frobenius `6.60902898358172e-5`。export再構成objectiveとの差は `5.181459357032165e-21`。これらは今回のfit内で計算・保存済みの診断値を転記しており、追加の数値再解析ではない。

| gate / 測定 | 結果 |
|---|---|
| SOURCE／input／契約／依存／approval／fresh output | PASS |
| fit収束・有限export・正rank8・J exact対称 | PASS |
| κ Hermitian／元g κ一致／保存XDF κ一致 | 3件PASS |
| J対称／U unitary、各8 frame | 16件PASS |
| full256 baseのraw CAR／K同値性 | PASS。誤差 `3.552713678800501e-15`、allowance `1.7591898453124768e-11` |
| **full256 frame0のraw CAR／K同値性** | **FAIL**。spectral error **`2.8302153250645086` Hartree**、allowance **`4.831545715791788e-10` Hartree** |
| full frame1–7のgate | 未到達、7件NOT_RUN |
| full raw／KのNα／Nβ保存 | 未到達、4件NOT_RUN |
| sector physical／CDF／K／XDF Hermitian | 未到達、4件NOT_RUN |
| sector36 adapter同値性 | 未到達、1件NOT_RUN |
| C-DF物理sector誤差（Hartree） | **未測定／null** |
| X-DF物理sector誤差（Hartree） | **未測定／null** |

登録された39診断のうち20 PASS・1 FAIL・18 NOT_RUN（後者に物理DF誤差の非gate測定2件を含む）。pipelineは最初のgate FAILでSTOPした。process exitcode0は失敗gateを記録して正常に停止処理を終えたことを示すだけで、科学的PASSではない。

このFAILは、固定したraw CAR経路とK候補経路がframe0で一致しなかったという結果である。この比較だけからどちらの経路が正しいか、誤りがadapter／独立reference／展開・basis conventionのどこにあるかは特定できない。閾値を変更せず、同じ分子行列を使った追加検算・原因切り分け・修正は行っていない。ffsim/SciPyの旧API障害は通過したが、Stage 1表現資格は成立していない。C-DFの物理精度やX-DFに対する優位性をこの結果から主張しない。

## 科学call、部分証拠、資源

[call summary](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/call_summary.json)。fitのinput load／fit／qualification／tensor reconstruction／maxabs／Frobeniusが各1で6計上単位、評価input／export loadが各1で2、評価のprefixが99、合計 **107／最大147**。実fit呼出し1・正常return1、science retry0／resume0、STOP後の新規科学call0。成功時 `83+8R` の規則は変更せず、今回R8でも成功経路147を実績として代入しない。

評価prefixはadapter1、構造Frobenius19、allowance21、physical build1、XDF build1、XDF matvec36、base build2、spectral norm2、frame build16。固定コードは8組のraw/K frame job全16件を先に登録し、executorへ並列submitする。F0 gate FAIL時には既存executorのcleanupで投入済みjobの終了を待つが、F1以後のgate／sector和／normを追加実行していない。台帳上の16件をF0の2件へ減らして報告しない。

[operator inventory](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/run01/operator_inventory.jsonl)に、physical／XDFのsector行列2、raw／K base2、raw／K frame0の2、計6 NPZを保存した。残りframeの行列はexportしていない。保存J/U／raw optimizer出力／call台帳／runtime／STOPをそのまま保持する。保存済みphysical／XDF行列が存在しても、停止後にnormを計算して未測定値を補っていない。過去のstate／energy／科学的PASSも流用していない。

起動時は32 CPU、MemAvailable約52.4 GiB、memory pressure0、GPU driver利用不可。[資源preflight](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/resource_preflight.json)。既存JAX／BLAS／vectorized CARを使い、frame pairは固定式により8 worker、BLAS4 threadの構成。恣意的なCPU/RAM/worker/wall capなし。新しい性能最適化を理由にSOURCEや科学条件を変更していない。

[GNU time](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/resource_time_run01.txt)はwall8.90秒、user26.65秒／system8.05秒、最大RSS582,444 KiB、swap0。[1秒間隔のprocess tree監視](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/resource_usage_run01.json)は約9.016秒、sampled aggregate RSS最大524,084 KiB、thread合計最大98。計測範囲・sample時刻が異なるためpeakを同一視せず、thread数を同時使用CPU数とみなさない。資源不足・timeoutによる停止ではない。

## 保全と次の担当

原本SOURCE25ファイル、契約、input hash、旧科学／技術artifactと元workspaceをbyte／Git blobで照合し、共有venv／cache／他ユーザーのjob・fileを変更していない。[保全記録](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/preservation.json)と[evidence inventory](../../artifacts/research_c_c1_stage1_compat_science/2026-10-25/evidence_inventory.json)を参照する。今回の科学結果／grant／ledger／marker／STOPも上書き・再利用しない。更新する現行索引・概要・manifestには今回の停止状態を追記する。

- **ユーザー／統括（次の判断）**：frame0同値性FAILの原因調査と人工再現・修正をCodexへ依頼するか判断する。今回のfit枠は消費済み。同じ分子export／行列を使った追加数値解析が必要なら、対象・call範囲を別途明示承認する。
- **Codex（別依頼後の作業案）**：raw CAR／一般K／係数展開／回転・spin・bit conventionを人工入力で切り分け、新しいSOURCEと人工検証を固定する。今回のraw・STOPは保全する。現時点でその修正／再解析は未実施。
- **ユーザー／統括（その後）**：新たなStage 1実行範囲を承認し、全gate資格と別測定したCDF／XDF誤差が揃ってからStage 2適格性を判断する。Stage 2回路build／compileとStage 3 PR／sampling／総資源評価は未承認。

今回の作業は部分証拠のcommit／push、remote SHA／SOURCE／成果物照合まで。**mandatory STOPを維持し、追加科学実行へ進まない。**
