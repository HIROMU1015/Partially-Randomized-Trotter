# Track A H6精度一致・測定込み資源比較 v2：一次結果・実行監査

2026-10-11。固定source `54986a9f54515c8c2289e38fa16fd81e2b6008c2` と公開v2契約に従い、ユーザーが認可した科学実行を一回だけ行った。原statusは **`H6_MATCHED_RESOURCE_STOP`**、reasonは `RuntimeError:MemoryError:`。retry/resumeなし。grantは消費済み、mandatory STOP。H8・候補追加・他の科学計算へ進まない。

## source・grant・入力・公開来歴

- [固定契約](track_a_h6_matched_accuracy_execution_contract_v2.md)・[sealed manifest](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v2/2026-10-11/sealed_manifest_v2.json)。manifest SHA-256 `99cd2055283d2bb35ec5b373d934b3107fac6997a013e5f523914df3216014b9`、digest `7693137a582dc80d2f221dc2ed8a51161ebd79f6f183f6a4bb9dbe4e0f20b19e`。公開準備 `8f9bf1484f409e434b7db156fbf5f65d9f2050f0`。科学source216 pathは実行後も固定bytesと一致した。
- [authorization-only grant](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/authorization_v2.json) 公開commit `2157da1d208a04102166fa1ae3c154490546c90f`、SHA-256 `a006c3b933db79d372a09366aeb0e7ae508842e96bd3d786e7d6fb9729fd2e39`。[起動前remote照合](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/remote_prelaunch_verification_v2.json)で253 pathを再取得照合後だけ起動した。source・入力・環境・資源条件を変更しなかった。
- [runner](../../scripts/resource_applicability/run_track_a_h6_matched_v2.py)・[supervisor/worker](../../src/trottertracks/resource_applicability/h6_matched_execution_v2.py)・[静的auditor](../../scripts/resource_applicability/audit_track_a_h6_matched_v2.py) はscience source commitに固定されている。
- 保存入力source `f37005f01b2be38c5993d6e82df91abe9c643d21`、結果 `554fc52add39c2c1b45b765a3135df76fda6f15a`、[入力監査](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_execution_v2/2026-10-10/execution_evidence_inventory_v2.json) `f52787a22542b31bd39fd004a8d3d71325bc56b0`。親33 pathと[snapshot](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/h6_input_snapshot.npz) bytesを保持。新SCF・DF分解・state生成・solverは行っていない。

## 一次artifact・保存範囲・欠測

- [原出力ディレクトリ](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/)：raw 5220 file、77980307 bytes。derived runtime cacheを除く全原記録を公開対象とする。
- [原terminal](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/execution_terminal.json)、[原inventory](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/execution_inventory.json)、[launch binding](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/launch_binding.json)、[coordinator limit receipt](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/coordinator_limits.json)。未保存の場合は下記欠測一覧で明示し、代用値や新terminalを作らない。
- [signal出力](../../artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/)：92候補中92件のsignal record、B3 normalization-only 168組中84組が保存されている。個別bias/u/normalization/shot情報は原recordを参照し、未保存値を推測しない。
- 保存wrapper record 902件、TASK_COMPLETEのcost row 902件。探索登録group 592／完了449、確認登録group None／完了0。[欠測・coverage](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/missing_coverage_v2.json)で候補・未完了task・未作成recordを区別する。最大1708 wrapperは上限であり、未登録の残task数を意味しない。途中でcompileしたがrecordが未保存の回数は推測しない。
- [外部観測](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/execution_observation_v2.json)、[静的audit](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/static_identity_audit_v2.json)、[bytes inventory](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/execution_evidence_inventory_v2.json)、[保全audit](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/preservation_v2.json)。原出力・過去のfreeze・監査を後から書き換えない。

## 条件と監査の解釈

linear H6／等間隔1.00 Å／STO-3G／12 modes／α3β3／sector400。tol-only1e-8・final_rankなし・cutoff0、signed generation-order DF rank19、採用済みweighted Hermitian projectionと同じ保存state。T=0.8、B0/B2 prefix5/10/15、B1 full19 S2/S4、q1/2/4/8、B2 r1/2/4・R=qr・K2/4、delta0.8/0.4/0.2/0.1、ε_sig=0.05/0.01/0.005/0.001の固定92候補を維持した。B3は168 normalization scalarだけでsignal/sampling/compileを追加しない。

signal先行screen、適格候補のexploration n8、精度別B2上位2のunion最大8を独立seed confirmation n32、paired X/Yと登録ordinary anchor、最大1708 measured wrapperとcompile guard1712は固定契約のまま。旧pilot標本をfresh平均にpoolしない。quantum shotは解析的Hoeffding会計で、measurement shot自体のsamplingは行わない。

coordinator AS soft2 GiB／hard8 GiB、signal・cost AS soft/hard8 GiB。signal CPU [0,2,5,6]・Numba/OMP4／BLAS1、costは最大2 disposable process・各thread1。RSS18 GiB／output2 GiB／log1 MiB／call・instruction guardを維持し、phase/total wall capはnull。元inventoryのRSS値は失敗batchのpeakを含まない場合があり、特に初期値0を実測peakと解釈しない。

静的audit exit codeは0。STOP時のPASSは保存bytesとidentity/limitの整合性だけを意味し、全signalや費用coverageの合格ではない。COMPLETE時は登録task・seed・paired wrapper coverageも検査する。環境事後一致はTrue。いずれもempirical numerical allowanceを厳密な誤差保証とせず、sample mean/SEをpopulation費用やformal winnerの保証としない。ground-state、化学的energy精度、全候補の資源優位、H8 GOを認定しない。

旧[v1 STOP](track_a_h6_matched_accuracy_result_v1.md)／source／消費済みgrant、旧[pilot 7 cell・32/36 wrapper](track_a_h6_technical_pilot_result_v2.md)、保存入力と既存dirty/untrackedを保全。**mandatory STOP。`H6_NOT_AUTHORIZED`／`DRAFT_NOT_AUTHORIZATION`／`next_stage_authorized=false`。** 今回の一回grantを再使用しない。科学的判断と追加runの要否はGPT・利用者へ戻す。

公開結果commitとGitHub再取得hash照合は[remote結果照合](../../artifacts/resource_applicability/track_a_h6_matched_execution_v2/2026-10-11/remote_result_verification_v2.json)で特定する。この文書はCodexによる事実と監査scopeの整理で、GPT独立科学レビューや承認済み次段階契約ではない。
