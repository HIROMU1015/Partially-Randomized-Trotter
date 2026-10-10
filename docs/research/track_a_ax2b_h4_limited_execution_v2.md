# Track A AX-2B：H4限定v3一回実行とGPT独立科学レビューへの引渡し

2026-10-10 JST。利用者の「次のGPTレビューが必要だと思われるところまで作業を進めて」に従い、[coverage修正・準備v3](track_a_ax2b_h4_coverage_preparation_v3.md)の新固定・別認可・同じH4 scope/capsの一回実行を行った。
親statusは `H4_LIMITED_STOP`、親reasonは `PHASE_WALL_CAP:correctness`、worker reasonは `None`。
correctness 6/8 cell、MP 12/16 record、explicit event 0/4 groupを保存した。
worker terminalは欠測のためworker reasonと最終call counterは取得できない。親による時間上限停止を、分子検証の成功や数値不一致の判定へ読み替えない。
これは実行statusと保存証拠の記述。科学的なGO/STOP、総u/精度適格性、shot・総費用・方式winner、H6認可は判定しない。

## 対象と固定来歴

linear H4 1.00 Å/STO-3G/legacy DF rank12・generation-prefix L_D=0/6/12、8 system qubit、Nα=Nβ=2・sector36、T=0.8。
保存binary64 H_DFと数学的に正規化した指定保存stateがtarget。旧8 cell、q1/q4（δ0.8/0.2）、B2/B3 R8/r2/K2/4/6、B1二次/四次PFを維持した。
energy/chemical accuracy campaign、候補探索・最適化、新Hamiltonian/state生成は行わない。

| 来歴 | immutable refとrepository path |
|---|---|
| 科学source | `b228f2307f5fea77f066ee13e11ba6d2d8b9bea7`、173件closure。[backend](../../src/trottertracks/resource_applicability/ax2b_molecular_ports_v3.py)・[runner](../../scripts/resource_applicability/run_track_a_ax2b_bound_v3.py) |
| seal保存commit | `d2b1511cd87ee51e2a08d0f7c5025e55e920745d`。[manifest](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_seal_v2/2026-10-10/sealed_preparation_manifest_v2.json)・[195件freeze](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_seal_v2/2026-10-10/preparation_source_freeze_v2.json)・[seal audit](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_seal_v2/2026-10-10/seal_audit_v2.json) |
| 別認可・実行base | `a699a74dbf7a10a4e43c32ed6b6d3cb9d402a708`。[別認可](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/authorization_v2.json)・[exact command](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/launch_command_v2.json)・[preflight](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/prelaunch_audit_v2.json) |
| 保存入力 | [旧snapshot](../../artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz)、SHA-256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。旧bytes不変 |
| 原terminal | [親](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/terminal_status.json)。worker terminal保存=False |
| 原log | [worker log](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/worker.log)。環境cache警告を原文保存し、失敗記録を削除しない |
| 実行監査 | [execution audit](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/execution_audit_v2.json)・[saved audit](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/saved_execution_audit_v2.json)・[stdlib helper](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/audit_saved_execution_v2.py) |
| 原coverage | [actual bounds](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/actual_coverage.json)・[比較receipt](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/coverage_comparison_v3.json) |
| 数値一次記録 | [input/reference](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/input_reference.json)・[primitive](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/primitive_validation.json)、各cell/MP/eventは[inventory](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/execution_inventory_v2.json)のpath/hashから追跡 |
| 前回STOP | [前回報告](track_a_ax2b_h4_limited_execution_v1.md)、commit79858db。source61091c2・manifest c56ebc4・旧grant/outputのbytesを保全 |
| 科学的判断の前提 | [GPT独立科学レビュー](track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md)・[H4限定契約](track_a_ax2b_h4_prelaunch_contract_v3.md)・[H6準備契約v2](track_a_ax2b_h6_pilot_preparation_contract_v2.md)。H6実行認可ではない |

source・manifest・別認可をそれぞれ結果前にcommit/pushし、GitHubからfresh bareへ取得してbytesを照合した。
sourceとmanifest/plan/input/capsはrun中に変更しない。manifest science_authorized/launch_allowed=falseは準備の履歴で、別v3 grantが今回一回を認可した。
前回grantは消費済みで再利用せず、専用launch_v2を用いた。今回grantも消費済み、retry/resumeなし。
準備時の[49 local metadata/mock tests](../../artifacts/resource_applicability/track_a_ax2b_h4_coverage_preparation_v3/2026-10-10/tests_v1.json)は工程証拠であり、分子の科学的PASSやimmutable CIではない。計算停止後はstdlib保存監査・資料整理・Git照合だけを行い、test・signal・sampling・build/compileを追加実行しない。

## 保存された事実と欠測

coverageはtuple/list正規化後のstrict canonical bytesが一致した。actual/expected SHA-256は `eacf9dec340e9b356090527597cfc1a277775350f050d5b50e2bb26e3c5e9609`。
前回はactual bounds全体が欠測だったが、今回は全boundsと比較receiptを保存し、値・型・順序・time/index・instruction/probe上界を照合した。
保存inputからload/native準備を行い、36-column sector referenceと独立occupation構成を照合した。原recordの列差指標は 1.832288481395828e-15。
登録179時間組×3 probeのprimitive記録のmax_errorは 1.7636654707726433e-15。いずれも実行窓内の技術的・経験的一致であり、厳密u上界やground-state認定ではない。

| 登録cell | correctness保存 | MP80 / MP120 | 完了cellの古典wall秒 |
|---|---|---|---|
| H4_B1_S2_q1 | [完了record](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B1_S2_q1_correctness.json) | [MP80](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B1_S2_q1_mp80.json) / [MP120](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B1_S2_q1_mp120.json) | 79.0 |
| H4_B1_S2_q4 | [完了record](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B1_S2_q4_correctness.json) | [MP80](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B1_S2_q4_mp80.json) / [MP120](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B1_S2_q4_mp120.json) | 286.8 |
| H4_B0_q4 | [完了record](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B0_q4_correctness.json) | [MP80](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B0_q4_mp80.json) / [MP120](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B0_q4_mp120.json) | 169.9 |
| H4_B2_K2 | [完了record](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B2_K2_correctness.json) | [MP80](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B2_K2_mp80.json) / [MP120](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B2_K2_mp120.json) | 551.1 |
| H4_B2_K4 | [完了record](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B2_K4_correctness.json) | [MP80](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B2_K4_mp80.json) / [MP120](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B2_K4_mp120.json) | 558.6 |
| H4_B3_K6 | [完了record](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B3_K6_correctness.json) | [MP80](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B3_K6_mp80.json) / [MP120](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v2/H4_B3_K6_mp120.json) | 77.9 |
| H4_B1_S4_q1 | 未完了・欠測 | 欠測 / 欠測 | 欠測 |
| H4_B1_S4_q4 | 未完了・欠測 | 欠測 / 欠測 | 欠測 |

完了recordにはnative stage trace、MPとの差、MP80/120差、raw/corrected signal、log_B、action counts、classical cell wall、cell終了時RSSを保存した。
未完了cellの計算結果を完了扱いしない。原MP traceとdecimal文字列を保持し、再fit・再最適化・精度閾値変更は行わない。
explicit代表eventは保存された範囲だけの主張で、全event平均の列挙・trajectory sampling・shot独立性の科学的証明ではない。
原terminalにない未完了call数は推定しない。[saved audit](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/saved_execution_audit_v2.json)のobserved_callsは None。
欠測は同auditのmissing_records、partial/invalid JSONはpartial_or_invalid_json_recordsに列挙する。
このrunではB1の四次PF q1/q4のcorrectness/MPが欠測、B2 K2とB3 K6のexplicit order0/2も欠測で、wrapper_cost phase記録はない。旧v5の四次PF・wrapper結果は別scopeの旧証拠として保持し、今回の独立MP/stage/control検証を完了扱いする代用にはしない。

## 古典評価資源と量子資源の分離

CPU3/worker1/BLAS1、phase900/1800/300秒・total3000秒、AS8GiB、output512MiB/log64KiBを固定。
原親wall 1802.0276073073037秒、raw 30 file / 17272882 bytes。
完了cell時点のrusage peak RSSから得られるworker peakの下限は 398544896 bytes。未完了区間を含む独立lifetime peakの測定ではない。
ASはaddress-space上限で、RSS上限・メモリ予約とは別。MP時間やJSON量は資源評価のための古典費用であり、量子回路費用に加算しない。
trajectory/occurrence/compileは契約上0。代表event/control検証の回路buildは登録scope内で許可したもので、compiled resource/shot/total-cost campaignではない。

## Gitと既存証拠の保全

[保全監査](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/preservation_audit_v2.json)で、2686件の既存ファイルを今回の索引追記だけ除いて照合した。旧science source/freezes/STOP、snapshot、既存dirty8件・未追跡14件を保持し、対象だけ明示stageする。
再開前にroot worktreeへ別のTrack B review文書が未追跡で追加されていた。[再開時保全記録](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution_v2/2026-10-10/resumption_preservation_v1.json)を分け、root HEADと既存5 reviewのbytes、再開時11 reviewのbytes・Git状態を保護する。root/Track Bの編集・stage・commitは行わない。

## GPTが判断する論点と停止境界

1. 保存された範囲でH4-N/A/E/Mのどの義務が満たされたか。欠測・未完了項目と、H6 technical前に必要な検証を明確にする。
2. 同じ保存target/stateに対するnative・独立occupation・MP80/120・stage照合の独立性と正規化が、主張の範囲に合っているか。
3. 経験的一致とu認定を分け、未認定uで許されるtechnical scope、追加数値検証の優先順位・古典予算を判断する。
4. explicit代表event/control記録と、未実施のwhole-trajectory平均・shot法則・期待cost標本を混同せず、必要な後続を定める。
5. 完了/STOP reasonを踏まえ、H4追加検証・計算方法/予算見直し・H6案の扱いを決める。H4実行完了や合成tests通過だけでH6を自動認可しない。

この文書はCodexの証拠引渡しとレビュー論点で、GPT独立科学レビュー結果や承認済みH6契約ではない。
現在はmandatory STOP、N/G=null、accuracy UNDETERMINED、numerical_allowance_certified=false。
`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION` / `next_stage_authorized=false`を維持し、科学修正・追加run・H6/H8/GPUを開始しない。
