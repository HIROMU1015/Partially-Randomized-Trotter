# 研究C：Proposed固定trajectory・F3-K1技術準備

実施日：**2026-10-28 JST**。依頼の推奨資料名にある20261027を識別子として保持した。起点 `8af9c2748bf2fd9b9b8ce691f7f1df0bee7a364a`、独立branch `codex/research-c-F3-K1-preparation-20261028`。

**Proposed `proposed_00`専用の接続・人工検証・独立7-call未承認契約を準備した。新規実H4科学callは0。** Proposed J/U・QPYの新decode、実κ再解析、実CAR reference／bridge／certificate生成、F1/F2再実行、sampling、C1再解析、CA/shots/総資源、詳細科学レビューは行っていない。

## 1．接続と最小差分

新[adapter](../../src/trottertracks/c_pr_aware_df_fixed_f3_k1.py)、[専用CLI](../../scripts/run_research_c_pr_aware_df_fixed_F3_K1.py)、[人工test](../../tests/management/test_research_c_pr_aware_df_fixed_f3_k1.py)の**3ファイルだけを追加**した。既存F2-K1／CAR／Arb／Gaussian／fitter／event実装の変更0。F2 runnerのBaseline固定scope・path・roleをProposedへ書き換えていない。

| 再利用処理 | F3固有の差分 |
|---|---|
| F2-K1 `POLICY/LIMITS/load_arrays` | 数学条件と7-call定義、固定H4/rank8 schema・finite・J exact symmetryを共用 |
| F2-K1 `execute` | F3のreference loaderとProposed F1のreuse bindingを渡してそのまま使用。継承されたjournalの`baseline_00` labelだけdispatch時に`proposed_00`へ置換 |
| K1 `exact_kappa/ball_coefficients/input_gate/kappa_bridge/build_reference/compare/compose` | **byte不変**。Proposed固有J/U・events・raw hashを渡す |
| `fixed_car.polar_bridge`、`SectorCAR.one/raw_frame`、`car_bilinears` | 同じ数学的定義・CAR代数・Arb192。新しい代数やmatrix exponentialは実装しない |
| `stage_b.prepare_arm/atom_operators`、`_ordered_exponentials/event_unitary` | `role=proposed`。元κ・frame・signed support・head/event順序を保持 |
| 既存`trajectory_gate/restore_events/reuse_gate/Calls` | Proposed seed、F1 pair、共通κ診断identityの薄いgateを追加。承認証明システムは追加しない |

F3の`run`は独立scope/fresh output・SOURCE/契約保存・failure/partial/STOPを持つ。旧F1/F2 runnerや使用済みcall枠は起動しない。

## 2．固定された対象とidentity

対象はB2 run02の**Proposed `proposed_00`一trajectory**。保存直鎖H4・隣接1.00 Å・STO-3G、通常number CDF rank8、Nα=Nβ2/sector36を含むfull8-mode/256、最後のancilla/full512。元frame順0..7、LD2、partial symmetric S2、δ=.1/s4/h=1/40、各half-head=1/80、r16/K0・保存16 events、own-DF ground oracle／状態準備除外。新ground/PF/RTE/DF評価は含めない。

共通P0（Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation1／qubits_initially_zero=false）、B2共通Gaussian/relative-head policy・declared phase・interleaved spin/Fock順序を保持。Arb192／python-flint0.8.0／FLINT3.3.1、位相込みfull_operator_2、global phase fitなし、allowance exact1/100000000。

| 固定入力 | SHA256 |
|---|---|
| 共通Stage A `prepare/input.npz` | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| Proposed `final.npz` | `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227` |
| Proposed保存trajectory JSON | `7329febc4c26cdb4eae2294d3917152329ec1c9683d9bca29d12801562c8b93a` |
| Proposed raw QPY | `1c19b44c07ccf0f79b7e5f177b2488d82070f6a27f5eee4a259b188078add95c` |
| Proposed F1 compiled/raw certificate | `417dc8545988ade7d43f6ff43a7cd5f6ebc3107c11cb2e1876b475d3eb8357e6` |

Proposed seedは **12704117278500360798**、Baseline seedは2939838941121001441。両armは異なるJ/U・law・seed・streamであり、同じindex0でもpaired trajectoryと扱わない。Saved SOURCE digest／cohort／export／16-event key順序に加え、Proposed seedをmetadata gateで照合する。Hash一致は技術的な同一性確認であり、PCG64 IIDの新保証ではない。

## 3．Proposed F1 certificateと共通shellの再利用

[F1科学結果](research_c_pr_aware_df_fixed_proposed_f1_execution_20261027.md)は保存Proposed raw/compiled、actual adjointsについて7/7 calls・3/3 PASS。F1科学SOURCEは `e7d22a404442234662ef6d117c55dd9868217c9e`、manifest SHA256 `0b10324fd59835dc371a65290dbd52e8909d9d85fbb44120421907f107f77701`。保存compiled/raw上界 **3.2092694515e-12**は既存報告からの引用で、今回再計算していない。

F3 triangleのcompiled/raw入力は **`F1-run01/compiled_raw.json`のみ**。F1のSOURCE・契約・STOP・resultとraw/compiled両QPYのhash、arm=proposed、index0、full512/全512列、operator2、phase_fit=false、Arb192、validated status・登録allowanceを照合する。Baseline B1 `kernel_compiled_raw.json`の誤流用、raw path/hash取り違えは明示的に拒否する。

共通head/X/Yの6QPY・6certificateはF1と同じpath/hash/scope（last-ancilla-only・既存terminal measurement規則）のreuse定義に一致させる。これらの再decode・再certificateはしない。Proposed F1を再実行しない。

Baseline F2-K1のraw/K1≤2.67040e-14、compiled/K1≤3.49618e-12はBaselineだけの既存結果であり、**Proposedのreference／bridge／certificateには一切代入しない**。

## 4．共通κ診断の適合性とbridge

共通h/g packetのpath/hashと既存`c1_stage1.corrected_kappa`を含むファイルSOURCEが、κ診断時から不変であることをpassive照合した。Python3.11.1／NumPy1.26.4、同じfloat64・縮約`prrq->pq`と数学的H/A定義、同じ192-bit K1 policyを維持する。

診断はbinary64とexact有理数で取得済みで、保存recordを192-bit数値計算の結果と混同しない。κの新数値診断は不要。次の科学実行では通常のκ取得に対し、K1の既存gateがliteral/H/A exact entriesを保存診断recordと照合する。今回その実数値比較を新たに行ってはいない。

同じtarget ID **`CAR_exact_kappa_H_polar_U_fixed_trajectory_v1`** をProposedにも適用する：V0=CAR(元κ,Proposed U)、V1=CAR(exact κH,Proposed U)、V2=CAR(exact κH,Proposed right-polar Q)。J、scalar、angle、signed support、event/head順、tail identity phaseは変更しない。

[F2-K1の証明](research_c_pr_aware_df_F2_K1_target_readiness_20261027.md)とコードをそのまま使う。κAのexact entry実虚L1上界a、full-Fock μ=8a、b=exp(|h|μ/2)、ProposedのU-local product上界からL=1+εUを得て、両half-headのtelescopingを **εκ≤L(b²−1)** で囲う。aが共通でもLはProposed固有なので、Baselineのεκ値も再利用しない。

U bridgeはProposed固有のJ/U/eventsからvalidated eta≥||U†U−I||2、eta<1を確認して求める。実対称J、finite U、signed K0 support・実angle・frame対応を科学callで検査する。失敗時に対称化・再分解・別targetへfallbackしない。eta≥1はこの十分条件で未資格という意味で、polar不存在の証明とは扱わない。

誤差計上もF2と同一：**εraw/V2=εraw/B0＋εκ＋εU**、**εcompiled/V2=εF1_compiled/raw＋εraw/V2**。B0のacb区間誤差、raw columnのrecentered radius、各bridge、F1 certificateをそれぞれ一度だけ含める。非unitary B0の距離に2のcapを適用しない。最終unitary endpointだけの距離capと区別する。

## 5．科学SOURCE・7-call契約・STOP

[新契約](research_c_pr_aware_df_F3_K1_contract_draft_20261027.json)はscope `C_PR_AWARE_DF_FIXED_F3_K1_V1`、`status=UNAPPROVED`／`executable=true`。数学SOURCEは旧142ファイルをbyte不変で継承し、新3ファイルを加えた**SOURCE145**へ固定した。今回の実科学実行許可はない。

- [SOURCE manifest](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-28/F3-K1-preparation/SOURCE.json) SHA256：`4711be5f75cb3e6965e87d51d443cf181f5fc5533859c7314790b02421ed95ce`。
- 契約SHA256：`7c9d7abb3445de1941f68e05444545fa553b2887e6a48a25017dc83c3229a23a`。
- Passive確認entry：`.venv/bin/python -B scripts/run_research_c_pr_aware_df_fixed_F3_K1.py --contract docs/research/research_c_pr_aware_df_F3_K1_contract_draft_20261027.json`。今回は`--execute`を使用していない。

| Call | 上限／処理 |
|---|---|
| input_arrays | 1：共通h/g/cとProposed J/Uのみ。F2のshape/dtype/finite・exact real J gateを再利用 |
| arm_replay | 1：role=proposed、元κで既存metadataを再生。回路buildなし |
| trajectory_load | 1：Proposed seed/source/export・保存16 signed K0 eventsを元順で復元 |
| CAR_reference | 1：共通診断identity・exact H/A・Proposed κ/U bridge・25 spin-count blocksのliteral B0指数化を含む |
| qpy_decode | 1：保存proposed_00 rawのみ |
| raw_CAR_certificate | 1：full512全512入力列／全出力・phase-sensitive validated operator2・allowance判定 |
| triangle_composition | 1：同一rawのProposed F1 certificateを一度だけ加える |
| **合計** | **独立最大7 logical units** |

Fresh output：`artifacts/research_c_pr_aware_df_fixed_car/2026-10-28/F3-K1-run01`、現在未作成。Retry/resume0、mandatory STOP。F1/F2の消費済みcall枠・既存runは使わない。科学実行の許可はチャットで足り、承認証明・原文保存・approval packet・新grantは要求しない。

SOURCE/input/role/seed/F1資格/κ診断identity、J・U・polar等の必要条件不成立は依存処理をSTOP。Certificateはupper≤allowanceでPASS、lower>allowanceでFAIL、区間が跨ぐ場合UNRESOLVED。Bridge未資格のnullを0へ置換せず、FAIL/UNRESOLVEDではtriangleへ進まない。完了したpartial・raw certificate・failure・Calls・resource・STOPを保存し、後続candidateや新namespaceでSTOPを回避しない。

## 6．人工検証・資源・保全

新テストはrole=proposed、Baselineとの取り違え、J/U hash、SOURCE/seed/law/order、F1 pairのqualified scope、同一raw、κ診断再利用条件、独立CAR・2 bridge、7-call E2E、PASS/FAIL/幅UNRESOLVED/null、partial/STOP/no-resumeを確認する。旧K1 suiteがexact H/A・非unitary増大・両head・25 blocks/full512・相対位相等を回帰する。全て人工数値入力で、保存H4はbyte/JSON metadataしか読まない。

最終結果は **新41＋回帰152＝193 distinct tests PASS、29 subtests PASS、skip0**。共通suiteは180 PASS／1 deselected、隔離general-K suiteは13 PASS＋29 subtests。最終logは`tests.log`／`general_k_tests.log`で、技術証拠の`tests.json`にhashとcoverageを記録する。

旧F2-K1の実行前専用test `test_passive_source_input_identity_no_decode` は、起点でF2-K1-run01が既に存在するため不適用。旧test/source/outputを変更せず、その1件をdeselectし、F2再起動がfresh-output gateで拒否される代替試験を追加した。旧一般Kは既存import隔離条件に従い別fresh processで実行する。既知C2並列STOP非決定性は未修正であり、修正済みとは扱わない。

初回人工試験で、誤ったcertificate pathを開く前の意味論照合が不足していることを検出し、F3側のreuse bindingを先に検査する順序へ修正した。CAR/Arb/bridgeの数学SOURCEは変更していない。この人工debugを実H4の失敗・retryと扱わない。

[技術証拠](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-28/F3-K1-preparation/)にSOURCE、最終test logs・初回人工debug log、passive preflight、reuse audit、preservation、資源plan、technical STOPを保存する。独立worktree・task venv/cacheを使用し、共有venv/cache/OS/scheduler/cgroup/他jobを変更しない。既存25-block／512-column process並列・process-local Arb/cache/recenterを将来そのまま使用する。CPU/RAM/worker/wallの人工capはない。

旧F2-K1 raw12・旧F2 raw6・F1 raw・κ診断raw、B2/C1等の保存結果は変更しない。新F3 bridge／raw-CAR／compiled-CAR値はすべて**未取得/null**である。

## 7．得られる保証・未資格事項・次担当

人工実装上のblockerはない。ただし実Proposed J/Uの必要条件と、certificateが1e-8内に収まるかは科学実行前には不明である。これらは固定call内のgateとして登録し、技術PASSで科学PASSを先取りしない。

将来F3がPASSすると、新たに得られるのは**保存Proposed proposed_00の固定1 trajectoryについて、raw／compiledからexact κH＋Proposed polar Qの理想CARへのvalidated位相込み上界**である。Baseline/Proposed両方で個別証拠が揃っても、全trajectory／全law synthesis、IID、元物理Hamiltonian完全保証、CA、保証shots、cost support、母平均費用・総資源優位は認定しない。固定index0の誤差値の大小を統計的な精度優位と扱わず、C1約83%は元の明示仮定付き点推定のまま。

次は**ユーザー／研究C**が新SOURCEと7-call契約の実行可否を別途判断する。**Codex**は明示的な科学実行許可後だけfresh F3-K1を一度実行し、証拠公開後にSTOPする。今回はcommit/push・remote SHA/SOURCE/input/成果物照合後、**mandatory STOP**。F3科学実行、sampling、F2/C1再解析、CA/shots/総資源、別分子、詳細科学レビューへ自動進行しない。
