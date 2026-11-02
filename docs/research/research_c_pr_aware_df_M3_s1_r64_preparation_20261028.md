# 研究C M3：Proposed PR s1/r64の統合技術準備

2026-10-28 JST。起点`a977adda5a9fa5a882ef8b79acb8c68cff8f0bb1`、独立branch `codex/research-c-M3-s1-r64-20261028`。ユーザーはM3-A→条件付きB→C→公開まで明示許可済み。人工試験・SOURCE固定後に追加承認なしで一度実行する。承認証明・原文保存・新grantは作らない。

## 固定範囲と再利用

保存直鎖H4・隣接1Å・STO-3G、通常number CDF8 Proposedのみ、Nα=Nβ2/sector36、system256/last-ancilla controlled512。元g補正κ、核scalar一回、元frame0..7/prefix LD2、tail identityを維持する。Partial ordered symmetric S2、δ=.1/s1/h1/10/r64/K0、m0..12/n=2^m、64 signed events/kernel。CA=.00159/DF reserve CA/8、βtotalπ/6・βstatπ/12、failure.01/axis.01/26、own-DF ground oracle・状態準備除外。

追加SOURCEは[M3 adapter](../../src/trottertracks/c_pr_aware_df_m3.py)、[runner](../../scripts/run_research_c_pr_aware_df_M3.py)、[人工test](../../tests/management/test_research_c_pr_aware_df_m3.py)の3件。旧科学SOURCE変更0。

- B1の`prepare_arm/configure/kernels`、CAR/general-K/atomと`reference_round/classify_round/summarize`を再利用。保存B1の同一J/U sector factors/H・ground・DF norm+guardを照合し、groundの新eigensolve/DF norm/fitはしない。新しいPF/RTE/13 exact-time評価だけを行う。
- B2の`reference_context/trajectory_reference/raw_gate/compiled_gate/adjoint_pair_gate/XY_gate/calibration`を再利用。Samplerは既存`independent_occurrence`で、M3専用namespaceから32非paired substreamsを作る薄い接続だけを追加。旧cohortとpoolしない。
- StrongBuilder/PrecisionBuilder・head内relative Gaussian・tail隣接同一basis共有、P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/qubits_initially_zero=false/MPFR160 declared phaseを維持。Cross-head/tail一般relative fusionなし。
- B2の共通head/X/Y raw/compiled QPY6件はhashとancilla構造を照合して再利用。新shell build/compileなし。32新complete kernelsと64actual inverse circuitsを保存し、全512列・全出力を検証する。位相fit/有限probe代用なし。
- C1の`qualification/assess_round/energy/cost_calibration`、既存Hoeffding/`compose_serial_cost/aggregate_totals`を再利用。新候補の13round値からshotsを算定し、旧shotsを流用しない。変更はserial反復数s1と共有校正SEの重みだけ。

新しい分解器、PR sampler、PF engine、Gaussian compiler、RPE solver、総費用集計器は実装していない。旧runnerを起動せず、既存grantやcall残枠も使わない。

## 人工・回帰

[最終log](../../artifacts/research_c_pr_aware_df_M3/2026-10-28/technical/tests_final.log)：**92 PASS（新13＋既存79）**、79.53秒。最後にscenario一致のpassive検査を追加し、影響する5件を再実行してPASS（8 deselectedはこのfocused再検査だけ）。旧B1/B2/C1/M2テストを含む。実H4 decode/科学call0で準備した。

人工64-event kernelでfull512 raw/compiled/reference・相対位相・actual adjoint・X/Yを実検証。別のmock E2Eで32 dispatch・128 QPY・call全数・32独立seed・13round×3lane・shell一回・n=2^m・共有校正SEを確認した。A不適格/UNRESOLVEDでB/Cを呼ばず、各stage correctness FAILでpartial/journal/STOPを保存しfresh再実行を拒否する。

初回人工fixtureのexclusive-write契約更新とtemp Git identityを修正した。実分子trial runによるdebugは行っていない。

旧C1 manifestと現起点の4 SOURCEにhash差がある。`rte.event_unitary`、`SectorCAR.one/raw_frame`、`_ordered_exponentials`、`atom_operators`へ後段で追加済みのoptional `arithmetic=None` hookで、今回使うbinary64既定分岐は維持されている。旧C1原本を変更せず、現起点81 SOURCEを新manifestへ固定した。差のhashはtechnical verificationに記録する。保存C1結果への保証移送はしない。

## 契約・call・STOP

[固定契約](research_c_pr_aware_df_M3_s1_r64_contract_20261028.json)、scope `C_PR_AWARE_DF_M3_S1_R64_V1`。

| stage | 上限 | 内容と進行条件 |
|---|---:|---|
| A | 61 | input/保存証拠・9因子・ground/law/代数gate、kernel1、exact13/reference13/classification13、summary |
| B | 325 | Aの全13適格時のみ。fullCAR/9因子/216atom/head cache、Gaussian9/relative4、保存shell6load/3gate、32×sample/build/reference/raw/compile/compiled/adjoint/XY/metrics、校正/δ価格比較 |
| C | 84 | B32件correctness完了時のみ。target/価格、serial26、evidence+旧2scenarioのqualification3/accuracy39/energy3/aggregate3/shared-SE3、比較4、summary |
| program summary | 1 | A不適格時もpartialなprogram結果を保存 |
| **最大** | **471** | Aで不適格なら62以下、B/Cなし |

1 logical unit内の行列列・atom・event・逆回路等は契約`logical_unit_contents`を参照。校正は32標本固定、各64 events。Bのadjoint unitはraw/compiled各inverse build/decode/checkを内包し、128 QPYと32 reference NPZを保存する。内部matrix columnを追加候補と扱わない。

Fresh output：`artifacts/research_c_pr_aware_df_M3/2026-10-28/run01`。Retry/resume0。Correctness FAILは依存call停止・開始済み独立job回収・partial保存。Aの十分条件不適格は物理的不可能性と断定せずB/C未実行でSTOP。Bの価格が高くても32件が正常ならCへ進む。科学開始後の修正/条件変更/別runは禁止。

## 数値・統計資格と資源

Matrix tolerance1e-10、complex signal2e-10、rtol0、phase fitなし。全law synthesis/implemented-law/費用supportはnull。C1の`ideal_exact_native`と`.001 A`を各合成/lawに仮定する`assumed_raw_signal_budget`だけを維持する。Evidence laneの未知項を0にしない。Actual CA・厳密enclosure・保証shots・ideal IID・母平均費用優位は未資格。

同じ32標本平均を全roundへ使うため、総費用SEは`W=Σ 2*N_m*2^m`倍とする。sqrt13で縮めず、32をshots倍率にしない。旧s4/r16と新s1/r64のcohortは独立・非paired。δ価格は旧kernel平均4倍と新平均を比較し、head境界/compile scope差から単純4:1を仮定しない。

Task-local venv/cache/output、read-only既存依存を利用する。A13独立sector評価、B32独立matrix chainsに既存resource plannerを適用。小行列BLASをworker数に協調し、Qiskit build/compile/actual-adjoint-buildは既存共通mutexで保護する。CPU/RAM/worker/wallの人工capは設けない。実際のaffinity/quota/MemAvailable/pressureを確認し、Bでは8件完了ごとに自process CPU/RSSとmemory pressureを記録する。共有設定・他jobは変更しない。

科学実行後は[結果報告](research_c_pr_aware_df_M3_s1_r64_execution_20261028.md)へ全13round、32校正、条件付き費用とRQ1〜8を記録する。新s/r/K/γ、追加arm/fit/H6/sampling/保証/詳細レビューへ進まず、公開remote照合後mandatory STOP。
