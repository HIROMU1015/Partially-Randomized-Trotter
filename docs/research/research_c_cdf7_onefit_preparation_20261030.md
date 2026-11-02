# 通常CDF7：一括許可に基づく1-fit実装と科学SOURCE固定

2026-10-30 JST。起点`0876449d7ce75e25277865834efbc3bafaed7b3e`。ユーザーは技術準備から保存H4・通常CDF7の初期化1件／optimizer最大1回／DF精度評価／公開までを明示許可した。科学実行前に本SOURCE・[契約](research_c_cdf7_onefit_contract_20261030.json)を固定する。Shared-basisの[非実行案](research_c_shared_basis_df_contract_unapproved_20261029.json)はexecutable=false/call0のまま保全し、専用Shared-basis fitterは作らない。

## 再利用と差分

新`c_cdf7_onefit.py`と専用CLIは、既存`c1_molecule_extension.factor`の`max_vecs=7/tol=1e-8/cholesky=True/optimize=False`を**一度だけ**呼ぶ。得られた7 frameを修正real Schur codecへ渡し、qualified x0を既存`c_pr_aware_df.fit`／ffsim0.0.83／SciPy L-BFGS-Bへそのまま接続する。Fitter内のinitializer hookは保存済みの同じrank7 J/Uを返し、新たなCholesky分解を計算しない。旧SOURCE・runnerを編集して再起動しない。

初期DF予算はfitの前提にしない。初期before／codec後の一般K/raw CAR同値性・codecのsector忠実性・J/U構造を必要条件とし、初期sector誤差＋guardが予算外でも、correctnessが通れば1-fitへ進む。この政策を新しい独立adapterに明示し、旧Stage Aの予算gateを変更・無効化していない。

目的は既存Stage A Baselineと同じ`f/f*`、`f=.5||g-g_DF||F²`、`f*=max(f_initial,1e-14 Ha²)`を初期値で一回固定。γ=0でpenaltyは加算しない。Scale設定に既存`scales`を使うため初期smooth massもmetadataへ入るが、lossや判定には使わない。正のscaleは最小化集合を変えないが、SciPyの絶対gradient／目的停止scaleを変える。CDF8とは初期値もf*も同じではない。

L-BFGS-Bはoptions=None、callback=None、jac=True、追加SciPy tolなし。固定SciPy1.14.1の既定ftol=2.220446049250313e-9、gtol=1e-5等を変更しない。ffsim tol=1e-8は初期化側の値で、SciPyのtol指定ではない。生x/jac、success/status/message/fun/nit/nfev/njev、正規化／raw objectiveとgradient、J/U、独立tensor残差を保存する。

SectorCAR、rank_assessment、representation_gate、元g補正κ、scalar一回、interleaved spin、engineering guardは既存実装を再利用。物理Hは一回構築し、before／initial／finalで同じH・contextを使用する。最終optimizer successと`sector norm+guard≤.00019875 Ha`を独立判定する。新loss/gradient/sector構築器・Gaussian・compilerを作っていない。

## 固定条件と科学call

保存直鎖H4・1.00 Å・STO-3G、Nα=Nβ=2／sector36、number CDF、一般実対称J、実直交U、**actual rank7／frame0..6**、元順LD2／tail2..6。原P-H4 bytes SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`を新[input copy](../../artifacts/research_c_cdf7_onefit/2026-10-30/input/P-H4.npz)へ保存した。初期・最終のtensor／sector／representation、保存CDF8 JSONの読み取り比較だけを実行する。原integrals生成・新ground・PF/RTE・sampling・shots・native・総費用は0。

| 段階 | 論理単位 |
|---|---:|
| Prepare：元input1、rank7 initializer1、codec1、tensor2、scale1 | 6 |
| Fit：snapshot1、既存fitter1、開始／終了diagnostic2、最終tensor1 | 5 |
| Evaluate：snapshot2、context1、物理H1、sector3、representation3、初期correctness1、保存比較load1、最終判定1 | 13 |
| 最終適格時のみ：exact tail mass／basis key補助診断 | 1 |
| **最大／DF未達時** | **25／24** |

内部iteration・gradient評価を別fitへ換算せず、1 optimizer invocationの中に含める。初期／最終のoperator誤差はHa、tensor lossはHa²、異なる量を混同しない。General-K/raw CAR component allowance1e-10／rtol0、objective一致`1e-12+1e-8|expected|`、codecの既存guard、既存`sector_operation_scale_guard8_proposal_v1`を固定する。すべてbinary64 numerical pilotで厳密enclosureや最終CA保証ではない。

Fresh `artifacts/research_c_cdf7_onefit/2026-10-30/run01`、retry/resume0。初期actual rankが7でない、SOURCE/input/codec/representation/solverが不整合なら部分証拠を残してSTOP。Optimizer不成功でも契約内の最終診断は行い、追加iteration・再fit・初期値fallback・rank変更はしない。CDF7_DF_ELIGIBLEはoptimizer success＋correctness＋最終予算PASS、CDF7_DF_NOT_METはcorrectnessを保った収束または予算未達。STOP.failedはcorrectness実行失敗を表し、通常のNOT_METをPASSと読み違えない。

## 人工・回帰と環境

**新8＋既存37＝45 unique PASS、29 subtests PASS**。[ログ・SOURCE・環境・verification](../../artifacts/research_c_cdf7_onefit/2026-10-30/technical/)。

- 実ffsimの人工PSD tensorをrank7 Cholesky初期化し、codecと112変数x0、実loss／gradientを確認。Minimizeはmockでnit0。初期予算NOT_METという人工gateでも同じx0で一度だけ接続し、codecの再pack／新初期化なし。
- 人工end-to-endは既存SectorCAR／物理H／general-Kをmockせず通し、初期予算未達→fit→最終適格／予算未達／optimizer不成功を確認。Codec・representation不一致ではfit前にSTOPし、fresh再使用を拒否。最大25／未達24の計上も確認。
- 既存γ0/γ.1 loss・codec、一般Kの非回帰を再利用。一般Kのpassive-import試験は数値依存をimportしたCAR試験と同一processでは適合しなかったため、旧前提を変えず別processでPASSさせた。科学許容差は変更しない。

Task-local `.venv`／`.venv-fit`のPythonを明示し、既存依存をread-onlyで再利用。評価Qiskit1.3.0、fit ffsim0.0.83/JAX0.4.38/SciPy1.14.1、ffsim2 SOURCE hashを固定。共有venv・cache・他jobを変更せず、cacheは本worktree内とする。開始前の空き資源を確認し、初期2 snapshotは既存plannerの並列・BLAS協調、1-fitは逐次依存のため一つのprocessで行う。CPU/RAM/worker/wallの新たな恣意的制限なし。

SOURCE35件・saved input8件・旧保護2153件を照合した。人工PASSは実H4の精度PASSではない。科学SOURCEをcommit/pushしてremote照合後、今回のチャット許可の範囲で追加承認なしに一度実行する。結果が適格なら後続PRの技術設計・未承認契約案までを準備し、PR科学実行は開始しない。結果公開・remote照合後mandatory STOP。
