# O1 M1：実回路・PF/finite-RTE信号検証の実行準備

2026-11-06 JST、SERVER 2（GPUA100-gleap02）。**実装・人工テスト完了、実H4未実行、mandatory STOP。** 基点は`74dc014627edfa3f7598747ded1b079ec0dbd6d9`。今回の指示は準備を許可し、実H4の回路生成・信号評価は別の実行許可を要求する。前段のoperator-only実行許可は拡張していない。

## 実装と既存コードの再利用

- [検証module](../../../src/trotterlib/o1_m1_circuit_signal_validation.py)と[明示CLI](../../../scripts/validate_o1_m1_circuit_signals.py)を追加した。import/`--help`はH4をロードしない。
- 固定`o1_m1_matched_split`は不変更。`extract_matched_df_tail`、既存Gaussian `BogoliubovTransform`、`build_one_body_gaussian_block`、diagonal primitives、`QiskitDFRTEEventCircuitBuilder`、`append_pauli_event`を再利用する。dense Gaussian fallbackでnative分解を検証したと主張しないよう、実行時にqiskit-natureを要求する。
- 独立CARとexterior-power参照は前段の[固定driver](../../../scripts/run_o1_m1_h4_operator_correctness.py)を再利用する。新しいDF fitや辞書探索はない。
- 共通の同一Gaussian T_eff半step回路をP/DF双方へ使う。tail identityは両辞書とも`faithful_identity_in_tail`に固定し、核scalarは一度だけ外側で作用させる。T_effの電子identityはGaussian一体block内に残す。
- `finite_rte_operator_moments`による解析的finite meanと、既存`paired_polynomial_action`による独立再帰を照合する。P/DFは同じdelta/q/r/Kだがlambdaとnormalizationが異なり得る。補正mean、raw複素信号、radius/phase、normalization、attenuationを別fieldに保存する。
- 実回路はfragment・I/Z/ZZ・符号・次数phaseを覆う**決定的な代表event**を検査する。RNG/samplingは呼ばない。H4全eventの総列挙や全回路meanの総和は行わない。全列挙との接続は下記の小人工系で検査する。

## 人工テストで検出・修正したphase

固定Qiskit1.3.0において、全Iの`PauliEvolutionGate`を直接行列化する経路でphase二重計上を観測した。人工eventのrotation angle `0.015922404298517844`に対し、実対角要素が`0.999493+0.03183943j`、独立参照が`0.99987324+0.01592173j`となった。これは物理H4の数値ではない。

新branchの[pauli_signal_adapter.py](../../../src/trotterlib/pauli_signal_adapter.py)の`append_pauli_evolution`に、全Iなら`global_phase -= coefficient*time`として直接追加する分岐だけを加えた。非identityは既存builderを使う。enclosing gateをcontrolled化した際の相対phaseも人工検査した。旧commit・旧結果・共有Qiskitを変更していない。

Gaussian basis自体にはvacuum/global phaseが付く。検査はその値を記録し、単一粒子作用とexterior-power Fock拡張の一致、保存Gの対角化、fragment operatorを別々に確認する。このbasis phaseはU†/controlled diagonal/Uで相殺する。**evolution/eventの比較をglobal-phase alignmentで合わせることはしない**。controlled operatorはancilla-lastの`diag(I,U)`へ直接比較する。

## テスト結果・環境

**14 passed in 5.16s**（既存6＋新規8）。実行したのは[旧人工テスト](../../../tests/test_o1_m1_matched_split.py)と[新人工テスト](../../../tests/test_o1_m1_circuit_signals.py)だけ。

- 複素G、正負square、元targetとDF targetが異なる2mode人工fixtureを再利用。
- 共通Gaussian保持回路、controlled relative phase、P/DF補正mean・normalization・attenuation。
- 核scalarと保持halfを含むcontrolled全step。意図的な保持phase不具合を入れたnegative controlでは実装差をFAILとし、finite-RTE誤差fieldが変わらないことを確認。
- 小人工DF辞書のorder0/2全eventを回路から厳密加重和し、既存finite momentsと照合。controlled全加重和も`diag(I,raw mean)`と一致。
- signed identity回転の単一global/controlled phase（正、負、微小係数）。
- 全比較の人工最大entry差はテスト内の`2e-13`未満。これは実H4用許容値ではない。
- テストは`np.load`とRTE sampling入口を禁止した。H4 snapshotの数値decode、実H4回路・信号評価は行っていない。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.venv/gaussian_deps \
XDG_CACHE_HOME=/tmp/o1-m1-signal-prep-cache-20261106 \
MPLCONFIGDIR=/tmp/o1-m1-signal-prep-cache-20261106/mpl \
/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-a-o1-m1-minimal-implementation-20261106/.venv/bin/python \
  -m pytest -q tests/test_o1_m1_circuit_signals.py tests/test_o1_m1_matched_split.py
```

既存専用runtimeをread-only利用：Python3.12.3、NumPy1.26.4、SciPy1.14.1、OpenFermion1.6.1、Qiskit1.3.0。新worktree内`.venv/gaussian_deps`だけにPyPI固定pure-Python wheelを展開した：qiskit-nature0.7.2（wheel SHA256 `9bc8900626b9a6f9c5dbe44589c53932408a661d61ad9003b8feadb30979cc05`）、qiskit-algorithms0.3.1（`d6c31d88a1ed65a9c6382c233ef10ecb9562e726f7b1b7823a0d178d63d29a0a`）。旧venvや共有packageは不変更。CPU/RAM/worker上限を追加していない。

## 別途実行許可を求める最小H4条件

以下は**実行準備であり未実行**。ユーザーがこの新SOURCEと条件で実行を許可してからCodexが実行する。

| 項目 | 固定する条件 |
|---|---|
| 保存input | commit `4591b8d75a2a6a55c74a51225f0c61648023c8ee`、`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| 系・split | linear H4、1Å、STO-3G、中性singlet、spin8、保存DF rank10。HD=T_eff、HR=V_square全10fragment、fragment LD0 |
| state/register | 元H保存`state`のみ、hash `8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`。DF_stateに変更せず、state再生成なし。full256、ancilla-last、途中sector射影なし |
| 最小診断grid | delta=0.1,0.05、q=2、r=2、K=0,2、計4点。T=q delta=0.2,0.1。最適化・探索・RPE scheduleではない |
| identity/保持 | faithful tail identity、同じGaussian T_eff半step program、核scalar一度 |
| circuit検査 | 全保存fragment/basis、共通T半stepとそのcontrolled版、決定的代表P/DF eventのnoncontrolled/controlled operator・相対phase、辞書/time別1代表の核phase込みcontrolled全step |
| mean/信号 | finite exact analytic mean、独立paired polynomial、元H stateの複素信号・radius/phase・B/attenuation |

算術判定式はSOURCE内に固定した。`eps=binary64 epsilon`、`m=4n^4+8d+16rank`、`S=|c|+sum|h_bare|+sum|W|+sum|h_eff|+sum_l |lambda_l|(sum|G_l|)^2`、`tol_op=gamma(m)*max(1,S)`。basisは`32 eps d max(1,max basis instruction count)`。implementationは`max(tol_op,tol_basis)*(1+q*r*(maxK+2))*(1+max delta)`。H4の実入力・program長から差を見る前に評価し、閾値・原精度を記録する。厳密rounding enclosure/energy allowanceではなく、人工atolの転用や結果後の閾値調整はしない。

PF誤差は`S2_exact vs exp(-it H_DF)`、finite-RTE誤差は`nominal corrected mean（独立CARのRとexact T半step）vs S2_exact`、実装差は`actual corrected mean vs nominal corrected mean`およびGaussian/eventの直接比較として別記する。保持回路の実装差をfinite-RTE誤差へ混ぜない。複素信号差、半径差、wrapped位相差も記録する。元HとDFの差は`exp(-it H_original) vs exp(-it H_DF)`として残し、元H targetをDFへ置き換えない。ground精度やenergy保証は認定しない。

実行許可後のcommand（この準備では実行しない）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.venv/gaussian_deps \
XDG_CACHE_HOME=/tmp/o1-m1-signal-run-cache-20261106 \
MPLCONFIGDIR=/tmp/o1-m1-signal-run-cache-20261106/mpl \
/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-a-o1-m1-minimal-implementation-20261106/.venv/bin/python \
scripts/validate_o1_m1_circuit_signals.py \
  --snapshot ../track-a-o1-m1-h4-operator-correctness-20261106/.venv/m1_input_snapshot.npz \
  --step-times 0.1 0.05 --q 2 --r 2 --K-values 0 2 \
  --output artifacts/track_a_o1_m1_circuit_signals/2026-11-06/run01/result.json
```

## 未解決・引渡し・STOP

実H4 Gaussian/controlled encodingとPF/finite meanは**未検証**。人工PASSはH4 correctnessや性能改善の証拠ではない。compile/transpileの費用評価、sampling実験、RPE shot・総費用、新DF/fit/rank探索は実装・実行対象外。

主agentが実装・検査した。AGENTSの委任禁止を維持し、sub-agentには委任していない。独立研究レビューとは呼ばない。既存の科学結果は変更せず、今回の直接関連報告だけを追加する。

**次の担当：ユーザー**が上記新SOURCE/条件の実H4実行を許可する。許可後は**Codex**がその限定検証を実行・公開し、mandatory STOPする。今回は実装・人工テスト・独立branch公開・remote内容照合までで**mandatory STOP**。
