# H6 Pauli-JW P2：保存trajectoryの部品価格（2026-11-01）

## 固定範囲

ユーザーが明示許可したP2のみ。設計commit `6f0759df7c1df7fd6d835922b0246753a8a98d6c` の
[検証設計](pauli_pure_rte_mechanism_validation_design_20261101.md) §5および
[実行準備](pauli_pure_rte_mechanism_execution_readiness_20261101.md) P2節を適用する。
P1結果 `bfb4ca75437a9b3605a8f57b5f82100615dd034f` をbaseとする独立branch/worktree。
旧SOURCE・結果・STOPおよびP1実装は変更しない。新しい承認証明・grant機構は作らない。

Linear H6、隣接1Å、STO-3G、neutral singlet、保存Pauli-JW 918非identity項＋constant、12 system＋ancilla、
T=.8、S2、q8/K2、P=0、central-rotation制御、cosine/+Im sineの測定付きfull Hadamard wrapper。
Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing追加指定なし。
対象は以下のconfirmation index0000のみ。価格によるindex選択をしない。

| ID | candidate | r | L_D |
|---|---|---:|---:|
| H32 | PR_p0078_q8_r32_K2 | 32 | 78 |
| H64 | PR_p0078_q8_r64_K2 | 64 | 78 |
| R64 | H6_p000_q8_r64_K2 | 64 | 0 |

## SOURCE・保存原本

Hybrid結果 `00958ae1945c780718b2deef07d5dab1096a5cd8`、SOURCE `887b69e449cb3c378286f09d0d7626599850fb67`。
純RTE結果 `10cb3e5cca7834ebf75301bd91bd093ca51d4a6b`、SOURCE `e755ecf581600192fd2745d7066e410cebd933d3`。
各commitのdata_archives.json、part/archive/member SHA、保存resultとQPYのraw/gzip SHAを照合する。
必要な18 memberだけを復元し、Fullは再保存しない。参照とhashを結果へ保存する。
科学SOURCEは実行前にcommitで固定し、結果source_and_inputs.jsonがその実commit SHAを記録する。
保存SOURCEのcode map、同一wrapper/compiler関数、saved individual metricsとcompiled QPYの対応を確認する。
model.jsonは固定hashのdescriptor確認にだけ使用し、state/reference/lawを新規評価しない。

## 最小拡張と意味論

- D1へcompact/full Pauli event共通viewと一program用integer interaction式を追加。未保存の確率・control_sidecar等を補わない。
- D2のcentral_control_bodyからtoken emitterを切り出し、同じ制御展開を実回路とdescriptor recorderで共有。
- 元append_pauli_evolution/append_pauli_event、両軸hadamard_wrapper、default_compiler、transpile_and_measure_costを再利用。
- archiveの元gzip復元関数を既存restore scriptから再利用。
- [kernel](../../src/trotterlib/pauli_mechanism_cost.py)、[runner](../../scripts/run_h6_pauli_p2_cost.py)、
  [人工tests](../../tests/test_pauli_mechanism_cost.py)。D1のgeneric controlはP2価格へ使用しない。

Fはsaved input/compiled QPYと個別価格を再利用し、新規QuantumCircuitへ再build・再compileしない。
復元したsource tokenをdescriptor recorderへ渡し、saved F inputの命令名・順序・wire・angle・phase・測定境界を照合する。
通常角度は1e-12以下、phaseは2π同値で1e-12以下。serialization名と科学的な入力意味を区別する。
Rは元forward/reverse halfの順序・時間とscalar phase、Eは保存event順序・符号・角度・負product/order phase、
Emptyはwrapper＋測定だけ。body global phaseはancilla P(phi)へ一度移す。
R64のRはscalar-onlyであり、Emptyとは区別する。

H32/H64のRはsource token/phaseの完全一致確認後、一度だけbuildする。
新規wrapperはR4、E6、Empty2の最大12program。同一semantic program hashはcompileを共有する。
Full6軸の意味論・価格照合が完了するまで、新規parts build/compileへ進まない。
利用可能affinity CPUと実際の12仕事に応じて独立processで並列compileし、全childをreapする。
各programは一度だけsubmit。科学compile失敗後の自動retryはしない。

## 人工検証と実行

人工固定2-qubit recordでcompact/full、product順序・signed rotation・order/scalar phase、
control後の相対phase、F descriptor一致、両wrapperとQPY roundtrip、R64型scalarとEmptyの相違、
入力共有、誤angle/wire/phase拒否、D1 interaction、既存compiler workerを検証する。
2026-11-01実装確認：`14 passed`、人工3-qubit wrapper compile 1件（H6科学compileとは別）。既存D2の関連人工回帰テストも7件PASS。
同テスト内でsampler/angle生成/generic control/compile禁止区間も検証。
実保存原本の事前技術照合：F全6軸の命令・parameter差0、保存metrics一致。新規科学compileはまだ0。
これは独立reviewではなく作成者による技術確認であり、科学的性能の証拠にはしない。

専用既存runtimeを読取利用し、環境やshared設定を変更しない。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-pauli-p1-bias-execution-20261101/.venv/bin/python -m pytest -q tests/test_pauli_mechanism_cost.py
PYTHONDONTWRITEBYTECODE=1 /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-pauli-p1-bias-execution-20261101/.venv/bin/python scripts/run_h6_pauli_p2_cost.py --execute
```

新規結果保存先はUTC execution dateによる
`artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/cost/`。
既存directoryがあれば再実行を拒否する。状態・SOURCE/input・意味論・phase・QPY・個別metrics・interaction・STOPを保存する。

## 会計と停止

RZ/CXについてΔinteraction=F−R−E+Empty、contextual retained=F−E、contextual event=F−R。
depthは各構成の記述指標のみ。全cohort平均や元Nとの積、新N/G/SE/rankingを計算しない。
固定programの削除操作とcompilerに条件付けた非加法性であり、独立した物理・compiler pass因果配賦ではない。
H32/R64はr不一致のfamily比較、H64/R64はq/r/K一致だがprefix/tail/law/eventが異なる。
代表trajectory、rare-event tail、母平均、一般的方式優位、全law実ρを保証しない。

原本欠落、identity/意味論/位相/compiler/価格境界不一致、12超過、旧証拠改変の必要、共有環境干渉ならSTOP。
通常の実装debug・人工test修正は範囲内。実科学失敗の自動retry・別候補/index/compilerへの変更は禁止。
完了後REPORT、GitHub result commit、remote SHAと全文を照合しmandatory STOP。
P1N/P3/追加精度/全cohort/母平均/論文/詳細科学reviewへ自動移行しない。
P1のSTOP_UNRESOLVED_COMPONENTSを保持する。次の判断担当はユーザー/GPT。
