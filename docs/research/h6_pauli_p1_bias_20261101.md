# H6 Pauli-JW P1：限定bias分離の実装と実行条件

2026-11-01 JST。利用者はP1の最小実装・人工検査・指定4候補の新信号・差分評価・公開を明示許可した。P1N/P2/P3、H2/H4、新N/G/SE/順位、sampling、回路build/compile、物理入力・状態・reference生成、詳細科学レビュー、主要claim採択は含まない。

正本は固定commit [6f0759dfの検証設計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6f0759df7c1df7fd6d835922b0246753a8a98d6c/docs/research/pauli_pure_rte_mechanism_validation_design_20261101.md)と[実行準備表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6f0759df7c1df7fd6d835922b0246753a8a98d6c/docs/research/pauli_pure_rte_mechanism_execution_readiness_20261101.md)。本書は承認の証明書ではなく科学条件の記録である。

## 固定対象と再利用

linear H6/隣接1Å/STO-3G/中性singlet、保存Pauli918非identity項＋constant、12qubit、T=.8、S2、q8/K2、主精度ε_sig=.0005。

|ID|保存候補|分割・r|
|---|---|---|
|H16|PR_p0078_q8_r16_K2|prefix78/r16|
|H32|PR_p0078_q8_r32_K2|prefix78/r32|
|H64|PR_p0078_q8_r64_K2|prefix78/r64|
|R64|H6_p000_q8_r64_K2|prefix0/r64|

Hybridは00958ae1945c780718b2deef07d5dab1096a5cd8、純RTEは10cb3e5cca7834ebf75301bd91bd093ca51d4a6bの結果を読む。model SHA256は6c66a2eeb4d99cf5bbd9137bbcb067275536be3201d5cc047225266dc5f58681、referenceは9956466a5cf3286af9960b792b455bb8bacc2532a34ac7a28b470475f92fcb7a。

保存Aのphysical_referenceとexact ν、Dgridのphysical_corrected_meanを読む。Aへν、DgridへB/νを再適用しない。保存B/bias/u/N/C/Gの新評価は行わない。archiveのspec/law/targetは選択memberだけをbyte/hash照合し、QPYは開かない。旧SOURCE/input/raw/grant/ledger/marker/STOPは読取専用。

## 数値処理とscope

- 新信号は共通B/C各1とDpoly4の6種類。B/Cは保持和／元の項別half sweep＋厳密tail作用を比較する。Dpolyは既存paired polynomial recurrenceを共通関数へ切り出し、normalization=1で補正平均を得る。旧raw meanは元のfinite normを渡す既定経路を維持する。
- B/Cそれぞれにtail exact actionを二つのhalf actionへ分ける予定済み整合検査を1回ずつ実施。合計8伝播まで。これは同じsolverのsemigroup自己整合であり独立した精度保証ではない。
- H6は全4096次元のcomplex128。NumPy1.26.4/SciPy1.14.1/Qiskit1.3.0、既存sparse_operatorとexpm_multiplyを使用。保存係数から部分作用素を組み立てるが、SCF/FCI/JW/DF・新しいfull-H referenceは生成しない。
- B=Cの一致目安は1e-10。各新信号・軸の診断分解能も1e-10に事前設定する。これは厳密な浮動小数点誤差上界ではない。保存包含のradiusと足し合わせた差の分解能以下はUNRESOLVEDとする。
- L0のB=C=AはH_D=0の構造的恒等式として確認し、保存Aをaliasとして使う。新full-H reference禁止を守り、実H6の独立したB/C伝播は追加しない。L0の数値oracle照合は人工系で行う。
- 差はouter=B−A、hold=C−B、trunc=Dpoly−C、grid=Dgrid−Dpoly。実部・+Im虚部・絶対値・分解能・閉合を保存する。絶対値の和による因果寄与率、新N/Gは作らない。grid差は実compiled残差ρではない。
- 保持78項はI/Zのみ。B/C不一致、保存条件不一致、非有限値、閉合不良はSTOP。分解不能ならUNRESOLVEDでSTOPし、高精度・別solver・新候補・再実行へ自動移行しない。

使用資源は実際の空きCPU/RAMを観測し、独立なDpoly4本を並列化する。worker数4は独立task数に由来し、恒久的資源上限ではない。共有環境・他利用者・既存科学runを変更しない。

## 検査・入口・保存先

[新kernel](../../src/trotterlib/pauli_mechanism_bias.py)、[既存recurrenceの最小拡張](../../src/trotterlib/pauli_signal_adapter.py)、[固定runner](../../scripts/run_h6_pauli_p1_bias.py)、[人工tests](../../tests/test_pauli_mechanism_bias.py)。人工系12件が通過した。科学実行前にcode/testsをcommitし、そのSOURCE SHAとinput hashを結果へ記録する。承認grantや承認台帳は新設しない。

```bash
.venv/bin/python -m pytest -q tests/test_pauli_mechanism_bias.py
.venv/bin/python scripts/run_h6_pauli_p1_bias.py inspect
.venv/bin/python scripts/run_h6_pauli_p1_bias.py execute
```

executeは新しい出力directoryだけを作成し、既存出力の再実行を拒否する。通常の調査にexecuteは使わない。

結果・SOURCE・数値検査・終了状況は artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/bias/ に保存する。実行後の正本は同directoryのREPORT.mdとJSONであり、本条件記録を実験後に書き換えない。

**P1結果の公開・remote本文/SHA照合後mandatory STOP。次の担当：ユーザー/GPTが必要なら次段階を判断する。P1N/P2/P3は開始しない。**
