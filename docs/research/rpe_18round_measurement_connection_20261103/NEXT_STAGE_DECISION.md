# 次段階判断

2026-11-03 JST / 技術準備完了、科学実行なし。

1. **確認範囲**：有限RTEの分布・期待値定義、controlled/Hadamardの符号とscalar、
   価格標本のpairing、既存decoderをSOURCEで追跡した。薄いadapterを追加し、
   人工fixtureと設定転記の15検査を通過した。既存3 decoder関数は変更なしで再利用する。
2. **残る条件**：保存physical μが実際の入力・trajectory分布・回路のmarginalであること、
   shot/axis独立性、外部alias window、入力状態の利用可能性。単位PF半径は未保証、
   Full7点不一致、Full δ=.02 control未発見、原Full診断fingerprint欠測は保持する。
3. **優先課題**：まず候補B、同一physical入力・同一trajectoryの理論平均と実回路平均の照合。
   分布平均が正しくても回路の符号・scalar・state埋込みが違えばV2の実装への適用が成立しない。
   この条件を少数requestで統計誤差なく切り分ける。
4. **最小新科学範囲（提案）**：δ=.02、保存m0/q1とm3/q8。
   Partial r1/K2のorder0/order2 conditional profile各2、Full r0/K0各1の計6request、
   12axis wrapperのideal平均。保存Hからphysical ground stateの再構成1回を含める。
   sampling/compile/費用計算0。全18roundの測定成功率を検証する実験ではない。
5. **準備状況**：旧SOURCE commit、H snapshot identity、固定設定、component selector、
   新call範囲、数値基準とSTOPを[計画](MINIMAL_SCIENTIFIC_VALIDATION_PLAN.md)に具体化した。
   新adapterは公開branchで固定する。H4 backend/科学runnerは未実装であり、実行readyとは表示しない。
   snapshot writerにstate vectorはない。保存byte同一性と同Hからのphysical再構成を区別する。
6. **担当・必要な許可**：ユーザーがBのstate solve・回路build・ideal signal評価を明示許可する。
   Codexは許可後、限定backendを実装・固定し、その範囲を実行して停止する。
   重要な独立詳細科学レビューには別許可が必要。今回は承認の証明用artifactを作らない。

V2両候補の`CONDITIONAL_SUCCESS`、R1のPartial約4.858%点選好と費用順位
`UNDETERMINED`、R2/R3の順位決定目的で非推奨はそのままである。
新しい成功率・費用順位・厳密半径・実準備correctnessを主張しない。

公開・remote照合後は**mandatory STOP**。ユーザーの新しい明示許可まで
新科学計算、sampling、compile、再解析、詳細レビューを開始しない。

成果物：[接続報告](IMPLEMENTATION_CONNECTION_REPORT.md)、
[前提と証拠の表](ASSUMPTION_AND_EVIDENCE_MATRIX.csv)、[計画](MINIMAL_SCIENTIFIC_VALIDATION_PLAN.md)、
[adapter](measurement_adapter.py)、[人工tests](test_measurement_adapter.py)、[検査記録](VALIDATION.json)。
