# Rを変数とするPF/PR総費用pipeline（技術完了）

2026-11-11 JST。**各RPE roundに有限のPR R_outer候補を置き、同一条件の供給済みN(T,R)/C(T,R)を照合し、候補費用・条件付きR選択・不足・不確かさを返すpipelineを実装した。** 人工/mockでのみ積和・R選択を実行した。実H4/H6の新RのN/C計算、保存数値再解析、価格予測/fit/sampling/build/compile、実データR最適化・実N×C/Gは0。Rの科学的最適値・採択は未決定。

成果：[module](../../../src/trottertracks/rpe_r_selection.py)／[A/C入出力仕様](INTERFACE.md)／[有限実行案](EXECUTION_PLAN.json)／[人工テスト](../../../tests/check_rpe_r_selection.py)。

## 既存成果と追加範囲

- 元B統合3b742fff2、H4 provider統合cb2b1d6b4、後続H4全13round Nの89c91c692を確認。既存B branchは後続N作業を含むため保全し、89c91c692から独立`codex/r-variable-rpe-cost-pipeline-20261111`を作った。Cの進行中`rpe-h6-provider-coverage-20261110`は読取調査のみで、変更・停止・merge待ちをしていない。
- 既存canonical IDはRを含み、task自体は同roundの複数候補を保持できた。旧`evaluate`はround/方式1候補に固定されていたため、選択層だけを新設。`pf_pr_rpe_pipeline.py`は変更せず、task/identity/measurement/plan/price policyとH6 signed allocation、元会計・比較器・未知joint uncertaintyの表現を再利用した。
- 元`aggregate_supplied_axes`はN×floatとfsumを使い、巨大Nの下位桁を保持できなかった。`exact=True`の小さい拡張を追加し、新入口は常に整数/有理数で積和する。旧APIの既定数値処理・科学結果は変更しない。
- H4凍結Mqと[H6凍結Affine](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae44283d169b61deace7e1c0d18c4538d4a3fc0c/docs/research/h6_df19_affine_price_provider_20261110/REPORT.md)はそれぞれR4/R16のdomain。別RのCに代入しない。providerや係数を変更せず、A/Cが新しく資格したfull candidate/domain/lawを受け取る。A/Cの完成を前提にせず人工recordで共通部分を完成した。

## 出力と科学的制約

`build_menu`は元taskのT/qPF/δ/prefix/K/formula、h/M/α/β/prior/decoder、H/state/snapshot、compiler/測定付きfull wrapper/P=0を保持し、Rとそれに依存するID/allocationだけを変更する。変更対象δはH4=.1/H6=1.6。H6の早期別δ・p5/R32等は固定のまま残す。固定PFをFull PF等へ読み替えない。

`coverage`はmetadataだけを検査する。`evaluate_menu`は供給済み正整数Nと単価をexactに積和し、各roundでRを選択する明示数値入口。実H4/H6で呼ぶことは別の科学実行範囲である。Nはfloatへ変換せず、binary64単価はその正確な有理数として扱う。JSONは分子/分母の整数で返し、巨大Nの1単位差や費用差を消さない。N側normalization・古典標本数を二重乗算しない。

価格direct/model/未検証外挿/null、PR path/cohort/law期待値、modelのRを含むfull domainを区別する。nullを取得済みとせず、候補を暗黙の固定R価格で埋めない。law/identityの矛盾は拒否し、未宣言lawや未知価格は不足を残す。PR有限cohortは明示policyの条件付きproxyで、全law期待費用へ昇格しない。

点最小は供給値・固定α等に条件付けた有限menuの選択である。N不成立候補は除外するが、欠測や未資格価格のある候補を無視して「全menu最適」とは表示しない。roundに候補が残らなければ全費用はnull。全round NのPASSだけで全RPE成功にしない。prior/alias/horizon/decoder/位相・半径/失敗配分/implementation-lawの上流資格と選択scheduleのbindingを別に受け取り、科学的採択は行わない。

共有fit/paired XY/cross-R依存と保存誤差証拠を受け渡す。未知のjoint covariance・systematic・選択後CIはnullのまま。全menuを覆う供給済み同時価格区間があるときだけ、exactな候補費用区間と厳格な区間優越を比較する。αtotal=.05、標本SE、観測maxから価格保証を生成しない。cross-R行列推定/伝播・選択後CIは今回未実装の科学・統計項目として明記する。

## テストと保全

**39 test method PASS**（B専用Python3.11.0rc1/NumPy1.26.4/Qiskit1.3.0環境）。主担当自身の技術検査で、独立科学レビューではない。

| 範囲 | 件数 | 記録 |
|---|---:|---|
| 新R menu・N/C照合・exact積和・R選択 | 12 | [TESTS.txt](TESTS.txt) |
| 元supplied-axis会計 | 15 | [ACCOUNTING_REGRESSION.txt](ACCOUNTING_REGRESSION.txt) |
| 元B統合pipeline・mock provider | 12 | [PIPELINE_REGRESSION.txt](PIPELINE_REGRESSION.txt) |

Rだけの変更、H6の11 signed piece配分、小R拒否、各roundで異なる人工の最小R、全recipe/law/SOURCE不一致、固定R modelの流用拒否、N不成立・欠測/価格資格不足、paired scenario、finite cohort/path/外挿、成功資格の独立性、未知共有共分散・供給済み同時区間を検査した。巨大Nは`2**1200+3`を含み、1単位差と非整数binary64単価の正確な積和をJSON round-tripで確認。人工値の性能をH4/H6の科学結果として解釈しない。

旧Bの静的保存fixture検査はそのまま再利用したが、実値の新積和・Rランキングには使っていない。SOURCE/input/raw/結果/grant/ledger/marker/KEEP/過去STOPは保全。他担当worktree・process・環境と共有設定は変更していない。

[CHECKS.txt](CHECKS.txt)に、固定model/N kernel/旧結果の保全、試験対象SOURCE、資料link、有限案のmetadata計数を記録した。

## 残る科学データと次の担当

[EXECUTION_PLAN.json](EXECUTION_PLAN.json)は未承認の有限案。H4 R4基準・H6既存固定scheduleのNは保存成果を先に照合し、同じ点を再計算しない。仮の比較menuはH4全13round R4/8/16/32、H6 m4..12 R16/32/64（m0..3は固定）。既存基準Nが再利用できる場合、新RのNはH4 39＋H6 18＝57 PR候補というmetadata上の範囲である。これはR候補の科学的採択や実行許可ではない。

- **Codex A**：別途許可された有限範囲のN(T,R)と精度・phase/radius・巨大整数証拠、law bindingを供給する。固定Rで取得済みの全round Nだけでは別RのNを埋めない。
- **価格担当A/C**：R依存Cの設計・有限calibration/独立validation・rare-order/law・共有不確かさを供給する。提案pilotはH4 R8/16/32×q8/16/32、H6 R32/64×q1/2/4の15条件。33 path/条件という仮有限案と精度の限界を明記し、実行前にSOURCE/モデル/必要精度を確定する。これで長q/全law資格を自動取得したことにはしない。
- **Codex C**：既に指示されたH6固定R provider統合/coverageを継続。本作業による変更依頼・停止はない。
- **ユーザー**：A/Cの証拠と有限案を確認し、次の科学範囲を明示判断する。**Codex B**は、その範囲が許可されデータ・成功/価格資格が揃った後だけ実N×C/G・実R比較を行う。

技術作業をcommit/pushし、remote SHA・主要成果物を照合して**mandatory STOP**。実N/C、実R最適化、適応的R追加、次の科学計算へ自動進行しない。
