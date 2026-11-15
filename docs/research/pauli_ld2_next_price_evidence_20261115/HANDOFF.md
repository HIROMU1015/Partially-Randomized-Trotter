# C2 → ユーザー・後続担当：LD2次の有限価格証拠

**現状では有限path変動とround移送のずれの大小を識別できない。**
**第一案はm9、fresh 2 path×X/Y、最大4 compile（B案）。**
**次に得るのは長q領域の2軸の有限cohort記述差と個別pathの違いであり、全law E[C]・CI・モデル資格ではない。**

[短いREPORT](REPORT.md)／[有限実行案](FINITE_EXECUTION_PLAN.md)。
使用branchは`codex/pauli-ld2-price-freeze-20261115`。Full-H規則案branch・D2 PR-DF branchは作業先に使わない。

m9はq512/R526/T51.2、特徴量[1,512,269312]。単一step較正S526とRが一致し、
未検証の長い反復・compiler境界の効果を調べる。m6 fresh2 path追加（A）も最大4 compileだが、
同roundの再現性を優先したい場合の代替であり、m7以後の空白は埋まらない。
選択は全13 roundに向けた証拠coverageを優先した計画判断で、総Gの支配項や統計的最適性を計算したものではない。
精度/信頼水準は未採択、m9のwall/RSSも不明。性能適合が未解決なら投入保留とする。

## 保存成果の受領と保全

5dacの較正10 record/18 compile、b74eの軸別モデルと4予測、f448の規則、
705fのholdout6 receipt/4 exact比較を固定参照として使用。6個別receiptのlocator・axis・
index・cohort・fingerprint・RZを静的照合し一致した。ADMIT/COMPAREやfitを再実行していない。
f448の文書に残る「未採択」は作成時点の履歴で、現指示と705fによりm5/m6記述規則の採択を確認した。
モデル判定の数値閾値は依然未採択。

m5 +0.044612%、m6 +0.157666%は元有限cohortの記述差として保持。
新cohortを足して旧比較を書き換えず、CAL/B0/W/holdoutの用途・元SOURCE・STOPを維持する。
本案のm9予測値は未計算・未凍結。旧係数からの2予測を後続の許可範囲で観測前に別公開する必要がある。
固定caller13abは元7-cell/n_pathsとM5/M6を固定しており、m9やm6 n2をそのまま実行できない。
限定driverの実経路・cohort/seed・予測locator・SOURCEと人工確認は有限実行案§2に具体化した。
新規候補はBの最大4 compileのみを第一案とし、Aとの合算取得は提案しない。

## ロードマップ差分

- 現在地：LD2はCAL18→2モデル/4予測凍結→独立holdout6/4記述比較の公開まで完了。
- 次の提案：従来の「次の価格証拠未定」を「m9新cohort2 path/XY最大4、観測前2予測固定案」に具体化。
- 残る不足：同roundの母平均精度、未比較round、長qの一般化、全law/rare/sampler距離、CI・正式E[C]/N/G。
  m9比較後も全13 roundの資格が揃うとはしない。希少イベント調査は保留。
- 次の条件：ユーザーの案採択と、予測計算・新取得・比較を含む明示許可、限定実装SOURCE・seed表・
  重複/資源適合の確認、観測前凍結。計画公開・commit/pushは科学実行許可ではない。

ロードマップ本体は統合担当へ委ね、本作業では編集しない。
今回は保存文書・結果の読取/locator照合とmetadataの計画整理のみ。新sampling/compile/再fit、
予測・誤差の再計算、CI・正式E[C]・N/Gは0。LD1/Full-Hへの移送0。
計画をcommit/pushしremote照合後に**mandatory STOP**。
