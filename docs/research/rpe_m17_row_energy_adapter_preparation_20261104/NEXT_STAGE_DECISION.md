# 人工adapter完了、H4科学実行は準備対象外

2026-11-04 JST。**行別丸め・signed energy・全prefix・guardを接続し、人工46 testsをPASSした。
H4の全support証明へ進むための不変領域はまだ得られていないため、科学runnerは準備しない。**

## 改善できたこと

現行kernelを変更せず、宣言された有限人工domainの全数検査でsigned residualを評価できた。
energyの増減、F/R間の非可換event、全prefix、最終observableの相関を失わずに合成し、
中間energy4でもsignal上端1を示す人工例が得られた。
行ごとのforest項・underflowは保持し、未知の相殺や√回数の累積を仮定していない。

## まだ解けていないこと

1. **H4のdomain**：有限binary64空間そのものは巨大で、全stateの列挙は実用的な標準案にならない。
   今回のadapterは候補domainを自動生成・探索せず、宣言domainの閉性を厳密に検査するだけ。
   実H4に小さい閉集合や有利な不変領域があるかはUNKNOWN。
2. **未知stateのradial**：行別の対称誤差上界だけでは、signed radialを非正にできない。
   人工の全数検査で得た符号を、H4の未知状態へ外挿できない。
3. **Runtimeの決定性**：有限tableの再利用には同入力のgraph/partitionの再現性が必要。
   今回の仮定は実runtimeの検査ではない。H4に適用するなら実observerで確認するか、
   許されるgraph変動を全て含むset-valuedな証明が必要。
4. **平均への接続**：guard用energyが小さくても、旧作用誤差の数値上界を捨てられない。
   既存平均bridge・strict予算・IIDモデルの保証範囲は独立に残る。

新しいH4 context load、norm/残差/energy/radial/guard/予算評価を実行して、
この不足を確認する段階には進めていない。
保存norm²はSOURCE/identity確認済みの既存区間を再利用する方針で、exact norm²診断も未実行のまま保全する。

## 次の最小案と担当

次段案は**Codex**による、列挙しない不変領域への一般化の技術検討。
次の依頼後、まずH=Iの小さい人工dense/non-Hermitian系で、
行別rounding-cellの包含または区間／二次形式の不変領域を作れるか確認する。
未知領域へ負のradialを仮定せず、符号を証明できなければ対称forest上界へ戻して限界を明記する。
今回のclosed-domain adapterは、構成した包絡を小さい全数参照へ照合するために再利用できる。
同じkernelの演算順を維持し、Hを追加する場合は必要性とobservableへの戻しの損失を示す。

人工段階で有利な全support証明が構成できた場合だけ、既存保存contextを使う最小科学SOURCEを準備する。
入力/hash/SOURCE、保存norm²/logical/law再利用、精度/pass/call範囲、runtime条件、
全prefix有限性・complex abs・座標guard・strict判定、出力・UNKNOWN/STOPを固定して公開する。
**ユーザー**はそのSOURCEの確認後、新しい実H4評価を別途明示承認する。
今回の人工PASSや次の技術準備依頼を科学実行許可へ読み替えない。
重要な独立科学レビューも別の許可範囲で、今回未実施。

将来重い評価が必要な場合、独立matrix/rowの証明を並列化し、共通immutable contextとbasis cacheを共有する。
BLAS/FLINT workerを協調させ、実資源・quota・共有負荷に基づいて構成を選ぶ。
CPU/RAM/worker/wallの恣意的な上限を設けず、共有環境や他jobを変更しない。
今回のrelation powerは全trajectoryを展開せず、domain sizeとlog q/log rに依存する。
q回の非自明Fraction積を作らないため、長列で巨大な分子分母を生成する方法とも異なる。

## 状態とSTOP

`scientific_execution_ready=false`。H4 guardは未保証。
`U_virtual_IID_upper=null`、`U_exec_upper=null`、全体`UNDETERMINED`を維持する。
旧ideal categorical包含の局所十分条件、旧numeric INCONSISTENT/u_num=null、SOURCE/raw/grant/ledger/marker/STOPを保全する。
branchは`codex/rpe-m17-row-energy-adapter-preparation-20261104`。
commitとremote SHA/tree/公開file照合は最終チャットで報告する。
**公開照合後にmandatory STOP。次段や科学実行を自動開始しない。**
