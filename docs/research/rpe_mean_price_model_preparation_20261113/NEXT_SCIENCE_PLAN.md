# モデルを全RPE費用へ使う前の科学検証条件

**未承認・未実行の設計。今回、検証点・候補・cohort・compile件数を選定していない。** Claude §4.3/7.3 T3の加法性やextensive残差も仮説として残す。旧C2の40/16 wrapper、直接8 wrapper pilot、レビューS4/S5等を自動実行しない。

## 設計すべき比較軸

|軸|固定する条件・切り分け|得るべき証拠|
|Full PF/Truncatedのq|同H/representation・公式・PF幅・fragment順・retained H・scalar規則・profile/lowering、q増大に対応する総T|固定＋stepの式に対するfull-wrapper directの校正と、係数固定後の別q holdout。B0/B0I/Fullは別domain。最大使用qへの移送根拠を別途確認|
|PRのR方向|q/T/Δ/準備/prefix/K/公式を固定し、合法allocationのRだけを変える|event角/order確率も変わるので、同c_eの流用を既定にしない。同law paired S/CとP0–P4で不足する境界寄与を検査|
|qとRの同時増大|同preparation/formula、q-only・R-onlyで説明した寄与を凍結|qR交互作用、outer/長距離compiler効果の独立確認。短q/Rの補正率を最大domainへ固定しない|
|Full-H pure RTE|one-bodyを含む全非scalar、head0、同K・q/R grouping・scalar一回|PR tailだけの係数と区別。同じ総event数でもgrouping価格が同じと仮定せず検査|
|新lowering|同compiler hashでもlowering/profile・wrapperが変われば別domain|旧receipt/旧C1 modelを新loweringへ移送する証拠。同一条件のdirect又は正式な同値性証拠がなければmissing|
|S＋残差|同一trajectory/axis/full wrapperでSとCを対にし、target lawを明示|E_ideal[S]と実samplerの残差を合わせるlaw bridge、有限cohort標本誤差、rare・cross-axis/round共有誤差を保持|

各domainで校正と独立holdoutのrole、seed/cohort・実SOURCEを取得前に固定する。既存C1の校正/検証/rare roleやS0a development roleを新独立holdoutへ付け替えない。候補familyをBが供給した後、必要な係数の識別可能性と必要なholdout方向を決める。校正と検証を混ぜてfitし直さない。科学的に不一致なら未資格を維持し、閾値緩和や自動retryで解消しない。

## 最小化できる計算

1. 固定receiptをそのまま再利用できるセルをSOURCE・physical/profile/law・V2 traceで受け入れれば、同じcompileを繰り返す必要はない。これはDの既存73件照合をCで再監査する指示ではない。
2. 既存Sやkernelの再実行が不要な場合は供給済み値/区間を利用する。ただし新しい残差との結合・比率・予測の数値評価は科学実行scopeへ数える。保存値だから無許可としない。
3. 旧connected-clusterの式・層別設計は使えるが、DF12等のraw κ(C)を現Sへ加えない。新domainに必要ならpaired C−Sの同条件校正と独立full-wrapper検証を選ぶ。RTE内部だけの局所検証はPF/piece/outer/wrapperの証拠を代替しない。
4. 大q/Rの全event列生成を価格モデル計算の前提にしない。既存有限状態E[S]と線形featureで足りる部分は再利用し、重いjoint量が必要なら直接有限検証と費用対効果を比較する。実性能・誤差は未確認のまま明示する。

## 価格精度・採用に不足する条件

- Bが同energy taskを満たすN又はN集合、比較family、許容差/主張水準を供給すること。固定nominalや構造proxyからcompile予算・winnerを確定しない。
- 校正誤差・独立holdout残差・long-q/R移送を分けること。許容精度はB既存のΣN b等に条件付けて決める。古い5%幅、レビューの倍率/1/3案を一律の正式閾値へ採用しない。
- ideal finite-Kと実PCG64/CDF joint law、希少order2、多数位置への蓄積誤差が価格期待値に与える影響を資格化するか、未資格systematicとして主張を限定すること。
- 同じ校正に由来する候補/round/XYの共分散と、全比較familyの同時区間を扱うこと。未知を0や独立性で埋めない。有限cohort診断を全law保証へ昇格しない。
- scalar/preparation/sourceの実整合はDの独立scopeで確認すること。Cのmetadata整合PASSからD T2の実scalar gateを解除しない。

これらを満たし外部科学producerが適用domain/validation/errorの根拠を供給してから、ユーザーがモデル採用と有限科学実行を判断する。現在のCモデルquoteはfalse資格のままで、Bの正式GはUNKNOWN。新S/C/fit/保存再解析/全Gを本計画から開始しない。公開照合後mandatory STOP。
