# 保存行列の精度切り分け：第一科学実行の固定計画

2026-11-03 JST。**準備のみ。SOURCEと範囲に対する別の明示許可後に実行する。**
[adapter](precision_adapter.py)・[driver](run_precision_isolation.py)・tests・本書を含む公開commitを固定SOURCEとする。
full SHAと固定URLは最終チャットで示す。承認証明用のpacket/grant/署名等は要求しない。

## 1. 固定物理条件・入力

linear H4/隣接1.00 Å/STO-3G/保存DF12、B physical state、8 qubit・4電子sector70、
Partial L_D3、δ=.02、m17/q131072/r32/K2、元の位相・scalar・normalization規約。
N411/axis、alpha=.016957089147912167、epsilon=.15244325851620388と保存PF/RTE/stat予算を保持。
これらの予算を新しく再解析しない。r32/K2/attenuationは保存tailへ既に含まれる。

| 入力 | 固定commit/path | SHA256 |
|---|---|---|
| 元RESULT | cd749066904d3f29b28b03c19ab69370372b157f / artifacts/rpe_m17_mean_validation/2026-11-03/run01/RESULT.json | 79fdf7e84f90c48230857b13f39f9becb30f19892e2e9c09d80805d346432a4a |
| 元7行列 | 同commit / artifacts/rpe_m17_mean_validation/2026-11-03/run01/operators.npz | e1a65238e9d9b61a6d0aa542ca7f509827b0a761fc751615f1f2018a0e7020df |
| 保存psi | e22d7718c47358ff650dbc04e8e72f6ab1c8bf41 / artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/physical_state.npz | 793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4 |

元m17実行SOURCEは128e9a5bf2630cd0333ec5f69b81b25def78eb79、library carrierは4591b8d75a2a6a55c74a51225f0c61648023c8ee。
入力はbyte identityとheaderを確認する。配列はread-onlyとし、psiを再正規化しない。
Task AではF/R、law_tail_mean、reference_tailとpsiだけを数値decodeし、A_prob/A_coeff/one-event平均を再構成しない。

## 2. A0：元と同じfloat64基準

保存F/R/tailから左結合`S64=R64 @ tail64 @ F64`を両経路で各1回組み、
NumPyのq131072乗とvdotで平均を各1個、計2個求める。
同じ演算式を使うが、元S自体は保存されていないため、**新たに組んだS64**である。
両S64を新結果のfrozen_steps.npzへ保存し、以後のA-stepは同じ成分を厳密なdyadic入力として使う。

元RESULTの各平均との差が両方1e-10以下なら基準再現とする。
超過すればINSUFFICIENT_EVIDENCEを記録して停止し、高精度jobは開始しない。
新S64を元の保存行列として偽って扱わず、基準再組立てに由来する限界を残す。

## 3. A-step：S64固定の後段評価

S64とpsi64の実部・虚部を`float.as_integer_ratio()`で正確なdyadicに変換する。
str(float)を高精度数へparseして入力を置換しない。mpmathの独立contextで各jobを評価する。
q乗はbinary powering、最後にstate-actionと共役内積を行う。
後段の差にはq乗だけでなくstate-action/期待値の丸めも含むと明記する。

精度ladderは**96→160→必要時224 bit**。
初期値は、診断1e-10に対応する34 bit、qの17 bit、次元70の7 bit、guard32 bitから
90 bitを見積もり、32 bit単位で96 bitへ切り上げた設計値。
これはconditioningを含む誤差上界ではない。64 bitずつ増やし、実際の収束を確認する。

まず両pathを96/160 bitで各1回、計4高精度平均を独立jobとして評価する。
両pathとも連続2精度のcomplex平均差が1e-13以下なら収束診断を満たす。
どちらかが満たさなければ両pathを224 bitで各1回追加し、160/224を比較する。
224 bitでも収束しなければINSUFFICIENT_EVIDENCEでSTOP。qを変えず、無制限の精度追加を行わない。
1e-13は診断結果の3桁下まで安定性を見る技術的規則であり、誤差上界・RPE許容差ではない。

## 4. A-factor：必要時だけpost-archive組立てを分離

A-stepが収束し、law/referenceの差が元1e-10より明確に大きい場合だけ実施する。
境界判定のため、差が1e-10から1e-13以内ならNEAR_THRESHOLDとしてSTOPし、自動で次へ進まない。

保存F64/R64/tail64/psi64を正確に変換し、**S=R tail Fの組立てから**高精度で評価する。
A-stepと同じ96/160、必要時224 bitの両path評価・収束規則を使う。
元tailのr乗やorder/rotation/normalizationの再計算はしない。
これはTask Aのpost-archive組立て検証であり、Task Bの平均tail再構成ではない。

## 5. 原因候補の事前判定規則

元1e-10の閾値と、新しい1e-13の収束/境界bandを混同しない。
bandは原因候補の判別を保留するためのもので、旧INCONSISTENTを緩める規則ではない。

| 観測パターン | 候補label・識別できる範囲 |
|---|---|
| 基準再現なし、収束なし、閾値近傍 | INSUFFICIENT_EVIDENCE |
| A-stepが収束し、高精度gapが1e-10を明確に下回る | POWER_ROUNDOFF_PLAUSIBLE。S確定後の後段丸めが診断超過を説明し得る。A-factor/Bは行わない |
| A-stepのgapが残り、A-factorで収束して閾値未満になる | CONSTRUCTION_ROUNDOFF_PLAUSIBLE。post-archiveのS組立てが候補。平均tail構成以前の原因は認定しない |
| A-factorでも収束したgapが閾値を超える | CONSTRUCTION_ROUNDOFF_PLAUSIBLEとIMPLEMENTATION_OR_DEFINITION_DIFFERENCE_POSSIBLE。保存factor間の差が残るが、入力生成時の丸めと定義差は未分離 |

単一原因・寄与率・数学的誤差保証を認定する規則ではない。
baseline→A-step、必要時A-step→A-factorのpath別signed complex差と、各段のlaw/reference gapを保存する。
高精度で両側が一致しても、共通入力の誤りや保存前の丸めを排除したことにはならない。

## 6. Call範囲・性能・出力

float64基準平均は2個。高精度平均は最初4個、最大で2 mode×3精度×2 path=12個。
scalarの差・収束・候補labelの評価を含む。B-1/B-2、別q/round、新H/state/DF生成、
sampling/shot/compile、統計予算・energy成功率・価格の新評価は0。
独立した別powerアルゴリズムも第一実行には追加しない。

LOCALの既存NumPy1.26.4/mpmath1.3.0を使用する。mpmathはPython backend。
人工70x70積96 bitの小測定は約.354秒だったが、実H4の所要時間・速度倍率を予測した値ではない。
binary poweringは不要な最後の平方を避ける。path/精度の4独立jobをspawn processで並列化し、
worker数は使用可能CPUとjob数から選ぶ。4は人工CPU上限ではなく初期独立job数である。
実行前にCPU/RAM/GPU/共有負荷を確認し、共有設定・他jobを変更しない。
CPU/RAM/wall-timeの人工上限は設けない。

結果候補：fresh `artifacts/rpe_m17_precision_isolation/2026-11-03/run01/`（準備では未作成）。
RESULT.jsonに入力hash、SOURCE、基準平均・drift、精度別結果、収束、段階・候補label、実資源を保存する。
frozen_steps.npzに今回のS64を2個保存する。途中の完了jobも保存し、元RESULT/NPZへ追記しない。
上書き/resumeは行わず、技術中断時は原記録を残す。

metadata確認は`python docs/research/rpe_m17_numerical_precision_preparation_20261103/run_precision_isolation.py --describe`。
**別の明示許可後だけ**、実行専用worktreeの固定SOURCEで次を使用する。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python docs/research/rpe_m17_numerical_precision_preparation_20261103/run_precision_isolation.py --run --output-dir artifacts/rpe_m17_precision_isolation/2026-11-03/run01
```

## 7. B-1/B-2へ進む条件とSTOP

Aで後段またはS組立ての差が十分に整理できれば、Bを無条件に行わない。
A-factorでもgapが残る場合、次の問いは保存tail構成以前にある。
B-1では、float64 scalarのどこを固定するかを先に決める必要がある。
例えばtheta64固定でcos/sinを高精度化する場合と、cos64/sin64自体を固定する場合は別の丸め境界である。
元angles/cos/sinのSOURCE replayやcomponent係数の取得を、既存の保存値と偽らない。
B-2でatanやorder重み/Bを数式から高精度再評価する場合は、有限精度samplerの入力lawとも別の比較になる。
いずれも追加の入力・SOURCE・callを固定する次案として返し、今回のTask A driverは実行しない。

旧numeric_diagnostic_status=INCONSISTENT、u_num_bound=null、RPE_tolerance_certified=false、
scientific_status=UNDETERMINEDを維持する。収束をu_num上界へ変換せず、統計残余を理由に閾値を緩めない。
正常完了・原因候補不明・未収束・入力不一致とも記録後mandatory STOP。
意図しない技術的中断の修正・再実行は既存ユーザー許可に従い、失敗記録・SOURCEを保全してfresh結果領域で行う。
正常な不一致・未判定を理由に条件を変更して繰り返さない。
重要レビュー、R2/R3、他roundへ自動移行しない。
