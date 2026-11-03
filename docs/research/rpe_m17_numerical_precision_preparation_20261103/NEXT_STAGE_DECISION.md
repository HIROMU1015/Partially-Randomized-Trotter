# 次段判断：Task Aで保存後の丸めを先に調べる

2026-11-03 JST。**adapter・人工テスト・第一科学実行計画の準備まで完了。実H4再評価は未実施。**

## 1. 現時点の判断

元m17は正常完了し、2経路差約6.26e-10が固定1e-10を超えた。
統計予算残余約8.60e-5とは別の診断であり、旧INCONSISTENTを変更しない。
後段q乗/期待値、step組立て、平均tail構成、float入力・定義差はいずれも候補で、現在はINSUFFICIENT_EVIDENCE。
[静的分析](NUMERICAL_DISCREPANCY_ANALYSIS.md)を参照。

最初に行うべきものは**保存済み行列に対するTask A**。
元7行列のうちF/Rと両tail、Bの保存psiで足りる。新H/state/DF生成は不要である。
ただし元S64は未保存なので、同じfloat64基準でSを組み、別archiveへ固定してから後段を比較する。
そのS固定と、F/R/tail固定で高精度にSを組む境界を分ける。

## 2. 固定できた第一実行範囲

- SOURCE：本document群・adapter・driver・testsを含む公開commit。full SHAと固定URLは最終チャット。
- 入力：元m17結果commit cd749066904d3f29b28b03c19ab69370372b157fのRESULT/7行列、
  B commit e22d7718c47358ff650dbc04e8e72f6ab1c8bf41のpsi。identifier/hashを[固定計画](PRECISION_ISOLATION_PLAN.md)へ記載。
- 物理条件：linear H4/1Å/STO-3G/DF12、physical state、8 qubit/4電子sector70、
  Partial L_D3、δ=.02、m17/q131072/r32/K2、保存N/α/β/epsilon。
- 基準：float64の両経路を元SOURCEと同じ積順で各1回。元平均の再現が1e-10以内でない場合はSTOP。
- A-step：S64固定で両pathを96/160 bit、必要時224 bit。収束差1e-13、未収束ならSTOP。
- A-factor：A-stepが収束してもgapが閾値を明確に超える場合だけ、同じprecision ladderで保存F/R/tailから評価。
- 最大call：float基準2平均、高精度12平均。差・収束・候補labelのscalar比較を含む。
  sampling/shot/compile、H/state/DF生成、B-1/B-2、他roundは0。

18人工testsはPASS。人工行列でq131072、正確なfloat変換、作用順、条件付き段階遷移、
default非科学経路、中断保全を確認した。実データの収束・原因は未判定。
この準備の推奨を、独立科学レビューや原因の正式認定と扱わない。

## 3. B-1/B-2が必要になる場合

A-stepでgapが消えれば後段丸め、A-factorでだけ消えればpost-archive step組立てが候補になる。
いずれの場合も、その観測だけで単一原因・厳密誤差上界を認定しないが、Bを無条件に行う理由はない。

A-factorでもgapが残れば、保存tail以前の構成差へ進む情報価値がある。
B-1の一部は保存A_prob/A_coeffとRESULTのorder確率/tau/Bを使える。
しかしcomponentごとの係数・確率は保存されておらず、元float rotation角/cos/sinも記録されていない。
scalar再実行や追加入力exportを必要とする部分を「保存済み」として扱わない。
theta64固定とcos64/sin64固定、数式からの高精度再評価B-2を分離し、入力とcallを別に固定する。
現在のTask A driverは平均tailを再構成せず、Bを自動実行しない。

## 4. RPEとの接続と残る不足

両精度の結果が近いことや、保存factorを正確に持ち上げたことは、
保存前の近似誤差・物理モデル誤差・真の平均に対する上界を与えない。
収束からu_numを捏造せず、u_num_bound=null、RPE_tolerance_certified=false、scientific_status=UNDETERMINEDを維持する。
RPE十分条件の認定には、どの入力境界に対する数値上界かと、他のmean差・既存予算との関係が別に必要である。
全18round成功、PRNGの厳密law、全回路の意味論、実backend/noise、費用順位は今回対象外。
旧V2のCONDITIONAL_SUCCESS、R1費用UNDETERMINED、A/run02/m17の記録とSTOPを保全する。

## 5. 次担当・許可・STOP

次は**ユーザー**が、この公開SOURCEのTask A（基準2平均＋条件付きの最大12高精度平均）を明示許可する。
**Codex**は許可後、[固定計画](PRECISION_ISOLATION_PLAN.md)の範囲で評価・記録・公開照合してSTOPする。
準備依頼は、新しい実H4累乗・mean差・収束の科学実行許可ではない。重要科学レビューも別許可である。
承認を証明するpacket/ledger/署名等は作らない。

意図しない技術中断の修正・再実行は既存ユーザー指示を適用し、元記録を残して新SOURCE・fresh出力で行う。
正常な未収束・不一致・未判定を理由に対象や閾値を変えて試行を続けない。
準備はremote照合後mandatory STOP。科学実行・B-1/B-2・sampling・compile・他round・R2/R3・重要レビューを自動開始しない。
