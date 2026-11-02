# 最小検証の選択肢とP1非実行計画

## Option比較

|選択肢|判断できること・支持/棄却|既存資産と新処理|負担・承認|
|---|---|---|---|
|0 構造整理のみ|情報依存と先行研究重複を明確化。cheap bias/構造定理がなければ安全な選択保証を保留。独立差分がないなら縮小/停止|本C0で実施。schema、SOURCE、式、既存方向を参照。新性能値は得ない|新数値処理なし。科学実行承認不要。正式な仮説採択はGPT・ユーザー|
|1 限定保存値pilot|構成長のみと有料適格性照会の違いが、既存固定域で選択損失/評価数へ現れるか。失敗なら単純規則を棄却/縮小|P1の16 Full endpoint。新選択、Oracle最小、G差/regret・取得量の集計。全G/shot/biasは保存値|新科学取得0、実装は小さい評価adapter。別SOURCE/input固定と数値再解析の明示承認が必要|
|2 新held-out検証|未使用物理条件/candidate域での移送と取得費用実測。再現しなければ一般化を棄却|新snapshot/geometry等が必要となり得る。真の未使用性を事前確認。必要なHamiltonian/signal/compile/samplingを問いと対応させて別契約|本C0では対象・予算を採択しない。Option1で利益・差分が示されるまで提案止まり。科学実行承認が別途必要|

**最小pilotはP1の1案だけを推奨候補とする。** PRの大規模partition sweepや新分子取得は提案を急がない。R3重複だけで研究価値がないと判断するなら、Option0で終了してよい。

## P1：最小qの適格性情報に価値があるか

固定元：f75a7c25…のdataset.json.gzとcandidate_index.csv。同じ保存政策のFullだけを使用し、H2の別DIAGNOSTICS政策やPRcohortは混合しない。

- H4/H6、linear1 Å、STO-3G、tol-only DF1e-8、rank10/19、T=.8、P=0。
- systemごとにFull prefix=rank、公式2nd/4th、q1/2/4/8。各8候補、計16。m5_best・B0/B0I・PRは対象外。
- epsilon=.001、alpha_axis=.025、通常signed bias・保存empirical allowance/native/compiler/状態政策を維持。
- 正確なcandidate IDとmetadata pointerはpilot_contract_DRAFT.json。候補選定は構造条件だけで行い、性能による追加/除外をしない。

### 固定する4規則

1. FIXED_S2：S2/q8を選択。
2. FIXED_Y4：標準Y4/q4を選択。
3. M0_LENGTH_ONLY：L=2*q*piece_count*(prefix+1)の小さい順、q、formula ID、candidate IDの順で1件選択。これはunmerged deterministic half-blockの相対数であり、scalar/control/wrapper込みcompiled Cではない。bias/適格性情報を見ない。
4. M1_PAID_FEASIBILITY：公式ごとにq=1,2,4,8の順で保存適格性だけを有料照会し、最初のELIGIBLEを1件残す。両公式の候補からM0と同じ順で選択。非ELIGIBLEでも後続qが単調改善する保証は仮定しない。UNDETERMINEDは記録して探索継続し、候補なしならABSTAIN。signal/price欠測は技術欠測として停止し、物理的不適格に置換しない。

M1は照会したstatus以外のbias/headroom/N/G/Cを受け取らない。最初の適格qより大きいqを見ないため、Gの最小性は保証しない。無料のmetadataと、有料signal評価を伴う規則を明示的に区別する。

### 評価と予算の数え方

評価側で全8候補の保存status/GからOracle最良tie集合を1つ作る。選択が適格か、G、absolute/relative regret、tie集合への一致、ABSTAIN、適格性の誤主張を2context×4規則で記録する。M0は適格性の主張をしないが、無効候補を返した割合は必ず提示する。

取得は各規則ごとにcold-startを仮定する。M1の照会は各context最大8候補。最終的に選択した候補の実行前検証/価格取得も省略しない。fixed/M0の候補にも最終適格性確認1件が必要で、M1が既に照会済みなら重複計上しない。価格は適格選択だけ2軸、Fullのためsampling0。Oracleは必要な全候補signalと適格候補価格を要求する基準として示す。

実際には保存値replayなので新signal/compileは全て0。この「反実仮想の取得件数」と「今回行う数値call件数」と「元取得の実時間」を別欄にする。欠測の元wallを件数から推測して実時間短縮を断定しない。

数値callの上限案はbootstrap1/load1/structural_features16/label_views16/selection_trace8/oracle_context2/decision_metrics8/summary1。合計53の外側event案。内部で新会計・shot・compile等は0。SOURCE実装前の設計予算であり、内部call閉包は未検証。実装時に増加が必要なら理由を記録して承認対象へ戻し、既承認と扱わない。

### このpilotが研究判断を変える条件

- 構成長のみが無効候補を返す、または大きな損失を持つなら、構成長だけのG選択仮説を縮小する。
- M1が両contextで適格候補を返し、固定2法の最良よりGを悪化させず、Oracleより少ない取得量なら、情報制約付き選択の工学的継続候補。ただし「全qで最小」「新規方法」「未知系で成功」へ昇格しない。
- M1の利益がない、ほぼ全signalが必要、保留を除外しないと成立しない、既知規則との差分がないなら、Option0の技術整理に留める。結果を見たrule調整・精度追加・新proxy fitはしない。
- 既存手法を同条件で実装した比較はP1に含まれない。その未実施を勝利と数えず、独立論文化判断は別に残す。

## 承認と停止

次の判断はGPT・ユーザー：Option0で閉じるか、P1を開発用pilotとして正式採択するか。採択後、Codexが既存exact算術を使った最小runner、SOURCE/input seal、人工検査、内部call表を完成させる。**実行承認はその固定契約に対して別途必要**。

C0の契約案はenabled=false、grant=null、runner_SOURCE=null。承認入力・grantを発行できる実行入口は含まない。science/compile/samplingの承認は含まれない。将来P1も失敗時のpartial保全、無断retryなし、正常/異常ともmandatory STOPを案とする。過去の研究に固定されたSTOP/authorizationは変更しない。
