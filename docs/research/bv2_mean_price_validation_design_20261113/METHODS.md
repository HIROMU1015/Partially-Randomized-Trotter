# 平均compiled価格の三方式：対象量と不足証拠

今回の記号・比較だけの設計で、実S/C・保存統計の再解析はしていない。S0aは同じ保存event列のS parity8件を確認しただけで、下記の平均法を資格しない。

| 評価軸 | A：閉状態平均S | B：局所期待値/connected-cluster | C：Streaming標本平均 |
|---|---|---|---|
| 直接求める量 | 定義済み状態modelのE[S]。compiled Cには別の全体残差が必要 | 選んだ基準の局所平均。E[S]近似、またはE[D=C−S]。full境界がなければ全Cではない | 保存/新trajectoryのS標本平均。保存同path compiled Cがある場合だけ別にC標本平均 |
| q/R時間 | 状態数dが一定なら既存二乗合成O(d³[Σlog r_j+log q+J])。kernel準備/状態閉包が前提 | 固定cluster≤3/K2ならorder-pattern formはpiece当たり14項以下。component条件付き校正/境界は別費用 | n本についてO(n[qR+qJ×head量])。全event生成/処理時間は残る |
| memory | kernel O(Jd²)、作業O(d²)。実状態が増えるなら保証なし | 固定局所table＋piece/boundary情報。component組合せが大きい場合の費用は別 | eventは逐次保持できるが、head/preparation、seeds/lawsは残る。compiled QC/DAG全体のmemoryは解決しない |
| 再利用SOURCE | MeanKernel/MeanPiece/full_wrapper_mean、元AccountingとS0a law/source binding | _iid_sequence_form/_sequence_form、旧calibration/holdout、local_residual_mean/combine_mean。旧係数は転用しない | S0a SavedEvents/stream_structure、既存iter_sample_rte_events（将来のsampling許可後だけ）、C1保存C/S |
| 適用条件 | 全合法supportに対し遷移と価格rewardの同値類が閉じ、piece/outerの法則が同じ反復で表せること | 同じlaw/conditional component・basis・angle、controlled局所unit、PF/RTE・piece・outer・wrapper境界 | 保存cohortのexact candidate/law/compilerが一致すること。新sampling時も元law/SOURCEを保持 |
| 誤差 | 物理縮約誤り、binary64順/零判定、数値伝播、compiler残差。閉包だけで厳密Cにならない | cluster打切り、境界、非局所compiler、law距離、校正分散/共有covariance、長q/R systematic | 標本誤差、希少event未観測、PRNG/実law、C−Sモデル誤差。S標本平均のSEはCのSEではない |
| 検証 | source上の全continuation/reward同値性と有限点の物理照合、別のmean参照。有限path一致だけでは証明不可 | disjoint校正とfresh full-wrapper holdout。q-only/R-only/joint方向を分離 | 既存summaryを優先、必要時だけ新cohort。X/Y同pathはpairedでnを2倍にしない |
| 未取得 | 実C1の小状態・physical kernel/precision/law証拠 | 現domainの局所S/D係数・境界・打切り/移送誤差 | 全lawへ認定できる価格mean・分散/CI、巨大q/R計算負担、モデル移送 |
| Bへ渡せる資格 | E[S]をFULL_WRAPPER_MEAN_COMPILED_PRICEへ直接渡せない | full E[S]+full E[D]と適格なlaw/domain証拠が必要 | 有限cohortはFINITE_COHORT_ONLY。全law期待値の宣言へ自動昇格しない |

## 物理SOURCEに対する閉包の論点

`pr2_matched_accuracy_m1_execution._prepare`→`prepare_df_partial_s2(identity_policy='extract_identity_phase')`を使うC1では、`df_rte_tail`はretained identity componentsをrandomized supportから除く。したがって、以前の人工identity rotationの角度反例は、そのままこのphysical supportでの到達例ではない。この制約自体はSOURCEから確認でき、今回新しいpreparation数値を生成して確かめたものではない。

合法な非identity K2 applicationだけなら、tail内phaseはTaylor order2のπと負productのπの逐次和になる。signed timeはrotation angleへ入り、同一|τ|の符号反転が同じorder確率を持つ場合もある。しかしpieceごとに|τ_j|/allocationが異なり、単一のIID lawへ平坦化しない。phaseの実値・binary64の累積順、pending Zと固定scalar/wrapperの相殺の扱いを省略できるとはまだ示していない。

必要な候補状態はpending basis（完全operations/inverse）、wireごとのpending Z、合法tail内active basisと相対phase、piece位置/outerの継続条件、terminal/scalar/wrapper。global_phaseやcount/depthはSの将来分岐に戻らない部分もあるが、全report状態を小さくできることと、S observableだけを縮約できることは別。全合法continuationについて同じ遷移・S増分・terminalになる同値性、あるいは一定長でのflush保証が必要である。

fixed register数は小さい状態行列の存在を意味しない。binary64は理論上有限bit集合だが、到達値/履歴がq/Rとともに増える可能性と、実用的な有限quotientは別問題。今回SOURCEだけでは十分条件を証明できず、physical closureはUNKNOWNのまま。単純flagやmodulo角を採用せず、新たな近似が必要なら別仮説としてユーザー判断へ戻す。

## 局所補正で失ってはいけない寄与

| 寄与 | 現C1 S | 平均/残差modelで必要な対応 |
|---|---|---|
| RTE内部 | active basis共有、controlled rotation/product、literal Z fold、tail phase | pieceのcomponent/order lawに対する局所平均と打切り誤差 |
| PF–RTE | 同じAccountingのpending basis/Z | 固定headを含む同じ局所unitでpaired C/Sを定義 |
| signed-piece | 11pieceの元順、負角、整数allocation | piece別law/両側head、interfaceを保持。単なるΣr_jへの圧縮不可 |
| outer | 同じcounterでq反復、途中finishなし | outer端の相殺・compiler非局所残差、q方向移送 |
| scalar/wrapper | scalar −cTは一回、X/Y suffix/H/measure/terminal | 固定費の二重加算を避け、全wrapper条件で残差検証 |
| compiler | Sでは1Q再合成等を完全再現しない | C−Sの同trajectory residual、独立全回路holdoutと系統誤差 |

旧connected-clusterのκ1/κ2/κ3は元unitのcompiled Cから得た量。現在のSへそのまま足さない。S基準なら同一window/boundaryのD=C−Sからκ(D)=κ(C)−κ(S)を定義し、全境界残差を合わせる。C1 affine補正か新local Dのどちらかを選び、重ねない。内部local_residual_meanはboundary/full_wrapperがnullであり、これだけではBへmean Cを渡せない。

旧H4 DF12/H5 DF9・主にuncontrolled中央RTE/L4,6,8の検証とdirty source hashは過去成果の正本として保全する。現DF10/19・controlled full wrapperへ係数/5%・2%等の旧閾値や独立SE合成を移植しない。

## 参照器・不確かさ・技術的な推奨

短期の参照器はC1保存compiled価格とS0a対応Streaming S。Sの標本平均を計算しても、Cの平均・分散はC−Sとの共分散を含む別問題である。C1が保存済みのmean/residual/paired covarianceは再利用し、同じ統計を計算し直さない。

新標本数は価格精度ε_C・信頼水準/採用policyと分散・lawの仮定から固定する。IID仮定下のs/√nは記述的/nominalな診断で、経験的sを厳密分散上界にしない。未観測order2を0価格とせず、必要なら別承認の条件付きlaw校正/stratificationと正しい重みを検討する。forced rare pathを通常cohortへpoolしない。X/Yは同path、同じ係数やboundary校正を複数roundで共有するため、CIを独立合成しない。

**現時点では最終方式を決定できない。** 小状態の証拠が得られるならAはイベント全生成を避ける候補、得られなければBがスケールする近似候補。Cは比較用の参照器として有用だが、巨大q/Rの直接全wrapper compileコストをなくさない。まず選択済みB domainでclosure/supportの限定確認を行い、成立しない/UNKNOWNなら必要な局所仮説と境界校正をユーザーが選ぶ。どれも本作業で科学的に採択していない。
