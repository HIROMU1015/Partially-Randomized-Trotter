# BF-1 minimal pilot proposal — execution NOT authorized

日付: 2026-10-04  
status: `PROPOSAL_ONLY_AWAITING_BF0_REVIEW`  
親decision: `PROCEED_BF0_DESIGN_AND_NOVELTY_AUDIT`  
`science_execution_authorized=false` / `runner_creation_authorized=false`

関連: [数学契約](bf0_mathematical_contract.md)、[claim監査とprovenance](bf0_prior_art_claim_matrix.md)、[BF-0外部レビュー依頼](bf0_external_review_request_20261004.md)。以下の数値は**結果前にreviewする提案値**であり、実測から選んだ有望点、freeze済みcontract、authorizationではない。

## 1. このpilotが答える問い

固定したnative五stage四次familyを、同じdomain、同じoptimizer、同じ係数評価予算で設計する。通常PFのerror/length objective O、既知leading-tail objective L、実際のfinite-RTE objective Fを比較し、randomized-tailの有限化を設計へ戻すことが、共通coherent-signal taskでdecision-relevantな差を作るかを調べる。

「係数が変わった」だけでは通過しない。O/Lで探索した係数もすべて同じfinite objectiveで再評価し、qと総RTE budgetを再最適化する。Fがその対照を越えなければ、旧P-Dと同じ選択一致として止める。

BF-1の到達点は、**I2 oracle-assisted developmentにおける有限信号とaction proxyの機構検査**である。compiled cost、新method、実用I0/I1 design algorithm、transfer、総costのGOは出さない。三文書のreview完了前はrunnerも作らない。

2026-10-04の利用者指定により、BF-0 reviewで問題が閉じた場合は、必要な最小修正と結果前固定・authorizationを経て**一回限りのBF-1**を行う。研究方針の全面再設計は、BF-0 reviewで重大な問題が判明した場合、またはBF-1終了後に行う。pilotの良否にかかわらず終了後に停止し、BのRQ・新規性・着地点を全面再評価する。

## 2. Development inputとtask案

| 項目 | 提案する固定値 / 規則 | 現時点の状態 |
|---|---|---|
| system | H4 linear 1.00 Å、STO-3G、8 system qubits、DF rank 12 | 既知development条件。新geometryは選ばない |
| Hamiltonian | Aのcurrent preparation contractで定義した同じDF表現・identity抽出・screening・fragment順序 | 記載を参照するだけ。snapshot未access |
| split | weight-ordered prefix `L_D=3`の一つだけ | split frontierやendpoint探索はしない |
| time | `T=0.8` | Aとの既知の接続点 |
| state | 同じDF Hamiltonianのnormalized ground stateをI2 referenceとして用いる案 | future state作成/identity固定は別authorizationが必要 |
| signal | \(z=\langle\psi\rvert e^{-iTH}\lvert\psi\rangle\)、数学契約のcorrected finite mean | energy誤差のclaimではない |
| primary accuracy | `epsilon_sig=0.01`, `alpha=0.05`; axis配分はepsilon/√2とalpha/2 | review対象の提案値 |
| bridge accuracy | `epsilon_sig=0.05` | primaryで生成した係数の再採点のみ。係数再探索禁止 |
| excluded | epsilon=0.002、新しいtime/state/geometry、H6/H12、held-out | 未認可。失敗時の自動追加なし |

stateのdegeneracy、eigensolverのtolerance、phase convention、state fingerprintは実行前contractで固定する。設計と評価に同じI2 referenceを使うため、training/evaluation分離、held-out性能、large-system deployabilityを主張しない。

NPZのliteral pathと既存文書上のidentityを選定することと、実物をresolve/stat/hash/loadすることは別である。今回は後者を一切行わない。将来使用するsnapshotのliteral path、documented bytes/hash、preparation source、rank/screening/split規則、state identityをmanifestへ固定する作業は未完了であり、science authorizationの前提に残す。M2 1.30 Åは使用しない。

## 3. Family、construction、allocation

- 研究対象は数学契約の`w=(a,b,1-2(a+b),b,a)`、実数、四次order等式、`max|w|<=2`のfamilyのみ。
- 全feasible枝、両符号を扱う。正係数側やSuzuki近傍だけに探索を寄せない。
- 主構成はF: exact-stage fusion後にfinite-RTEを挿入する。S、別kernel、processor、general forward/reverse-sweep係数の探索は追加しない。
- `r_j`はlower-bounded largest-remainder ruleで一意に決める。自由なstage allocation探索、stageごとのK探索はしない。
- inputに対して一step templateが成立し、tail occurrenceが全q-step境界を越えて融合しないことをadapter reviewで確認する。成立しなければ本pilotは実行せず、contract reviewへ戻す。

order残差のscreenと数値bias guardは数学契約に従う。near-zero cutoffでstage数を減らす最適化は行わない。

## 4. 三つのdesign objective

三armともI2への同じaccessを許す。次の差は**設計objectiveの差**であり、採点metricの差ではない。`N_a`のceil、axis allocation、infeasible判定、tie規則は共通にする。

| arm | 設計に使うobjective | 何をcontrolするか |
|---|---|---|
| O: ordinary PF | 理想PF signalのbias、`B=1`、簡約したexact PFのDF/tail exponential数を用いたerror/length objectiveをqで最小化する | 通常PF用に同一familyを探索する対照 |
| L: known leading-tail | 理想PF biasに保守的`E_tail`を加え、`log B_L=q sum_j (lambda_R h gamma_j)^2/r_j`、leading work `C_L=n_D+q R_bud`を用いてq/Rで最小化する | 既知absolute-time・normalization modelで十分か |
| F: finite task | 実際のfinite `M,B_K`のaxis biasと、期待event workから数学契約の`G_action`をq/Rで最小化する | finite-tail構造を設計に返す効果 |

Oのtail exponential一つを一factorと数えるのは**formula-length design surrogate**であり、tail exact evolutionを安価に実装できるという仮定や、Oのphysical resource claimではない。DF側は各native blockを数え、`exp(-it H_D)`一つへ置換しない。Oだけに勝ってもGOにはしない。

Lのaxis bias surrogateは`b_L,a=|nu_ideal,a-z_a|+E_tail`。`E_tail`は同じsigned times、r、共通Kから計算する。Fのcancelled biasをLの設計時へ戻さない。Lのleading normalizationはK>=2のsource展開に合わせる。LとFの差にはbound保守性も含まれるため、Fのwinning coefficientをLと共通のfinite採点で比較し、単にboundが緩いという説明を除外できるか報告する。

Oのprimary scoreを具体化すると、`n_ideal`を簡約後のDF/tail exponential数、理想PF biasから得た`N_ideal,a(B=1)`を用いて、`min_q n_ideal sum_a N_ideal,a`とする。L/Fのprimary scoreはそれぞれの`min_(q,R_bud) sum_a N_a C_a`とする。feasibility marginが正でない、normalization/shot数が数値表現不能、resource上限に達した場合はinfeasible/unfinishedを明示する。失敗を大きな有限scoreへ黙って変換しない。

探索終了後は、**各armの選んだ一点だけでなく、各armが評価した全係数**を同じfinite `G_action`で再採点し、その係数ごとにq/Rを再最適化する。Fのclaim候補は、O/Lの全探索集合と固定baselineのbest finite scoreに対して判定する。橋渡しepsilon=0.05も同じ係数集合を再採点するだけとする。

## 5. Optimizerと係数評価budget案

三arm共通のdeterministic one-dimensional constrained searchとする。stochastic optimizer、追加seed、多重restartは使わない。domainを`(s,d)`表示し、exact制約を満たすfeasible曲線の各枝を対象にする。

1. 各armに共通の16初期点を与える。Suzuki五stage点とzero-embedded Yoshida三stage点の2点、残り14点は全feasible曲線のarc lengthによる層化点とする。各connected componentへ最低1点を与え、残りを長さ比例で配分する。component/点のtieはs、次にdの昇順。端点・交点の扱いとarc-length精度は実装前に列挙仕様へ固定する。
2. 各armは追加16点までrefineする。feasibleな隣接点を両端に持つ曲線intervalについて、両端の最小objectiveが小さい順に選び、arc-length midpointを評価する。tieはcomponent index、左端s、左端dの昇順。既評価pointは再計上しない。
3. すべての両端scoreがinfeasibleの場合は、同じ固定順序で最長intervalを二分する。全候補を失敗として保存し、geometry、domain、budgetを広げない。

各arm最大**32 unique coefficient evaluations**、合計96。feasible曲線のcomponent数や端点処理が16点budgetと両立しなければ数値探索を始めずdesign reviewへ戻す。midpoint解のbranch確認・制約screenに失敗した候補もbudgetを消費し、後から成功点だけを32点として数えない。

このoptimizerはglobal optimumの証明を与えない。32点はmechanism pilotの予算であり、新methodの成立を保証する探索密度ではない。同じ初期点・refinement rule・評価上限でobjectiveだけを変える。共通初期点の結果を三armに共有してよいが、logical evaluation budgetはそれぞれに計上する。

optimizerはdeterministicのため探索seedは`NOT_USED`。将来の再現用master seed案は`20261004`、必要になったnumerical probe用substreamはpurpose/arm/cell indexから分離し、M1/M2 seedを使い回さない。BF-1でtrajectory samplingは計画しない。state作成のrandom startも使わない案とする。

## 6. 小さいq/R/K policy

| parameter | proposal |
|---|---|
| q | `{1,2,4,8}`; delta=`{0.8,0.4,0.2,0.1}` |
| R_bud per outer step | `{5,10,20,40,80}` |
| r_j | 数学契約の固定rounding。`R_bud<n_tail`はinfeasible |
| K | 共通`K=2`を基本とし、下記の事前escape policyだけで共通`K=4`を許す |
| all parameter tie | objective最小、次にq、R_bud、K、s、dの昇順 |

K4は自由な追加gridではない。各係数/q/Rについて、K2の解析的全tail remainder `E_tail,2`を先に計算する。**`E_tail,2 > epsilon_axis/4`かつ両axisの理想PF biasが`<=epsilon_axis/2`**の場合だけK4 escapeを許す提案とする。この条件は同じI1/I2情報で三arm・固定baselineへ適用し、結果の良否やarm名で変更しない。

eligibleなcellのK2/K4は、探索時には各arm自身のobjectiveで扱い、終了後の再採点では共通finite objectiveで比較する。Fの有限signalやscoreでO/Lの探索順位・refinementを決めない。K4で改善しなかったことを理由にK6、r_j個別探索、R=160、q=16へ広げない。escapeの条件・epsilonの使い方はpreregistration前にreviewで確定する。K4へ移ったcell、boundが保守的だったcellも全件記録する。primaryとbridgeのK eligibilityは各accuracyに同じ規則を適用し、算出済みmeanを再利用する。

## 7. Fixed baselinesと強い通常PFの未解決事項

以下の四つを**proposed fixed reference**とする。採択・wrapper検証済みという意味ではない。探索予算を使ってbaselineの係数を調整せず、q/R/K policyは全armと同じにする。

| baseline identity | purpose / obligation |
|---|---|
| native second-order S2 | 現行kernelに対する基本対照 |
| Yoshida three-stage fourth-order | 同一familyのzero-stage boundary対照。source half-listからの全list展開を照合する |
| Suzuki five-stage fourth-order | 同一familyの既知interior対照。analytic係数を使う |
| Morales v3 Table I left column, unprocessed 21-stage eighth-order, spectral-norm optimum | 既知の強いhigher-order S2 composition対照。Table I右列/eigenvalue optimum、v1 17-stage、processed Table IIとは別identity |

Moralesの版・列は結果を見る前に上記へ固定する提案である。published coefficientsの転記、full-list ordering、高次order residual、signed-time DF adapter、normalization/costの照合は未実施である。現行`morales_8th_list()`の呼出しでこのbaselineを実装済みとは扱わない。21 tail occurrencesではR=5/10/20がinfeasibleとなり得ることも、同じbudget下で報告する。

現行`new_4th_m2_list()`は八桁係数のlegacy referenceである。同じexact四次familyの強い対照として使うには、係数sourceと精度・order残差の扱いを事前reviewで閉じる必要がある。本提案はそれを無断修正したり、精度gateで黙って除外して優位性を作ったりしない。mandatory baselineとされた場合はproposal/budgetを改訂してfreezeし、追加実行を自動認可しない。

Ostmeyer Eq.40のgeneral fourth-order、processed法、SPRINTのnear-integrable法は五S2 unprocessed classと異なる。BF-1は**このclass内の係数設計機構**に限定するので、全PFでbest、強いpublished PF一般に勝ったとは主張しない。それらを別class対照として必要とするかはBF-0外部reviewの未解決事項であり、許可なくalgorithmを増やさない。

## 8. Resource / numerical budget案

| 項目 | hard cap proposal |
|---|---|
| new molecular systems/geometries | 0。既知development snapshot一つのみを将来の別authorizationで使用 |
| coefficient evaluations | 32/arm、3 arms、96 total |
| K2 finite cells | 最大 `(96+4) x 4 q x 5 R = 2000` |
| K4 escape finite cells | 最大2000。eligibility外は評価しない |
| primary + bridge total distinct finite cells | 最大4000。epsilon変更でmean計算を繰り返さない |
| ideal PF reference cells | 最大400 coefficient/q cells、exact targetとstateは各一つ |
| CPU | 最大2 workers、各BLAS thread=1、合計8 CPU-hours、wall time 4 hours |
| memory | aggregate RSS 8 GiB、per worker 4 GiB |
| GPU | query/useとも0。fallbackも禁止 |
| trajectory / circuit / compile | 0 / 0 / 0。BF-1ではmatrix/state-action meanとanalytic action workだけ |
| science checkpoint output | B専用directory、最大128 MiB。future manifestでpath固定。A cache不使用 |
| retries / expansion | science retryなし。cap到達でSTOP、失敗・未評価cellを保存 |

4000はlogical上限であり、全cellが実行可能という見積もりではない。O/L候補の最終finite再採点をこのcap内へ含める。exactに同じB source/input/state/coefficient/q/R/K identityの重複だけはB内でmemoizeしてよい。Aのruntime/cacheやcompile mapをnew BF scienceへ流用しない。

数値誤差guardの提案は`u_axis <= 0.01 epsilon_axis`、threshold近傍のratio uncertaintyは1%以下とする。ただし実行前に、そのguardを支えるremainder・係数丸め・operator評価のbound/検証方法と計算費用を固定する必要がある。都合よく二重精度結果をexactと宣言しない。これを予算内で支えられない場合は`UNRESOLVED_NUMERICAL_MARGIN`で停止する。

現在は上記capを満たす実装性能の証拠がない。scienceを用いたbenchmarkで予算を試すことは本proposalのauthorizationに含まれない。実装設計reviewで不適切と判明した場合は、結果を見る前にproposalを修正して再reviewする。

## 9. Primary metric、materialityと機構検査

primary metricは数学契約の`G_action=sum_a N_a E[C_action,a]`。feasibilityを満たすcellのbest値と、`(bias, log B, deterministic actions, expected random actions, q, R_bud, K)`を報告する。full Pareto setはこの固定grid・評価済み係数に限る。単なるscalar score以外に、どのfeasible q、tail負担、frontier点が変わったかを示す。

materialityのreview案は、F探索集合のbest scoreが**O探索集合、L探索集合、四fixed referenceの共通finite再最適化best**に対して少なくとも5%小さいこととする。この5%はBF-1 action proxyに対する探索的閾値で、M2の10%から転用した値でも、新規性・compiled resourceの閾値でもない。採用する値と理由は実行前reviewで確定する。

feasibility、best q、Pareto membershipも事前固定した同じ参照集合で記録する。ただしqの変化や新Pareto点だけを見て、結果後にBF-Cの条件を追加しない。resource差とは別のdecision routeをBF-Cへ採用するなら、比較対象、判定閾値、数値guard、tie処理を事前登録で固定する。固定しないrouteはsecondary機構診断に留める。四fixed referenceのformula選択を記録することと、別familyを探索することを区別する。

加えて、numerical guardを含めたratioの不確かさが1%以下であることを要求する。deterministic/randomのaction重みをそれぞれ1から±2%動かす事前固定の四corner診断でも改善方向が残ることを要求する案とする。これはanalytic proxyの局所的感度検査であり、actual gate costの誤差保証ではない。DF block別basis costの違いはこの検査だけでは閉じない。

機構検査は新しい係数を追加せず、評価済み同一係数について、O→L→F objective変更がどの項に由来するかを分解する。F係数を通常PF errorだけで採点した値、leading log-normalization、actual finite normalization、finite bias、integer allocation、native workを併記する。次を区別する。

- 既知Gamma/leading modelだけでFのdecisionを説明できる。
- finite bias/normalization/roundingが同じfamilyのPareto decisionを変える。
- 予算内のoptimizer routeやoracle情報にだけ依存して良く見える。

最後の場合は方法成立へ進めない。allocation ruleを固定しているので、独立した新allocation algorithmを発見したとは主張しない。classical coefficient-design費用もCPU-hours/peak memory/evaluation数で報告し、quantum proxy改善から除外して隠さない。

## 10. BF-A / BF-B / BF-Cと全結果後mandatory STOP

BF-0 reviewが通った後に、入力・source・判定規則を固定した一つのexecution IDに対して科学実行を一回だけ認可する案とする。失敗、中断、resource cap到達でも終了後に停止し、科学retry、retuning、追加条件による救済は自動認可しない。以下は結果の分類案であり、事前登録で境界を閉じる。

| outcome | case / stop status | 停止後の全面再評価で検討すること |
|---|---|---|
| 通常PF/leading modelと実質同じdecisionで、finite-tail固有の機構差が支持されない | Case BF-A / `MANDATORY_STOP` | B-Fを主研究として停止することを第一候補とする。固定budgetのnegative resultは一般的不可能性ではない |
| 係数・finite-tail負担の機構差はあるが、共通参照のresource差がmateriality未満でdecision routeも不通過 | Case BF-B / `MANDATORY_STOP` | 機構noteへの縮小。明確な理論的選択原理がある場合だけ限定継続の価値を評価する |
| semantic/accuracy/source gateを満たし、共通再最適化後のmaterialityまたは事前登録したdecision routeと、finite-tail固有の機構差を満たす | Case BF-C / `MANDATORY_STOP` | partial-randomized-task coefficient designを正式な主研究候補として再設計する価値を評価する |
| closest-artが実質同じmethodを既に解いている | pilot前STOP：`NARROW_TO_REPLICATION_NOTE`または`STOP_DUPLICATIVE` | 新method claimを取り下げ、BF-1を実行しない |
| bestがdomain/grid上限、数値margin未解決、adapter/baseline不備、resource cap到達、評価未完了 | `INCONCLUSIVE_STOP_FOR_REVIEW` | BF-A/B/Cへ無理に分類せず、不足点を報告する。grid・algorithm・geometry・seedを自動追加しない |

BF-A/Bの境界は係数差の有無だけで決めず、同一情報・参照で説明できるmechanismとdecisionの関係を記録する。係数差のnumerical同等性、score regretの許容幅、q/Pareto tieの扱いは実行前に固定する。未固定のままpilotを開始しない。

bestがR/qの下端にある場合も、未評価領域へ最適性を一般化しない。upper-bound hitやdomain capに接するcaseで、上限を増やすことをGO判定へ含めない。

BF-1が全gateを通っても、statusはI2 development pilotのlocal evidenceに留める。**全outcome後にSTOPし、研究BのRQ・新規性・着地点を全面再評価する。** execution recordには`mandatory_stop_reached=true`、`next_stage_authorized=false`、`automatic_next_stage=null`を残す。BF-CからBF-2 actual compile、BF-3 independent validation、oracle-free design ruleへ自動進行しない。BF-A/BからB-Sや別algorithm co-designへも自動移行しない。random alternative、split co-design、held-out取得、GPU、共通API変更、algorithm採択、commit/pushは別scopeとする。

## 11. 実行前に未固定の事項

本proposalを実行contractへ変えるには、次を結果前に固定し、reviewと明示的authorizationを別途受ける必要がある。

1. BF-0最後の二novelty claimと通常PF baseline scopeに対するreview判定。
2. 入力snapshotのliteral path/documented identity、DF preparation/source、state作成規則・identity。現物accessの許可。
3. coefficient domainの全枝/arc-length列挙、floating-point representation、order残差からsignal guardへの対応。
4. Fのnative adapter仕様、negative time、controlled identity phase、baseline転記/ordering、有限meanとの意味論照合。現行APIが検証済みという前提は禁止。
5. O/L/F objective、共通finite再最適化、K4 policy、tie、seed、全予算、metric/materialityをfreezeしたmanifest。
6. 数値guardの根拠、capを守る実装設計、B専用source/output paths、source/environment identities、実行authorization。

現在はこれらを閉鎖した実行manifestもrunnerもない。三BF-0文書のreviewが終わるまでscience codeを作らない。必要なfuture B固有codeは`src/trottertracks/algorithm_codesign/`へ置く設計とし、`src/trotterlib/`の共通変更が必要なら小さな別変更として提案する。
