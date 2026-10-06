# block synthesis pilot案：BS-0.5 amendment v2

2026-10-06 JST。**PROPOSAL_ONLY / implementation_authorized=false / RUN_READY=false / mandatory STOP。**
旧[v1案](block_synthesis_small_pilot_proposal_v1.md)の採用ではなく、[BS-0.5監査](bs05_method_target_design_audit_v1.md)に沿った改訂案。
旧三文書・旧JSONのbytesは履歴として保持し、以下のtarget分類・arm条件・primaryを現在のreview対象にする。

## domainの役割を固定する案

旧案のu={1,3}、τ={1/8,1/4}、2-system＋1-control、K=2,r=1を拡張しない。
normalized λ=1、Pauli q0,q1、R_P(θ)=exp(−iθP/2)。分子geometry/basis/DF rank/split L_Dは適用外。
targetそのものは未生成、domain値の実行採用はGPT review待ち。

| stratum | finite O target | 件数 / evidence role |
|---|---|---|
| commuting | Rc=(ZI+u IZ)/(1+u)、P3(−iτ Rc) | 4 **control**。method成功数へ含めない |
| noncommuting | R1=(ZI+u XX)/(1+u)、P3(−iτ R1) | 4 performance |
| boundary | R2=(IZ+u XX)/(1+u)、W=R_XZ(π/2)、P3(−iτR2/2) W P3(−iτR1/2) | 4 performance |

合計12 target中、performance evidenceは**8**。signed time/phase/identityは別semantic fixture案で、性能domain追加ではない。
C targetは各caseの[ordinary event分布](bs05_ordinary_finite_rte_baseline_v1.md)から得る同じjoint Φと補正b。
boundaryも同じU_e2 W U_e1のcontrolled channelを用いる。MだけからΦを推測しない。

## O/C比較表を分ける

| 層 / profile | 比較条件・目的 |
|---|---|
| O1 ordinary finite-RTE | 固定event distribution＋task-tuned deterministic native synthesis。phase/B2/branch count/十分shotsまで戻す |
| O2 gatewise **PAI一種類の案** | ordinary joint lowering後の各native Pauli rotation channelへ既知PAIを適用。signed estimator weightをouter bへ掛ける。Sparseと二つ並べた探索はしない |
| O3 standard sparse/operator LCU | 同じ有限M、D0∪D1、同operator残差、情報access、objective・budget。既知手順を強い対照にする |
| O4 Pauli-LCU | D0の同finite M。同phase/残差/会計。T=0に退化してもshots/CX/1C/contextを数える |
| O5 structured candidate（除外） | 現構成はO3と別deltaなし。GPTが独立差を先に定義しない限り戻さない |
| C1 full-channel Sparse PS | 同joint Φ、同controlled atom辞書、signed channel係数・diamond残差。operator winner競争に入れない |
| C2 positive probabilistic synthesis | 同joint Φ、正係数・同diamond残差・outer b保持。既知single-qubit結果のjoint liftingは未検証、span/precision不適格も保存 |

現案はO4 profile＋C2 profile。12×6=**72 target/profile records案**であって、科学row・solver call・synthesis key数ではない。
performance側O32/C16、commuting control側O16/C8。
各profileの固定precision候補から複数Pareto vectorを保存できるため、72を測定総数上限としない。
O5の復活、grouped LCUの追加、別辞書は別GPT scope判断が必要。84-row pilotへ自動復帰しない。

PAIは旧SPと同じcheap notch Δ=π/4の案を一つだけ使う。
native R_Q(θ)ごとにθ0=Δ floor(θ/Δ)、θ1=θ0+Δ、θ2=θ0+πを固定し、real gを
`Σg=1, Σg cos θj=cos θ, Σg sin θj=sin θ`で定義するchannel補間案。
exact notchはg=(1,0,0)。coefficient enclosure/phase/source semantic確認は未実施。
qj=|gj|/γ、γ=Σ|gj|、選択weight gj/qj。whole branchのcorrected weightはb×全native位置のweight積。
branch条件付きmoment/rangeを先に算出し、outer branch分布で平均する。outer-path相関を独立gate平均の積で置換しない。
PAIはjoint channelを再現してO taskを満たす対照であり、operator-only分解の発明ではない。
数値keys・call cap・実装版は未固定。PAI採用自体もGPT確認待ちでscienceを認可しない。

D0={ζP}、D1={ζ R_P(σπ/2)}、ζ∈{±1,±i}、σ=±1の旧辞書案を保つ。
literal184、controlled atom T cap2、extra workspace0は**template案**。生成・actual cost・semantic passを意味しない。
phase-preserving重複集約・lazy生成・cacheを全armへ公平に与える。
CではDのcontrolled joint channelsを使い、system-only channelの保証は流用しない。
operator residualとdiamond residualは同じ数値だから同精度とはせず、補正weight込みの共通signal biasへ戻す。

## primary：multi-resource Pareto

primaryは`(G_T, Nshots, G_CX, G_1C, workspace, C_classical)`。
Clifford/CXを恣意的な重みで合算しない。workspaceはcontrolと追加workspace/peak live qubitsを併記する。
C_classicalは取得・辞書・合成・solver・検証・samplingのCPU/wall/peak-memory/callsを別欄で保存し、共通resource modelで比較する。
CPUとmemoryが相反する場合もtrade-offとし、一方だけ選んで「古典cost改善」にしない。

同じtarget層・input・ε/α・context・precision候補・情報access・budgetの適格vector間でのみdominanceを判定する。
全主要座標で悪くないかつ一つstrict改善が必要。Tだけの改善、係数/gammaだけの改善、commuting勝利だけではGOを作らない。
共通項で厳密に同一の座標は相殺を明示。その他のuncertainty intervalが重なればpoint dominanceとrobust dominanceを区別する。
厳格判定はcandidateのupper≤referenceのlowerが全座標で成立し、一座標strict（後段materialityを採用するならその閾値も結果前固定）。
未取得座標はnullであり0ではない。その座標込みのdominanceは未決定。ineligible/failed/未freezeを同点扱いしない。
strict改善のnumeric margin・classical cost model・materialityは未固定であり、今回GO thresholdを勝手に設定しない。

Pauli-LCUでG_T=0でもwinnerにしない。zero denominatorのratioはnull、絶対差と全vectorを示す。
T低下とshots/CX増加はtrade-off。各層に複数Pareto点が残ってよい。
unknown state preparationは共通感度記号のまま保存し、結果後のhardware weight/cost係数でwinnerを作らない。
O/C間の差はfull-channel要求の追加資源診断で、new-method improvementや最小必要総costではない。

## grouped LCUを強い候補としてレビューに残す

[Wada et al. v1](https://arxiv.org/pdf/2512.06260v1)は同一LCU項のgroupingでcircuit/samplingを交換する既知対照。
原文targetはKρK†（およびnormalized ratio）。linear Tr(ρM)へのadapter、PREPARE/SELECT、phase、group size、追加ancilla、十分shotsは未定義。
現ΦはE[Ctrl(U)•Ctrl(U)†]であり、M•M†とも違う。grouped CP mapをそのままC1の代替としない。
旧extra workspace0では非singleton groupを同条件の無料対照として扱えない。
同targetへの有効なadapterとworkspace scopeが閉じればstrong comparator候補、閉じなければscope差を明示する。
「一回小pilot」を維持するために既知強い対照を無視して一般的優位をclaimすることも、今回勝手にarmを増やすこともしない。

## 実行前の未固定事項と停止

ordinaryは形式的分布・列・ledgerまで定義済み。**actual synthesis sequences/counts/error guards、budget数値が未了**なので実行contractはまだ閉じていない。
各arm solver/objective/version/tolerance/budget/seed、precision採用、数値enclosure、α/ε/bias配分、shot/key/call capは要review。
gatewise PAIという単一案、positive/channel adapter、generic LCUの最適化目的（多資源frontier取得法）も採否対象。
旧40min/30CPUmin/1GiB/16MiB/process1/run1/retry0は未承認cap案を保持し、今回はそれを利用したrunを行わない。

選択肢は、applicationを設計する価値があるか、独立deltaを別reviewするか、このrouteを閉じるか。
GPT判断→必要な仕様freeze/実装承認→実装＋focused semantic確認→source review→別one-shot authorizationが揃うまで進めない。
どの段階のレビューも科学実行を自動承認しない。今回commit/pushしてmandatory STOP。
BF/BM closure、SP-0.5/SP-1の原result/marker、Track Aを変更せず、新科学処理0のまま戻す。
