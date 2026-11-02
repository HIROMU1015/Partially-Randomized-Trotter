# SP-1 wrapper accumulation / placement：結果前契約 v1

2026-10-06 JST。**source最終review用。科学実行0、RUN_READY=false、実行未認可。**
GPT review `APPROVE_SP1_CONTRACT_DIRECTION_WITH_REQUIRED_AMENDMENTS_BEFORE_SOURCE_FREEZE` に従う。
[原文](inputs/sp1_contract_gpt_review_20261006.txt)のSHA256は
`60481446a066058e2cc10d01e6663031acdb8f688b36b88f042ed6fb2153d520`（10,407 bytes、CRLF保持）。
[f974f8e7の提案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f974f8e7caa5a2769255c0483d7e40caae0a3930/docs/tracks/algorithm_codesign/sp1_wrapper_preregistration_proposal_v1.md)
は履歴として保持し、本契約でfusion・metric・実adapterを確定する。
数値正本は[contract_v1.json](../../../artifacts/track_b_sp1_wrapper_source/2026-10-06/contract_v1.json)。

## 目的と非claim

SP-0.5で確認されたprimitive synthesis trade-offが複数controlled rotationsへ累積したときに
消失・維持・反転するか、登録されたD-like/R-like populationへの選択的PAI placementで
その境界が変わるかを、共通finite-confidence会計で判別するdevelopment/mechanism pilot。

新規性判定、最小必要shot数、actual compiled wrapper費用、DF-native improvement、
actual finite RTE、一般的なD/R配置則、独立validation、論文成立を主張しない。
Sparse PSの積L1二次モーメント・回路T費用crossoverは既知であり、
[前段の一次claim監査](sp1_wrapper_preregistration_proposal_v1.md#問いと一次claim監査)を維持する。
全結果を保存してSTOPし、研究方針・必要な次検証はGPT側で判断する。

## 固定入力・独立作業場所

branch `track-b-sp1-wrapper-source-review-20261006`、
worktree `/home/abe/Project/prt-worktrees/track-b-sp1-wrapper-source-review-20261006`、
基点 `f974f8e7caa5a2769255c0483d7e40caae0a3930`。必要text/JSONだけのsparse checkout。
今回明示copyした資料は上記review原文一つ。A/root・共有APIの実装を変更しない。

| 入力 | 固定identity／用途 |
|---|---|
| SP-0.5結果commit | `e57c1fdd28589422e9c973e34e53f6c725b921e9` |
| 元result JSON | `artifacts/track_b_sp05_economics_result/2026-10-06/v1/result.json`、SHA256 `d92c3a3d618219f3f46c3cc8143a49b085772e3d1943036511bf39c792dfb759` |
| 元tool identity JSON | SHA256 `803d902a18d925a9565749fbc64422dca4979bfa6c0bd8b438b28de18d3aa984` |
| 保存synthesizer | pygridsynth 2.0.0、wheel SHA256 `30b5b15e9383a8ea8510d54f28e2de0385d102cb54fd4440071abf4525027568` |
| 数値runtime | B専用の既存Python 3.10.12 / mpmath 1.3.0。全28 package versionsとpy source treesをlaunchで照合 |
| 再利用field | 23 literal key rowsのsequence bytes/hash/T数/error guard、保存PAI係数区間 |
| 再利用しないfield | SP-0.5 J・分類・集約moment/cost。旧PAI/J/guard関数は再実行しない |

native operator errorは10^-6、catalogueはexact kπ/4（k=0,…,7）一つ。
新target synthesis・catalogue・precision追加は0。
23件は異なるliteral keysであり、23独立角度とは呼ばない。
入力は既知development evidence。SP-0.5 result/contract/source/authorization/markerを編集しない。
B-F限定negative closure、BM現adapterのnew-method closure、旧BM-1/16-cell未実行、過去STOPを維持する。
M1/M2はTrack Aのsource-bound local evidence。M2 H4 1.30 ÅをBのheld-outにしない。

## domain・target・共通fusion

ancilla |+⟩、system |0⟩の2 qubit。X/Y ancilla測定をRe/Imとし、
各wrapperの理想平均νをtargetにする。mask間で同じν、異なるtemplate/n間のGを競わせない。
ordered logical列は左から右の時間順。

| template | n | ordered列・分布 |
|---|---|---|
| A：same-angle, alternating-noncommuting-generator accumulation | 8,16,32,64 | 全D、+3π/16、Z/X交互、path1・weight1 |
| B：mixed angles | 同じ4値 | 全D、(+π/16,+π/8,+3π/16,+1/5 rad)のcycle、Z/X交互、path1・weight1 |
| C：selective placement | 同じ4値 | (D:+π/16 Z ; R:σπ/8 X ; R:σπ/8 Z ; R:σπ/8 X)をn/4回。wrapper単位σ=±1、各確率1/2・weight1 |

maskはNONE/D/R/DR。Cのcoinは有限列挙するtoy distributionで、finite RTEではない。
D/Rにはgate数・angle・sign・outer randomnessが同時に割り当てられている。
結果の解釈はこの登録populationの配置効果に限定する。配置の主要証拠はC。
A/BはR=NONE、DR=Dというduplicate controlで、独立positiveへ数えない。

logical controlled R_P(θ)をjoint factors
R_(I⊗P)(θ/2)、R_(Z⊗P)(−θ/2)へlowerする。negative time・relative controlled phaseを保持する。
system phaseをlowering前に消さず、完全joint II因子だけidentity channelとして扱える。

**logical → joint-space lowering → canonical exact fusion → placement → PAI** の順序を固定。
fusionはrole/maskを参照せず、隣接する完全に同じjoint generatorのsigned角度をexactに加算し、0を削除。
commuting reorder、mask別pass、D/R labelによる阻止・並べ替え、結果後strategy変更を行わない。
unexpected mixed-role fused factorは全maskでrejectし、未reviewのrole付与を採択しない。
この登録domainではその分岐は起きない。

[静的監査](../../../artifacts/track_b_sp1_wrapper_source/2026-10-06/static_fusion_audit_v1.json)は
12 wrappers / 16 paths、pre/post各960 native位置、candidate0・actual fusion0・cross-role opportunity0。
全pathのpre/post sequence、signed exact angle、D/R lineageを保存済み。信号・資源scoreは計算していない。
科学実行時は48 mask rows・96 axes・64 path-mask records、最大128 native factors/path。

## 保存係数adapterと数値guard

保存されたg1/g2区間のexact rational midpointを取り、tilde g3=1−tilde g1−tilde g2とする。
γ=Σ|tilde g|、p_j=|tilde g_j|/γをexact Fractionで構成し、Σg=Σp=1を保持する。
枝choicesはpath条件付きで独立。共有drawを追加しない。samplingは行わない。
true coefficientの保存区間からの全L1 displacement e_iを別のnumerical biasへ戻す。
true/implementedの両方を覆うγbar_iについて

\[
u_a\leq\sum_\omega p_\omega|b_\omega|
 \Bigl(\prod_i\bar\gamma_i\Bigr)\sum_i e_i/\bar\gamma_i.
\]

これはsigned channel積をtelescopingした上界。u_a>10^-8ならnumeric failureとして停止する。
δ_iはgenericではmax(saved δ_upper, 2 saved η_upper)。saved ηの丸めにより2ηを下回らせない。
exact catalogueのδ=0はprojective channel identityの代数的確認により、NONE/PAIへ共通適用。
guardは固定されたSP-0.5入力の信頼性を前提にし、独立の再合成／再guard検証ではない。
全Fraction会計を70桁directed decimal enclosureで保存する。signalのpoint値をintervalと呼ばない。

## primary metric・accuracy/confidence・分類

primary名は **fusion-normalized additive synthesized-primitive T cost**。

\[
C_T^{\rm add}=C_{\rm fixed,T}+\sum_i C_{T,i}^{\rm stored},\qquad
G_T^{\rm add}=C_{\rm init,T}+\sum_{a=\Re,\Im}n_a E[C_T^{\rm add}].
\]

logical exact fusion後の保存primitive T/T†数を加算する。
primitive境界を越えるClifford/T cancellation・reducer・actual wrapper compilationは行わない。
固定prep/basis/control/measurementはCliffordなのでT費用0、per-shot fixed T=0、batch init T=0。
Cliffordや全gate費用が0とは言わず、depth/workspaceの性能claimもしない。

W=b_ω Γ_ω sign、Γ_ω=Πγ_i、V₂=Σp_ω b_ω² Γ_ω²、M=max|b_ω|Γ_ω。
E[C]とE[W²C]を別保存し、外側pathとの相関を保持する。
合成biasは Σp_ω|b_ω|Γ_ω Σ_i E_branch[δ_i]。
対象外の通常合成誤差もPAI weightで増幅される。PF/RTE bias=0はsynthetic targetの定義による。

complex ε=1/20、familywise α=1/20、96 axesの各α_a=1/1920。
ε_a=ε/√2の下側75桁有理数をcontractに固定し、2ε_a²≤ε²を機械確認する。
norm-one ±1 outcome、Var≤V₂、centered range≤2Mの共通Bernstein規則で

\[
s_a=\epsilon_a-b_{\rm synth,a}-u_a,\qquad
n_a=\max\{1,\lceil(2V_2+4M s_a/3)\log(2/\alpha_a)/s_a^2\rceil\}.
\]

logはrange reductionと256-term exact rational atanh series / tail upper bound。
shotは実測せず、**bound-based sufficient count**。minimum necessaryとは呼ばない。
shot cap=10^9/axis。超過をclipせずSHOT_CAP_EXCEEDED、s≤0はBIAS_BUDGET_EXHAUSTEDとして保存。
両軸適格時だけGと同wrapper NONE比を作る。NONE不適格・zero-costにpositive ratioを作らない。

| NONE比interval | row classification |
|---|---|
| upper≤.95 | MATERIAL_GAIN |
| lower≥1.05 | MATERIAL_LOSS |
| 全区間が(.95,1.05)内 | NO_MATERIAL_SEPARATION |
| その他のboundary overlap | NUMERIC_INCONCLUSIVE |

5%はこのsynthetic resource modelのmateriality。統計的significance／publication thresholdではない。
runnerはresearch GO・次stageを決めない。row cap/bias-ineligibleは保存する。
全48 rows処理完了はSP1_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW、partial/numeric/cap failureは
INCONCLUSIVE_MANDATORY_STOP_NO_RETRY。いずれもSTOPし、retryしない。

## 実装・semantic確認・保存

実装は共有libraryをimportせず、B専用namespaceに置く。

| source | 内容 |
|---|---|
| [wrapper_sequence.py](../../../src/trottertracks/algorithm_codesign/synthesis_placement/wrapper_sequence.py) | exact角度、joint lowering、共通fusion、static audit |
| [wrapper_adapter.py](../../../src/trottertracks/algorithm_codesign/synthesis_placement/wrapper_adapter.py) | 保存列・区間adapter、bias、2-qubit point channel |
| [wrapper_accounting.py](../../../src/trottertracks/algorithm_codesign/synthesis_placement/wrapper_accounting.py) | 前段のexact rational共通kernelを保持 |
| [wrapper_launch.py](../../../src/trottertracks/algorithm_codesign/synthesis_placement/wrapper_launch.py) | source/A/path/tool照合、exclusive marker、caps |
| [wrapper_result.py](../../../src/trottertracks/algorithm_codesign/synthesis_placement/wrapper_result.py) | inventory/duplicate/STOP validation。score再計算なし |
| [runner](../../../scripts/tracks/algorithm_codesign/run_sp1_wrapper_pilot.py) | static planと別承認後だけのone-shot runを分離 |

59 focused local testsはsigned lowering、X/Y/Z basis/inverse、controlled U/−U phase、
共通fusionとzero deletion、保存notch/sequence identity、tiny signed-channel compositionと81枝独立列挙、
小さいouter coin、係数bias伝播、shot/bias/cap、direct-A拒否条件、partial inventory/STOPを確認する。
full suite・registered domain資源/信号sweepは0。local passであり、CI・外部再現・研究結果ではない。
[記録](../../../artifacts/track_b_sp1_wrapper_source/2026-10-06/focused_tests.txt)。

科学実行後のshapeは[result_schema_v1.json](../../../artifacts/track_b_sp1_wrapper_source/2026-10-06/result_schema_v1.json)。
runtime inventory checkerはcomplete時の48/96、duplicate positive禁止、eligible shot capとSTOPを確認する。
schema全体をJSON Schema engineで実行するというclaimはしない。
全ordered native列・lineage、各path p/b/Γ/conditional cost/moment、係数/p/sign/γ、
保存sequence bytes/hash/count/guard、各軸bias/u/margin/shots/G/NONE比/分類を保存する。
point signalはmpmath 80 dpsのsigned channel逐次合成（3^128列挙なし）。
ideal/finite meanとresidualは診断専用で、primary shots・順位に戻さない。
nonfinite、trace residual>10^-50、residual>saved synthesis+coefficient guard+10^-50なら停止する。
point diagnosticは独立のfinite-time certificateではない。

## launch境界・資源上限

新science source Sをcommit固定して最終reviewする。現在のauthorizationはpending、source_commit=null。
SP-0.5 authorization/markerを流用せず、review commit HEADから実行しない。
最終review通過と別明示実行指示の後だけ、新worktreeで**Sの直接子A**を作れる。
Aの変更可能pathは本contractのauthorization JSONと任意の
`docs/tracks/algorithm_codesign/sp1_execution_authorization_receipt.md`のみ。
clean HEAD、親[S]、source/contract/tool identities、runs1/retries0/STOP、明示指示をlaunch時に照合する。
pending authorizationでは保存データ採点・matrix構築・marker前に拒否する。

wall1200 s / CPU900 s / peak RSS1GiB / output8MiB（markerを含む）/ 1 process / retry0。
0.1s POSIX watchdogでwall/CPU/RSSを確認し、CPU OS capと追加AS2GiB guardを使う。
preflight後、科学評価前にexclusive markerを作る。消費後の例外もretryしない。
観測run時間はruntime/static-plan準備から計測し、科学会計・診断・serializationをcap内で確認。
resource snapshotはモデル＋最初のserialization後、最後のserialization/persistence前。
異常後のbounded failure receipt保存は科学評価を含まず、markerは削除しない。
出力過大/serialization failure時は保存できたrow数と小さいfailure receiptを残す。

今回はここまでをsource review用に公開してSTOP。実行authorizationは作らない。
分子NPZ、Hamiltonian/DF、trajectory、GPU、新合成・catalogue・target・precision、
旧16-cell pilot、次stageへ進まない。positiveでも自動認可は一切ない。
