# SP-1後：finite coherent-signalのblock合成・設計review

2026-10-06 JST。**DOCS_ONLY_DESIGN_PROPOSAL_AWAITING_GPT_REVIEW / mandatory STOP。**
利用者から受領したGPTの[研究計画原文](inputs/sp1_post_run_gpt_research_plan_20261006.md) §13を、
数学・実装仕様、[claim/対照表](block_synthesis_claim_and_baseline_matrix_v1.md)、
[有限pilot案](block_synthesis_small_pilot_proposal_v1.md)へ具体化した。
これは新手法の採択、新規性認定、source review済み実装、実行authorizationではない。

## 受領判断と固定証拠

主RQ案は、同じ有限精度coherent-signal taskで合成対象・block単位・誤差配分をそろえたとき、
合成費用・測定負担・古典取得費用の釣合いを改善できるか。
短いfinite-RTEの補正後平均作用素をphase-awareな低T回路の線形結合へ変換する案を第一構成候補とする。
一般のjoint分解、first-moment補間、LCU、atom結合、cost×momentは既知。現候補の独立差は未確定。
方針・RQ・論文着地点・追加検証の採否はGPT側、Codexは受領scopeの仕様化だけを行う。

固定SP-1 result commitは`9d2bb1fa439748b02084bd9fbc9b10a705328f8a`。
[正式結果とscope](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9d2bb1fa439748b02084bd9fbc9b10a705328f8a/docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md)は
2-qubit synthetic A/B/C、n={8,16,32,64}、四mask、native error 10^-6、signal ε=.05のlocal evidence。
分子geometry/basis/DF rank/split L_D/PF delta窓は適用外。48 rows/96 axes完了、42適格、6モデルshot cap。
Cの外側はweight1の符号coinでactual finite RTEではない。CのD/R単独material gainは0、DRは8/16でgain、32でloss。
この限定結果を維持し、二層RTE×PAIの失敗、selective placement一般の不可能性へ広げない。
十分shot countと加法的primitive費用は、最小shots/最良whole-unitary再合成の下界ではない。
旧raw result・classification・marker・authorization・sourceは変更しない。

## 数学対象を三つの型へ分ける

| 型案 | 再現対象 | 係数条件・保持する情報 |
|---|---|---|
| `FiniteMeanTarget` | M=Σω pω bω Uω、全ρのTr(ρM) | pは確率、bは補正weight、Mは一般にnonunitary。signed time、cutoff、順序、normalizationを保持 |
| `ScaledControlledChannelTarget` | T_B=B Σω pω C(Ctrl(Uω))、B>0 | exact TP辞書分解ならΣj aj=B、Σ|aj|≥B。first momentとは別制約 |
| `UnitaryAtom` / `OperatorLCU` | Mtilde=Σj aj Vj | Vjはphase込みunitary、real aj。Σaj=1を要求しない。qj>0はnonzero係数support上で正規化 |

channel targetはsystemだけでなくcontrol ancilla込みを使う。projective unitary同値をatom同一性に使わない。
TPのsum条件をoperator LCUへ移植しない。旧SP-1 `Gate`の条件を外して上書きしない。
複素係数は±1/±i phase-bearing atomへ分けるか、一般phaseをcontrol側へ明示的に実装して費用を数える。
zero coefficientはsampling supportから除く。M=0は別recordにし、gamma=0で割らない。

目標はoperator norm residual ||Mtilde−M||≤δblock。正解state一つのsignalへのfitは採用しない。
finite Mとexact evolutionの差δsim、dictionaryの近似δblock、実gate列のδimpl、numerics uを別欄に保存する。
単一blockのdense I2 residualは小型oracle評価であり、a priori大系保証とは区別する。

## current finite-RTEとの意味論接続

normalized tail Rhat=Σl pl Pl、Pl²=I、signed dimensionless時間τ、分割r、even cutoff Kなら、
参照する補正後平均はM=[P_(K+1)(−iτ Rhat/r)]^r。
K=2は三次P3であって二次P2ではない。
[固定Sのrte.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d01ed9a332ebc5b66ed08acf56214a9b9c0236d/src/trotterlib/rte.py)の
`finite_taylor_operator`、`finite_rte_corrected_operator`、`finite_rte_operator_moments`をtextとして確認した。
import・呼出しは0。式を新しい実装検証済みreferenceとは呼ばない。

一microstepのnormalizationは

\[
B_K(x)=\sum_{k=0,2,\ldots,K}\frac{|x|^k}{k!}\sqrt{1+\frac{x^2}{(k+1)^2}}.
\]

補正前event平均はM/B_K(τ/r)^r。Mとこのattenuated meanを混同しない。
scalar、符号吸収、eventのphase、作用順序をrecordに保持する。
複数blockのM2 M1は別々にfinite化した積であり、P_K(A)P_K(B)をP_K(A+B)へ置換しない。

候補手順は、閉じた小Pauli代数の係数mapで有限多項式を積和し、phase付きatom辞書への係数問題へ渡すこと。
これは現時点ではPauli多項式処理と標準LCU最適化の構成案。標準対照にも同じ集約/reuseを与える。
全trajectory列挙を避けたことだけで独立method deltaとはしない。辞書取得・再利用・memoryの差は未証明。
DF fragment全体をconstant-size supportと仮定しない。実DFのbasis/逆basis/contextは将来の別review事項。

## samplingと有限confidenceの会計

real aj、j~q、Hadamard ±1 outcome Y_(j,a)ならX_a=(aj/qj)Y_(j,a)。
各軸の平均はRe/Im Tr(ρMtilde)、V2=Σ aj²/qj、range bound Mmax=max |aj|/qj。
canonical q=|a|/gammaではV2=gamma²、Mmax=gamma。shared draw等は今回の未採用自由度に加えない。
正の混合も元RTEのbω/Bを消さず、joint-spaceの実装を通じたfirst-moment biasを戻す。
nonunitary Mそのものを、無補正の確率混合や単一unitaryだと扱わない。

全armへ同じfinite-confidence規則を与え、各軸の残余を

\[
s_a=\epsilon_a-b_{{sim},a}-b_{{block},a}-b_{{impl},a}-u_a>0
\]

とする。operator residualから各軸biasへの変換とRe/Imのfailure allocationは結果前固定する。
point signal/oracle varianceをprimary shotsや係数探索へ使わない。
cost-aware q∝|a|/sqrt(C)は既知のmoment×cost理想化であり、Bernstein/range/ceil込み最適性ではない。
C=0 atomでこの式を使用しない。最初の提案は全arm canonical qで比較し、cost-aware探索の追加は別判断とする。

P=Mm…M1、Ptilde=Mtilde_m…Mtilde_1のerrorには

\[
\|Ptilde-P\|\le\sum_j\Bigl(\prod_{k>j}\|Mtilde_k\|\Bigr)
\|Mtilde_j-M_j\|\Bigl(\prod_{k<j}\|M_k\|\Bigr)
\]

を使う。finite block normを1と仮定せず、boundaryと実装のerrorも同じ順序で戻す。

資源は(G_T, Nshots, G_Clifford/CX, workspace, C_classical)を保存する。
準備・basis/control-phase/readout/context、辞書生成/solver/certification/sampling費用を別欄へ置く。
Pauli-LCUのT=0部分回路だけでは勝利にならない。未知の準備costは感度Pで表示し、勝つPを後から選ばない。
additive費用を使う場合はその名称を保持し、actual compiled claimは実gate列の連結/reduction検証後に限定する。

## 将来APIと情報accessの案

実装候補namespaceは`src/trottertracks/algorithm_codesign/block_synthesis/`。今回は作成しない。

| 将来API案 | 入出力と独立verification |
|---|---|
| `build_finite_mean_spec` | signed Pauli coefficients/time/K/r/order → sparse operator + normalization/provenance。全trajectoryを要求しない |
| `make_phase_aware_dictionary` | 固定gate-template manifest → phase付きatom/cost/workspace/error identity。baselineも同じものを使用 |
| `fit_operator_lcu` | 同一target/dictionary/residual/budget → coefficients、solver status。standard optimizerも同条件 |
| `verify_block_residual` | 固定係数と実gate列 → 独立interval residual、phase、composability。solver自己申告をcertificateにしない |
| `account_coherent_task` | coeff/q/verified error/context/ε/α → 共通sufficient shotsと資源vector。I2信号は診断専用 |

I0はPauli入力・rational time/cutoff・固定gate templates。I1はそこから得るsparse係数/構造情報。
I2は小型dense operator/channel/finite signalの独立評価。
どのarmが辞書/係数取得にI2を使ったかと取得costを記録し、oracle-assistedをoracle-freeと呼ばない。
実装・synthetic semantic tests・科学pilotには別source/authorizationが必要。現段階はAPI設計のみ。

## 今回の停止点と次review

GPTに[具体pilot案の未固定表](block_synthesis_small_pilot_proposal_v1.md)のtarget・辞書・baseline・会計・予算をまとめて採否判断してもらう。
独立新定理の完成だけを小型試験の条件にしない。同時に一般joint構想だけで無制限探索へ進めない。
[設計manifest](../../../artifacts/track_b_block_synthesis_design/2026-10-06/preparation_manifest_v1.json)に二入力のidentityと保護pathを記録する。
入力計画のsandboxリンクやChatGPT引用番号は原文に保存するが、GitHubのsource attributionとは扱わない。
BF/BM closure、SP-0.5/SP-1 one-shot、Track Aの原稿/証拠境界は保持する。
新matrix/trajectory/library生成/solver/synthesis/compile/tests/GPU/NPZ/Hamiltonianは0。
資料をcommit/pushした後、利用者/GPT reviewへ戻してmandatory STOPする。
