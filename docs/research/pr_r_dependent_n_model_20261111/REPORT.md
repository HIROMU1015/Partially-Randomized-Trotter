# PRのR依存性とN側評価モデル（2026-11-11 JST）

各roundのR_outerだけを指定してN(T,R)へ接続するmetadata plannerとPR batch providerを追加した。既存のfinite RTE作用、signed schedule、phase/radius/Hoeffding、巨大整数N検証を再利用し、固定R用の科学SOURCE・実行入口は変更していない。今回の証拠は人工行列・mock・metadata検査だけである。新Rの実H4/H6 signal/B/N、保存結果の新数値再解析、最適化、価格/G計算は0。公開照合後mandatory STOP。

独立branch `codex/pr-r-dependent-n-model-20261111`、独立worktree `pr-r-dependent-n-model-20261111`で作業した。固定成果`89c91c692e1d35de94d2e7afbe890aabe6d80b7e`を基点に、B/Cのbranch・worktree・環境・プロセスを変更していない。

## R依存モデルと再利用

1 outerのsigned係数をw_j、PF反復数をq=q_PF、δ=T/q、piece配分をr_jとする。H4はS2でw=(1), r=(R_outer)。H6は保存m5_bestの11 signed pieceを同じ順序で用い、既存`allocation(w,R_outer)`が各r_j≥1、Σr_j=R_outerを与える。Rをpieceごとに重複適用しない。全event数はq_PF R_outerであり、q_RPE=2^mとは別の量である。

保存tailのlambda_Rと正規化tail行列Gを固定して、各pieceのsigned τ_j=lambda_R δ w_j/r_jを作る。K2の既存finite kernelは

- P₂(τG)=I−iτG−τ²G²/2+iτ³G³/6（paired TaylorのK+1次まで）、
- b₂(τ)=sqrt(1+τ²)+(τ²/2)sqrt(1+(τ/3)²)、
- log B(T,R)=q Σ_j r_j log b₂(τ_j)、A=exp(−log B)

を用いる。b₂は偶関数だが、P₂はsigned τを使う。上記は`finite_taylor_operator`、`finite_rte_distribution`のSOURCE式の記述であり、新しい分子数値を計算したものではない。

corrected信号はpieceごとのP₂のr_j回適用、raw信号は同じ更新と各eventのb₂除算を独立columnで伝播する。exact-tail columnを内部対照に保ち、raw=corrected/Bを事後定義しない。Rを変更すると有限Taylor近似のcorrected信号も変わるため、旧biasやNをB比だけで移せない。scalar/identityは既存規約で総Tに一回会計し、fragment/product/application順を保つ。

小さい|τ|ではlog b₂(τ)=τ²+O(τ⁴)なので、log Bの主項はq lambda_R² δ² Σ_j w_j²/r_j。S2ではlambda_R² Tδ/Rに比例する主項になる。ただし、これは局所漸近説明であり、計算入口は既存finite式をそのまま使う。整数allocationの段差、非可換な積、bias方向、disk条件があるため、R増加に対するNの単調性・最適性は保証しない。

| 処理 | 再利用先 | 今回の差分 |
|---|---|---|
| ID・recipe・共通N/C record・欠測batch | `pf_pr_rpe_pipeline` | 不変。R変更時に既存canonical IDとstructureを再作成 |
| signed係数・ABS_TIME_PER_OUTER配分 | `h46_m5_plan` → frozen `h46_new2_core.allocation` | 不変。H4は1 piece、H6はR≥11 |
| context/reference/half-step/finite mean | `h6_df19_rpe_d2`の汎用作用関数 | 不変。H6固定D2 guard/analyzeやH4固定26候補guardは拡張しない |
| signal disk・phase/radius・Hoeffding | `h4_df10_rpe_d1.analyze_bound_point`＋common connection | 不変。H4/H6のbindingを渡す |
| 巨大Nの整数検査 | `verify_h4_df10_rpe_shot_integers.integer_check` | 汎用純粋関数だけを再利用。旧H4 exportや科学runnerを起動しない |
| R入力・一括N leaf | 新`pr_r_dependent_n.with_round_R/RBatch` | caller指定Rのみ。optimizer/価格providerなし |
| metadata CLI | 新`plan_pr_r_dependent_n.py` | 科学実行機能なし。Bが承認後にproviderを接続する |

## Nへの接続と数値精度

既存disk D_correctedはsigned biasを一度会計し、座標allowance、full-H reference offset、保存state residual×Tを含む。raw diskはA D_corrected＋normalization margin等。共通geometryはrho_lower=A−raw_disk>0、beta_det=asin(raw_disk/A)、beta_det+beta_stat≤beta_totalを検査し、epsilon_axis=rho_lower sin(beta_stat)/sqrt(2)、N_axis=ceil(2/epsilon_axis² log(2/alpha_axis))へ接続する。

actual implementation/law誤差と厳密ground同定は未資格のまま残し、actual Nはnull。ideal laneの数値PASSはこれらの保証ではない。AはN側の信号半径に一度入る。CへBやAを再乗算しない。

新入口はRに応じた演算数を明示する。pieces数をJ、event数をE=qRとすると

- stages=q[2(prefix+1)J+J]+2E（head作用・exact-tail作用・finite更新・raw除算）、
- roundoff=64·2^-52·(1+maximum_norm)·(1+stages)、
- u=max(1e−9,10·vector_closure+roundoff)+reference_pair_difference、
- log_B_margin=max(1e−12,64·2^-52·(1+|log B|)·(1+stages))

を保存する。既存のfloorと算術を用い、Rループをstage数へ追加した経験的allowance `R_AWARE_EMPIRICAL_ACTION_COUNT_V1`である。rigorousな累積誤差上界とは扱わない。旧候補との精密比較には、この同じpolicyで保存旧signalをscalar再評価する必要があるが、今回は実施していない。

reference差gate1e−10、exact-tail norm/独立closure gate1e−9を維持する。保存2経路のcomplex値も直接照合し、報告差と観測差の大きい方をallowanceへ渡す。極小τでfinite bがbinary64の1へ丸まる場合やτのunderflowは、B=1としてPASSさせず未資格とする。overflow/underflow・非有限値・位相/半径FAILを記録し、R変更や別精度での科学retryへ自動進行しない。巨大Nは保存epsilon/alphaに対するDecimal140/180桁のceil一致を検査し、旧binary64整数から上方向だけ補正する。十進文字列も出力する。この整数検査もsignal diskのrigorous資格を与えない。

## 固定条件とinterface

H4は保存DF10・S2/δ=.1/PR prefix1/K2、PF B0/prefix7を保持する。H6は保存DF19・m5_best/δ=1.6/PR prefix7/K2を保持する。h=.1、13round、化学精度、各方式αtotal=.05/各軸.05/26、beta_total/stat=.4/.2、prior/decoder/energy shift0、保存H/state/sector、P=0は凍結taskからコピーする。

H6ではm0..3のT=.1.. .8はδ=1.6より小さく、正整数q_PFでこのδを維持できない。そのため新R入口はH6 m≥4に限定する。m0..3の旧prefix5/R32・各deltaとPF候補は変更しない。これらのR変更はprefix/delta条件を含む別の科学判断を要する。

`with_round_R(base,{m:R,...},source)`は各roundに1候補を持つtaskを返す。複数Rを同roundで比較するときは別taskとし、同じ費用和へ重複投入しない。変更candidateには新IDとallocation/event数を付け、未評価と記す。[H4例](EXAMPLE_H4_TASK.json)・[H6例](EXAMPLE_H6_TASK.json)はmetadata例であり、科学条件として採択していない。

`partition_saved(task,records)`は同じIDだけでなくcommon identity全体を照合し、旧RのNを除外する。保存FAILはFAILのまま残り、共通`run_missing`が再評価前にSTOPする。新providerはR mapへ明示したroundのPRだけを受け、他PR/PFやH6初期roundの不足を新規計算で埋めない。

`RBatch`はBのcontext factoryを受け、H/state/snapshot bindingを検査する。referenceは同H/state/Tの2経路保存値をSOURCE付きcallbackで渡せる。callbackなしなら既存の新reference2経路を明示的に計算するため、将来の承認範囲に含める。referenceはR非依存、half-stepは固定δ/recipeで共有し、候補ごとのPR伝播を並列化する。worker数は既存のCPU/RAM観測に従い、独自の小さな上限は設けていない。

出力は従来の`identity/plan/actual/diagnostics/source/origin`である。planにN_X/N_Y、各軸disk/radius/beta/epsilon、diagnosticsに全signal/logB/B/attenuation・allocation・R-aware allowance・reference provenance・整数検査を保存する。C record schemaは不変。[Codex B HANDOFF](HANDOFF.md)に具体的な接続・入力責任を示した。

## 技術検証と最小科学案

[TEST_LOG](TEST_LOG.txt)：人工/mock/metadataの12テストPASS。S2のR4/8/16とm5のR11/16/22を非可換2-level人工系で既存spectral finite meanと照合し、corrected/raw差<2e−13、独立closureを確認した。H6 signed配分とqの区別、候補ID/旧R除外、異なるroundのR map、一括`run_missing`、H4/H6人工scalar、未知actual、数値range/FAIL・retryなし、SOURCE保全を検査した。実H4/H6保存signalを用いたN再解析ではない。

[最小科学計画](EXECUTION_PLAN.json)は未採択・未実行。例としてm12のH4 R4→8とH6 R16→32を1候補ずつとし、新finite mean2・内部対照2、新R scalar2と保存baseline scalar2、XY計8軸を提案する。既存2経路referenceをbinding付きで再利用できれば新reference0、満たせなければSTOPして別途最大4 reference actionの範囲を判断する。旧N原本とH/stateは保全し、PF再評価・sampling/compile/価格/G・候補最適化は行わない。

担当：Codex Bが本interfaceと既存loaderを接続し、ユーザーがR menuと新数値allowance policyを含む科学実行範囲を判断する。許可前に実分子の新R計算や保存baseline再解析を開始しない。既存STOP/KEEP/grant/ledger/markerを保全し、今回の公開・remote SHA/主要成果物照合後mandatory STOP。
