# S3から基底エネルギーの全体RMSEへ：接続条件

2026-11-13 JST。**S3は固定raw mean・保存solver proxy・ideal IIDモデルに対する標本診断であり、十分N／保存Hamiltonianへのglobal RMSE boundは未取得。** 最小の追加証拠は、全事象のloss契約と有効なpopulation risk上界、proxyとtarget基底エネルギーの距離、raw値・lawの適用誤差である。全mean再計算や共通有効Hamiltonianの新証明を一律に要求する必要はない。

今回はSOURCE調査・数学的条件整理と既存8技術回帰／field照合のみ。**保存mean・bias・RMSE・CIの再計算、trial原本の再集計、実分子計算、sampling/S3/価格/Gは0。** 新評価器は不要と判断し、既存コードを変更していない。[不足証拠表](ERROR_GAP_MATRIX.md)／[Bへの引渡し](HANDOFF.md)／[有限科学案](NEXT_SCIENCE_PLAN.md)／[技術確認](TECHNICAL_CHECKS.txt)。

## 1. 確認したSOURCEと固定条件

| 固定commit | 実際に確認した内容 |
|---|---|
| B S3 `2084791b1b09cff65e07fa8566bb734e1e6bff8e` | `rpe_s3_final_plan_20261113/run01/` REPORT/HANDOFF/PRIMARY/LOSS_ENVELOPE/EXECUTION。trial archiveを開かず保存指標を引用 |
| 計画 `ee1d456dcd1cb47ae1f404b01a62447ede22a8c1` | SCIENTIFIC_PLAN：proxy採択、UNDEFINED主loss、[0,4096]副loss、confidenceと固定一回実行 |
| B code `ab12b69120323379b0aae572e7f62692a5819125` | `rpe_decoder_statistics.py`のsimulate_prepared/simulate/exact_counts/summarize/trial_budget、`rpe_decoder_fast.py`のdecode経路 |
| caller `7486b91582c01e0313257c03ca9f58e9c3326847` | `scripts/run_h4_s3_statistics.py`：43,819固定、同trial主副評価、観測support検査とnull資格 |
| A契約 `bba150834ca30bdb186fbfd58c55a8d51177b855` | `h4_pr_physical_error_contract_20261113/` REPORT/PHYSICAL_INPUTS、全13行field/SOURCE・m3/m4欠測 |
| A結果 `169d3316dd008eb98dd9be56fe4ecefc5c1a7d60` | `h4_pr_s1_s2_rmse_results_20261113/` REPORT/ROUND_SUMMARY、元S0/m1と新10roundの来歴 |
| C law `d72dc4e6228bb380f4d1d000e5e030794fce7164` | `finite_k_sampler_law_alignment_20261113/` REPORT/MATH/LAW_CONTEXT/SCIENCE_PLAN、既存TV helpersと人工テスト |
| 入力生成 `97d25cc431516d720ce91c7cb4f148e061d8ea34` | `scripts/h4_tol_only_input.py:generate/projected/load`。生成・eighは起動せずSOURCE本文だけ |
| 既存prior `cefc01622a3cd75dd16d30de7b9fb68274f3eddd` | `h4_df10_rpe_d1_20261110/REPORT.md`、`h4_df10_rpe_d1.coefficient_prior`のexact係数境界 |

対象はLinear H4/1.00 Å/STO-3G、保存tol-only 1e−8 DF10/weighted-Hermitian completion、sector(2,2)/dimension36、S2/L_D1/K2/ΔPF=.1、h=.1/M12/P0、元exactκ・H/state/snapshot・law/prior/decoder/profile。εE=.00159360010199040 Ha、εqpe=εtrot=.00079680005099520 Ha、minkowski/finite_B_diagnosticは変更しない。

## 2. S3で取得したものと限界

以下は[固定S3 REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2084791b1b09cff65e07fa8566bb734e1e6bff8e/docs/research/rpe_s3_final_plan_20261113/run01/REPORT.md)の転記で、再評価ではない。

| 指標 | 保存結果・意味 |
|---|---|
| 反復 | 固定43,819、seed20261113、1回 |
| 解決条件付きRMSE点推定 | **.0006847202619833286 Ha**、保存proxyに対する観測値 |
| 未解決 | 0/43,819、97.5%二側CP上端 **9.999788982664716e−5** |
| 主population RMSE CI | **null**。解決条件付きRMSEのCIもnull |
| 副population MSE envelope | **[0,28.963512732533932] Ha²**、RMSEスケール上端5.381775983124338 Ha |
| 十分N／global bound | 未資格/null。branch_error_probability=null、prior_alias_qualification=false |

副評価は未解決loss∈[0,4096] Ha²・support仮定・粗いMC精度を条件とする感度envelope。特定の全事象energy推定器のRMSEではない。field名`population_RMSE_CI`だけで資格を昇格しない。この上界は目標達成を支持しない。主点推定が小さいこと、prior外0件、未解決0件も代替証拠にならない。

解決事象D・未解決U、pU=P(U)、target E*とし、全事象のloss LUが定義された場合に限り、

\[
R(E_*)=E[1_D(\widehat E-E_*)^2]+E[1_U L_U]
=(1-p_U)r_D^2+p_U E[L_U\mid U].
\]

条件付き**標本値**はpopulation rDの上界ではなく、CP区間はpUだけを制御する。UNDEFINEDのままでは後半が定義されない。零観測はpU=0を証明しない。一般に偶数shot・内点確率のBinomialでは両countが半数になるzero座標事象も正の確率を持ち得る。今回はこの確率を実H4へ数値適用していない。

特定fallbackを採らず、宣言した有限energy補完familyの全てに対するrisk envelopeを評価する選択肢はある。ただし、全support・補完family・targetとの整合性が必要。任意loss区間から自動で単一energy誤差変数を作ってMinkowskiを適用しない。

DECODEDはzero/tie未解決でなかったという実装statusであり、branch正解oracleではない。priorは選択に使われず、prior外0観測もalias不在を証明しない。直接の全riskを厳密に評価できれば誤branchのlossもそこへ含められるが、稀な誤branchを有限MCで見逃した可能性は別に制御する必要がある。

## 3. Target・stateの資格

HDFを今回の保存completed DF Hamiltonian、E0,DFを同sector基底エネルギー、E*を保存solver proxy、ψsを保存数値stateとする。DF前の同じ有限基底分子Hamiltonian Hmolや連続系の物理分子は別対象。**固定契約の「元H」はPF/有限RTEの前の保存HDFであり、Hmolへtargetを変更しない。** Hmolへの主張を求める場合は追加DF誤差が要る。

E*=−2.1663874486347634 Ha、保存residual=1.9513566490707353e−15 Haは89c91c6:records[5].diagnosticsの値。生成SOURCEは保存36次元sector行列に`scipy.linalg.eigh`を適用し、`eigenvalues[0]`／`vectors[:,0]`を選び、独立occupation行列・Hermiticity・residualを数値検査している。**最小固有対を意図した数値証拠は既にある。** 一方、行列組立誤差・固有値包含・最低固有値の順序の認証はなく、生成時からground_state_certified=false。

正規化ψと正確なHermitian Hに対する認証residual r=||(H−E*)ψ||なら、dist(E*,spec H)≤r。しかし、これはどの固有値かを決めない。H=diag(0,1)の励起stateでもresidualは0である。今回のresidualはbinary64観測値で、厳密rの上界とも未同定。この反例は保存stateが励起stateだという判定ではない。

既存係数prior [−4.994249624417489,−2.0985459369977186] Haは保存Hermitian DFモデルへのexact-rational/Gershgorin＋変分境界であり、無視しない。h=.1のprincipal条件に対する既存根拠はあるが、その条件に必要な局所位相条件や全shotのbranch成功を証明したものではない。priorだけでは狭いground包含にもならない。

最小target証拠は、同じHDFについての**狭いE0,DF包含区間**。直接risk経路では保存stateが厳密固有stateであることや新overlap証明を一律に要求しなくてもよい。保存stateの効果はrawモデルに入り、targetとの距離を別に制御できる。

Hmolまで主張する場合は、同じbasis/sector/scalarで |E0,mol−E0,DF|≤||Hmol−HDF|| を裏付ける必要がある。tol-only1e−8はそのままenergy誤差ではない。既存weighted projectionはfloat norm＋DecimalのPASS_ENGINEERING、representation_error_certified=false。DF truncationとcompletionを合わせた認証上界とは区別する。basis誤差等の新予算も自動採用しない。

## 4. 13roundの物理誤差

全13行にraw/corrected/reference/exact_tail、logB/normalization、3 bias成分、phase/radius、allowanceとSOURCEがある。過去の数値gate PASSを確認したが、今回は再判定していない。m3/m4の観測norm欠測とSOURCE gateによる条件付きempirical診断を維持する。

\[
z_{corr}-z_{ref}=(z_{exact\ tail}-z_{ref})+(z_{corr}-z_{exact\ tail}).
\]

第一項は固定head/tail S2分割のPF寄与、第二項はfinite-K mean寄与。このPRはtailをdiscardしない。complex signal空間の分解であり、energy biasの分解ではない。正のBについてraw≈corrected/Bの独立伝播closureが保存される。Bは測定減衰と統計負担を表し、単独ではenergy bias・law誤差の上界にならない。

S3はrawを周辺Bernoulli平均にしたので、**その固定rawに表れたPF/finite-K/stateの影響を含む**。S3のriskへ同じ寄与を再加算すると二重会計になる。一方、rawの数値不確かさ、解析的ideal meanと実samplerのlaw差、target proxy差は伝播していない。S3の`includes_physical_systematic_error=false`は、この適用誤差を資格付けしていないという意味で読む。

局所disk→局所phaseには原点を除く半径などの条件が必要で、さらに全decoder energyへはbranch分岐を含む論証が必要。局所phase/Tを全energy biasにしない。認証residualならDuhamel型関係
`|<ψ|exp(−iHT)|ψ>−exp(−iE*T)| ≤ |T| r`
をreference signalに使えるが、energy RMSEの式ではない。保存residualの認証値としての代入も行っていない。

exact-tail固定S2は一つのunitary stepの反復として表せるが、finite-K corrected stepは一般に非unitaryでRもround依存。共通有効Hermitian H・共通固有stateは未確定。ただし、固定13 rawの全riskを直接扱う際の必須前提ではない。原論文の単一有効位相モデルに基づく解析ルートで必要となる。

## 5. 全体riskへの二つの接続経路

### 5.1 固定Nの全riskを直接評価する経路

全事象に定義した同じenergy estimator Z、固定N、同targetで、plug-in count law Qと評価対象law Pを区別する。BがQへの有効なpopulation二乗risk上界 RQ⁺を取得し、全RPEデータのTV(P,Q)≤Δが認証され、同じlossが0≤L≤Uなら、

\[
R_P(E_*)\le R_Q^+(E_*)+U\Delta,\qquad
\operatorname{RMSE}_{P}(E_{0,DF})
\le \sqrt{R_Q^+(E_*)+U\Delta}+d_{target},
\]

dtarget≥|E*−E0,DF|。confidence付きRQ⁺なら結論も同confidence・同仮定の下に限定され、無条件の数学的保証へ昇格しない。未知Δやdtargetを0にしない。数値・law項を分ける場合は条件付きの三角不等式で扱い、PF/finite-KそのものをΔへ二重計上しない。

認証raw座標差 |μm,a−νm,a|≤ηm,aと独立shotモデルがあれば、Bernoulli couplingとdata processingから
`Δ ≤ min(1, Σ_(m,a) N_(m,a) η_(m,a)/2)`
という保守的経路がある。branch不連続にもLipschitz仮定なしで有効だが、大きければ不有用な上界として残す。1/2は±1平均からBernoulli確率への写像。Δは**energy推定1回の全shotデータlaw**で、MC43,819反復をここへ掛けない。経験的uを認証ηと読み替えない。

Cのcategorical/event/trajectory TV helperは再利用可能だがF0は未実行。解析的重みとの差、実PCG64 joint性、回路作用差は別に残る。Cの同path X/Y価格pairingもS3の独立shot/X/Y仮定を証明しない。理想量子測定のIID条件付きclaimと実sampler/実backendのclaimを分ける。

### 5.2 元のεqpe・εtrotを個別に資格付けする経路

共通確率空間上の基準推定量Z0と近似推定量Zについて、

\[
\|Z-E_{0,DF}\|_2\le
\underbrace{\|Z-Z_0\|_2}_{\epsilon_{trot}\text{に対応させる成分}}
+\underbrace{\|Z_0-E_{0,DF}\|_2}_{\epsilon_{qpe}\text{に対応させる成分}}.
\]

各項が既定budget以下ならεE/2＋εE/2で結合できる。独立性はMinkowski自体に不要だが、**Z0のlaw/state/normalization/N/decoder/全事象出力、coupling、各L2量の上界**は要る。S3はZの周辺標本のみで、Z0もcoupled差も評価していない。

共通有効Hと対応する正しいenergy branch Eeffを別途資格付けできる場合は、`||Z−E0,DF||₂ ≤ ||Z−Eeff||₂ + |Eeff−E0,DF|`という従来型の分離も可能。そのためには同じ有限shot/law/decoderに対する前半のRMSE上界と、後半の認証energy biasが必要である。現在のfinite-K各roundからこれらは取得されていない。同じ全事象Zなら`MSE=Var(Z)+(E[Z]−E0,DF)²`も恒等式だが、そのbiasにはfinite-shot/branch/量子化の偏りも入り得るので、PF・finite-Kのenergy biasと同一視しない。

厳密単一位相を基準にする場合も、同Bでattenuateした基準と単位振幅基準ではQPE統計が異なる。normalizationをどちらの成分へ置くかを先に固定する。saved-state full-H referenceを基準にすればstate混合が残る。実装都合で選ばず、未決定の科学契約としてユーザーへ返す。

直接全riskルートは個別budgetの達成証明を自動で与えない。既定等分budgetへtarget/law/数値項を加えるなら、既定εtrot内等への明示配分又は総和条件が必要で、予算を二重使用しない。現在の結果から「余裕」を再計算して新配分を採択していない。

## 6. 最短の次段階と今回の完了範囲

まずBが**全事象lossとpopulation精度契約**を確定し、Aが同HDFの**target包含を取得する有限案**を準備する。既存13 mean/ref/logBは再利用する。次に主張水準に応じraw数値誤差とC lawの扱いを揃える。条件付き数値診断、confidence付きモデルrisk上界、認証された物理global boundを別々に表示する。

新しい全mean、P2/P3、新PF、追加S3反復、共通有効Hの全面再証明を最初の必須作業にしない。分散型CIへ替えるだけでrare branch lossを十分抑えられるとも限らない。Bの現exact_countsは小分布用で、13round全count直積を扱う完成solverではない。

Bがrare-eventの解析的上界を検討する場合、zero事象の一部は既存Binomial PMFで扱える。各axisが独立で、偶数shotならそのroundのzero座標確率は `BinPMF(N_X/2;N_X,p_X) × BinPMF(N_Y/2;N_Y,p_Y)`（奇数shotなら0）。さらにround独立なら全roundのunionは `1−Π_m(1−p_zero,m)`。これはモデル上の式であり、今回は実H4へ適用していない。branch-tieと誤branchのlossはこの式だけでは覆わない。元BのBinomial部品は再利用できるが、全branchのtail評価・認証算術は別の技術課題である。

既存8人工・保存fixture回帰と固定S3/null/law/13round fieldのpassive照合はPASS。新scientific資格は付けていない。REPORT・matrix・HANDOFF・有限案を公開し、remote SHA/tree/主要成果物照合後mandatory STOP。
