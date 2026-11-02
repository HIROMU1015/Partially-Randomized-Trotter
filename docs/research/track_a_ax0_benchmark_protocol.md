# Track A AX-0：比較実験・情報分離仕様

2026-10-09。**将来実験の設計であり、実装・計算認可ではない。** [研究契約](track_a_ax0_research_contract.md)、[モデル対応](track_a_ax0_model_correspondence.md)、[計算予算](track_a_ax0_compute_budget.md)と併用する。

## 1. task、誤差、資源

主taskは登録された有限DF Hamiltonian Hと同一状態ψに対する
`z(T)=<ψ|exp(-iHT)|ψ>` のcomplex signalである。各方式は同じH・ψ・T・要求精度・failure allocationに比較する。状態は主にground-like referenceとし、基底状態であることと小さいRayleigh residualだけを混同しない。

B0：constant/full one-bodyとprefixを残してtailをdiscard。B1：全DF deterministic。B2：同じprefixのdeterministic backboneとcanonical finite-RTE tail。B3：prefix0のrandom-dominantであり、one-body/scalarもすべてrandom化した方式ではない。

corrected平均z_corr=B z_rawを参照する。axis aのbiasをb_a=|z_corr,a−z_a|、参照・平均作用の数値不確かさをu_a≥0とし、h_a=ε/√2−b_a−u_aとする。α_real=α_imag=.025、per-candidate complex failure総量.05を主契約に固定する。

\[
h_a>0\quad\text{for both axes},\qquad
N_a=\left\lceil\frac{2B^2\log(2/\alpha_a)}{h_a^2}\right\rceil,
\qquad G_{\rm RZ}=N_{\rm real}\bar C_{\rm cosine}+N_{\rm imag}\bar C_{\rm sine}.
\]

これは登録bias/数値予算の条件下でのHoeffding shot会計。Bはexact finite normalization。biasの絶対上界を用いたoperational判定はconservative eligibility、参照biasを使う判定はoracle-assistedと表示する。等号・非正margin・overflow・未解決uは不適格または判定不能であり、0 shot/0費用にしない。

既存PM-2の再現では当時のu=0、binary64値、roundingを維持する。新研究のu導入を既存statusやledgerの修正に使わない。数値不確かさが境界を跨ぐcellは`eligibility_undetermined`とする。

主量子指標はstate preparationなしのfull measured wrapper期待RZ総数。補助はRZ depth、CX count/depth、total depth、circuit size、one-shot cost、N、B、signed bias components。任意角rotationやT/Toffoliとは同一視しない。ΣN×depthはresource indexで、並列schedule・device latencyを定義しない限りwall timeではない。

物理quantum shotごとにfresh IID canonical trajectoryをdrawする意味論を維持する。32 cost trajectoryを同じ回路としてN回再利用する実験は別のsampling契約となる。本研究のNは量子実験を実施したshot数ではない。

## 2. 系、rank、時間、精度

| 条件 | 主規則 | 追加条件の意味 |
|---|---|---|
| H4 | 既存1.00 Å、STO-3G、DF rank12、T=.8を開発に使用。1.30 Åの固定5構成も既観測の診断 | H4の再収集を先に要求しない。既観測geometryを新held-outと呼ばない |
| H6 | 主に線形1.00 Å、STO-3G。AX-2でrank policy・candidate cap・modelをfreeze | H4モデルの初回移送検査。pilotの露出条件を記録 |
| H8 | 同じ主要geometry/basis/rank policy、H6までで凍結したmodel | 改良後modelの独立確認。H8参照値を見る前に予測recordを保存 |
| geometry診断 | 1.30 Åは構造/geometry依存を検査する必要がある場合だけ事前登録 | 水素鎖以外への一般性、strong correlation全般を代表するとは言わない |
| signal anchor | ε={.05,.01,.005,.001}を維持 | 旧.005〜.1を超える厳しいmargin・適格性を検査。.0001は数値誤差と予算確認後の別stretch判断 |
| 主時間 | T=.8、atomic units | 既存比較を接続 |
| 長時間 | T=3.2を限定診断として候補化。採用理由・subsetをH6結果前に登録 | 累積normalization/PF誤差が移送結論を変えるか。全直積は実施しない |

新H6/H8のDF rankは固定12でも旧configの11/15でもなく、**同じDF truncation error規則から得られるactual rank L**を使う。提案規則は`df_rank=None`、明示的な正の`df_tol=η_DF`、内側の追加切捨てなし。η_DFは `T_max η_DF ≤ .01 ε_min` の共通予算から設定し、OpenFermionの返却truncation valueが使用version・integral conventionでこのnorm allowanceを保証するかをAX-2の生成前に一次定義とsourceで確認する。

repositoryの`_low_rank_kwargs`はdf_tol≤0を拒否するため、`df_tol=0`でfull rankという計画は採用しない。OpenFermionの[公式API](https://quantumai.google/reference/python/openfermion/circuits/low_rank_two_body_decomposition)はthreshold/final rankとtruncation errorを区別する。仕様・norm対応を確認できなければ、rank capなしの明示toleranceと表現誤差未認証の扱いを登録し、full integral-H精度について主張しない。rankが予算を超えた場合も結果後にcapを変更せず、別target/versionへ設計し直す。

全方式の精度参照は**同じ保存DF target**である。finite DF表現誤差、B0 discard誤差、PF、finite RTE、数値、samplingを別記する。DF targetと元の化学Hamiltonianとの差をsignal誤差に黙って吸収しない。H4 rank12 legacyと新rank policyは別層であり、単純なサイズだけの比較と称さない。必要なら新policy H4接続はAX-2の最小検証として別認可する。

energy chemical accuracy（例えば.0016 Hartree）は本研究のε_signalではない。RPE接続にはenergy window/aliasing、state overlap、round schedule、PF energy bias、finite-RTE・synthesis・統計予算、preparationを新契約で定義する。単一Tのphase/Tだけでenergy-estimation達成と報告しない。

## 3. 候補生成と探索境界

fragment orderingはsource-definedの固定weight ordering、tieはoriginal fragment indexで決め、全方式で共有する。ranking weightとexact λ_Rを別保存する。target/reference値やcompiled費用に合わせて順番を変えない。

prefix集合はサイズ対応fraction F={0,1/8,1/4,1/2,3/4,1}から、各`floor(fL+.5)`とその±1を[0,L]へclipしてdeduplicateする。L_D、fraction、λ_R、tail weight fractionを記録し、同じL_DをB0/B2へ与える。B0のL_D=0もone-body-only discard baselineとして許可。L_D=LのB0はB1と同一ならaliasを残してdeduplicateする。B2のempty tailはB1へ、L_D=0はB3へ分類する。

| 方式 | 合理的な探索自由度 | 不要な自由度 |
|---|---|---|
| B0 | 共通prefix、q、PF order/synthesis | r/Kを割り当てない |
| B1 | 全prefix、q、PF order/synthesis | r/Kを割り当てない |
| B2 | 共通interior prefix、q、R、even K、backbone/synthesis契約 | B0/B1の探索費用を抑えるための制限を課さない |
| B3 | prefix0、q、R、even K、共通one-body/control契約 | 全Hamiltonian random化へ意味論を変えない |

初期random latticeはq=2^i、R=2^j、0≤i≤J_q、i≤j≤J_R、r=R/q（正の整数）、K={2,4,6}。δ=T/q、τ=λ_R T/Rとする。qとRの効果を分離し、same Rでqを変える比較を含める。K=6の実装接続・数値適合はAX-2で確認し、不成立なら事前にK={2,4}へscopeを限定する。

同じL_D/T/R/Kではτ、B、E[n_rand]はqに依存しないという定義上の関係を監査項目にする。この条件下のq差はouter PF、deterministic反復、basis/boundary/controlの影響を識別する。これは今後の検査設計であり、保存結果を今回再解析した結論ではない。

**J_q/J_Rとdirect cell quotaは未確定U04**。確定手順はAX-1の旧候補coverageと保存境界、AX-2の数値headroom・登録時間/RSS・回路長profileから決め、main reference bias/costを見る前にfreezeする。旧q≤8/r≤64を高精度実験の十分な探索上限とはしない。全方式に同じq capを使用し、高次PFのstage費用は別に予算化する。

accuracy到達候補がなくcapに接する場合、`not_reached_within_registered_search`とする。資源最良がcap上なら`boundary_limited`、finite precision等で判定できなければ`undetermined`。方式全体の不可能性・真の最適性は主張しない。結果後の無制限なq/R/K追加は禁止。追加が情報価値を持つなら別protocol/versionとしてreviewし、新しい探索と区別する。

### direct比較集合

同じfinite lattice Xに全モデルが予測を出す。direct参照集合X_directはmodel予測・参照signal・参照compile費用に依存させない。AX-2でquotaを割り当て、次の順で候補IDを確定する。

1. B1/B0の登録baseline（全共通prefix×q、登録PF order）と、B2/B3のprefix/q/R/Kの端点およびsame-R pairをmandatory setとする。
2. 残りは(method,prefix,q,K,R-bin)で均等round-robin、cell内はcandidateのcanonical identity文字列順で埋める。モデルの順位を使わない。重複はfingerprintで除く。
3. mandatory setが予算に収まらなければ、結果前にlatticeを縮めて再登録する。評価後の削除で集合を小さくしない。主張はX_direct内の比較に限定する。

モデルがX\X_directから選んだ構成は主regretを測れない。別途選択構成を評価する場合はmodel-selected supplementary setとして明示し、direct集合の最良性・coverage検証と混ぜない。ineligible cellでもcost予測誤差の診断対象に残すかを事前登録し、accuracy結果に基づくcost-validation biasを避ける。

### 強いbaseline

coreは全方式の二次PF/native wrapper。ここだけなら「指定二次PF実装との比較」がclaim上限。一般的な決定論法への優位を狙う場合は、少なくともB0/B1のstandard fourth-order PFを同一signal精度でq最適化し、対称controlled synthesisを検査する。四次の負時間・scalar phase・inner-H_Dの意味論を確認する。

repositoryにはP-Dの高次formula/負時間検証とdeterministic cost機能があるが、現行full wrapperを高次へ接続する完了証拠ではない。AX-2で小規模一致検証、AX-3で登録q集合、AX-4でfreezeした同じbaselineを使う。control最適化がB2 backboneにも適用可能なら同じ選択肢を与え、B1だけ/ B2だけに利得を付けない。

Simon–Loveの[対称PF control](https://arxiv.org/html/2511.13855v1)、PR原論文App. A.4との関係を確認する。一律のRZ半減補正ではなく実装とreference作用で検証する。費用は別のbuild/compile/correctness budgetに含める。強いbaselineを計算できなければそのclaimを縮めて閉じる。

## 4. trajectory統計とrare order

旧32 trajectoryは保存H4との接続用の初期cost sample数。cosine/sineで同じtrajectoryを使い、seed/index/step hierarchyを確認する。one-shot covariance Σと固定Nから
`Var(G_hat)=(N_real² Var(C_cos)+N_imag² Var(C_sin)+2N_real N_imag Cov)/n`
を計算する。deterministic costは単一compileの値。reference signalはtrajectory MCではなく平均作用の決定的計算なので、cost SEをsignal/quantum shotの不確かさに加算しない。

AX-1は保存32のSE/covariance・感度のみを報告し、samplingを追加しない。新AX-3/4では32を初期batchとし、(a) 10% materialityの判断や方式比較がcost uncertaintyで決まらない、または(b) 未観測Taylor orderの寄与を1% one-shot cost以内に抑えたと確認できないcellに、独立seedの追加96を一回だけ許す計画とする。最大128、予算内、追加criteria/seedをmain参照前に登録する。

finite order probabilitiesとRからrare trajectoryの確率を計算し、未観測確率とpre-transpile gate上限等に基づく寄与上限を別記する。SE=0やbootstrapだけでは未観測rare orderの不存在を保証しない。上限が得られない場合はrare-order uncertainty unresolvedとし、formal winnerを主張しない。必要なorder-stratified検証は既存sourceを再利用できるが、canonical unweighted平均とsampling weightを混同しない。追加strataは別予算・別事前登録とする。

SEと±2SEは診断表示であり、formal simultaneous CIではない。formal勝者を主張する場合は有限cost上限を持つbounded confidence手順とcomparison数に応じたfailure allocationを結果前に追加登録する。初期研究の完成条件にformal family winnerを要求しない。

## 5. モデル・選択metric

cost単位・scopeが一致するcellだけで、e_C=(C_pred−C_ref)/C_ref、log(C_pred/C_ref)、過小評価u_C=max(0,(C_ref−C_pred)/C_ref)を出す。zero reference costは比率N/A。axis別、method/prefix/q/R/K/精度別の分布とcoverageを報告し、平均一つで隠さない。

参照eligible集合F_ref⊂X_directを作り、各モデルは同じX_directから予測eligibleな構成x_hatを選ぶ。primary regretは
`G_ref(x_hat)/min_{x in F_ref}G_ref(x)−1`。
先にfalse acceptance（参照不適格を選ぶ）を判定し、その場合regretを通常の有限値として報告しない。F_ref空は比較不能、モデルが全候補を棄却する場合はabstention/missed opportunity。tieはcandidate identity辞書順。

参照shots×予測cost、予測shots×参照cost、両方予測、両方参照の四通りを分ける。operational biasモデルがない場合、後二者を埋めずcost-only条件付きregretを出す。H4 fit後のregretは事後開発診断であり、未使用予測性能ではない。

連続error/regretを主結果とする。**10%過小評価・10%regret**は予算見積もりや構成選択を実務上変えうるずれを識別する設計上のmaterialityであり、理論定数や論文採択基準ではない。**1% rare-cost寄与・1% εの数値目標**は10%選択差の解釈を支えるheadroomとして採る。境界の不確かさはundeterminedとする。既存M2のunderestimate fraction定義はそのまま残し、新metricへ対応するとき分母の差を表示する。結果を見て閾値を変更しない。

## 6. development・評価・情報漏洩

H4はモデル開発、H6は初回transferと必要な改良、H8は凍結モデルの確認という構成を維持する。H4二geometryを「H4校正＋未使用H4検証」と再命名しない。校正入力はH4 1.00 Åを主とし、既観測1.30 Åは診断・必要なら改良入力として表示する。

H6 technical pilotはwall/RSS/形状等の技術情報の取得とscience値の露出を分ける。modelの選択・修正にH6 bias/costを用いた場合は該当scopeをdevelopmentとし、同scopeを後で未使用H6 testと呼ばない。隔離が保証できなければH6全体をdevelopment扱いとし、独立確認をH8に置く。

H8のHamiltonian構造情報を許容入力に使うことと、reference bias/Cを使うことを区別する。参照値を開く前にmodel/source、feature、fit、selector、candidate集合、予測record、seed、metric、unknown処理をfreezeする。H8を見た後の改良・候補追加はexploratory。H8後に「新held-out」を探すことを予算に含めない。

## 7. 実行前に確定する未解決事項

| ID | 確定する内容と方法 | 段階／確定前にできないこと |
|---|---|---|
| U01 | 旧`paper_d6`名称と同梱高次PF原稿F6の版・出典・符号／PR原論文archiveのroundingを一次資料で追跡 | AX-1対応監査。未確認のPF式をPR原論文保証として利用不可 |
| U02 | 別saved event/basis metadataの所在、fingerprint、allowlist。なければ欠測確定 | AX-1開始前〜coverage監査。exact transition会計を埋めない |
| U03 | positive DF tolerance、norm allowance、actual rank metadata、同じtargetの保存規則 | AX-2生成前。H6/H8 targetを生成・採用不可 |
| U04 | J_q/J_R、direct quota、強いbaseline集合、長時間/geometry診断の有無 | AX-2 profile後、main参照前。大規模候補評価不可 |
| U05/U06 | primitive sector保存性、reference/平均作用の数値u、overflow/phase検証 | AX-2 correctness。H6/H8精度適格性・.0001不可 |
| U07 | 割当core/RAM/GPU/時間/disk、checkpoint上限 | AX-1はsaved解析の上限、AX-2はscience上限。予算なしのrun不可 |
| U08 | 高次PF・対称controlの接続と小規模作用一致 | AX-2。強い決定論法に対する優位claim不可 |
| U09 | operational axis-bias/shot predictorの入力・保証/経験性・取得費用 | AX-1で可否判定、AX-3予測前freeze。truth-free総費用/eligibility claim不可 |

各未知量の確定は別成果物で記録し、本書や既存契約を黙って更新しない。negative resultは正常な終了であり、GOはモデル性能やPR優位ではなく、correctness・比較可能性・情報価値・登録予算を基準にする。
