# 研究C：RC-DF型L2の強度・停止条件の技術診断

2026-11-01 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。
起点：335a7e04951e009bd6910afaecd1a3107e0d61a4。

**現在のRC2は有効な固定pilotとして保全するが、十分に最適化された強いRC-DF対照とは認定しない。** α=.1は初期penalty値を揃えた政策で、勾配の効き方・最終誤差・最適強度を揃えたものではない。L2がほぼ変化しなかった主因は、保存証拠だけでは強度・停止・表現構造に分離できない。次は追加fitではなく、同じ初期点とRC2／PR保存終点の**0-fit・14-call一次診断**を推奨する。

比較政策は**B4：共通の物理DF予算をadmission条件とする比較**を優先し、その内側で強度規則と両armの一次最適化精度を事前固定する。予算制約を直接解く新optimizerや、予算内で最適なρを得たとの主張は行わない。現在のB1比較をB4で最適化済みと改称しない。

今回の実H4 decode・新規再解析・fit・PF/RTE・ground・native・shots・科学callはすべて0。実施した計算は人工行列／人工二次関数とmockだけである。重要GPT科学レビュー、独立新規性、論文化GO、研究全体の正式終了判断は対象外。共通運用指示2026-10-25版§8.1/8.2/8.4/10/10.1/10.3をworkspace原本で確認し、承認証明・保存承認文・新grantは作成しない。

## 1. 固定条件と、読み取った保存値

[Stage A実行報告](research_c_regularization_comparator_execution_20261101.md)、[事前設計](research_c_regularization_comparator_design_20261101.md)、[旧PR Stage A](research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md)を参照。RC2実行SOURCE150f904f6975868abe09966dda8ab41722e2a7cc、元PR科学SOURCEe1085c95f5c01b68566e17d543ace5832cb94061。今回これらのfitを再起動していない。

対象は保存直鎖H4、隣接1.00 Å、STO-3G、Nα=Nβ=2・sector36、通常number CDF rank8、一般実対称J・実直交U・元frame順・元g補正κ・scalar一回。同じqualified J/U/x0、LD2のtail規約、ffsim0.0.83・修正real Schur codec・joint L-BFGS-B/options=None。Stage AにはPFの時間・δは非適用、PFの数値を評価していない。

以下は**保存JSON／報告の値の引用**である。終了配列から新しいnorm、gradient、固有値、trajectoryを計算していない。

| 指標 | γ0 | RC2 α=.1 | PR-tail γ=.1 |
|---|---:|---:|---:|
| Exact tail λ [Ha] | 2.2097618367788057 | 2.209761871954849 | 1.4424298855756243 |
| Tensor Frobenius | 5.607958698344648e-9 | 4.942278943464885e-9 | 6.285301032493658e-9 |
| Sector operator norm [Ha] | 7.018855038843621e-9 | 5.8389056735133376e-9 | 9.039161828383144e-9 |
| Engineering guard [Ha] | 3.278781198621627e-9 | 3.2787811814051313e-9 | 3.2564531608037766e-9 |
| 保存DF予算CA/8=.00019875 Ha | PASS | PASS | PASS |
| Optimizer success / nit | true / 97 | true / 97 | true / 1129 |
| nfev=njev | 107 | 106 | 1228 |
| 終了gradient L2 | 3.104515097647238 | 1.5563974379521772 | 4.673093479443538 |
| 終了gradient∞ | .7423484410886623 | .4222446600957083 | 1.4825504139166679 |
| 終了combined objective | 7.200029332968556e-9 | .10000199116572639 | .04262376967322518 |
| 停止理由 | relative reduction | relative reduction | relative reduction |

RC2の[P初期値・終了値](../../artifacts/research_c_regularization_comparator/2026-11-01/run01/fit/final_diagnostic.json)は4.410884790420646→4.410972371783312 Ha²。初期tensor objective2.1839632053132894e-9→終了1.2213060585293254e-17、combined1.1→.10000199116572639である。従ってcombinedの改善とPの微増は矛盾しない。L2単独の単調減少を解く問題ではない。γ0とRC2のnitが同じでも、同じ軌跡・同じ停止機構の詳細・penalty無効を示さない。

RC2の初期combined gradient∞9754.561788218592も保存されているが、lossとpenaltyの内訳は未保存。既存値から「正則化gradientが小さかった」とは判定できない。旧run01は17 units消費済み、変更・残枠再利用はしない。

## 2. 目的関数・単位・勾配

既存[RC2](../../src/trottertracks/c_rcdf_comparator.py)と[PR目的](../../src/trottertracks/c_pr_aware_df.py)は、ffsimの同じlossと解析ADを使う。

\[
g_{DF,pqrs}=\sum_{tij}U^t_{pi}U^t_{qi}J^t_{ij}U^t_{rj}U^t_{sj},\quad
f=\tfrac12\|g-g_{DF}\|_F^2,\quad
F_C=f/f_*+\alpha P/P_0,\quad P=\sum_{tij}(J^t_{ij})^2.
\]

f*=2.1839632053132894e-9 Ha²、P0=4.410884790420646 Ha²、α=.1。
F_Cとraw f+ρPは正の定数f*倍で同じ最小点を持ち、ρ=αf*/P0=4.951304123962429e-11。**fとPはともにHa²、ρ・α・Fは無次元**。Raw係数の小ささだけで弱いとは判定しない。目的全体の正定数倍は、数値solverの停止経路まで不変にするものではない。

空間Z_RCDF=J、spin-lift K=J/2を混同しない。Pはdiag一回・off-diagonal二回。既存ffsimのpackは回転48→J上三角80の128変数で、独立J parameterのweighted penalty gradientは

\[
\partial_{J_{ii}}(\alpha P/P_0)=2\alpha J_{ii}/P_0,\qquad
\partial_{J_{ij}}(\alpha P/P_0)=4\alpha J_{ij}/P_0\ (i<j).
\]

U座標への直接penalty gradientは0。ただしUがtensor lossの地形を変えるため、joint解で間接的にJの再配分が変わることはある。PRのpenaltyもJだけに依存し、直接U gradientは0である。

PRはF_P=f/f*+γ(Mε/M*)²、γ=.1、ε=1e-8 Ha、M*=2.2093719370722074 Ha。LD2以後の非identity Z/ZZ係数c=TJに対するMε=Σ√(c²+ε²)である。**同x0で両penalty寄与=.1**は再現可能なB1政策だが、微分・曲率・方向を等しくしない。Headを罰しないtail目的と全frame L2という対象範囲の差もあり、norm形状だけの因果ablationではない。

任意の方向dに沿う変化はdF=df/f*+αdP/P0。Lossを下げるためにPが上がる方向も受容される。終点で二つのgradientが相殺しているのか、loss側が大きいのか、L2側に残る下り方向があるのかは、保存された合成normだけでは分離できない。

## 3. PとDF表現の構造：減少余地と限界

### 3.1 同じtensorを保つ方向

Q_ti=u_ti u_ti^Tとすると、各frameでΣ_iQ_ti=I。従って

\[
J_t\mapsto J_t+a_t\mathbf1\mathbf1^T,\quad\sum_ta_t=0
\]

は全g_DFを保つ。s_t=Σ_ijJ_tijとしてa_t=−(s_t−s̄)なら、方向微分dP=2Σ_ta_ts_t=−2Σ_t(s_t−s̄)²≤0。非零なら固定tensorでL2を下げる方向が少なくとも一つある。これはrank探索や修正candidate採用ではなく数学的性質であり、**保存H4にこの変更・勾配評価を適用していない**。この方向が零でも、全てのtensor-null方向が消えたとは言えない。

同じbasisを持つ二frameならJ1→J1+D、J2→J2−Dもtensor不変。人工2/4軌道で、head=A・tail=0をhead=tail=A/2にすればtensor不変・P半減だが、tail λは0から正に増える。これは**小さいfull L2が小さいtail λを含意しない**反例である。実H4で起きた現象の再現でもRC-DF一般の無効性の証明でもない。

全tensorの保持は、固定κ/scalarで総Hamiltonianの保持を意味するが、HD/HR、PF splitting、tail law、attenuation、native価格の保持を意味しない。Frameをまたぐ再配分を合法なPF無変更最適化と呼ばない。

### 3.2 固定精度下でPが動きにくい可能性

各frameのQ_tiはFrobenius直交規格化される。固定Uのmap A_t:J_t→Σ_ijJ_tij Q_ti⊗Q_tjはisometryで、\|A_tJ_t\|F=\|J_t\|F。Frame間はrangeが重なるので、全A=[A1 … AR]には冗長性があり得る。

\[
\|g_{DF}\|_F^2\le R\sum_t\|J_t\|_F^2=RP.
\]

従って非零tensorを高精度で保ちながらPを任意に0へはできない。実H4がこの下限や固定Uの最小norm解に近いかは未評価。

対称係数をj_diag=Jii、j_off=√2 Jijとwhiteningすれば、固定Uのraw ridge条件は(AᵀA+2ρI)j=Aᵀg、特異方向の係数はσ/(σ²+2ρ)倍となる。2ρと実際のσ²の大小、targetの投影、tensor-null方向の成分によって感度は違う。**H4のSVD・Hessian・固有値を新たに計算していない**。この式は構造の説明で、新しいfitterの実装ではない。

Joint Uは非凸。列符号／列順とJの対応変換、同basisの分配、指数写像の周期性など表現冗長性があり、parameter gradient normは座標依存である。L2で動きやすい配分がtailに不利な可能性、tensor精度との交換が必要な可能性、solverが下り方向を進み切らない可能性を区別する。強度増加がDF予算内で有効なλ改善を生むとの保証はない。

## 4. 実際のffsim／SciPyの停止

[修正codec接続](../../src/trottertracks/c1_ffsim_corrected_codec.py)は元ffsim fitterを使い、SciPyへmethod='L-BFGS-B', jac=True, callback=None, options=Noneを渡す。**boundsなし、SciPy tolなし**。ffsimのtol=1e-8は初期分解設定であり、このfitのftol/gtolを1e-8にしない。目的のADは既存ffsim/JAX x64、有限差分gradientではない。

実環境SciPy1.14.1の`_minimize_lbfgsb`で既定値を確認した。

| 設定 | 既定値・意味 |
|---|---|
| ftol | 2.220446049250313e-9、factr=1e7×binary64 eps |
| gtol | 1e-5、projected gradient∞の基準 |
| maxcor / maxls | 10 / 20、limited-memory履歴／line search回数 |
| maxiter / maxfun | 15000 / 15000、保存fitはこれらの上限停止ではない |
| bounds | Noneなのでprojected gradientはこの問題では通常のparameter gradientと一致 |
| OptimizeResult.jac | 最終combined objectiveのgradient。Pだけのgradient、勾配ノルム、Hessianではない |

減少率条件は(F_k−F_{k+1})/max(|F_k|,|F_{k+1}|,1)≤ftol。Gradient条件とは独立に停止し得る。Line searchや有限精度で改善が小さくなってもsuccessが返る場合があり、保存messageは相対減少停止だった。H4の各iterationのloss/penalty内訳、最後のstep、line-search trialは保存されていないため、97という回数の詳細原因は復元不能。[SciPy1.14.1仕様](https://docs.scipy.org/doc/scipy-1.14.1/reference/optimize.minimize-lbfgsb.html)、[同版SOURCE](https://github.com/scipy/scipy/blob/v1.14.1/scipy/optimize/_lbfgsb_py.py)

保存された3候補のgradient∞はすべて1e-5を超える。これは**既定gradient基準の達成証拠がない**という具体的限界であり、数学的なstationarity不存在や将来の改善量を確定するものではない。Optimizer success、一次条件、大域最適、P減少、DF予算は別命題。

**定数寄与による単純説明を採用しない。** 正の大きな定数Cは分母がCに支配される領域で停止を早め得る。しかし今回のRC2終点F≈.100002、PR終点≈.042624、γ0終点≈7.2e-9はいずれも1未満。非負Fの相対減少停止を成立させる最後の対では、直前値が1以上からこれらの値へ大幅減少したとは考えられず、分母のfloor1が支配する。RC2の約.1を単に定数として引いても、そこで減少率の閾値は緩まない。そもそもPは厳密な定数でもない。微分・経路・丸めへの影響は別に未判定である。

人工二次関数q_C(x)=C+.5 x₀²+.5×10⁻⁴ x₁²、x初期=(1,1)を同SciPy設定で解いた。

| 人工C | nit | gradient∞ | 停止 |
|---:|---:|---:|---|
| 0 | 3 | 2.334972257076045e-8 | projected gradient |
| .1 | 3 | 2.334972257076045e-8 | projected gradient |
| 1e9 | 1 | 9.999000000005e-5 | relative reduction、success=true |

[人工記録](../../artifacts/research_c_regularization_first_order/2026-11-01/technical/synthetic_stop.json)。大きいCの例を実H4の因果説明へ転用しない。またftol=0でも減少0なら0≤0で条件成立するため、「相対停止を無効化して必ずgradient収束」の修正ではない。単にtolを締めた片arm再fitを強い公平比較へ昇格させない。

## 5. 強度・停止の比較政策

| 政策 | 長所 | 今回揃わないもの／注意 |
|---|---|---|
| B1 初期penalty寄与一致 | 同x0で結果非依存・安価・再現可能。現pilotは有効 | gradient、曲率、最終DF誤差、最適ρは一致しない |
| B2 初期gradient寄与一致 | 初期の実際の摂動を診断できる | J/U尺度・座標と方向・曲率に依存、U直接gradient0。Norm一致だけで公平全体を確定しない |
| B3 最終DF誤差一致 | 特定の誤差水準で比較できる | 事前保証不能、fit/選択が増える。結果後のρ選択にはselection bias、完全一致は共通CAの必要条件ではない |
| **B4 共通物理予算admission** | **最終精度要件の下で追加価値を問う研究目的に適合** | 現ffsimはsector制約を直接解かず、Pareto最適や予算使い切りは保証しない |

**一つ選ぶならB4を優先する。** κ/scalar・rank/class・元順LD2・同x0・DF予算/guardを共通にし、強度と停止の選定規則を結果前に宣言する。同じCAに対してDF誤差の完全一致までは求めない。強い対照の比較には、両armの共通座標での一次条件も確認する。1e-5は既存SciPy既定gtolというengineering指標に限定し、厳密stationarity／大域最適の証明に使わない。

現段階では新ρを選ばない。α=.1／γ=.1を変更せず、まず一次成分を診断する。B2を係数の自動再設定ルールにせず、B4の下で次の実験条件を選ぶ材料にする。必要なら後で共通の最適化精度確認を新契約にするが、L2側だけ追加最適化して旧PR側を十分収束済みと仮定しない。

次の3種類は異なる比較である。(i) 同じoptions=Noneの固定pilot、(ii) L2終点依存性の一arm診断、(iii) 両armの一次最適化精度を共通化した比較。(ii)は原因調査として意味があっても、(iii)を代替しない。多数ρ grid／反復STOP変更は提案しない。

## 6. 選定した次の最小科学実験：fit0、一次診断14 units

[未承認契約](research_c_regularization_first_order_contract_unapproved_20261101.json)と[独立entry](../../scripts/run_research_c_regularization_first_order.py)を準備した。今回このentryは**passive preflightだけ**を実施し、run01を作成していない。追加fitを決める前に、初期の強度差・保存終点の残存gradient・tensor-null方向を切り分ける。保存合成normだけではできないため、この限定再解析には別途明示科学実行許可が必要。

| 順序 | 内容 | 最大units |
|---|---|---:|
| 1 | 同じg、初期x0/J/U、RC2終点、PR終点の読込 | 1 |
| 2 | 3点の既存unpackと保存J/Uの忠実性、再logmなし | 3 |
| 3 | 各点で既存base／RC2／PRのvalue+gradientを各一回 | 9 |
| 4 | gradient差分・U/J block・競合方向・限定null方向の一次診断 | 1 |
| 合計 | **新fit／optimizer iteration／独立tensor誤差・sector・PF評価0** | **14** |

三つの目的を全点で評価するのは、初期の寄与一致と、同じ保存geometryに対する両penalty方向を分離するため。異なる終点同士の合成normだけを比べない。Cross-evaluationは新candidate生成や再最適化ではない。既存ffsim loss/AD内部のtensor再構成は9評価に含まれるが、独立Frobenius/max-entry誤差やsector normは再評価しない。γ0終点、Hessian、SVD、有限摂動、探索、歴史的trajectory推定は対象外。

取得するもの：各value/gradient、lossとweighted penaltyの差分、各U/J block norm、内積・cosine（零normではnull）、penalty/loss norm比、共通gtolのengineering判定、Σa_t=0のidentity移送方向のnormと方向微分。方向が零なら明記し、全null空間についての結論は出さない。DF予算は既存PASSを引用し、再計算・新admissionはしない。Raw combined objectiveの値を目的間の性能順位に使わない。

初期RC2寄与がPRより小さいことが判明しても、それだけで強度不足を証明しない。合成gradientが大きく、下降成分が残るなら共通停止精度の検証を優先する材料になる。成分が拮抗し物理精度との交換が示唆されるならB4の強度政策を検討する。ただし局所一次診断から改善量、最適ρ、強いRC-DFの到達性能は予測しない。判断不十分ならUNRESOLVEDのまま研究Cへ戻す。**結果に応じた自動fit／ρ選択は0。**

固定g/点のhash、元fit identityと新評価identityは契約・SOURCEに独立記載。Fresh outputはartifacts/research_c_regularization_first_order/2026-11-01/run01、retry/resume0、初回correctness FAILで後続停止・部分証拠保全、完了／FAILの両方でmandatory STOP。0-fitなのでsolver／停止policy変更はなし。次のfitが必要ならさらに独立した契約と許可が必要で、旧17-callの残枠を使わない。

固定実装module/entryのcommitはa41329d7e860415ddfc21df5b02728804846d16d。[SOURCE39](../../artifacts/research_c_regularization_first_order/2026-11-01/technical/SOURCE.json) SHA256は328a84c93b89e1cf02fd23784fa53f0435b857ce490c6996713b5fd7416c47fd、契約SHA256はb7820771dbb67de704348867745b268e60b5787a27c109961ae5e6a1102f727b。Source39には旧36件と新module/entry/test3件を固定し、test末尾の空行整理は最終公開treeのhashで記録する。入力9件はcontract内hashで固定した。

今回実行したpassive command（科学call0）：

~~~bash
PYTHONDONTWRITEBYTECODE=1 \
 /tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python -B \
 scripts/run_research_c_regularization_first_order.py \
 --contract docs/research/research_c_regularization_first_order_contract_unapproved_20261101.json
~~~

明示許可後の同entryには上記に`--execute --output artifacts/research_c_regularization_first_order/2026-11-01/run01`を追加する。今回は実行していない。環境と空き資源の再確認、SOURCE/input preflight、公開照合とmandatory STOPを実行指示の一部として維持する。

## 7. 最小実装・人工検証・保全

[c_rcdf_first_order.py](../../src/trottertracks/c_rcdf_first_order.py)は既存目的を呼ぶだけの保存点診断で、loss／fitter／codec／optimizerを再実装していない。既存ffsim pack/unpack、PR/RC2 value/gradient、io、Calls、finishを再利用。新規部分は勾配成分の集計・固定14-callの別entryのみ。旧fit bridgeのoptions=Noneを変更しなかった。診断にfitは必要ないため、停止policyを上書きできる新adapterも追加していない。

[人工試験](../../tests/management/test_research_c_regularization_first_order.py)で2/4軌道・det±1/π・実ffsim AD、対称J gradient係数、強度linear scaling、tensor-nullとλ反例、人工SciPy停止、passive import、fresh output・STOP・部分証拠・no optimizerを確認。人工E2EではH4形状と同じrank8/norb4の**乱数で作ったtensor**も用いるが、実分子データは一切読まない。人工テストのPASSはH4の科学的判定ではない。

テスト結果・固定SOURCE/input・byte preflight・原本保全の照合記録は[technical](../../artifacts/research_c_regularization_first_order/2026-11-01/technical/)に保存する。新試験13＋既存RC2回帰22＝35 PASS、既存codec/PR回帰88 PASS、合計123 PASS。科学結果の再計算ではなく、ローカル人工試験でありCI／外部独立再現とは称さない。

確認時MemAvailable約53.7GB・load約1.1、既存JAX CPU cached ADを使用。人工はtiny matrixで、BLASだけ自process内1 threadにしてJAX poolと競合を避ける。科学callや対象を並列に増やさず、任意のCPU/RAM/worker/wall上限は追加しない。Task-local cache/tmpのみ作成し、既存venv・共有設定・他jobは変更しない。将来の9小規模value/gradientはcompile/cacheを再利用する方が適し、process増殖による起動・重複compileは不要。

旧SOURCE36・旧run01・元Stage A・契約・STOPを含む115ファイルのGit blob一致を確認した。起点に存在したファイルの変更はPROJECT_MAP/docs README/scripts READMEの索引追記のみで、旧H2/H4/H6の数値SOURCE／raw／科学契約／STOP変更は0。公開時にはremote SHA/tree・公開本文も照合する。旧証拠inventoryを書き換えて新結果を混ぜない。

## 8. Stage Bへの価値と接続

**固定pilotの下流情報としての価値はあるが、今回は一次診断を先行する。** 現在の候補を使ってPF/RTEを測れば「初期寄与一致という固定政策で得たJ/U」の性能差を知れる。ただし強度の最適性・両armの十分な最適化を確認する作業にはならず、強いRC-DF baseline比較へ昇格させられない。

後続は既存[c_pr_aware_df_stage_b.py](../../src/trottertracks/c_pr_aware_df_stage_b.py)のprepare_arm／candidate_factor／atom_operators／tail_gate／kernels、CAR/general-K、M3/M4の13round評価・RPE接続を再利用可能。RC2の保存J/U・元h/g/cで表現一致→元順LD2 tail law→own-DF ground/alias→PFと補正RTE・attenuation→phase十分条件、の順になる。新RC2固有のground・誤差が必要で、PRの値を移送しない。

比較案は保存PR M4と同じδ=.1、m0..12、partial S2 s1/r128/K0・LD2・own-oracle/状態準備除外。これは**将来案**で、今回この条件を採択／実行したわけではない。両armのDF予算とphaseを通過した条件で初めてshots側の比較材料が揃い、native価格取得はさらに別段階。λだけで下流資源を決めない。今回Stage B/Cのentry・科学契約を増やしていない。

## 9. Q1〜Q10と次の判断

| 問い | 回答 |
|---|---|
| Q1 Pがほぼ減らない理由 | Combinedはtensor loss大幅減少を優先でき、P減少自体は停止条件でない。具体的な主因の分離は未了。 |
| Q2 α=.1の公平性 | 同x0の初期penalty寄与一致。Gradient／曲率／最終誤差／最適ρは揃えない。 |
| Q3 強度・停止・構造 | 三者とも候補。停止messageとgradient未達は確認済み、構造上の制約と非同値性は数学／人工で確認。実H4の寄与分解は未判定。 |
| Q4 固定pilotとして有効か | YES。事前登録した17-callの結果を保持。 |
| Q5 強い十分最適化RC-DF対照か | 未確立。原論文交互solver全体の比較でもなく、PR側も一次基準達成未確認。 |
| Q6 政策 | B4共通物理DF予算を優先。結果前の強度規則と両armの共通一次精度を別に規定し、完全誤差一致は不要。 |
| Q7 最小新計算 | 原因分離には保存3点の14-call・fit0一次診断を推奨。追加fitはまだ選定しない。 |
| Q8 Stage Bの価値 | 固定pilotの性能測定にはYES。強いbaseline資格の代替にはならず、今回は診断先行。 |
| Q9 技術準備 | 独立診断SOURCE、人工E2E/回帰、passive preflight、未承認14-call契約を用意。実H4計算0。 |
| Q10 次の担当・許可 | ユーザー／研究CがB4方針と14-call限定再解析の実行可否を判断。Codexは明示許可後だけ同契約を一度実行・公開・STOP。重要レビュー／新fit／B/Cは別判断。 |

**今回の完了後はmandatory STOP。** 新科学計算・新fit・Stage B/C・重要GPT科学レビューを自動開始しない。
