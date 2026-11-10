# H6 DF19 RPE D2：4round・8候補の実行準備

2026-11-10 JST。**計画のみ。添付D2指示§5・§9に従い、新reference／finite mean／bias／phase／radius／N／Gは未計算。科学実行は別途明示許可待ち、公開照合後mandatory STOP。** 候補とcall範囲は[PLAN.json](PLAN.json)、本段階の静的確認は[CHECKS.txt](CHECKS.txt)。これは承認証明書やgrantではない。

## 推奨する最小menu

[D1完了結果][D1]の条件をそのまま再利用する。Linear H6、隣接1.00Å、STO-3G、保存tol-only 1e−8／DF19／weighted-Hermitian completion、12 mode・Nα=Nβ=3のsector400、保存数値state、P=0。targetは保存DF-Hの同sector ground energy、CA=.00159360010199040 Ha。厳密ground同定・実装/law誤差の未資格事項を維持する。

保存prior `[−9.695399206821989,−3.1355322139814668] Ha`、h=.1、βtotal/stat=.4/.2 rad、M12・13round、各方式αtotal=.05・各軸α=.05/26、無energy shiftとD1のalias判定を転記する。prior／horizon／h・α・βを再評価・最適化しない。

各時点PF/PR各1構成、**計8候補**を提案する。全て既存`4th(m5_best)`の同じsigned係数・fragment順。候補IDは既存`h46_long_time_signal.cid`のtuple順と`json.dumps`を変更せず算出したmetadataであり、適格性の証拠ではない。

| m / T | q_RPE | q_PF / δ | Full19 PF ID | PR prefix / R_outer / K | PR ID |
|---|---:|---|---|---|---|
| 0 / .1 | 1 | 1 / .1 | `dec32b474f3fc33ddfab` | 5 / 32 / 2 | `f2b156ebaeb00c2c033b` |
| 1 / .2 | 2 | 1 / .2 | `7791db0dc2fab94ce734` | 5 / 32 / 2 | `9fe878b5ed57560be583` |
| 2 / .4 | 4 | 1 / .4 | `e7200ae9e16ba879a78d` | 5 / 32 / 2 | `307350517bc4a1b4f9ff` |
| 5 / 3.2 | 32 | **2 / 1.6** | `3f813a46e17461bfe484` | 7 / 16 / 2 | `72255c43facdfb8b4bee` |

短時間3点のPRはD1 T=.8のprefix5/R32/K2を継承する。m5はD1 T1.6のFull19／PR7/R16/K2を**q_PF=2、δ=1.6**で延長する。新しいδ=3.2や広いq/R/K探索を増やさず、長時間への接続を検証する選択である。PFとPRを同じq_PF/δで比較する。q_PF2とq_RPE32を混同しない。

R_outerは各outerの全11 signed pieceへ`ABS_TIME_PER_OUTER`で整数配分し、各piece最低1 stepを保つ。全interrogationのevent数はq_PF R_outer。K2は既存paired numeratorの3次までを意味する。負時間piece、tail identity、scalarを保存規約から変更しない。このmenuのPASSを先取りせず、FAIL時にもq1/q4、別prefix/R/Kを自動追加しない。

## 保存値で対応できる部分と不足

| 入力 | 再利用／新規範囲 |
|---|---|
| H/state/snapshot/DF receipt | D1と同一。H `223628b0…`、snapshot `99440a59…`、state `664c6335…`。完全なIDはPLANのscientific_binding。新H/DF/state生成・ground solveなし |
| prior・RPE政策・D1 m3/4 | `45607a5086721ea1d21b1ed2d7edfe870549f0c8`のresult.jsonをそのまま読み、再計算しない。D1のSOURCE `4c928b90…`と結果公開commitを区別 |
| tail scalar／lambda_R | 保存`H6_context.json`のprefix5/7を再利用。p5: scalar .09856554679631149、lambda_R .9182185130575018。p7: .0194666612606853／.1649333839054104。context SHA `61d87e2fe7e03892f7db6a075a891d9b83af834832b083ec3de3b75e8a7e9d38`とD1 H/state bindingを照合済み |
| **T=.4 full-H reference** | 旧[固定δ結果][FIXED]に保存値がある。`T=.4, cell.id=B1_p19_q2_2nd`を選択、他4 rowのreferenceも同一。値と元empirical allowanceをPLANへ転記。旧readiness metadataのH/state/snapshot/DF receiptがD1と一致し、旧sourceの`reference_pair`由来を確認。候補が異なってもfull-H/state/time referenceは共有できる |
| T=.1/.2/3.2 reference | 調査した保存系列に一致なし。**3時点だけ新しい保存DF-H actionが必要**。T=.4/.8/1.6のfull-H referenceを再evolveしない |
| 8候補のPF/PR finite mean | 同一candidateの保存値を確認できず、8件が新規対象。T=.4には旧S2/Yoshida・異なるq/Rのsignalがあるが、今回のm5/q1 signalとして代入しない |
| compiled価格 | 調査した長時間studyのsignal／探索価格／再利用価格と旧固定δ系列では**今回の8候補に一致するCは未取得**。旧T=.4の別公式・qの価格も代入不可。CとGはnullのまま、追加価格処理なし |

contextの所在は固定SOURCE索引`7a8bc7f132e166390c943e6b32c92cc6abe1baf4`、`artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json`の`T20_reference_files/H6`から解決する。保存snapshot/receiptは同索引の`saved_inputs/H6`。内部絶対pathやraw eventを今回の成果へ公開しない。これらの索引・原本・過去STOPを書き換えない。

## 既存pipelineの再利用と必要な小変更

再利用SOURCEは本branchのD1結果commit `45607a5086721ea1d21b1ed2d7edfe870549f0c8`。[長時間study][LONG]の次のkernelは同commitに存在し、`scripts/h46_long_time_signal.py`は元carrier `155529e2…`と同じbytesである。

| 既存コード | D2での使用方法 |
|---|---|
| `ax2b_h6_saved_completion_loader_v1.load_h6_snapshot` | 保存completion・全binding・state bridgeを保持して読み込む。生成契約や旧controllerは起動しない |
| `h46_new2_core.occupation_one_body` | 保存spin-sector順のdΓ行列。sector外の非零遷移を拒否する既存実装を利用 |
| `h46_m5_plan.weights/allocation` | 既存m5 signed順序・ABS_TIME_PER_OUTER配分。別の係数表を新造しない |
| `h46_long_time_signal.evolve_batch` | deterministic half-step順序、exact-tail control、corrected/raw別伝播・closure・scalar一回の規約を再利用する。下記の作用backendだけを薄く置換する必要がある |
| `rte.finite_taylor_operator/finite_rte_distribution` | paired K+1次numeratorとBを既存関数で計算。乱数・trajectoryは生成しない |
| `run_pf_common_run.qualify_record`のallowance部分 | norm/closure・stage数によるu/margin政策を再利用。**有限時間のepsilon走査・alpha_axis=.025・shot算定は呼ばない** |
| `h4_df10_rpe_connection.bind_saved_round` | expected_system/rank/bindingをH6へ明示。D1専用H6 ID guardは変更せず、新しい入力層から共通binderへ接続 |
| `h4_df10_rpe_d1.analyze_bound_point` | D1と同じreference診断・disk・normalization・phase/半径・既存Hoeffding。価格なしの`prices=[]`で使用できる。独立したRPE solverは不要 |

**旧一括CLI `h46_long_time_signal.py signal`、`context()`、`run_signal()`をD2へ直接起動しない。** 旧全格子／T1.2・2.0／過去の出力先を扱い、`spectrum()`および`matrix_only_tail_coefficients()`から新しい`eigh`も行うためである。D1の固定4 ID guardを緩めて新候補を通すこともしない。

新しい固有値問題を避ける最小経路は次のとおり。これは**実装すべき薄いaction adapterの仕様**であり、今回D2の科学runnerを実装・実行したという意味ではない。

1. 保存係数を既存ladder routineでsector作用行列へ表す。`A0=dΓ(h)`、`A_l=λ_l dΓ(G_l)^2`、`H_sector=cI+ΣA`。これは保存Hの作用表現であり、SCF/DFや新しい物理Hamiltonianの生成ではない。400次元で扱い、full Fock行列を作らない。
2. 新reference3時点だけ`expm_multiply(−iTH_sector,state)`、同じ3時点を`scipy.linalg.expm`で別経路照合する。固有値・固有vectorを求めず、保存E_stateからreferenceを生成しない。両経路の差は数値診断であり厳密ground証明ではない。
3. deterministic blockの`exp(−itA/2)`を`expm`で得て共通半stepをcacheする。PRでは`F_p=Σ_tail A_l−s_p I`、`L_p=F_p/lambda_R`に、既存`finite_taylor_operator(L_p,lambda_R t/r,K)`をr回作用させる。raw列は各short-stepでB除算、corrected列と別伝播する。exact-tail controlは同じF_pの行列指数。最後にPFはc、PRはc+s_pのscalar位相を全時間で一回だけ適用する。
4. `evolve_batch`の出力schemaへreference/corrected/raw/total_signed_bias/log_B/allowance_evidenceを接続する。8 candidate出力、PR4件にはexact-tail内部対照を持たせる。raw=corrected/Bを事後代入して独立closure診断を消さない。
5. D1と同じscalar helperへ渡す前に、入力層・作用backendの差分だけを2-level人工系で検証する。非可換ブロック、負時間、scalar一回、K2/normalized polynomial、raw別伝播、q_PF対q_RPE、H6 bindingを対象とする。既存solverテストの全面再実装や実H6での事前試行は行わない。

D2の実装・人工確認を固定したcommitを将来の科学SOURCEとして記録する。本書の再利用SOURCEや計画公開commitを、未実行の科学SOURCEと偽らない。既存D1 SOURCE・helperを変更する必要はない。ユーザーが本menuとcall範囲を許可すれば、その範囲内の通常実装・人工テスト・debugと科学評価を一括して進める計画とし、承認証明書や中間の再承認を増やさない。

入力adapterは新recordのsignalとu/marginを既存helperの`saved_signal`／`saved_allowance`／`saved_finite_time_endpoint.axes.*.allowance`欄へ整形し、`origin=NEW_D2_FINITE_MEAN`と新結果pathを別欄に記録する。旧有限時間endpointを評価したことにはしない。`candidate/binding/compiler/prices=[]`を付し、共通binderへH6の期待binding・m・hを明示してから`analyze_bound_point`へ渡す。D1結果ファイルの値・ID・target IDは書き換えない。

## 固定する数値政策・判定・call範囲

allowanceは共通PF政策のfloor1e−9、倍率1を維持する。stage数は既存の`q_PF×2(prefix+1)×11`。closure・最大中間normに基づく旧roundoff式を再利用し、新referenceの別経路差を非負guardとして加える。T=.4はPLANの元reference allowanceを縮小せず下限として保持する。log_B marginも旧式のfloor1e−12／stage依存値と保存値のmax。経験的allowanceであることを維持し、観測誤差をrigorousな上界へ格上げしない。

具体的には`roundoff=64×2^−52×(1+maximum_norm)×(1+stages)`、新reference時点の`u_a=max(1e−9,10×closure+roundoff,old_u_a)+reference_pair_difference`とする。T=.4はpairを再評価せず、元のempirical allowanceを`old_u_a`へ保持してmax部分だけを用いる。`log_B_margin=max(1e−12,64×2^−52×(1+|log_B|)×(1+stages),old_margin)`。これは未知の科学誤差を0にする規約ではなく、既存の数値pilotへの追加guardである。

D1と同じ`hypot(|bias_X|+u_X,|bias_Y|+u_Y)`にreference/state位相差とT×保存residualを加え、margin/closure込みraw diskを接続する。E_stateは診断専用でprior/branch選択に使わない。新roundにもD1と同じtask ID `H6_DF19_GROUND_CA_CONDITIONAL_D1`を保持し、別欄のstageをD2とする。歴史的`binding.T=.8`は入力reference時刻であり、実発展時間はcandidate.Tとm/hで明示する。

| 承認後の科学call単位 | 固定範囲 |
|---|---|
| 新full-H reference時点 | .1/.2/3.2の3時点、2経路＝6 state action。保存.4は読取のみ |
| PF/PR candidate finite mean | 上記8件。PR4件のexact-tail内部対照を含む。保存D1候補の再評価0 |
| phase/radius/disk/N | 8件×ideal/actual lane。X/Y別N。actualの未取得implementation/law距離はNone |
| prior/horizon/新H・DF・state/eigensolve | 全て0 |
| sampling/build/compile/価格fit/G/RPE成功率simulation | 全て0 |

この表の件数は科学対象の範囲でありCPU/workerの独自capではない。共有context・半step cacheと時間ごとの独立jobを利用できる。BLAS並列・process数は実空き資源とtaskの実測に合わせ、他worktreeや共有設定を変更しない。

判定と停止条件：

- SOURCE/input/H/state/fragment順・saved tail context・時刻bindingの不一致は当該campaignを停止し、結果を保全する。参照元の古いgrant・ledger・markerを再使用・再発行しない。
- reference別経路差>1e−10、exact PF norm差>1e−9、raw/corrected vector closure>1e−9、非finiteは既存数値gateとして不成立を記録する。同じ時刻のreferenceが不適格なら両方式を未資格とする。thresholdを緩めず、自動retry・backend変更をしない。
- 正raw半径、βdet+βstat≤.4、既存branch条件を満たしたときだけ条件付きNを採用する。不成立候補はN=nullで残し、有限menu内の他候補は独立に報告できる。actual未知量を0に置かない。
- 初期branchは保存priorと無shift政策、角度の符号はφ=−TEから確認する。数値E_stateへbranchを合わせる補正や、欠けた全roundを含むRPE成功率simulationを行わない。
- 全4roundが両方式でPASSでも、D1と合わせて最大m0..5の6roundまで。m6..12（T6.4..409.6）のsignal/N/C、全体phase/branch/failure条件と全Gは未確定。αは全13round分を維持し、評価した6roundへ再配分しない。

## 価格担当との接続・次段階

価格への結合keyは`candidate_id, system, H/state/snapshot, completion/rank_policy, formula/signed-order, prefix, q_PF, δ, R_outer/allocation, K/law, T, m/q_RPE/h, compiler, control, full_Hadamard_wrapper_with_measurement, axis, P=0`。compiler期待値はD1と同じ`1239304c…`、Qiskit1.3.0／rz,sx,x,cx／opt1／seed17／symmetric_directional。これは将来のCを照合する条件であり、今回の8候補をcompile済みという意味ではない。

H4側の長時間価格研究には介入せず、方法の公開後にH6 domainへの資格を判断する。今回の新候補にはC未取得なので、別時刻価格の角度非依存性や保存C×q_RPEを仮定しない。モデルが必要なのは同一H6条件で直接cacheがなく、T/δ/q_PF/prefix/R/K/law等を変更する場合。追加compileが必要かは、H6で適格なモデル・native計数・保存cacheで閉じる範囲を先に検討してから決める。

**次の最小範囲**は本menu8件のD2科学実行のみ。完了後、成立したm5構成を出発点としてm6/T6.4を次の長時間判別点にする案を検討できるが、qや候補を未評価のまま固定したことにはしない。m6..12の一括実行、価格fit・D3／全G、詳細科学レビュー、Main Claim変更へ自動進行しない。

本段階は既存branch `codex/resource-eval-h6-df19-rpe-preparation-20261110`を継続。旧SOURCE/input/raw/結果/KEEP/STOP・H4作業・共有環境は保持する。新規科学moduleをimportせず、既存blob／JSON欄とmetadataだけを確認した。本担当自身の静的照合であり独立科学レビューではない。

公開remote SHA・主要成果物の全文一致を確認後**mandatory STOP**。次の担当は**ユーザー**：上記8候補・4round・call範囲の科学実行を判断する。**Codex**：別途許可後に薄いaction adapterと必要な人工確認を仕上げ、許可されたD2範囲だけを評価する。

[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/45607a5086721ea1d21b1ed2d7edfe870549f0c8/docs/research/h6_df19_rpe_d1_results_20261110/REPORT.md
[LONG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[FIXED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h6_fixed_delta_result_v1/2026-10-15/signal_summary_v1.json
