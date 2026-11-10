# H6 DF19：RPEエネルギー推定の接続準備

2026-11-10 JST。**静的調査・保存欄の転記・接続設計まで。新signal／bias／N／G、最適化、sampling、回路build/compile、RPE simulation、詳細科学レビューは0。公開照合後mandatory STOP。**

H6の保存Hamiltonian/state/referenceと、有限時間PF/PRのsignal・直接compiled価格は再利用できる。H4担当のadapterも共通算術を既に実装しているため、新RPE solverは不要である。ただしH4/DF10専用の入力guardはH6を拒否する。必要な変更は対象bindingの明示と薄い入力層の共通化であり、数値係数や保存Nの移植ではない。

成果物は本報告と[保存4点の入力projection](SAVED_POINTS.json)。前回の[shot調査][shot]・[価格再利用調査][price]・[価格SOURCE索引][price-index]を再利用し、H6固有の差分だけを追加した。[H4 adapter][h4-code]と[準備報告][h4-report]は調査中に固定commit `e91021ae2db0fba755ba7a6404c18e7e003c129f`でGitHubから取得可能になった。H4のbranch/worktree/sourceへ変更は加えていない。

## 1. 採択済み条件と保存入力

今回採択された条件は、Linear H6／隣接1.00Å／STO-3G、保存tol-only DF19のground energy、1 kcal/mol、αtotal=.05、既存保存state、P=0、測定付きfull Hadamard wrapperのcompiled RZ×十分N、RPEである。既存`config.CA=1.59360010199040e-3 Ha`を1 kcal/molの換算値として参照する。旧`TARGET_ERROR=CA/10`は使わない。

元の分子Hamiltonianへの誤差保証は対象外。**保存DF-H自身をtargetとするので、元Hやcompletion前のHとの差をenergy reserveとして追加しない。** 保存weighted-Hermitian completion・signed weights・fragment順・scalarを維持する。保存stateを使う条件付き数値pilotを基本案とし、厳密groundの証明や新しいstate solveを初回要件に追加しない。

| H6保存入力 | 固定値・可用性 |
|---|---|
| H／snapshot／state binding | H `223628b06c794f4a2a7a84db532e19ed60915f7e86898de66347129a5aeaffa9`、snapshot `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`、state `664c63350c2510518b0ec561f804641447d39f75e68b960192c31d37540265f3` |
| rank政策・sector | `tol_only_1e-8`、保存観測rank19、12 system qubit、Nα=Nβ=3のspin sector400。H4の8 qubit／sector36やlegacyのsector70と区別 |
| state証拠 | 保存solver energy `−3.2360662799087647 Ha`、residual `6.1060388500553216e−15`、`ground_state_certified=False`。記録の引用で、固有対を再評価していない |
| reference | 同じ保存H/stateのfull-H信号。T=.8のreference.npzと保存receipt、長時間studyのT1.2/1.6/2.0 reference欄を利用できる。scalar/global phaseを含む |
| 原本可用性 | 既知のH6 snapshot・DF receipt・snapshot receiptの3ファイルとT=.8 reference.npzを読み取り、保存SHA256と一致。NPZ配列展開・state-actionは0 |
| 再現単位 | [長時間PF/PR結果][long]の`scientific_provenance.json`、`signal_provenance.json`、`completion_provenance.json`。公開carrier `155529e20a8fac39e00786c40b57830b71203c4f`を元実行SOURCEに付け替えない |

αtotal=.05は今回の利用者指定である。[元論文][paper]の`η=1−α`のoverlap不足量や、別方式比較のfailure=.1とは別の記号・条件として扱う。全研究に共通する論文固定αを根拠にした採択ではない。

## 2. 保存signal・価格のcoverage

[長時間study][long]にはH6のT=.8/1.2/1.6/2.0、S2／標準Yoshida4／m5_best、Full prefix19、B0/B0I/PR prefix3/4/5/7/11/15がある。基本q_PFは1/2/4/8/16、T1.2では6、T2では10も含む。T1.6/2.0のq32追加はS2のみ。PRのR_outerはS2=4/8/16/32/64、Y4=8/16/32/64、m5=16/32/64、K=2/4/6である。

`R_outer`は一outerの全signed pieceへの総event配分で、`ABS_TIME_PER_OUTER`・各piece最低1を保持する。m5の負時間pieceを正時間へ置換しない。Kは最大偶数event次数で、paired numeratorはK+1次。PRのnormalization Bは全signed piece・全outerの積であり、scalar/identityは一回だけ適用する。

**q_PFはT内のPF step数、q_RPE=2^mはRPEの時間倍率。** `δ=T/q_PF`とbase time hを別欄にする。T固定のq_PF格子から、h固定でTが増えるRPEの価格・信号を補間しない。

| 保存系列 | 直接価格・modelの資格 |
|---|---|
| 旧T=.8共通PF/PR | 旧H6 D136にはmodel価格も残る。登録全域を直接compile済みとしない。長時間studyが再利用した価格も、元receipt／model区分を辿る |
| 長時間PF/PR `155529e2` | `reused_compiled_prices.json.gz`、`exploration_costs.json.gz`、q32の`extension_parallel_costs.json.gz`等に直接取得した候補。価格取得域は全signal格子の部分集合。PRの探索n9と独立確認を分離する |
| 今回の4保存点 | 下表のFull PFは直接決定論価格、PRは直接paired finite-sample価格。RZ/CX/三depth/sizeの保存欄を転記。新平均・CI・順位・総費用は算出していない |
| H6の未登録時間／構成 | この固定資料では、全RPE時間域を保証するH6固有のq価格モデルは確認できない。新H6 model qualification又は必要点の価格が別途必要。価格なしを0にしない |
| Track D／legacy／H4 | D C6〜C8のlaw・pairing・model検証の方法、legacy18-roundの会計、H4のadapterを再利用する。CDF8/DF12/DF10の係数・N・改善率をH6の価格にしない。D q64完了と旧総費用への未反映の区別も維持 |

最小接続例として次の**既存4点を転記**した。これは新しい候補最適化や費用rankingではなく、未採択の有限pilot案である。PF baselineはFull19と明示し、B0/B0Iの切捨てPFを黙って代用しない。

| 保存ID | T／formula／q_PF／prefix | R_outer/K | 価格 |
|---|---|---|---|
| `76601611270ed9f0677d` | .8／m5_best／1／Full19 | — | 保存common-nativeのdirect receiptあり |
| `2606efdab20e3b8522e5` | .8／m5_best／1／PR5 | 32/2 | EXPLORATION n9、独立CONFIRMATION n33 |
| `a4f5ffc0f320644dadfa` | 1.6／m5_best／1／Full19 | — | EXPLORATIONの決定論n1 |
| `2d2af6ca830dff14a695` | 1.6／m5_best／1／PR7 | 16/2 | EXPLORATION n9、独立CONFIRMATION n33 |

各点のreference/corrected/raw、total_signed_bias、log Bと、保存endpointのu/log_B_marginが存在する。`SAVED_POINTS.json`はこれらをそのままcopyし、`N_X/N_Y/rpe_round/full_rpe_resources=null`を付した。旧allowanceを取り出したεsignal=.0005・αaxis=.025は出典の記述であり、今回のεE・RPE配分ではない。B、attenuation、新bias、N、Gを算出していない。

他の構成にも保存signalがあるが、価格は別である。例としてS2／δ=.1のT=.8ではFull19 q_PF8 (`1f707cc04a8cb24a0970`) とPR4 q_PF8/R4/K2 (`68356304cb48aacf195e`)に直接価格がある。T1.6ではFull19 q_PF16 (`c40daca9e3df16a35f1e`)は直接価格あり、対応PR4 q_PF16/R4/K2 (`1d393ea8c9c1b2ce418d`)はsignalのみで、この固定長時間価格表では価格未取得。信号の存在からCを補わない。

## 3. h・prior・β・round数の推奨候補（未採択）

**第一候補はH4のD1案に合わせたh=.8、無energy shiftの条件付きpilot。** 保存.8/1.6をm0/1へ対応でき、短時間の新signalを作らず接続を確認できる。h=.8を正式に選ぶには、そのprincipal energy branchと安全余裕に保存DF groundが含まれるという従来と同水準のalias前提を明示する。保存solver energyは数値的参考であり、energy priorの厳密包含証明ではない。h=.8と広いprior[-10,10]を同時に採用しない。

広いpriorや小さいbase timeをH4/H6共通で使う場合の**代替はh=.1**。保存.8/1.6はm3/4に対応するが、m0..2等が欠ける。旧h=.02は方法再利用の候補に留め、この4つの保存時刻はdyadic roundに一致しないため、保存値だけの初回には優先しない。T1.2/2.0をh=.8又は.1のroundへ無理に対応させない。

βtotalは旧実装に合わせた`.4 rad`を第一候補とし、βstat・deterministic余裕は保存reference/stateを含むdiskの資格に応じて既存の残余法で決める。旧`.02/.02/.36`やDの`π/6,π/12`をH6へ固定転記しない。必要条件はβdet+βstat≤βtotal、正のraw半径、strict branch条件βtotal<π/3である。

round範囲は、選択h・βに対して既存`required_rpe_round_horizon`で

\[
M=\min\{m\ge0:\beta_{\rm total}/(h2^m)\le\epsilon_E\},
\qquad T_{\max}=h2^M
\]

を**次回一度だけ**評価する。今回は新しいM/Tmaxを計算していない。旧13/18roundを正式採用しない。初回のαは全round/両軸一様`.05/[2(M+1)]`を推奨し、旧cost感度配分はH6価格coverageが揃った後の比較候補とする。各PF/PR方式がそれぞれ同じαtotal=.05を満たす比較であり、両方式のNを共通にする必要はない。

| h候補 | 保存値と一致する時間 | この固定資料で不足する時間 |
|---|---|---|
| .8 | m0=.8、m1=1.6 | 使用するならm≥2（T≥3.2）のsignal/価格。1.2/2.0は別時間 |
| .1 | m3=.8、m4=1.6 | m0..2と、使用するm≥5。1.2/2.0は別時間 |
| .02 | .8/1.2/1.6/2.0の一致なし | 新たに選ぶ全round。DF12の保存時間列をH6へ移さない |

この表は時間metadataの対応で、signalの外挿・horizon最適化・性能評価ではない。

## 4. Nに必要な入力とH4との差分

共通入力は`target_id, candidate_id, binding, compiler, T, q_PF, δ, R_outer, K, formula, prefix, saved_signal, saved_allowance, prices`。RPE側で`m, q_RPE, h, βtotal/βstat, α_X/α_Y, law_model`を別に与える。信号と価格は同じcandidate・law・whole-wrapper単位へ結合する。model価格なら`model_id/source/coverage/q_variable`と時間・δのdomainも必須である。

H4の[assess_supplied_round][h4-code]へ渡すのは、**同じDF-Hのground target位相へ向かうstate-specific corrected signal disk**、同一完成済DF-Hに対するreference/stateの誤差、log B marginの効果を含むraw normalization error、implementation/law誤差、IID前提と出典である。未知を0にしない。保存の軸bias/uは、そのままoperator normや厳密diskではない。PF Fullでも保存state/referenceの資格は共通に必要で、PRだけへ追加負担を課さない。

rawは`z_corrected/B`、Bは全interrogationのnormalization。正のraw半径とphase余裕から既存`c1_stage3_accuracy._assess_raw_signal`／`rpe_resource_accounting._hoeffding_shots`を使う。今回targetはDF-Hなので`df_accounting='own_DF_target_no_DF_energy_shift'`とする。元H用adapterにDF差を加算してtargetを変えない。φ=−hE、E=−φ/h、scalar/identity一回、decoderへ真のEを渡さない規約を維持する。

| H4と共通に使う部分 | H6で異なる部分／最小変更 |
|---|---|
| H4の保存point schema、時間binding、disk→N、α union、軸別ΣNC、既存decoder/horizon | system/rank/H/DF/state binding、sector400、19 fragment、completion、保存candidate/cohort/reference |
| `assess_supplied_round/failure_budget/aggregate_supplied_axes` | 算術の複製は不要。H4の`inspect_saved_point/bind_saved_round`がH4/rank10を固定しているため、期待system/rankを明示引数又は小さい共通入力層へ分離する |
| raw/corrected、normalization一回、direct/modelの区別 | H6の数値値・allowance・signed piece配分を供給。H6をH4と偽装してguardを通さない。H4のguard/defaultは保全する |
| 軸別Nの会計と未知CI=null | H6のCを使い、探索n9と確認n33をpoolしない。RZを主指標とし、量子shot Nに古典価格標本数nやBを重ねて掛けない |

H4固定commitの人工15件は既存成果として参照する。今回同じテストやsolverを複製せず、**H6 source codeは追加しなかった**。H6の入力転記と共通schemaの静的照合を先行し、上記小変更はH4/H6の次回実装へまとめる。現H4 adapterを無変更でH6へ適用済みとは報告しない。

## 5. 最小のH6科学計算案（未実行・別途承認）

**保存4点だけのscalar campaignを最初の上限とする案**である。ユーザーがh/prior/β・有限候補・従来の数値state仮定を採択した後、horizon／α配分、reference位相・radius・disk、十分Nと凍結直接価格への接続をまとめて評価する。PF/PRを新規に探索しない。

1. H4担当の固定adapterを小さく共通化し、H6 bindingと採択targetを登録する。追加人工確認はH6入力を正しく受け、H4や別H/stateを拒否する差分に限る。既存15件を再実装しない。
2. 上記4点の保存reference/corrected/raw/bias/u/log Bだけで、DF groundへの条件付きphase/radiusとerror allowanceを確認する。保存数値stateの資格を超えた厳密ground成功保証は主張しない。誤差条件が閉じなければNは未資格とし、初回campaign内で追加eigenproblemやsignalを自動実行しない。
3. 採択したh/βから既存関数でMと全roundαを設定し、保存点に該当するrowのN_X/N_Yを算定する。αaxis=.025を全roundへ反復しない。必要なら保存Cを使った**該当roundのみのRZ会計**までを同じ許可範囲に含める。全round coverageがないので全energy成功／全Gは未成立と表示する。
4. 未評価roundと価格domainの不足を報告し、公開照合後mandatory STOPする。初回は新H/state/reference生成、signal evolve、sampling、build/compile、fit/bootstrap、RPE測定simulationを必要としない。

**全RPEへの次の拡張は分離する。** まず欠けたT_mと、採択したround構成だけのfinite mean／state-reference誤差評価を既存`h46_long_time_signal.evolve_batch`等で行う範囲を定める。保存点を再evolveせず、H/stateは再生成しない。高次signed構成やr/K/prefixの再選択は別の最適化scopeである。

新compileが必要になり得るのは、同じwrapperの保存価格/cacheがないT/δ/q_PF/prefix/R/K/law/順序/control、又はcompiler等を変える場合。`h46_long_time_native.compile_cached`はwrapper fingerprint＋compilerの一致時だけ再利用する。**別TのRZ数を角度非依存と仮定したり、保存単価×q_RPEを無条件に採用しない。** H6用のmodel資格や既存native計数で足りる部分を先に評価し、全長回路compileを既定にしない。Dのnative bridgeや絶対価格係数はそのまま移せない。

全roundのN／phase／energy／α条件とC coverageが揃った時点で初めて、各方式の`G_RZ=Σ_m,a N_m,a C_RZ,m,a`を比較する。PRのfinite-cohort平均、未観測tail、model bias、共分散不足を0とせず、保証水準を点推定／条件付き評価として明示する。

## 確認・保全・次の担当

固定SOURCE本文と保存JSON欄を読み、H6入力4ファイルの可用性／hashを確認した。保存値の転記は元blobとfieldごとに照合し、4点とも新N/Gがnull、価格cohortが分離され、内部絶対path・raw event/QPY・秘密情報が含まれないことを確認した。H4 adapter・長時間報告・価格調査の3資料はGitHub取得全文bytesとローカル固定blobが一致した。研究moduleをimportせず、既存テストの再実行もしていない。人工15件PASSはH4報告の引用であり、本担当のH6数値検証結果ではない。

H4が前回worktreeを使用中だったため、固定`fd76a3ed`から独立branch `codex/resource-eval-h6-df19-rpe-preparation-20261110`を作成した。変更は本報告・保存欄projection・入口索引のみ。旧SOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOP、H4作業、共有環境は保全する。公開remote SHAと主要2成果物の全文bytesを照合した後、**mandatory STOP**。

次の作業（未実行）：**ユーザー**はH4/H6の準備を併せてh・alias/prior・β・有限候補と上記scalar campaignの実行範囲を採択する。**H4担当Codex／本担当Codex**は許可後に共通入力層とH6差分の人工確認をまとめ、指定された科学評価だけを行う。今回の指示をその実行承認には扱わない。

[shot]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fd76a3edaf5b6fee9cadb556ba57efc7c46ed778/docs/research/rpe_shot_error_connection_20261110/REPORT.md
[price]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/caa592a807c533c36e06a94d2fd4e62af4a38734/docs/research/rpe_price_reuse_survey_20261110/REPORT.md
[price-index]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/caa592a807c533c36e06a94d2fd4e62af4a38734/docs/research/rpe_price_reuse_survey_20261110/SOURCE_INDEX.md
[long]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[h4-code]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e91021ae2db0fba755ba7a6404c18e7e003c129f/src/trottertracks/h4_df10_rpe_connection.py
[h4-report]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e91021ae2db0fba755ba7a6404c18e7e003c129f/docs/research/h4_df10_rpe_preparation_20261110/REPORT.md
[paper]: https://arxiv.org/html/2503.05647v1#S2
