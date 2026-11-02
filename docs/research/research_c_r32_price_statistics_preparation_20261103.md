# 研究C：update005/r32 native価格の独立校正・統計設計

2026-11-03 JST。起点 `f200e45afecc0a80b9f8a895b68c75ba4dda5dd5`。
**技術接続・人工検証は完了。推奨案は新規独立640件、95%近似t区間、D費用の0.5%を半幅目標とする固定N校正（最大5,845 logical units）。これらの統計条件と実行価値はユーザー／研究Cの採択待ちであり、今回の科学実行は0。** 同等性・非劣性marginは未指定／nullを維持する。

[未承認契約](research_c_r32_price_statistics_contract_unapproved_20261103.json)／[実装](../../src/trottertracks/c_r32_price_statistics.py)／[entry](../../scripts/run_research_c_r32_price_statistics.py)／[人工テスト](../../tests/management/test_research_c_r32_price_statistics.py)／[検証結果](../../artifacts/research_c_r32_price_statistics/2026-11-03/technical/verification.json)／[scalarだけの標本数計画](../../artifacts/research_c_r32_price_statistics/2026-11-03/technical/scalar_planning.json)／[SOURCE](../../artifacts/research_c_r32_price_statistics/2026-11-03/technical/SOURCE.json)。

## 1. 保存証拠・対象と問い

| 固定資料 | 結果収録commit／科学SOURCE | 今回の役割 |
|---|---|---|
| [r再調整報告][R]、同run result／execution_summary | `f200e45afecc0a80b9f8a895b68c75ba4dda5dd5`／`1cdeaa8bd55eb07a1d5b8cf2080259e929d405d1` | r32は登録r16/32/64/128を比較した後の最小点。既存32件は探索・計画用 |
| [同じ新J/Uの強D][D] | `38b8d0df39b8943178ecef611194819cff1c2169`／`5f27539ae0ab88181d9cfcd45badb54daf855c28` | 固定D kernelと同じ会計・誤差laneの比較分母 |
| [初期Evidence Map](research_c_evidence_synthesis_and_decision_readiness_20261031.md)、[最新Evidence Map](research_c_evidence_map_updated_20261102.md)、[貢献候補](research_c_contribution_positioning_20261102.md)、[Claim統合](research_c_claim_contribution_alignment_20261102.md) | 対応原本の固定来歴を継承 | DF目的、候補固有会計、Dとのtradeoffと未資格の区別。重要科学レビューを新規に実施しない |

保存直鎖H4・隣接1.00 Å・STO-3G、通常number CDF8、signed対称J・実直交U、Nα=Nβ=2/sector36、system8/full256・controlled512。元g補正κ、核scalar一回、元spin/frame順を保持する。Selectedは未収束有限endpoint `update005`、SHA256 `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`、入力snapshot SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。

PRは元frame順L_D=2・partial ordered symmetric S2・s1・δ=h=.1・r32・K0・m0..12。Dは同じJ/Uの全9因子ordered S2/s1、r/L_D非適用。共通CA=.00159 Ha、DF reserve=CA/8、βtotal=π/6・βstat=π/12、RPE failure=.01・axis=.01/26、own-DF ground-state oracle・状態準備費用除外。P0はQiskit1.3.0/opt1/seed17、rz/sx/x/cx、approximation degree1、qubits_initially_zero=false、MPFR160 declared phase・位相fitなし。Head相対Gaussian／tail exact basis_key共有とfenced serial会計を維持する。

既報値はPR r32総RZ106752458375、shots15202、W13979232、kernel平均7636.5、総RZ記述的SE1192089996.44。Dは総RZ105532782860、shots13728、kernel11026、PR/D点比1.01155731406。今回これらをrawから再集計せず、CI・p値・母平均の新推定も付与しない。「確実に高価格」「同等」「低価格の可能性なし」といずれも断定しない。

## 2. Estimand・モデル・選択

価格変数Xは、固定J/U/r32・signed K0 sampling・固定P0 compiler・融合規則から得る**一つのcomplete controlled h-step kernelのRZ数**。理論的な対象lawを「一様な64-bit seedからPCG64を初期化して32 eventsを生成する実装law」と定義し、μ_impl=E[X]を対象にする。実際の事前固定hash seed群は、このlawからの独立擬似乱数replicationというモデルで解析する。Deterministic PRNG実行そのものにランダム性や厳密独立性を証明した、という意味ではない。

主推定対象は`C_PR=W_ideal μ_impl+S_ideal`、`Δ=C_PR−C_D,ideal`、`R=C_PR/C_D,ideal`。W・shell・D・shotsは保存済みの条件付き値として固定する。Dは固定compiler/kernelなので、今回のclassical trajectory校正に関する標本分散項はない。Dの科学的／合成誤差を0と認定したものではない。

| 区別 | 今回の扱い |
|---|---|
| 実装seed lawとideal IID law | μ_implの統計推定をideal母平均・law距離へ移送する保証はない。Ideal RPE重みに実装価格を入れた条件付き会計である |
| Classical校正と量子shots | 新640件は価格校正。実量子測定0。保存RPE shots/Wは再計算せず、640やrをWへ追加乗算しない |
| Compiler | version・seed・profileを固定した価格関数。Compiler政策や再最適化探索を混ぜない |
| Full-law correctness | 各標本のraw/compiled/adjoint/X/Y PASSは必要。全lawのsynthesis/operator上界・actual CAは別で未資格 |
| 問いA | 選定済みr32対Dの期待価格差。新holdoutだけで推定し、既存32件をpoolしない |
| 問いB | r集合内の母平均最小性は対象外。主張するには各rの独立校正と事前定義した同時区間／多重比較制御が必要 |

r32を過去の最小点として選んだselection effectは、探索的32件を推論から分離することで扱う。新cohortと過去cohortの独立性モデルを前提として、選定後に固定したr32だけを検証する。過去標本との平均、再ランキング、結果依存のr選択や標本追加は行わない。

## 3. 統計方法と資格

推奨する主解析は固定N=640における`x̄ ± t_(.975,N−1) s/√N`。分散・平均・記述的SEは既存`b2.calibration`を再利用し、既存SciPyのt分位点で区間を構成する。[NISTの平均区間式][NIST]に従う。正規IID母集団ならt式の理論に対応するが、本価格分布の正規性を仮定済みとは扱わない。有限価格分布に対するCLT/tの**近似区間**であり、N640だけで被覆率95%の厳密保証を主張しない。稀なtrajectoryや強い歪みは依然見落とせる。

W>0、D>0なので同じ端点を`W×端点+S`、`−D`、`÷D`へ変換し、総費用・差・比の区間を得る。Dの固定分母に対する一標本問題であり、独立二標本のWelch式や比の分母推定誤差は導入しない。同じ校正を全13roundに使うため総SEは**W×SE_kernel**。√13や√20でさらに縮めない。

| 登録する返却状態（すべてモデル内・近似） | 条件 |
|---|---|
| NOMINAL_MODEL_PR_LOWER／HIGHER | 登録精度を満たし、診断screenがclearで、Δ区間上端<0／下端>0 |
| UNRESOLVED_SIGN_NOT_EQUIVALENCE | 区間が0を含む。同等性・非劣性の認定ではない |
| UNRESOLVED_PRECISION_TARGET | 区間半幅/D>.005。Nを増やさずSTOP |
| UNRESOLVED_DISTRIBUTION_DIAGNOSTIC | moment skewness絶対値>2、または最大delete-one平均変化>区間半幅。全標本・近似区間を残すが符号結論を保留 |
| UNRESOLVED_ZERO_OBSERVED_VARIANCE | 観測分散0。ゼロ幅区間を保証として返さない |

Skewness=2等は保守的な**提案された工程診断**で、CLT成立や被覆率を認証する定理ではない。外れ値を除去・winsorizeせず、正当な大費用標本を棄却しない。非整数／負値／NaN／欠測、seed重複、native correctness不一致はデータ不正としてFAIL/STOPする。成功例だけを集める再標本化は禁止する。

観測min/maxをsupportとせず、true support・Hoeffding型非漸近価格boundはnull。固定event数とcompilerから有用な数値supportを導出する作業を今回の条件へ無断追加しない。Bootstrapや逐次信頼列も追加しない。固定Nは任意停止を避け、既存処理に最も小さい接続で済む。

主検定的判断は**ideal会計のRZ一件のみ**。RPE failure=.01を価格alphaへ流用しない。Assumed laneと残り5指標は、同じ640件から平均・SE・affine totalsを記述するだけで、別の有意差・同時区間・優位判定を出さない。同等性・非劣性・科学的重要差はmargin未採択のため判定不可。

## 4. N640の根拠・費用・採択待ち

旧rawを開かず、既報scalar SE1192089996.44とn0=32、D105532782860だけを使い、`H(n)/D=t_(.975,n−1)×SE_old×√(32/n)/D`を計画値として計算した。これは今回明示されたscalarベースの技術計画で、旧価格の新科学推定ではない。旧分散が今後も同程度という未検証の計画仮定を含み、旧gapの選択バイアスもある。

| 新N | 予測95%半幅/D | 役割 |
|---:|---:|---|
| 160 | .9977% | 安価だが既報約1.16%gapに近い精度。非正規近似の懸念も大きい |
| 256 | .7865% | 中間案 |
| **640** | **.4960%** | 推奨。0.5%半幅目標の最小計画N630を既存32件単位に丸めた20組 |

0.5%は約1.16%の探索的gapを切り分けるための**推奨推定精度**で、重要差・同等性marginではない。実分散や希少tailにより精度未達／判定不能となり得る。正式power解析、成功確率、実験後の精度保証ではない。平均がD付近ならN640でも符号未確定で正常終了する。

既存混合r16/32/64の96回路検証はwall1531秒という保存運用記録がある。640件には相応のnative/full512検証負担があり、数時間級となる可能性を見込むが、r混合・cache・同時負荷が違うため正確な所要時間やspeedupへ比例換算しない。最大640compile、2560QPY、640のfull256 referenceを保存する。実行前に空きCPU/RAM/diskを確認する。

Confidence95%・0.5%目標・N640・診断政策は**契約内に具体化した提案**。ユーザー／研究CはCodex-Bのupdate064結果を踏まえて採択／延期／別設計を判断する。別設計を選ぶなら実行前に新SOURCE/契約へ固定し、結果後に変更しない。今回の準備完了は科学実行許可ではない。許可はチャットで十分で、承認証明・原文保存・新grantは不要。

## 5. 最小接続・実行範囲・calls

既存の`c_rte_r_retuning.prepare`（候補/一般K/CAR9/保存ground等）、`c_pr_integrated_performance.native`、`c_pr_aware_df_stage_b2.calibration`、`c_pr_aware_df_m3.serial_cost`、`aggregate_totals`、`trajectory_seed`、P0/StrongBuilder/QPY/Calls/STOPを再利用する。旧実装ファイルは一切変更せず、新module/entry/testだけを追加した。

Native helperが固定32件を扱うため、640件を20組に分けて呼ぶ。最初にfull CAR/Gaussian/head相対/shellの同一arm cacheを準備し、以降は共有する。各組内部の32件を実資源に応じて並列化し、既存BLAS調整・Qiskit build/compile/adjoint排他を維持する。組を逐次にするのは同じ可変builder/cache・既存排他と資源計画を再利用するためで、価格による逐次選択ではない。CPU/RAM/worker/wallの人工上限を新設しない。

新master=2026110302、cohort=`update005_r32_prospective_v1_batch00..19`、axis=`R32_PRICE_HOLDOUT_ONLY`。組内sample0..31、global ID0..639。既存のSHA256→64bit seed関数とPCG64を変更しない。SOURCE manifest hashもseed keyへ含め、並列順によらず再現する。新640seedの重複と旧3r×32＋保存r128×32の128seedとの一致を検査する。**Seed key非重複と独立性証明は別**。[NumPy1.26の並列乱数説明][NUMPY]も擬似乱数stream分離の前提・限界を示す。ここでは既存seed設計を新しいSeedSequence実装へ置き換えない。異なるseedで同じevent列になっても正常なlawの反復であり、棄却しない。

| 論理call | 上限 |
|---|---:|
| 既存Stage A prepare | 20 |
| 固定r32 law/config・law gate・normalization | 3 |
| 共通full CAR/Gaussian/shell | 35 |
| 20組×（9種×32件＋組の記述的calibration1） | 5780 |
| 保存r32 gate・seed非重複・会計binding・全件validation・主統計・副記述・summary | 7 |
| **合計** | **5845** |

9種はtrajectory sample、kernel build、independent reference、raw、compile、compiled、actual adjoint、complex X/Y、metrics。Full512を成分allowance1e-10／signal2e-10・rtol0・位相fitなしで検査する。旧factor/ground/referenceは許可後に整合確認し、新fit/optimizer/DF誤差/PF-RTE誤差/ground eigensolve/shot式は0。既存32件の価格raw/QPYや保存r128を新校正へpoolしない。Accounting bindingは保存shots/plansとshell metricsを既存serial関数へ渡し、W/Sを一度固定する。

途中は組完了数・既存の組別記述的calibrationのみ保存し、区間や判定を覗いて打ち切らない。最後まで640件そろわなければ主推定はしない。Correctness FAILでは依存callを停止して開始済みtaskを回収し、partial/source/journal/QPY/STOPを保全する。Retry/resume0、失敗標本の置換0。精度・符号・分布診断のUNRESOLVEDは科学的正しさFAILと区別し、追加標本を自動生成しない。

## 6. 人工・回帰検証と実行準備

**42 tests PASS（新22＋既存20、5.95秒）。** 人工平均でlower/higher/符号未確定、近似t式とaffine差/比、共有SE、shell一回、強い歪み・高influence・精度未達・観測ゼロ分散を確認した。640件のmock end-to-endでは20組完了前の統計判定なし、poolなし、compile correctness FAILのpartial STOPを確認した。旧native32 mockを再利用して新cohortの実dispatch・QPY保存とsame-arm cacheも確認。SOURCE drift・候補資格・seed衝突・fresh/retry/resume・passive時NPZ未decodeも検証した。

最初の人工試験は進捗ファイルへの二回目のexclusive saveで2件FAILした。組ごとの`batchXX/completion.json`へ保存先を修正後、上記全42件がPASS。旧科学SOURCE・STOPは変更していない。これはlocal人工検証で、外部CI再現やH4科学性能の証拠ではない。

Passive preflightはSOURCE70/input31/依存10packagesと候補・過去STOP・統計条件・seed keyの一致を確認しPASS。新run01は未作成。科学SOURCE manifest SHA256 `f6f0be47a453349209237694826637cd289b57a00f7184d2cee2d6241172314b`、契約SHA256 `4a78a3756243aa310df60a51d32a4c43bd318056c2fcf536da924f72b3e904a4`。Scientific Git SOURCEは本報告・契約・moduleを含む公開commitで識別する。

将来のentryは次の通り（今回は**--executeを使用していない**）：

```bash
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
 scripts/run_research_c_r32_price_statistics.py \
 --contract docs/research/research_c_r32_price_statistics_contract_unapproved_20261103.json
```

上記はpassiveのみ。別途の固定範囲の採択・科学実行許可後にだけ`--execute`を付ける。Fresh outputは`artifacts/research_c_r32_price_statistics/2026-11-03/run01`。Unapproved filename/metadataを許可証明に変更する手続きは不要である。

## 7. 残る限界・次の担当

μ_implの区間が狭くなっても、全law synthesis、implemented-vs-ideal signal/価格距離、actual CA、保証付き量子資源優位、selectedのJ/U同時収束、他分子への一般化、r集合内母平均最小性は未資格のまま。近似区間の被覆率もvalidated保証ではない。重要な論文主張に非漸近母平均保証が必要なら、真のsupport等を使う独立設計が後続課題となる。

**ユーザー／研究C**：Codex-Bのupdate064結果と今回の実行価値を確認し、confidence・精度・N・診断政策を含む5,845-call案を採択・科学実行許可するか判断する。**Codex-A**：明示許可があれば新SOURCE/入力/fresh outputを確認して一度だけ実行し、近似統計結果・未資格理由・証拠を公開してSTOPする。今回はそこへ進まない。

Branch `codex/research-c-r32-price-statistics-preparation-20261103`。共有研究概要・PROJECT_MAP・manifest等は変更せず、本書を統合担当への導線とする。Bの新候補・最適化コード・契約・成果物には触れない。旧SOURCE/input/raw/grant/ledger/marker/STOPを保全し、commit/push後にremote SHA/tree・本文・差分を照合して**mandatory STOP**する。正式共通運用指示2026-10-25版§8/8.4/10を現物確認して適用した。

[R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f200e45afecc0a80b9f8a895b68c75ba4dda5dd5/docs/research/research_c_rte_r_retuning_execution_20261103.md
[D]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/38b8d0df39b8943178ecef611194819cff1c2169/docs/research/research_c_same_ju_strong_d_execution_20261102.md
[NIST]: https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm
[NUMPY]: https://numpy.org/doc/1.26/reference/random/parallel.html
