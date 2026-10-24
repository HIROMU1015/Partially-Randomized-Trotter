# 研究C・C1 Stage 3：共通CAでの総資源比較の技術準備

2026-10-25 JST。起点 `47d4032b4622c2b5060c19d7d2bf4789f822b64d`、branch `codex/research-c-c1-stage3-preparation-20261025`。

**技術実装・人工検証を完了した。分子科学実行は0。共通CAの達成、資源優位性、Stage 3の実行条件はまだ採択・検証していない。**
[未承認契約](research_c_c1_stage3_contract_unapproved_20261025.json)は推奨案である。
正式共通運用指示2026-10-25版をユーザーの追加後に読み、チャットの許可と科学的SOURCE・入力の再現性を区別した。新しい承認証明・grant・認証処理は作成していない。

## 1. 再利用する証拠と実装

対象は保存P-H4、直鎖1.00 Å・STO-3G、Nα=Nβ=2／sector36、system full256／controlled512。通常CDF number・元g補正κを保持した保存rank8と保存XDF rank10を使う。新fit・Hamiltonian生成は不要。

[Stage 1](research_c_c1_stage1_eval_only_execution_20261025.md)の物理sector operator normはCDF `7.666027761392501e-5` Ha、XDF `3.3751558354542566e-15` Haである。これは旧結果の引用であり、今回再計算も新CA条件への数値判定もしていない。
[Stage 2](research_c_c1_stage2_native_execution_20261025.md)のT=1/10、one-step ordered S2、P0でnative接続はPASSした。しかしDF精度は異なり、HF信号は基底エネルギーの証明ではない。literal/sharedの費用は一致しており、未観測の追加削減を仮定しない。L_D／delta窓はこれら旧局所検証では非該当。

| 再利用する実装・データ | 適合範囲と今回の接続 |
|---|---|
| `c1_stage2_native.NativeModel/NativeBuilder`、一般K adapter | one-body・元frame順・κ・scalar・Gaussian向き・controlled phaseを保持。CDFをsquare型に偽装しない |
| 保存sector行列3、CDF full base＋8 frame＋raw_total | 将来の固有状態・referenceに再利用。Stage 1の物理DF normや全gateを反復しない |
| `xdf_car_generator`、Stage 1 CAR | XDFのfull frame行列が未保存なので、将来その11 generatorだけ再構成する |
| `prototype.MixedAtom/make_tail`、`rte.make_rte_config/iter_sample_rte_events/finite_rte_corrected_operator/finite_rte_attenuation` | 符号付き一般KのI/Z/ZZ分解、有限RTE law・event・平均・減衰を再利用。新sampler・Taylor法は作らない |
| `circuits.BasisProvider/build_event_sequence` | full Gaussianを使用。同じbasis内の合法な共有のみ。singleton support-completionによる新削減は今回使わない |
| `required_rpe_round_horizon`、既存`_hoeffding_shots`、`_vectorized_reconstruction` | horizon・有界±1標本・任意roundのbranch復元を再利用 |
| `gates.compiled_wrapper`、`f2_execution.circuit_metrics` | phase保持P0、RZ/CX count・filtered depth・total depth・sizeの既存定義を再利用 |
| 旧`evaluate_rpe_round_candidate`／DF cost provider | square DF型、effective-PF固有状態、旧PF係数・短q資格に依存するため直接流用しない。一般Kとの間に小さい数値接続を追加 |
| 旧literal RZ token価格・短q affine proxy | compiled full interrogation費用として使わない。旧scope／精度の結果は移送しない |

## 2. 実装した範囲

- `c1_stage3_connection.py`：既存native modelを元順の決定論prefixと一般K tailへ分離。tail I項を一度だけ確定位相へ抽出し、sampling normから除外する。既存RTEをD-forward half／RTE／D-reverse halfへ接続する。新規fitは呼ばない。
- 同module：occurrenceごとにfactoryを呼ぶ直列native接続と全wire barrier、独立X/Y読出しsegment、直列費用展開。PRでは同じtrajectoryをq回再使用しない。旧HF wrapperはHF準備・one-step metadataを要求するため、読出しだけの最小segmentを用意した。
- `c1_stage3_reference.py`：元CAR generatorからPF／有限RTE平均を構成し、matrix powerで長roundを扱う。PF誤差・補正後RTE誤差・observed attenuationを別に返す。既存binomial／RPE復元へのvirtual marginal接続を追加。
- `c1_stage3_accuracy.py`：signal単位への変換、位相・shot・エネルギーへの条件付き接続、既知supportに基づく費用区間、両軸／全roundの総費用集計。未知誤差を0扱いしない。資格なし・区間重複では優位を出さない。
- `run_research_c_c1_stage3.py`：既定はSOURCE／保存入力のbyte確認のみ。別途許可後のevidence評価entryはfresh JSON出力を要求し、固定精度条件とscopeを照合する。分子pipelineは上記関数と既存loader/compilerの組合せであり、今回分子を自動生成・実行する一体runnerは追加していない。

新しいfitter、Gaussian合成器、数値線形代数・RPEアルゴリズムは実装していない。重要な科学条件の採択後、既存loaderからevidenceを生成する呼出し順をその採択条件で固定して実行する。今回のentryは未知の合成誤差やstate preparationを解決するものではない。

## 3. 最終精度との対応

初回推奨は各armの**自身のDF Hamiltonianのsector基底状態oracle**を仮定し、物理sector基底エネルギーを最終共通targetにする。状態準備費用は未知・除外する。従って比較できるのは条件付きinterrogation総費用であり、状態準備を含む無条件の総資源ではない。これは既存評価計画のstate-preparation除外scopeと一致する。HF準備がoracleを代替するとはしない。

各armのDF誤差上界をd_DFとすれば、同sectorのHermitian演算子に対するWeylの関係から基底エネルギー差はd_DF以下。一方、DF以後の誤差はまず同じsignal単位へ変換する。ideal finite-RTE平均の正規化をB、A=1/Bとすると、

```text
ideal observed z0 = A exp(-i t E_DF)
δraw = A (εPF,op + εRTE,corrected,op) + εsynth,raw + εlaw,raw
ρlower = A - δraw > 0
βdet <= asin(δraw/A)
εcoordinate = ρlower sin(βstat)/sqrt(2)
Naxis = ceil[2 log(2/αaxis) / εcoordinate²]
βdet + βstat <= βtotal < π/3   (全round)
|Eestimated - Ephysical,ground| <= d_DF + βtotal/tmax
```

最後の式は、oracle・初期alias branch・IID shot・全roundの決定論誤差資格を満たす場合に限る。RPE branch条件は[PR論文v1 Appendix B、Lemma B.1](https://arxiv.org/pdf/2503.05647v1)と既存復元処理に対応する。signal円板の幾何とHoeffdingの合成は今回の保守的な接続であり、論文の最適定数を主張しない。

DFのHa単位の誤差をsignal normへ直接足さない。own-DF oracleを採用した場合、DFをWeyl energy biasで一度数えるので、同じDF誤差のDuhamel signal biasを重ねない。別の共通physical状態を使うならこの設計を変更する必要がある。PF誤差を単一のeffective Hamiltonianのenergy shiftと同一視しない。RTE normalizationによる減衰も、エネルギー誤差に単純加算しない。

`binary64_operation_scale_guard8_v1`と旧Stage 2のcomponentwise toleranceは、それぞれ元の数値同値性検査にだけ適用する。長round全trajectoryのspectral norm保証には移送しない。今回のnumerically-qualified pilotは厳密な誤差証明と区別する。

## 4. 未採択の初回pilot案

| 条件 | 推奨案 |
|---|---|
| 最終共通目標 | CA=`159/100000` Ha。CA/100は要求しない |
| DF配分 | 共通reserve CA/8、推定側7CA/8。実際のCDF/XDF誤差を一致させない。新条件への判定は将来実行時 |
| 時間／round | base δ=1/10、βtotal=π/6、βstat=π/12、m=0..12、q=2^m。h=δ/s、coherent repetitions=q*s |
| 決定論baseline | 各armの全frame ordered symmetric S2、s∈{1,4} |
| PR候補 | 各armの元outer順prefix L_D=2、s∈{1,4}、r∈{4,16}、K∈{0,2}。Kは有限Taylor order、rはkernel内microstep数 |
| 候補数 | 各arm2決定論＋8 PR、両arm計20。各candidateは同じs/r/Kを13roundで検査 |
| alias／状態 | E_DF∈[-10,10] Haを将来検査。own-DF sector ground oracle。符号はz=exp(-itE)、復元後E=-phase/δ |
| failure配分 | 全体0.05。量子統計0.04／4選択protocol／13round／2軸、費用区間0.01／16PR候補／6指標 |
| law | 探索初回はideal finite-RTE IID lawを条件として推奨。実NumPy finite-bit sampler誤差は未資格。実装lawを対象にするならεlaw未知の間はN／総費用なし |
| circuit費用 | serial native kernel＋独立ancilla準備／読出しsegment。state preparation除外。P0、Qiskit1.3.0、opt1、seed17、basis rz/sx/x/cx |
| cost calibration | PR候補ごと32独立kernel標本、共通master・arm別stream・task対応ID。古典calibration回数をquantum shotsへ掛けない |
| virtual RPE | 各armの決定論／PRで最良の資格付きcandidateを選び、各1000 virtual marginal試行。選択後の独立seed。実quantum実行は行わない |

固定binary64でobserved attenuationが0へunderflowする場合は、巨大な補正後matrix powerを作らず当該候補を不適格とする。virtual binomialのshot数がNumPyのint64範囲を超える場合は、黙ってclipせず診断未実施と記録する。いずれもCPU/RAMの人工制限ではない。

登録gridは初回のfeasibility確認用で、全candidateがCAに到達するとは予測しない。長roundでは減衰と誤差で全滅し得る。その場合は`INFEASIBLE_REGISTERED_GRID`でSTOPし、round別r/Kや細かいsへの探索拡張は別判断とする。旧研究のround別scheduleをCDFへ無資格移送しない。

補助的なraw誤差配分案はdeterministic room `A sin(π/12)`のPF1/2、RTE1/4、synthesis1/8、law1/8。これは診断用配分で、採否gateは上記の合計signal／phase条件とする。事後に閾値を変えない。alias窓・CAの厳密な代表値・oracle・failure配分・lawモデル・cost scopeは研究Cによる採択が必要。

## 5. 費用の公平性と資格

両armに同じnative Gaussian・一般K Z/ZZ・exact隣接basis共有・角度結合・P0を適用する。元frame順を勝手に最適化しない。tailはfull frameのまま扱う。literal/sharedの未観測差を加算しない。

長qの巨大回路を列挙・global compileしない。各hのnative kernelをcompileし、全wire barrierで区切ってcoherently反復する実装方式を比較する。countはその方式の加算、depthは**明示的な直列scheduleの和**であり、全体最適化後のdepthではない。PRでは各occurrenceがfresh IIDで、同一sampleを長qに繰り返す方式は使わない。

```text
E[Caxis,m] = (2^m s) E[Ckernel] + Chead + Creadout,axis
Gmetric = Σm [NX,m E[CX,m] + NY,m E[CY,m]]
executions = shots = Σm (NX,m + NY,m)
```

ancillaのhead Hとreadoutは別segmentとしてcompileする。空のX-shellをまとめてcompileしてH/Hを消し、実回路で必要なHを無料にしない。measurement／classical backend job数はnative quantum gate費用と区別する。主指標は期待総RZ count、副指標はRZ/CX depth・CX count・total depth・size。異なる指標を根拠なく合算しない。

Stage 2のs=1 deterministic full kernelはSOURCE／IR／time／compiler identityが一致する場合だけ再利用する。HF準備込みwrapper費用は使わない。他のhやPRの費用は新しく計測する必要がある。

費用の区間は、既知の全law-support上の上下限と独立calibrationを仮定するHoeffdingへ接続した。観測min/maxや標準誤差を有界supportの代用にしない。support資格が未確定なら点推定のみ。同じcalibrationをround間で再利用するとき区間幅は反復数に比例して展開し、独立誤差として縮めない。候補・指標全体への同時配分を行い、期待総RZの区間が重なる場合は優位不明とする。未資格synthesisをsample最大値だけで全law上の上界としない。

## 6. 将来の実行順・科学call・高速化

契約のcall表は**未承認の登録上限案**で、composite logical unitの内部処理を併記した。今回これらは一つも実行していない。

1. fresh output、固定SOURCE／保存fit／元入力／環境のbyte preflight。新fit0。
2. 保存sector3・CDF full10の再利用、未保存XDF full generator11だけ構成。ground eigenpair3とalias/oracle条件を検査する。
3. 全中間generator/tailのsector閉性が証明・検査されるまではfull256 referenceを使用する。総Hが保存sectorを持つことだけで、PF/RTEの中間operatorを36次元へ射影しない。
4. arm別のimmutable local exponential cache、arm×roundのexact evolution26件を共有し、20 candidate／260roundのPF・RTE・減衰を評価する。独立candidate／armを利用可能CPU・RAMに応じて並列化する。BLASとのoversubscriptionを避ける。
5. feasible候補だけ新kernel／head／readoutをbuild・compileし、raw controlled／actual adjoint／X/Yとcompiled phase・unitarityを確認する。長round synthesis errorの資格を満たさないなら総費用は出さない。
6. 全roundの同CA gate後にshots／costを集計、区間資格付き選択、virtual RPE、CDF/XDF比較。新条件・candidateの追加や失敗後のretry/resumeを行わない。

数値correctnessの推奨：raw／compiled controlled最大成分誤差1e-10、unitarity1e-10、X/Y絶対誤差2e-10、virtual reconstructionは宣言したphase/energy区間で判定。独立referenceは元CARと既存event referenceを使い、candidate K展開をreferenceへ戻さない。これらのcheck toleranceと、科学的なlong-round signal error上界は別である。

固定call上限の詳細はJSONの`scientific_calls`を参照する。新compile最大517（PR512＋新deterministic2＋head/readout3）。再利用不可なら勝手に余分なcompileをせずSOURCE／scope不一致としてSTOPする。候補不適格は依存costを省略し、registered gridの残りのfeasibility検査を続けられる。実装correctness FAIL、入力不一致、予算外call、非有限値は最初で依存処理をSTOPしてpartial evidenceを保全する。

CPU／RAM／worker／wall-timeに人工上限を設けない。保存小行列、matrix power、exponential cacheを使い、長q回路と全trajectory列挙を避ける。実行時の空きCPU／RAM／GPUとlibrary対応を調べてbackend・並列度を選び、共有環境・他ユーザーのjobや設定は変更しない。GPUの速度向上は未測定で主張しない。

## 7. 人工検証と完了境界

新規人工23件＋Stage 2回帰27件、一般K回帰13件＋29 subtestsを実施。最終件数・所要時間は[人工証拠](../../artifacts/research_c_c1_stage3_technical/2026-10-25/)を参照する。係数・符号・I抽出・元順・phase・fresh factory・barrier・X/Y順序、実RTEの人工tail、2×2人工CAR相当generator、PF／RTE／減衰の分離、shot式・alias／oracle条件、全roundとfailure配分、費用の範囲・区間・加重・比較、virtual binomial/RPE接続を確認した。

一般Kの旧import検査は、他の人工数値testでlibraryがload済みの同一processでは13件がtearDownで失敗した。import-cleanlinessを調べる元の目的に合わせfresh processで独立実行して全PASS。科学SOURCEを変更して回避していない。

環境は既存task専用Stage 2 venvをread-only再利用し、52 package lockを参照した。新しい依存追加・共有venv変更なし。cache／tempはこのworktreeに分離。SOURCE manifestは関連数値／回路コードと既存lockだけを対象とし、無関係な全file監査や承認台帳は追加しない。

**残る科学判断**：上記pilot条件、oracle・状態準備除外のscope、ideal lawモデル、全law上のsynthesis誤差と費用support資格。これらの保証が不足する場合でもfeasibility／点推定を記録できるが、厳密な同CA総資源優位を宣言しない。

ユーザー／研究Cが推奨条件を採択・変更し、公開SOURCE・保存入力・契約・fresh outputに対する科学実行をチャットで別途明示許可する。Codexは許可後にのみ登録範囲を実行する。今回はremote照合後**mandatory STOP**。実H4 decode・新operator/PF/RTE/shot/総資源の計算・実H4回路build/compile・新fitは0、旧SOURCE／結果／grant／ledger／marker／STOPを保全する。
