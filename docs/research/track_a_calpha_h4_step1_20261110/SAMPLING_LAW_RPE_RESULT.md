# C-α／α0-P：固定gate lawの資格化・全18round会計・価格の到達範囲

2026-11-10 JST。担当：Codex。

**固定K0 builderの理想gate意味論と有限bit CDFに対する接続を囲み、同じH4の全18roundで条件付き位相・半径保証と必要shot会計を得た。X/Y合計は113,252 shots。native emitterのscalar集約差を加えても値は変わらない。一方、全roundのcompiled期待価格は未取得で、総compiled費用の改善は未確認。**

今回の対象はLinear H4／1.00 Å／STO-3G、元H_physと保存元H state、保存DF12、既存generic frame、generation-prefix LD3、Nα=Nβ=2／sector36、正時間partial-S2、paired K0、δ=.02、r4、prep費用除外。新C-αのsystematic .14 rad、stat .24、reserve .02、全体 .4を維持した。別分子・別law・r/LD探索はしていない。

保証の対象は**compile前の固定gate意味論**。有限bit層は、既存CDFへfresh独立53bit一様入力が入るという条件に依存する。固定seedの決定論的PCG64 streamが数学的IIDであると証明したものでも、compiler／実機誤差まで含む95%保証でもない。重要な新定理の正式採択やC-αの新規性・費用優位を認定したものではない。

## 1. 許可・既存成果・実行SOURCE

更新された `.agent/docs/comment.md`「固定sampling law資格化と保証付きPR＋RPE総費用評価」の明示許可に従い、通常の技術修正・数値資格・条件成立後の全round会計と限定価格取得をまとめて行った。前回の再構成モデルの候補を無条件に固定lawへ移していない。

固定基点は **`1b1449c9d8f04e4b574bc8a7c6b677572bfb0346`**。同じ専用branch `codex/track-a-calpha-h4-step1-20261110` とworktreeを継続利用した。旧run01〜run07、SOURCE、失敗時SOURCE、raw、REPORT、STOPは不変更。前回の[保証改善報告](MACROSTEP_GAP_RESULT.md)と[理論資料](reviews/track_a_calpha_macrostep_gap_limited_scientific_review_20261110.md)を再利用した。

元H／state／DF／sectorは既存 `load_private_inputs` の同一性検査で結合し、元stateの再solve・別state採用・入力の再正規化・phase変更はない。private state・積分・HDF5・private operandは公開対象に含めない。`research_project_instructions_unified_2026-11-09.md` は確認済み環境では未取得のまま、Codex自身の読了とは扱わない。

今回の新しい出力は以下。各rawに実行SOURCE hashを記録し、旧出力を上書きしていない。

| 出力 | 内容 | 結果 |
|---|---|---|
| [run08](../../../artifacts/calpha_h4_step1/2026-11-10/run08_sampling_law/result.json) | projection順とsampler順の同一性を誤って要求した初回 | 技術例外。SOURCE・logも保全 |
| [run09](../../../artifacts/calpha_h4_step1/2026-11-10/run09_sampling_law/result.json) | 全324 K0 atomのcontrolled skeleton、gate基底、有限bitと再構成差 | 接続を包含、二点成立 |
| [run10](../../../artifacts/calpha_h4_step1/2026-11-10/run10_rpe_accounting/result.json) | 未取得16roundの上界、保存端点の再利用、全18round会計 | 条件付き保証・113,252 shots |
| [run11](../../../artifacts/calpha_h4_step1/2026-11-10/run11_price_probe/result.json) | 新K0 m0の1 trajectory、X/Y full no-prep compile | 両軸価格取得。期待価格ではない |
| [run12](../../../artifacts/calpha_h4_step1/2026-11-10/run12_cost_scope/result.json) | source命令数の厳密期待会計と長round価格の成立性 | compiled総費用は未取得 |
| [run13](../../../artifacts/calpha_h4_step1/2026-11-10/run13_native_scalar/result.json) | step builderとordinary native emitterのscalar集約差 | 全18roundの条件を維持、shot数不変 |

## 2. 再利用と最小差分

固定SOURCEの `prepare_b2`、`df_rte_tail`、`df_partial_s2`、`df_rte_qiskit`、RTE canonical `_make_event`、`df_trotter.ops`、generic diagonalizer、CAR／exterior lift、native ordinary emitterを再利用した。既存Gaussian compilerやsamplerを改変していない。

新規は[gate/CDF比較helper](../../../src/trotterlib/calpha_sampling_law.py)、[raw半径会計adapter](../../../src/trotterlib/calpha_rpe_accounting.py)と今回の限定runner。既存K2専用会計driverをそのまま使うと旧成分枠やnormalizationが混ざるため、raw半径を直接受ける薄いadapterが必要だった。内部の必要shot式は既存 `_hoeffding_shots` を呼び、sin／logまで囲んだ整数切上げでも確認した。

[full no-prep価格adapter](../../../src/trotterlib/b2_p1_compiled_cost.py)は `origin/codex/track-a-b2-p1-full-no-prep-cost-20261107` から**原文のまま**再利用。新compilerや新しい価格推定器は作っていない。

## 3. 固定lawと再構成モデルの対応

### 3.1 Gate列とfull-space意味論

固定generic diagonalizerを入力とする `BogoliubovTransform(U.T)` は、今回全frameで `RZ` と隣接 `XXPlusYY` の列だった。各gateのinverseが角度の符号反転と逆順になることをSOURCEと生成物で確認した。理想gate意味論ではこの列をVとするとVはunitaryであり、

`Q_l = sign_l V Z_support V†`

は全Fock空間でHermitian involutionとなる。個別atomのNα／Nβ保存を要求していない。隣接するnumber-preserving Gaussian gateであることから、one-particle作用とexterior liftの対応を使える。vacuum phaseは比較参照でだけ取り除き、実際の `V D V†` では元から相殺する。

全324 atomを既存canonical event／controlled builderへ通し、inverse basis→controlled RZ/RZZ→forward basisの順序、対象qubit、係数sign、`2*rotation_angle`、control状態1、K0のevent phase=+1を確認した。controlled relative phaseを事後fitしていない。control-0ではbasis inverse／forwardが相殺し、control-1では固定eventが作用する。既存Hadamard wrapperのH→controlled evolution→H（YはSdg→H）はX=Re(z)、Y=Im(z)を与える。

[固定gate profile](../../../artifacts/calpha_h4_step1/2026-11-10/run09_sampling_law/gate_profile.json)に、全basis列・diagonal primitive・atom・controlled確認を保存した。密な9-qubit行列の全324個生成や新しいsignal simulationはしていない。

### 3.2 丸めoperandとの差を消さない

実際のgate列を基本区間演算で評価し、旧run07のliteral exterior liftとの差を囲んだ。basisのpolar projectionや再直交化による置換はしていない。元sectorから出る成分を保持するtotal-number blockで接続し、atomごとのtrajectory射影・dropも行っていない。全Fock空間のbasis比較はone-particle作用とtensor／exterior積のnorm評価でも記録した。

最大のbasis差は全frameで一様にmachine epsilon程度ではなかった。例えばfragment7のN=4 block差上端は約6.24e-8、fragment9では約2.08e-8。これらを0扱いせず、各係数・rotationの重みを掛けて平均event差へ入れた。対応する全Fock/exterior比較上端の最大表示は約1.12e-7。

head比較では `A D A†−B D B†` を、同じscalar cに対する `D−cI` で代数的に分離した。`||A||=1`、`||B||`、`||A−B||`、`||B B†−I||` とdiagonal phase差で囲む単純な積の三角不等式であり、入力gateやphaseを変更する処理ではない。この比較でheadの短時間性を保ち、巨大な一様basis上界を全headへ無条件に加算する保守性を避けた。

保存projectionはcomponent IDの辞書順、実samplerはextraction順だった。run08は両者の順序一致を誤って要求し停止した。run09では**比較側だけ**を `(fragment,support)` で対応させ、sampler順とCDFは変更していない。また135係数に微小な丸め差があり、最大絶対差2.89e-18、L1差4.32e-17だった。旧tail hashとも一致しないため、byte完全一致とは報告しない。係数差はevent差へ含め、同じ固定SOURCEの再構成誤差として扱った。λ_Rと抽出identityは旧値と一致した。

### 3.3 確率・角度・finite-bit

K0 eventの回転角は既存SOURCEの `atan(λ_R δ/r)`、実gateはsignedなRZ／RZZの角度2倍を使用する。binary64角を正確な実数parameterとするgateモデルと、数学的な正確なarctanを分け、両者の差を囲んだ。

NumPy 1.26.4の `choice(replace=True,p=...)` は累積和、末尾での正規化、right searchを使う。[固定版SOURCE](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_generator.pyx)。PCG64のdouble変換は上位53bitを使う。[固定変換SOURCE](https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_common.pxd)。この規約から、CDF端点a,bに対する正確なgrid質量 `(ceil(2^53 b)−ceil(2^53 a))/2^53` を取得した。nominal確率との差のL1は約 **7.63390e-15**。

これは**各drawがfresh独立一様grid入力である条件付きの確率law**。実装の決定論的seed streamの独立性を暗黙に数学的定理へ追加していない。実際のPCG64 samplerは変更せず、今回の保証評価で乱数列の実samplingは行っていない。価格probeの4 eventだけは別途既存samplerで取得した。

## 4. 二点の保証と証拠水準

再構成モデルが収縮である旧run07の包含を再利用し、lawとの差εを全時間で

`ε (1−ρ^q)/(1−ρ)`、`ρ≥||S_rec||`、`ρ<1`

として加算した。law側は理想unitary eventの確率平均と完成headなのでnorm≤1。これは既存sector追補のtelescoping恒等式へ既知の収縮係数を残した評価である。前回の状態defect精密化も、レビュー式(3.6)の元となる同じ恒等式・Duhamel残差評価を使い、新しいlawやPF familyを採用していない。重要な新定理の優先権・正式採択は主張しない。

共通の接続項上端：head合計6.92507e-14、tail basisの一event差1.51182e-13、scalar phase差2.03384e-18。有限grid・角度込みのmacrostep差上端は **6.740752e-13**。

| 値（角度・finite CDF込み、compile前gateモデル） | m0 | m17 |
|---|---:|---:|
| 旧run07の再構成disk上端 | 3.44259196e-7 | 0.01633582704 |
| law接続誤差上端 | 6.74075194e-13 | **6.91405344e-8** |
| 固定lawのdisk上端 | 3.44259870e-7 | **0.01633589618** |
| 半径下界 | 0.999983886452 | **0.110238066328** |
| 位相上界 (rad) | 3.44265299e-7 | **0.129423069406** |
| 正半径・0.14 rad条件 | 成立 | 成立 |

ordinary native emitterのhead／tailは同じ既存関数へ委譲するが、最上位scalarだけはq step分を一つの `PhaseGate(−(c+identity)T)` に集約する。run13でこのbinary64 phaseとstep-builderの別々のscalar phaseとの差を有理数で取得し、誤差へ加算した。m17のphase parameter差は `5/35184372088832` rad。追加後も位相上界は **0.129423069408 rad**、半径下界 **0.110238066328**で条件成立、shot整数も不変だった。

| 水準 | 今回の資格 |
|---|---|
| 正確なgate basis・係数確率・角度のideal K0 | 再構成との差を囲み、二点条件成立 |
| 保存binary64再構成モデル | 前回run07の包含を再利用 |
| 固定builderのuncompiled gate意味論 | 全support skeleton・unitarity/involutionの構成・数値包含を接続 |
| finite-bit確率・角度 | CDF grid質量とbinary64角parameterを含む。fresh IID grid仮定付き |
| ordinary native emitter | 同じblock委譲とscalar集約差を接続 |
| compilerが生成した全長gate列の厳密誤差 | **UNKNOWN** |
| noise・実機成功確率 | **UNKNOWN** |

個々のprimitiveがsector36を保存するという不要な条件を導入せず、full lawの差を元sectorからの接続に含めた。平均の微小なspin非保存を厳密zeroに修正していない。

## 5. 全18roundの条件付き保証・shot会計

m0/m17のraw信号・operator評価は再実行していない。保存端点の保証を再利用し、未取得m1〜m16の上界だけを評価した。tail log・matrix exponentials・共通operatorを共有し、16回の新preparationや巨大回路simulationは行っていない。

既存Hoeffdingのcoordinate誤差式をraw半径へ直接適用した。attenuation=1、signal_error=0は**取得済みraw半径を渡すためのAPI表現**であり、未取得の実装誤差を0としたものではない。追加のRTE normalizationは掛けていない。各軸α=1/720、36軸で合計1/20。sin／logの区間上端を整数切上げし、既存helperの整数と比較した。

| m | raw半径下界（表示） | systematic位相上界 (rad) | X shots | Y shots |
|---:|---:|---:|---:|---:|
| 0 | 0.9999838865 | 3.44266e-7 | 515 | 515 |
| 1 | 0.9999677731 | 6.88546e-7 | 515 | 515 |
| 2 | 0.9999355473 | 1.37715e-6 | 515 | 515 |
| 3 | 0.9998710985 | 2.75454e-6 | 515 | 515 |
| 4 | 0.9997422126 | 5.51003e-6 | 516 | 516 |
| 5 | 0.9994844878 | 1.10239e-5 | 516 | 516 |
| 6 | 0.9989692261 | 2.20629e-5 | 516 | 516 |
| 7 | 0.9979394535 | 4.41866e-5 | 517 | 517 |
| 8 | 0.9958829079 | 8.86171e-5 | 520 | 520 |
| 9 | 0.9917817862 | 0.000178215 | 524 | 524 |
| 10 | 0.9836280509 | 0.000359508 | 533 | 533 |
| 11 | 0.9675192148 | 0.000723975 | 550 | 550 |
| 12 | 0.9360740702 | 0.001468079 | 588 | 588 |
| 13 | 0.8761599360 | 0.003019040 | 671 | 671 |
| 14 | 0.7673777520 | 0.006389579 | 875 | 875 |
| 15 | 0.5878996040 | 0.014363323 | 1490 | 1490 |
| 16 | 0.3426655337 | 0.036848787 | 4385 | 4385 |
| 17 | 0.1102380663 | 0.129423070 | 42365 | 42365 |
| 合計 | — | 全18点≤.14 | **56,626** | **56,626** |

表の値は表示用の丸め。正確な包含端点とnative scalar差込みの最終値はrun10／run13 rawの有理数文字列を使う。必要合計は **113,252 shots**。これは十分shot数であり最小shot数ではない。測定を実行した結果ではない。

branch条件は既存nearest-branchの帰納条件 `3β<π` を使い、β=.4でmargin約1.94159 rad。初期prior[-128,128] Haに対し `128δ+β<π`、margin約0.18159 radを確認した。最終energy誤差上界は **0.000152587890625 Ha**、既存目標0.00015936001019904 Ha以内。未知energyをbranch選択へ入れず、今回新しいdecoder実行・phase oracleによる事後選択も行っていない。

したがって、固定gate・state promise・fresh IID trajectory／±1 outcome・上記branch条件の下で成功確率下界19/20を得る**条件付きRPE会計**である。compiler／実機の95%成功を認定していない。

## 6. 価格取得と総費用の未取得範囲

### 6.1 実測したm0の1 trajectory

全round会計成立後、既存sampler seed20261110でK0 eventを4個取得し、その固定1 trajectoryをX/Yで共有して両軸をcompileした。native→controlled full wrapper→測定まで、状態準備を除くwhole回路が対象。価格によるtrajectory選択や有利なseedの探索はない。

compilerは既存価格条件に合わせ、**Qiskit1.3.0、rz/sx/x/cx、optimization1、seed7、backend／coupling mapなし、all-to-all**。compiler hash `a0301472e2fb95ba82d3ce80dace0d6de9637ccc74b1887d8271ca1c3e5c04d8`、backend `no_backend`。

| compile後の指標 | X | Y |
|---|---:|---:|
| RZ count | 6156 | 6156 |
| CX count | 3320 | 3320 |
| RZ depth | 2566 | 2566 |
| CX depth | 2700 | 2700 |
| whole depth | 5587 | 5587 |
| circuit size | 11725 | 11725 |

9 qubits／1 measurement、native803 instructions。2 compileで6.764秒、process peak RSS583,988 KiB（import等を含む）。この1 trajectoryの価格は**期待値ではなく**、standard error／CI／期待価格／総compiled費用はnull。旧K2の別trajectory価格と単純比較して優劣を決めない。

### 6.2 Source命令数の期待会計と長round制約

有限CDF質量から、各basis fの確率P_fとgate列長L_fを使い、r4 tailの期待命令数を厳密に計算した。

`E[C_tail,source] = 4 + 2 Σ_f L_f [P_f + 3 P_f(1−P_f)]`

これは実builderの隣接同basis取消をindicatorで数えたもので、短回路価格のQ倍ではない。1stepの決定論headは728 source instructions、tail期待値約127.559176、step期待値約855.559176。各q、最上位scalar一回、X/Y wrapper一回を明示して全shotへ重み付けすると、source命令数期待合計は約 **1.01187186e13**。正確な有理数をrun12に保持した。

この座標ではhigh-level Gaussian gateやcontrolled rotationも1命令であり、**compiled RZ／CX／T／depthとは異なる**。compiled費用の上界・期待値・費用優位の代用にはしない。

m17のnative命令数はtrajectoryによらず **101,711,873〜133,693,441** の構造範囲になる。さらに、現行のmaterialized canonical hasherでは、全回路の決定論head部分のJSONだけでも **2,366,030,741,504 bytes** を保持する必要がある。これは短回路の時間／RAMの経験外挿ではなく、固定serializerの定義展開と反復個数からの下界であり、確認時available RAM約57.5 GBを超える。

既存の同一hash-policy streaming実装と実H4 Q2の保存検証（`96efc6123b8bea3ff66d5dea729e0dcc3b9908b4` の `docs/research/b2_q2_fingerprint_h4_20261107/REPORT.md`）も確認した。streamingはこのJSON allocationを減らすが、flat circuit・wrapper copy・whole compiler DAGを消さない。既存の分割compileやboundary fusionはwhole価格と一致しない事例があるため、代替価格へ無断採用しない。

今回m17の巨大whole compile／価格MCは実行していない。あらゆる方法で不可能と断定せず、**現在適格なpipelineで全round期待価格が取得できることは未確立**と結論する。短roundのMCを追加しても長roundの不足を埋められないため、1 trajectoryの接続確認を越えて無目的な価格標本を増やさない。

### 6.3 比較可能なもの・未比較のもの

| 方式 | 同じ条件との対応／今回の扱い | 最終compiled総費用 |
|---|---|---|
| 既存K2 finite-RTE PR | 同じ元H/state/DF12/LD3/sector36の旧P1参照。保存7,805,784 shotsは旧engineering allowance下で、今回と保証の水準・予算意味が同一ではない | 未比較 |
| 既知paired K0 PR | 同じlaw／設定／入力／gate列なら今回α0-Pと物理信号・価格が同じ | 同一回路の価格。別の物理利得はない |
| C-α保証を適用したα0-P | 固定law接続と全18roundの包含付き条件付き会計を取得。保証に由来する正当化の改善 | 113,252 shotsとm0の限定価格まで。全Gは未取得 |
| 強Full PF | 同じ元H/state/sector36の旧／HP記録を参照。旧8armの18,540 shotsは条件付き数値記録で、今回新たに資格化・再計算していない | 同じcompiler／全round価格が今回未取得 |
| Truncated PF | 元Hとの差・精度・成功条件を揃えた今回の適格な保証／価格を取得していない | 未取得 |

強PF参照は `origin/codex/track-a-b2-p1-hp-branch-shots-20261107:docs/research/track_a_b2_p1_hp_branch_shots_20261107/REPORT.md`。旧STOPと後続HP成果を区別し、各原SOURCEを保全した。shot数の大小だけで総費用順位を出していない。K2→K0の既知law変更による物理的減衰差と、同じK0への保証適用の寄与を混同しない。

## 7. 精度・検証・資源・保全

新しい人工確認は[gate/CDF 4件](../../../tests/test_calpha_sampling_law.py)、[raw会計／branch 3件](../../../tests/test_calpha_rpe_accounting.py)、[価格adapter／source期待会計 3件](../../../tests/test_calpha_price_connection.py)の計**10件PASS**。全Fockの小人工gateとnumber blockの一致、inverse、CDF規約、非unitary参照のhead差、整数shot、prior端近くのadversarial位相で既存decoder、K0角の拒否、両軸測定、短い全event列を列挙したbasis取消期待値との一致を確認した。分子入力・旧runをtestとして再実行していない。

CPU affinity32、RAM約62 GiB／available約53 GiB。独立13 frameの区間演算は13 process、BLAS／OpenMPは各process1threadとして過剰並列を避けた。共通operatorと反復／geometric sumを共有し、q回の逐次信号作用や巨大回路を展開しない。共有package・venv・他ユーザーjobは不変更。GPUを使用せず、未測定のspeedupは主張しない。

run08の技術例外は当時のSOURCEを `executed_source/` に保全し、比較mappingを修正したrun09を別namespaceへ保存した。旧run01〜run07の再実行ではない。run09 5.242秒、run10 59.447秒、run11 6.764秒。run12／run13は保存値・scalarからの会計／接続であり、新signalやground solveはない。

原始H／state／DFの科学identityと、全runの実行SOURCE／raw hashは各結果とmanifestから追跡する。private配列・HDF5・private operand・秘密情報はcommit対象外。公開後のcommit固定はlocal実行の証拠を保つもので、immutable CIや外部再現が完了したとは扱わない。

## 8. 判断と次の担当

取得した証拠では、**compile前の固定α0-P gate lawへ保証を接続でき、全18roundの条件付きRPE成功・shot会計まで進めた**。総compiled期待価格と、強Full／Truncatedを含む同条件比較は未完成であり、C-α独立研究の新規性や総費用改善の最終判断は残る。

**Codexの次の技術候補**は、同一gate列・同一compiler・同一fingerprintを保つ長round pricing経路でflat circuit／wrapper／whole DAGの保持を扱い、その期待価格と不確かさを取得すること。これは今回未実施の大きい技術課題として具体化したもので、短回路価格の外挿では代替できない。

**ユーザー／GPTの次の判断**は、今回の保証対象・telescoping精密化と証拠を研究成果としてどう採択するか、同じ既知K0との差分が保証・会計の研究寄与として十分か、長round費用取得へさらに進む研究価値があるかを判断すること。Codexは重要な新定理の正式採択・研究の統合／終了を代行しない。

成果・旧STOP・失敗履歴を保全し、長round価格の技術的な不足を明示して**mandatory STOP**。全compiled総費用が小さい、実機95%成功、新しいsamplingアルゴリズムを実証した、という主張は行わない。
