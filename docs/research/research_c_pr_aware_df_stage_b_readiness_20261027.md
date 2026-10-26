# 研究C：PR-aware DF Stage B接続・人工検証・限定実行準備

2026-10-27 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。起点：`9ae5942d02572498b6ee1b5f309ca504ed58c6ef`。

**Baseline／Proposedの2種類の一般J CDFを、共通のPR law・sector reference・native合成へ接続した。新15＋既存41＝56 tests PASS。最初の科学実行はB1だけ、最大141 logical units・2 arm×13 roundを推奨する。** 今回の実分子decode・再解析・sampling・回路build/compile・新fit・科学callはすべて0。B2/Cは実行しない。

[未承認B1契約](research_c_pr_aware_df_stage_b_contract_draft_20261027.json)／[SOURCE・人工log・passive確認・technical STOP](../../artifacts/research_c_pr_aware_df_stage_b/2026-10-27/technical/)。実装SOURCE：`366cb090a2c05c6d7ffeafdba2443bfc64b8a035`。正式共通運用指示2026-10-25版§8.1/8.4/10/10.1/10.3に従い、承認証明書・承認原文保存・新grantは作成しない。科学実行は別途の明示許可後のみ。

## 1. Stage Aの固定成果と今回の境界

[公開Stage A報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9ae5942d02572498b6ee1b5f309ca504ed58c6ef/docs/research/research_c_pr_aware_df_stage_a_x0_v2_execution_20261027.md)をGitHub connectorでも取得し、local blobと照合した。直鎖H4・隣接1Å・STO-3G、Nα=Nβ2／sector36、通常number CDF8、元frame順prefix LD2、共通初期値、γ0/.1、既存L-BFGS-B options=Noneでの保存結果は以下のとおり。

| 保存値 | Baseline | Proposed |
|---|---:|---:|
| exact tail mass [Ha] | 2.2097618367788057 | 1.4424298855756243 |
| DF norm＋guard [Ha] | 1.0297636237465247e−8 | 1.229561498918692e−8 |
| optimizer success／nit | true／97 | true／1129 |

削減率34.72464491113348%、`STAGE_A_PROXY_AND_DF_PASS`、30/30 COMPLETE、retry/resume0は保存記録の引用で、今回の再計算ではない。停止理由は相対目的減少で、gradient収束・大域最適性を前提にしない。

科学SOURCEは`e1085c95f5c01b68566e17d543ace5832cb94061`、数値実装SOURCEは`5a185964090f2cb8d19d445d41cf520c4d8ed2c7`。両`final.npz`のbytes・NPY header・SOURCE／result／STOPの来歴を固定する。NPY header確認はJ/U/x/jacのshape・dtype・orderだけを読み、科学配列payloadをdecodeしない。

新しいattenuation、PF／補正RTE誤差、信号、native費用、shots、同精度総資源、新規性は未取得／未資格。過去のCDF/XDFのPF誤差・operator・費用やsampling分布を、この新しいJ/Uへ流用しない。

## 2. 接続と再利用

```
Stage A baseline/final.npz ─┐
                           ├─ general-K ─ native model ─ original prefix LD2
Stage A proposed/final.npz ─┘                 │                ├─ head
     original h/g/c → original-g κ           │                └─ tail I/Z/ZZ
                                             │                   ↓
                                 common precise Gaussian      fresh ideal RTE law
                                 (B2 only, opt-in)             (B1, no sampling)
                                                                ↓
     new raw SectorCAR generators → own-DF H/ground → partial S2 / corrected RTE
                                                                ↓
                                  2 arms ×13 round: PF/RTE/A/signal + feasibility
```

CDF/XDF固定backendを継承してlabelだけ入れ替える方式は採らなかった。そこには旧fit export schema、XDF生成、旧operator・QPY・metrics再利用、CDF/XDF固定cacheがあるためである。新adapterとB1-only runnerを追加し、下表の計算関数を直接再利用する。旧SOURCEは編集しない。

| 既存SOURCE／関数 | 再利用内容・適合範囲 |
|---|---|
| `c_pr_aware_df.coefficients` | Stage Aと同じgeneral-J→非identity Z/ZZ係数massを独立に照合。loss・γは変更しない |
| `c1_general_k_adapter` | 一般実対称J、number convention、interleaved spin、元g補正κ、I/Z/ZZ、負／微小非ゼロ係数を保持。rank-one square化なし |
| `c1_stage2_native.NativeModel/NativeBlock` | base＋元順8frame。2 armは`CDF_baseline`／`CDF_proposed`として別source identityで扱う |
| `c1_stage3_connection.split_native_model/configure_rte` | 元順prefix、tail identity抽出、既存辞書・paired finite-RTE再生成 |
| `pr_dictionary_validation.prototype`／`trotterlib.rte` | exact係数、λ、正規化、確率、cutoff・event・PCG64 samplerの意味を維持。B1ではsamplerを呼ばない |
| `c1_molecule_extension.SectorCAR` | 新J/Uごとに独立raw density base/frameを構成。既存general-K sector経路と各因子／sumを比較 |
| `c1_stage3_reference.reference_kernels/reference_round` | 新DFのCAR generatorsからpartial S2と補正有限RTE、反復二乗法、sector operator誤差・oracle信号を評価 |
| `c1_stage3_molecular.MolecularBackend._ground` | 新physical／Baseline／Proposedのsector eigensolveとground保存だけ再利用。旧backendのprepareは呼ばない |
| `c1_stage3_feasibility.classify_round/summarize` | 既存の登録十分条件を満たすか判定。shots関数は呼ばない |
| `c1_stage3_integrated.Calls/resource_plan` | 既存のcall計上・FAIL/STOP・実資源による並列度選択だけ再利用。旧20候補runnerは起動しない |
| `c1_h4_strong_baseline.spin_adapted_eigh/StrongBuilder` | 両CDFのκに同じexact spin-lift基底政策、head各半分の既知relative Gaussian／analytic vacuum phaseを適用 |
| `c1_h6_native_precision.PrecisionBuilder` | headとtail双方で同じexact-zero/LAPACK/atan2 Gaussian emitter。旧full-frame providerの低精度経路を使わない |
| `c1_stage3_native_bridge.partial_kernel` | 修正済み符号規約を再利用。旧connection内のsign-dependent angle検査は呼ばない |
| `c1_h6_compiler_precision`／`CompilerProfile`／`circuit_metrics` | 固定P0／MPFR160 declared phase／RZ・CX・depth等。今回の使用は人工回路のみ |

新SOURCEは`c_pr_aware_df_stage_b.py`（arm・law・reference／native接続）と`c_pr_aware_df_stage_b_runner.py`（B1限定orchestration）、入口は`run_research_c_pr_aware_df_stage_b.py`。新fitter、CAR代数、Taylor law、Gaussian合成器、compiler、shot推定器は実装していない。

## 3. Tail・law・identityの意味

各armで元`g`からκを補正し、核scalarをbaseに一回だけ保持する。LD2はone-bodyの後の元frame F0/F1を決定論headとし、F2..F7をtailへ置く。weight順sort・frame探索はしない。

number fragmentは`D_l=.5 Σ_ijστ J_lij n_iσ n_jτ`、`K_ab=J_(a//2,b//2)/2`。既存adapterの`c_Za=−.5 Σ_b K_ab`、`c_ZaZb=K_ab/2`とStage Aの`coefficients`が一致することを人工検証した。identityは正確な決定論位相へ抽出し、λ_Rには混入しない。残る符号付き回転Pauli atomのabsolute massがλ_Rである。

新しいJごとにtail／確率／正規化を再生成する。source hashとarm labelをcache／frame identityへ含め、別armの分布やGaussian cacheを流用しない。B1では新λをStage A保存exact massとも固定allowanceで照合する。閾値はmass比較のrelative1e−12／absolute1e−14、matrix component1e−10の数値整合検査で、厳密丸め証明ではない。

K=0でも1 microstepあたり1 signed random rotationがあり、r=16なら1 occurrenceは16 eventsである。`τ=λ_R δ/(s r)`、`A_m=(1+τ²)^(−r s 2^m/2)`という既存paired-RTE式を人工値で照合した。保存H4のAは今回計算していない。

zero tailは既存`deterministic_only_certified`分岐を使い、空のrandom law／A=1を扱う。1e−200程度の人工非ゼロ係数もtailへ残ることを確認した。正係数の確率underflowを含む実装上の異常は既存lawの規則でFAILとし、thresholdで削らない。

将来B2のtrajectory keyはarm/source/round/axis/shot/occurrenceを含む。同一sampleを全coherent occurrenceで繰り返す政策にはしない。数値referenceはideal finite-RTE IID lawを条件付きの定義として評価する。PCG64・binary64で実装するsamplerとの差の上界は未資格で、implemented-law errorはnullを保持する。

## 4. PF／RTE・信号とsector reference

新J/Uから各armのbase＋8frameを構成し、各因子とsumを独立general-Kへ照合する。tailについても、raw CARの和と`Σ|c_a| signed_atom_a + c_I I`の一致、atomのHermiticity／involutionを確認する。Stage AのDF norm＋guardはhashで固定した同一最終J/Uに対応する保存値として再利用し、DF誤差を測り直さない。

B1の参照は全因子がNα/Nβを保存するsector36で行う。これは個々のGaussian gateをsectorへ射影する回路検証ではない。人工full空間とsectorのcomplete-factor／PF／RTE一致を別に検証した。legacy SectorCARとQiskit full空間のbasis順序は異なるため、`bit_reverse_indices(8)[sector.basis_indices]`を明示し、B1のreference cacheにも保存する。

時刻h=δ/s=1/40、headの元順half sweep→exact tail→逆順half sweepがideal partial S2。中央tailだけを既存補正有限RTEへ置換したkernelと比較する。mごとにn=s2^m、t=δ2^mとして

- PF誤差：`||U_PF(h)^n−exp(−i H_DF t)||₂`
- 補正RTE誤差：`||U_corrected(h)^n−U_PF(h)^n||₂`
- raw complex signal：`A_m <ψ_DF|U_corrected(h)^n|ψ_DF>`

を別々に保存する。元物理H・Baseline H・Proposed Hのground／aliasは新たに確認する必要がある。各DF自身のground-state oracleを仮定し、状態準備費用は除外。物理DF誤差[Ha]と無次元operator／signal誤差を根拠なく加算しない。

人工反例では同じ総H=X+Zに対し、baseline head=−H／tail=2Hは可換でPFがexact、proposed head=X／tail=Zはλが小さく減衰は改善してもPF誤差が増える。これは依存関係を反証する人工例で、同じCDF fitの実分子結果ではない。

全13 roundで正の有限A、既知`εPF+εRTE≤sin(π/12)`、保存DF reserve内なら`POTENTIALLY_FEASIBLE`。十分条件未達は`INFEASIBLE`（物理的不可能の意味ではない）、必要値不足は`UNRESOLVED`。未知の合成／law誤差は0にせず、潜在的適格性をCA達成へ昇格させない。二値判定を結果に合わせて変更しない。

## 5. Native接続と公平性

両armで、κのexact interleaved spin-lift基底、保存CDF frameのfull U、PrecisionBuilder、StrongBuilder head内relative fusion、P0 opt1/seed17/rz,sx,x,cx/approximation1/qubits_initially_zero=false、MPFR160 phaseを共通にする。baselineだけ旧Gaussianへ戻したり、proposedだけsingleton基底へ置換したりしない。

head各半分の相対変換は既存`V_following† V_previous`とanalytic vacuum phaseを使用する。tailは同じPrecisionBuilderのfull-frame基底を既存event emitterへ渡し、隣接exact同一basisだけを共有する。head/tailを跨ぐ一般相対融合は未実装のため無効とし、その追加削減を仮定しない。P0が実際に行った最適化だけを将来metricsに反映する。

`build_kernel`は修正済みnative bridgeを呼び、核scalar・tail identity・signed event角度・chronological順序を維持する。`trajectory_reference`はGaussianとは独立なCAR指数と既存`event_unitary`から参照を作る。`assess_native_probe`はancillaを含むraw／compiled／reference、compiled/raw、actual adjoint、unitarityを全operatorで比較し、phase fittingを行わない。人工X/Y wrapperもcomplex expectationと照合した。

これらは人工の1 trajectory単位の検証であり、実H4・全sampling lawの合成上界を資格付けしていない。将来B2では新J/Uのfull512 controlled回路・新独立referenceで検証し、旧H4のoperator・signal・metricsを使い回さない。

費用はcomplete controlled partial-S2 kernel一体compileのRZ/CX/depth等を取得する。head/tailやeventを個別compileした価格の和を全trajectory価格とみなすproxyは拒否する。`cost_point`は有限bit校正のsample mean／sample standard errorだけを出し、N=1のSE、ideal母平均、全law support／保証interval／資源優位はnullとする。sample最大値を厳密support上界へ昇格させない。

## 6. 人工・回帰テスト

| 試験 | 結果 |
|---|---:|
| 新Stage B：general J/spin/order/identity、負・微小・zero tail、mass/law、分布再生成、独立occurrence seed、full/sector、PF反例、negative-event full controlled／adjoint/X/Y/P0、cost proxy拒否、141-call end-to-end／STOP、header／byte preflight | 15 PASS |
| 既存strong-baseline＋feasibility回帰：phase・Gaussian方向・frame順・compile故障、未知誤差、sampling／十分条件、fresh／STOP等 | 41 PASS |
| 合計 | **56 PASS** |

試験入力はすべて人工配列・人工operator・mock metadataで、Stage A保存科学配列は使っていない。初期人工検証で、frame-only candidateのcomplex accumulator dtype、sector/full bit順序、測定を除いたX/Y Operator化、metricsに含まれる追加欄を修正した。旧SOURCE／科学allowanceは変更していない。

再現時は既存evaluation env（Python3.11.1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、Nature0.7.2、OpenFermion1.6.1等）をread-only使用し、task-localのTMP/MPL/Numba cacheで下記を別process実行する。

```bash
"$EVAL_PY" -B -m pytest -q --tb=short -p no:cacheprovider tests/management/test_research_c_pr_aware_df_stage_b.py
"$EVAL_PY" -B -m pytest -q --tb=short -p no:cacheprovider tests/management/test_research_c_c1_h4_strong_baseline.py tests/management/test_research_c_c1_stage3_feasibility.py
```

## 7. 推奨する最初の科学実行：B1のみ141 units

候補条件はStage Aのfuture policyから維持する**未承認案**：保存H4・1Å/STO-3G、CDF8両arm、sector36、元順LD2、partial S2、δ=.1/s4/r16/K0、m0..12、CA=.00159／DF reserveCA/8、βtotalπ/6・βstatπ/12、candidate failure.01・axis.01/26、own-DF oracle・状態準備除外。

| logical unit分類 | 最大数 |
|---|---:|
| 共通input／Stage A evidence／sector context／physical operator | 4 |
| final J/U load／native model | 2＋2 |
| raw base＋8frame／各factor表現gate | 18＋18 |
| arm sum／tail law／tail gate／Stage A mass照合 | 2＋2＋2＋2 |
| physicalと両DFのground／alias | 3＋3 |
| partial-S2／corrected-RTE kernel pair | 2 |
| 2×13 exact own-DF evolution | 26 |
| 2×13 PF/RTE/A/signal評価 | 26 |
| 2×13 feasibility判定 | 26 |
| 両arm summary／比較 | 2＋1 |
| **合計** | **141** |

call分類はrunnerのLIMITSと一致し、人工完走で照合した。raw_factor／factor gateは各18個を個別計上し、tail gateは1 arm内のatom確認をgrouped unitとして契約に明記する。logical call、2候補、26 roundセル、calibration sample数を混同しない。

**B1のsample・実回路build・compile・shots・総費用・新fitはすべて0。** 全lawの平均operatorを既存有限Taylor referenceで評価するため、Monte Carlo sampling誤差は導入しない。binary64のoperator残差は数値pilotで、strict enclosureではない。

fresh outputは`artifacts/research_c_pr_aware_df_stage_b/2026-10-27/B1-run01`、retry/resume0。入力／SOURCE／表現／tail／mass／非有限値／ground-residual／alias不一致では依存する後続処理をSTOPしpartialを保存する。候補の位相十分条件未達では登録roundと他armを継続する。ground/alias未確認のままoracle信号を進めない。両armの正しさ確認後、各arm13 roundを独立並列処理する。

保存成果はfactor gate、ground、tail atom／係数／符号／確率／正規化／identity、kernel／H／basis mapping、26セルのPF/RTE/A/complex signal、候補分類、call/resource/STOP。保存DF誤差は別表示し、合成・実装law誤差・cost support・shots・CA・総費用はnullを維持する。

## 8. B2の最小拡張案とStage Cへの未取得証拠

B1先行が合理的である。固定政策のPF／RTE十分条件が満たせない場合に、大量のnative build/compileを先行する情報価値が低いためである。B1結果を確認してからB2の採否を決め、B1契約でB2を自動起動しない。

B2には今回の接続を利用できるが、実H4の科学実行契約は未固定である。点推定の初回案は32独立trajectory/arm、計64校正samples、各16 microsteps、最大64 distinct kernel build／64 P0 compile。新full512 reference・raw/compiled・actual adjoint・complex X/Y・metricsをsampleごとに確認する。32はprecision保証の標本数ではなくpilot案で、事後的な標本拡張はしない。

B2の最終call上限は、B1後に採択するfull correctness／shell／Gaussian cache／校正範囲へ整合させた別契約で固定する。今回固定した上限は**B1の141だけ**である。B2実行runnerを完成済みと主張せず、今回のnative接続・単一probe検証・cost-point helperをその再利用部品とする。

有限sampleの点推定からideal-law期待費用や全support保証は出せない。全law合成上界、finite-bit law差、費用support、全13 roundの統計roomとshots、共通CA・期待総RZ/CX/depthはStage Cへ進む前に別途取得・資格付けが必要である。B1でのPOTENTIALLY_FEASIBLEや単一native probe PASSはその代用ではない。

## 9. 保全と次の担当

passive preflightでSOURCE39件・保存input8件とNPY header・環境版を照合し、Stage A97ファイルのbytes不変と新B1 output未作成を確認した。新しい4 SOURCE／testファイルと技術資料だけを追加し、旧数値SOURCE・Stage A raw/STOP・旧契約・旧grant/ledger/markerを変更しない。manifest／保存input／環境をpassive preflightし、共有venv・cache・他jobへ影響する変更はしない。科学計算はsector36／既存expm・matrix powerを用い、実資源に応じた2 arm並列とBLAS協調を行う。CPU/RAM/worker/wallの人工上限は追加しない。

研究C／ユーザーはB1案・141-callの実行可否を判断する。明示許可後に限りCodexが固定SOURCE・入力・fresh outputを照合してB1を実行し、結果を返してSTOPする。B2/C・追加fit・科学条件変更・新規性最終判断・GPT詳細科学レビューへ自動進行しない。今回も公開・remote照合後、**mandatory STOP**する。
