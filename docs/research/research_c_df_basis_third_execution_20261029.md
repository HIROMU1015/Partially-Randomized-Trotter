# 研究C：保存γ0第三DF・141-callでnative入口の技術STOP

2026-10-29 JST。保存直鎖H4・1.00 Å・STO-3G、通常number CDF rank8 γ=0、Nα=Nβ=2/sector36、元frame順L_D=2、partial ordered symmetric S2 δ=h=.1/s1/r128/K0、m0..12。Own-DF ground-state oracle、状態準備費用除外、CA=.00159 Ha、DF reserve=CA/8。

**固定493-call契約を一度実行し、141 attempts＝140 COMPLETE＋1 technical FAILでSTOPした。** Sector36の一般K/独立CAR、own-ground/alias、全13round PF/RTE、条件付きRPE/shots/損益分岐まで完了したが、native入口の`full_law_gate`が`KeyError: 'components'`となった。新しいadapterが既存B2へ渡すtail metadataの必須fieldを欠いていた。測定されたoperator非同値のFAILではなく、full signed-law gateは未判定である。

Sampling／kernel build／compile／QPY／DF holdoutは0。新fit/optimizer/rank探索/DF誤差再計算/retry/resumeも0。契約のSTOPに従い、修正SOURCEでの再実行や残枠の再利用は行わない。Native価格、外部予測誤差、条件付き総費用はnullのままである。

## 固定SOURCE・入力・preflight

| 対象 | 固定値 |
|---|---|
| 公開実行SOURCE | `14270239ad2f448feeedb4751e4174a3b571a430` |
| Remote tree | `5144a33976ff4784d09c5a927de66df86adc4919` |
| Scope | `C_DF_BASIS_SWITCH_THIRD_GAMMA0_V1` |
| [契約](research_c_df_basis_third_contract_unapproved_20261029.json) SHA256 | `b3af0d017bb9e018380b30f031c4814dbd28c7dc1ee84f0b6d5b4715f4b4833a` |
| [SOURCE112](../../artifacts/research_c_df_basis_switch/2026-10-29/technical/SOURCE.json) SHA256 | `fa65695c5f8f3c402c7618e831296d59f006d1f6e4a8a7a4653adf86ab445912` |
| 保存γ0 J/U | `artifacts/research_c_pr_aware_df/2026-10-27/run01/baseline/final.npz` |
| Export SHA256 | `f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02` |
| 元fit SOURCE | `e1085c95f5c01b68566e17d543ace5832cb94061` |
| 元P-H4 SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| Fresh output | `artifacts/research_c_df_basis_third/2026-10-29/run01` |
| 実行branch | `codex/research-c-basis-third-gamma0-execution-20261029` |

[設計・人工検証](research_c_df_basis_switch_optimization_design_20261029.md)で用意したrunner、CAR、PF/RTE、RPE、native referenceを再利用し、実行SOURCEは無変更。元g補正κ、scalar一回、frame順、一般対称J/実直交U、数値allowance、P0条件を維持した。`unapproved`という歴史的ファイル名/metadataを変更せず、チャットの明示許可に基づき実行した。承認証明・承認原文保存・新grantは作成していない。

[実行preflight](../../artifacts/research_c_df_basis_third/2026-10-29/execution_preflight.json)はSOURCE112・保存input93・旧保護1667件、契約hash、依存版、fresh output不在を照合しPASS。固定済み131人工/回帰PASSの証拠を確認し、実データの成功と同一視しなかった。準備時の許容条件を今回の結果で変更していない。

## 完了した検証と未実施部分

| 検証 | 状態 |
|---|---|
| 旧2cohort訓練64行・共通interceptの6metricモデル固定 | COMPLETE、第三DF読込より前に保存 |
| 保存DF予算 | 旧sector誤差＋guard=1.0297636237465247e-8 HaのPASSを再利用。新norm計算なし |
| Sector36一般K/独立CAR | base＋8 frameの9因子PASS |
| Own-DF ground/alias | PASS、energy=-2.1663874524547952 Ha、residual=2.8265151640433296e-15、alias margin=1.6179938779914944 |
| Own tail law、符号/identity/mass、exact-key geometry | PASS |
| PF/RTE/phase十分条件 | 13/13 POTENTIALLY_FEASIBLE |
| RPE | evidenceはUNQUALIFIED、登録2仮定はCONDITIONAL_CA_PASS |
| Native GO | GO_FIXED32。実価格や損益分岐の正負による選別なし |
| Full256/sector36 raw-CAR factor照合 | 9因子＋sector sum PASS |
| Full256 signed-tail law gate | **TECHNICAL FAIL、metadata欠損で未判定** |
| Gaussian/shell/native32・full512 circuit gate | NOT_RUN |
| DF holdout・第三DF MAE/RMSE・native価格・総費用 | NOT_RUN／null |

[Representation CSV19行](../../artifacts/research_c_df_basis_third/2026-10-29/presentation/representation19.csv)はsector因子9＋full/sector因子9＋sector sum1の保存残差を転記した。Full/sector因子のPASSを、生成していないcontrolled回路のPASSと扱わない。物理DF guard・これらのbinary64 residualは厳密enclosureではない。

## 全13roundと条件付きshots

PF/RTEは第三DF自身のHamiltonian/lawで新しく評価し、旧Proposed/P4のPF・ground・shotsを移植していない。誤差はsector36の無次元operator誤差で、HaのDF誤差とは別である。表のNは各roundの**X軸/Y軸それぞれ**の十分shots処方で、実際にshotsをdrawした数は0。

| m | PF operator error | Corrected RTE error | Attenuation | N/axis：ideal | N/axis：assumed budget |
|---:|---:|---:|---:|---:|---:|
| 0 | 6.243235e-5 | 5.952529e-5 | 0.9998092744 | 512 | 514 |
| 1 | 1.245275e-4 | 1.189495e-4 | 0.9996185852 | 512 | 514 |
| 2 | 2.463848e-4 | 2.370805e-4 | 0.9992373160 | 513 | 515 |
| 3 | 4.722605e-4 | 4.677079e-4 | 0.9984752136 | 514 | 516 |
| 4 | 8.046709e-4 | 8.880172e-4 | 0.9969527523 | 516 | 518 |
| 5 | 9.699112e-4 | 1.499472e-3 | 0.9939147902 | 520 | 522 |
| 6 | 1.310800e-3 | 2.055257e-3 | 0.9878666102 | 528 | 530 |
| 7 | 2.475065e-3 | 2.713857e-3 | 0.9758804396 | 543 | 545 |
| 8 | 4.510221e-3 | 5.040407e-3 | 0.9523426324 | 575 | 577 |
| 9 | 8.979491e-3 | 9.667520e-3 | 0.9069564894 | 645 | 648 |
| 10 | 1.794428e-2 | 1.932518e-2 | 0.8225700737 | 815 | 819 |
| 11 | 3.584732e-2 | 3.876478e-2 | 0.6766215262 | 1304 | 1309 |
| 12 | 7.162597e-2 | 7.902517e-2 | 0.4578166897 | 3380 | 3396 |

全rowはPOTENTIALLY_FEASIBLEだが、未知のsynthesis/law errorを実際に0と認定したわけではない。[精密値13行](../../artifacts/research_c_df_basis_third/2026-10-29/presentation/rounds13.csv)、[3lane×13roundのRPE39行](../../artifacts/research_c_df_basis_third/2026-10-29/presentation/RPE39.csv)に保存した。CA=.00159 Ha、βtotal=π/6、βstat=π/12、αaxis=.01/26、αtotal=.01は固定のまま。

| Lane | 保存判定 | 全X/Y shots処方 | W | S_RZ |
|---|---|---:|---:|---:|
| evidence | UNQUALIFIED_ROUNDS | null | null | null |
| ideal_exact_native | CONDITIONAL_CA_PASS | 21,754 | 35,925,928 | 76,139 |
| assumed_raw_signal_budget | CONDITIONAL_CA_PASS | 21,846 | 36,090,788 | 76,461 |

後二者はそれぞれexact native＋ideal finite-RTE IID仮定、および合成/law誤差各≤.001A_mという未資格の登録仮定である。Energy側の保存数値条件1.2783276208742716e-3 Haは両仮定でのpilot値で、actual CA/保証shotsはnull。Evidence laneを条件付きPASSへ昇格させない。

条件付きserial会計は`C_RZ(μ)=W μ+S_RZ`まで得られたが、μは未測定であり総費用を算出していない。Ideal仮定のM4-r128/P4に対するRZ損益分岐価格はそれぞれ**6516.678205111361／7666.394848380813**。Assumed-budget仮定ではM4比6515.664971799729。[全6指標×4比較の24行](../../artifacts/research_c_df_basis_third/2026-10-29/presentation/breakeven24.csv)に保存した。これらは未測定価格が満たすべき条件で、費用優位の観測ではない。

## Basis-switchの観測とDF holdoutの未完了

第三DFのexact λ_R=**2.2097618367788057 Ha**、P_same=0.36641952146278217、期待basis切替数=**80.46472077422668**、期待runs=81.46472077422666。期待Z/ZZ event数は62.856425715367514／65.14357428463248。Basis keyごとのqとframe別qは[law descriptor](../../artifacts/research_c_df_basis_third/2026-10-29/run01/reference/law_descriptors.json)に保存した。実trajectoryの切替数はsampling未開始のため未取得。

既存2DFの保存値はM4 Proposedでλ=1.4424298856 Ha／期待切替58.38932609、tail-P4でλ=1.5161659324 Ha／65.23827188（[148-call報告](research_c_df_basis_surrogate_repair_execution_20261029.md)の引用）。今回のlaw descriptorからnative価格差・因果効果・新DF汎化は判断できない。

[共通interceptの固定モデル](../../artifacts/research_c_df_basis_third/2026-10-29/run01/training/frozen_models.json)はM4/P4の64行だけで、第三J/U読込前の最初の8 callsにおいて保存された。RZの訓練結果はmean=14561.953125、A=`5931.298256478658 + 140.9086509146342×switches`、B=`5829.280151524529 + 68.84587732548052×overlap_path`。訓練18モデルは[CSV](../../artifacts/research_c_df_basis_third/2026-10-29/presentation/training18.csv)にも転記した。

第三DFのnative価格がないため、これらの式を新DFで評価/調整せず、DF holdoutのMAE/RMSE・資格判定は未実施である。第三DFの教師を使うintercept推定は行っていない。旧CのSURROGATE_NOT_QUALIFIED、静的DFが少数かつ同一H4系列であるという限界、head/basis固定費未分離は不変。

## STOP原因と人工testの限界

`c_df_basis_third.stage_reference`の`tail_gate`は、既存B1 tail reportと`atoms_catalog`を返すが、`components`を付けていない。`c_df_basis_third_native.execute`はそのdictをそのまま既存B2の`law_gate`へ渡す。B2はλ/identity/atom数に加え、`saved['components']`の確率列と実tail確率列を照合するため、このfield参照でKeyErrorとなった。λや位相allowanceを緩める問題ではない。

P4の既存`tail_record`にはcomponent_id/probability/coefficient_abs/sign/basis_id/frameの列があり、新adapterでこの接続を落としていた。今回のartifactではschemaを修復・上書きしていない。[失敗記録/traceback](../../artifacts/research_c_df_basis_third/2026-10-29/run01/failure.json)と[console原本](../../artifacts/research_c_df_basis_third/2026-10-29/science_console.log)を保存した。

準備時の新493-call人工end-to-endではB2 `law_gate`をmockしていたため、**新adapterのmetadataを実B2関数が受け取る境界を検証していなかった**。別の人工full512回帰は旧P4 adapterの正しいmetadataを使用しており、この欠落を捕捉しない。131 tests PASSを実データ接続の完了と扱えない理由をここに記録する。

次の最小技術修正候補は、既存P4と同じcomponent-record構造を第三DF metadataへ接続し、新第三DFの人工metadataを**mockしない実B2 law_gate**へ渡す回帰試験を加えること。新fitter/compiler/samplerや許容差変更は不要である。この候補は本実行SOURCEに適用しておらず、未承認の再実行も行っていない。

## 科学call・資源・原本

| 段階 | COMPLETE | FAIL | 状態 |
|---|---:|---:|---|
| T：旧2DFの訓練固定 | 8 | 0 | COMPLETE |
| A：第三DF reference | 64 | 0 | COMPLETE |
| RPE | 48 | 0 | COMPLETE（evidence未資格は維持） |
| Price/GO | 8 | 0 | COMPLETE |
| Native入口 | 12 | 1 | full_law_gateで技術STOP |
| Native32以降・DF holdout・最終runner summary | 0 | 0 | NOT_RUN |
| **合計** | **140** | **1** | **141 attempts／上限493** |

[Kind別call表](../../artifacts/research_c_df_basis_third/2026-10-29/presentation/calls.csv)、[原journal](../../artifacts/research_c_df_basis_third/2026-10-29/run01/science_calls.jsonl)、[STOP](../../artifacts/research_c_df_basis_third/2026-10-29/run01/STOP.json)。消費済みrun01をresumeせず、未実施分を「残枠」として別runへ移送しない。

実行前32 affinity CPU、空きRAM約52.5 GiB、load約1.1、CPU/memory pressure avg10=0を確認。既存resource planはsector13並列/native32並列、BLASはworkerごとの既存調整。今回はnative並列開始前に停止した。新しいCPU/RAM/worker/wall capなし、他job・共有環境変更なし。Task-local cache、準備venvはread-onlyで使用した。

GNU time実測wall **5.06秒**、user54.95秒、system47.63秒、max RSS745924 KiB、swap0、exit1。[実測](../../artifacts/research_c_df_basis_third/2026-10-29/process_time.txt)。Runner内wall4.706135903秒、CPU98.318664038秒。短いprefixの実測であり、native32完走時間や高速化倍率を推定しない。

[Postflight](../../artifacts/research_c_df_basis_third/2026-10-29/execution_postflight.json)でSOURCE112、input93、旧保護1667件、旧SOURCE103の不変を照合した。Raw35件をhash固定し、STOP/旧結果を保全した。Runnerは最終`result.json`へ到達していないため、そのrawファイルを捏造せず、別の[公開用summary](../../artifacts/research_c_df_basis_third/2026-10-29/execution_summary.json)をparentに保存した。CSVも完了済みJSONの表示転記だけで、追加PF・回帰・価格予測の科学callはない。

## 次の担当とmandatory STOP

**Codexの次の候補作業**は、component metadata接続と実関数を通す人工回帰の修正を、旧SOURCE/raw/STOPを保全した別SOURCEで準備すること。科学実行を再開するなら、今回完了済みのPF/RPE/訓練証拠をどこまで再利用するか、新SOURCE・新fresh output・新call範囲を具体化する必要がある。

**ユーザー／研究C**がその修正・新しい限定実行を判断する。今回のPF十分条件PASSやnative GOから、cost-aware新fit、η採択、第三DF価格/総資源優位、一般化・因果効果へ進まない。

本実行のSOURCE・部分結果・call・STOPをcommit/pushし、remote SHA/treeと成果物を照合後、**mandatory STOP**。Retry/resume・追加科学計算・詳細GPT科学レビューは実施しない。
