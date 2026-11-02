# 研究C：同じ新J/Uの強決定論PF・公平比較の技術準備

2026-11-02 JST。起点 `aced1647b81e8d28091058cdc7b6e3e1ee54896a`。今回は技術調査・最小接続・人工検証・未承認契約の固定のみ。**新しい実H4 PF/RTE・shots・native build/compile・総費用計算・保存NPZ配列の数値decodeは0。科学実行へ進まず、公開照合後mandatory STOPする。** 別Codex-Bの資源寄与分析ファイルは編集していない。

## 1. 比較条件・保存候補

[統合実行](research_c_pr_integrated_execution_20261102.md)で事前規則により選ばれた`update005`を両方式で使う。直鎖H4・隣接1.00Å・STO-3G、number CDF rank8、Nα=Nβ=2/sector36、8 system＋last ancilla/full512。元g補正κ・nuclear scalar一回・spin/frame順0..7を保持する。新fit・rank/γ/LD/順序探索は行わない。

- selected J/U：`artifacts/research_c_pr_integrated/2026-11-02/run01/selected.npz`、SHA256 `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`。
- 元h/g/scalar snapshot：`artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz`、SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。
- 保存された独立SectorCARの同一J/U/G/H：`artifacts/research_c_pr_integrated/2026-11-02/run01/physical/update005.npz`。
- 元sector basisとown-DF ground：`artifacts/research_c_pr_integrated/2026-11-02/run01/physical_reference.npz`／`artifacts/research_c_pr_integrated/2026-11-02/run01/C/ground_candidate.npz`。
- 新PR科学SOURCE：`980b514c81eb415fe053f558d7d3c0b8cfb5b9b4`、結果公開は起点aced1647b8。元SOURCE manifestと契約、543-call STOP・selected identityを別々に保持する。

保存JSONの既報値は、DF sector norm5.233373518021879e-9Ha＋guard3.227392702910976e-9Ha、CA/8=.00019875HaにPASS、κ=`original_g_corrected`/scalar=`nuclear_once`である。今回これらを再測定・再解析していない。Header確認はJ/Uがfloat64各(8,4,4)、sector generatorsがcomplex128(9,36,36)、groundがcomplex128(36,)というschemaだけで、NPY payloadを読んでいない。候補はU一次未達の有限endpointであり、最適解・J/U同時収束点と呼ばない。

| 条件 | 保存新PR | 準備した新強決定論D |
| --- | --- | --- |
| J/U・物理DF誤差 | 同じselected、同じengineering guard | 同じselected、保存値再利用 |
| evolution | partial ordered S2・元順LD2＋finite RTE | 全9因子のordered symmetric S2 |
| 時間 | δ=h=.1、s1、m0..12 | 同じδ/h/s/round、n=2^m |
| RTE固有項 | r128/K0、Aと補正RTE誤差は保存結果 | r/Kは非適用・null。A=1、RTE/law誤差0は決定論の定義 |
| RPE/energy | CA.00159、DF reserve1/8、βtotalπ/6、βstatπ/12、failure.01/axis.01/26 | 同条件、同じX/Y・own-DF ground oracle・準備費用除外 |
| native | precise spin-lift κ、head内relative/exact basis-key tail共有 | precise spin-lift κ・元full-frame U、全sweep relative fusion |
| compiler | Qiskit1.3.0、P0 opt1/seed17/rz,sx,x,cx/approximation1、qubits_initially_zero=false | 同じP0、MPFR160 declared phase、global phase fitなし |
| 価格 | 固定32 finite-bit trajectoryの標本平均/SE | **固定決定論kernel 1件**。標本数32を割り当てない |
| 会計 | 各occurrence独立、fenced serial、head/readoutはinterrogationごと1回 | 同じserial metrics/fence・head/readout規約 |

合法な融合範囲の違いを方式の実装として明記する。PRに未実装のcross-head/tail相対融合を追加せず、その削減を費用から差し引かない。Dも1 h-step内の既存最適化までで、全round長回路の再compileやkernel境界を越す未観測融合を仮定しない。

## 2. 再利用と最小差分

旧[M2同一J/U強決定論](research_c_pr_aware_df_M2_same_JU_deterministic_execution_20261028.md)、[強XDF baselineのCA会計](research_c_c1_h4_strong_baseline_ca_resources_execution_20261026.md)、[証拠整理](research_c_evidence_synthesis_and_decision_readiness_20261031.md)を確認した。新sourceは次の3件のみで、旧M2の71 SOURCEファイルはbyte不変である。

- `src/trottertracks/c_same_ju_strong_d.py`：1保存候補のinput/reference接続、条件付きdispatch、比較。
- `scripts/run_research_c_same_ju_strong_d.py`：default passive、別許可後の明示`--execute`。
- `tests/management/test_research_c_same_ju_strong_d.py`：人工・mock接続試験。

旧M2 entryは2 arm・旧A/B1/B2/C1/external XDFへの専用provenanceを固定しており、そのまま再起動しない。既存`validate_arrays/sector_match/fusion_gate/evaluate_accuracy/numerical_compare`とnative/ref/PF/RPE/serial primitivesを呼ぶ、1候補用の薄いorchestrationを追加した。新optimizer、DF再構成法、Gaussian compiler、shot式・集計器は作っていない。

| 再利用 | 今回の使い方 |
| --- | --- |
| M2 validate/sector/fusion gates | selectedと保存物理J/Uのexact array binding、full256 CAR9因子/sector36/sum、元frame・relative方向 |
| b.prepare_arm・StrongBuilder | `ld=8`で全frame model、roleは旧APIの`proposed`タグを使うがsource identityは新selected hash。PR split/r/KをDへ移植しない |
| molecule.PF_rounds | 保存DF Hのeigenpairを1回だけ計算し、13 roundの独立exact referenceと反復二乗法へ共用 |
| StrongBuilder/precision/P0 | 元spin-lift κ、U、I/Z/ZZ・scalar、全sweep相対Gaussian、control0 identity/control1 S2 |
| full CAR/full synthesis/B2 XY | 全512入出力、raw/compiled/reference、actual adjoint、相対位相、complex X/Y、ancilla-only shells |
| M2.evaluate_accuracy | Frobenius/both-matrix norm growth・shell伝播、既存RPE・固定N独立margin・energy・serial総費用 |
| AccuracyTarget/aggregate_totals | PRと同じ理想合成の反実条件laneを別評価。物理DF誤差Haとsignal誤差を根拠なく単純加算しない |

保存DF norm・guard・physical ground/aliasはprovenanceを検査して再利用し、元physical HamiltonianやDF normは再計算しない。保存own-DF groundの残差/正規化/energyと最低固有値を新S2 exact referenceのeigenpairで確認する。新PRのPF/RTE/attenuation/shots/価格は保存値を参照し、再実行しない。

## 3. 二つのD laneと公平性の限界

主たるD laneは旧M2と同じ**合成誤差を含むlevel B数値pilot**で、kernel/head/X/Yのfull512 Frobenius残差と両行列のnorm growthを13 roundへ伝播し、整数shotsのmarginを確認する。丸め・expm・reference全体の厳密enclosureへ昇格させない。

加えて`ideal_exact_native`はDにも合成が理想的である**反実条件**を明示し、保存PR idealと同じ宣言モデルで比較する。実際の未知誤差0の認定ではない。決定論law誤差0/A=1は構造的な定義、理想合成0は仮定として区別する。二つのlaneは同じ測定済みkernel価格とserial-cost recordsを共用する。

比較は事前登録6件：newPR ideal/newD numerical、newPR assumed/newD numerical、newPR ideal/newD ideal、oldPR ideal/oldD、oldPR assumed/oldD、newD numerical/oldD。新PR/旧PRの改善比は統合結果の保存値を引用する。したがって、旧J/U内、新J/U内、DF表現更新のPR自身、DF表現更新のD自身を分けられる。

新PR ideal総RZ207919150188と旧D総RZ101825929455は既報の比較値で、同一J/U比較ではない。今回、新Dの値や順位は未取得。新Dが有利でも、全決定論DFの大域的最適性やPR-aware目的だけの因果効果を示さない。PRの全law synthesis/実装law誤差・費用support/母平均優位は未資格のまま。PR標本平均とD固定価格の比は条件付き点比較で、保証付き同CA優位・有意差ではない。

## 4. 人工・回帰検証

[テスト集計](../../artifacts/research_c_same_ju_strong_d/2026-11-02/technical/test_results.json)は**134 PASS＝新16＋既存118、skip0、deselect2**。新suite 39.881秒、既存suite 18.255秒。人工134件を実H4の科学的PASSやCI/外部再現と同一視しない。

新suiteでは、人工rank8/signed一般J・非自明U、full512のraw/compiled/reference/adjoint/X/Y、13 PF/RPE/価格/二lane/6比較を通し、最大156 callsを実際に通過した。単一D build/compile・4 QPY、元scalar/κ・frame順・relative fusionの継承を確認した。入力hash/候補identity/DF資格/ground欠損/旧J/U・compiler・PR target不一致、phase/frame/compile不一致時のpartial STOPと後続metrics抑止、PF不適格時のnative省略・nullを確認した。未資格synthesis/fixed-N・numeric rangeは既存M2/H4/H6人工回帰を再利用する。

人工検証で旧`prepare_arm`のrole制約を検出し、新candidateをsource hashで結びつけたままlegacy roleタグを適合させた。追加fault fixtureのSOURCE serializerとimmutable dataclassの扱いも技術試験中に修正し、最終16件を再実行した。初期失敗logもtechnical配下に保全している。

Deselectの2件は、過去の固定2026-10-26契約を後日のcheckoutで再照合する実データbyte-only試験と、人工tempにGit identityがない固定N試験。旧契約/SOURCE/testsは修正せず、前者は同じlegacy preflight/条件assertionを人工SOURCEへ適用する新試験、後者は既存M2のsynthetic commit wrapperで置換した。現SOURCEのpassive preflightは別途実環境でPASSしている。

## 5. 未承認156-call契約・順序・STOP

[固定案](research_c_same_ju_strong_d_contract_unapproved_20261102.json)。Scope `C_SAME_SELECTED_JU_STRONG_DETERMINISTIC_S2_V1`。契約SHA256 `f199202333a376edf7fd8a428fa77aa7b094155eb73ea33feea69b22e13e77bd`、SOURCE manifest SHA256 `1e416afe6737ad4b705ba8a273acd6d1147e0ba15c6401643a24fa02bbbdbb64`。SOURCE75、直接保存入力27。次回のfresh outputは`artifacts/research_c_same_ju_strong_d/2026-11-02/run01`で、今回作成していない。

Aはinput/selected/保存physical J/U/DF予算/ground-aliasを照合してCARとの対応を確認。Bは全frame S2の13 round・ideal phase/energy十分条件。既知PF roomが不適格なら全13を保全しnativeへ進まない。Cに進める場合だけ、1 kernelと共通shell検証・合成誤差・shots/serial価格・登録6比較を評価する。

| logical unit | 上限 |
| --- | ---: |
| evidence_load | 1 |
| input_load | 1 |
| fit_load | 1 |
| reference_load | 1 |
| candidate_binding | 1 |
| car_context | 1 |
| raw_factor | 9 |
| sector_gate | 1 |
| ground_load | 1 |
| ground_gate | 1 |
| sector_eigenpair | 1 |
| native_model | 1 |
| PF_kernel | 1 |
| PF_round | 13 |
| full_reference | 1 |
| reference_sector_gate | 1 |
| gaussian_build | 9 |
| relative_gaussian | 16 |
| shell_load | 6 |
| shell_decode | 6 |
| shell_structure | 3 |
| shell_segment | 3 |
| kernel_build | 1 |
| fusion_gate | 1 |
| compile | 1 |
| operator_decode | 2 |
| segment_assessment | 1 |
| adjoint_build | 2 |
| adjoint_decode | 2 |
| adjoint_gate | 2 |
| XY_gate | 1 |
| metrics | 1 |
| round_synthesis | 13 |
| round_accuracy | 13 |
| energy | 1 |
| round_serial_cost | 13 |
| aggregate | 1 |
| ideal_round_accuracy | 13 |
| ideal_energy | 1 |
| ideal_aggregate | 1 |
| comparison | 6 |
| result_summary | 1 |
| **合計** | **156** |

最大：fit/optimizer/DF誤差再計算/physical ground eigensolve/PR再計算/量子sampling=0、DF-sector eigenpair1、PF kernel1/round13、D kernel build1/P0 compile1、new QPY4。Relative Gaussian16は固有境界の上限で、厳密に同じ回転がcache hitなら実績は減る。内部library演算は旧M2と同じbundle定義で数え、各callが一つのBLAS演算を意味するわけではない。

SOURCE/input・candidate binding・保存DF予算/guard/ground-aliasの不一致、CAR/native/位相/Gaussian/順序/reference/compile/adjoint/X-Yのcorrectness FAILでは部分証拠を保全して依存後続をSTOPする。PF十分条件不成立は`INFEASIBLE`（物理的不可能性ではない）、合成誤差・fixed-N・数値範囲等の未資格は`UNRESOLVED/null`。未知を0補完せず、反実条件の成功で数値laneの未資格を上書きしない。Retry/resume0、追加s/rank/fitや旧543/1759-call残枠の流用なし。正常/FAILとも公開照合後mandatory STOP。

## 6. 環境・保全・実行規模

既存read-only evaluation venv `/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python`を再利用し、NumPy1.26.4/SciPy1.14.1/Qiskit1.3.0等10依存と既存lockを固定した。[実環境preflight](../../artifacts/research_c_same_ju_strong_d/2026-11-02/technical/environment_preflight.json)で数値library未import・実NPZ decode0・科学call0を確認した。SOURCE75は旧M2 SOURCE71＋新3＋既存package entry1で、旧71のhash差分は0。

[保全照合](../../artifacts/research_c_same_ju_strong_d/2026-11-02/technical/preparation_checks.json)で旧raw3079件、旧SOURCE20組（統合207含む）・契約・grant/ledger/marker/STOP、旧138科学recordsの不変性を確認した。今回の技術準備では科学synthesis/status/manifest/noteを更新しない。新pathを4索引に追加するだけで、Bの分析資料は編集していない。[公開起点照合](../../artifacts/research_c_same_ju_strong_d/2026-11-02/technical/base_remote_checks.json)はaced commit/tree・統合報告/M2報告の公開bytesと一致した。

現在の資源照合はaffinity32・MemAvailable約54GB。将来は独立raw/compiled/各actual-adjointの4つのfull512 Operator decodeを、実際のaffinity/quota/空きRAMに基づく既存resource_planで並列化する。Qiskit build/compile/inverse生成は共有equivalence処理の既存条件に従い逐次、512以下の小行列BLASを協調する。4は独立処理の件数で、任意のCPU上限ではない。CPU/RAM/worker/wallの人工上限や共有venv/cache/OS・他ユーザーjobへの変更はない。所要時間・speedupの実H4測定は今回行っていない。

## 7. 残る未解決事項・次の担当

技術的blockerは現時点でない。**新J/U全frame DのPF/合成誤差/整数shots/native価格/総資源・同一J/Uの順位は未評価**。DF予算は既存数値guardの資格、全law PR保証や厳密CA・母平均優位は未資格である。

次は**ユーザー／研究C**が、公開SOURCE・selected hash・本契約・fresh run01に対する最大156-call限定科学実行を明示許可するか判断する。許可された場合のみ**Codex**が、A→B→条件付きCを一度実行し、保存新PR/旧PR/旧Dとの登録比較、証拠公開、remote照合、mandatory STOPまで行う。新fit・PR再計算・H6・重要科学レビューはその範囲に含めない。今回の技術準備から科学実行へ自動進行しない。

Branch `codex/research-c-same-ju-strong-d-preparation-20261102`。Commit/push・remote SHA/tree/公開本文/SOURCE/旧結果照合後、**mandatory STOP**。
