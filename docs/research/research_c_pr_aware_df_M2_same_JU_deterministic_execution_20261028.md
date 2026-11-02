# 研究C M2：同じ保存J/Uの強決定論S2・限定科学実行結果

2026-10-28 JST。**253/253 logical calls COMPLETE、correctness FAIL0、retry/resume0。決定論Baseline/Proposedともfull512・adjoint・X/Y、全13roundのRPE/共通CA数値十分条件がPASSした。** 同じJ/Uに対する保存PR主仮定の総RZは、今回の決定論s1に対してBaseline31.401866倍／Proposed5.306017倍だった。今回の登録費用・仮定条件では決定論側が低費用で、PR内部の改善を強決定論に対する優位とは扱えない。

## 1．SOURCE・契約・承認範囲

- 起点：`23499d0d84678a53af3c9bccb819901063e9d16a`。
- 技術準備後に固定した科学実行SOURCE：`a6ffc23d94f6488a4769596f089a91c895d49214`。
- Branch：`codex/research-c-M2-same-JU-deterministic-20261028`。独立worktree：`/tmp/prt-c-M2-same-JU-deterministic-20261028`。
- [固定契約](research_c_pr_aware_df_M2_same_JU_deterministic_contract_20261028.json) SHA256：`b36f61f77060b04237eb970d140765b51f1f5f4091645644c56a28b93bc68f24`。
- [SOURCE71 manifest](../../artifacts/research_c_pr_aware_df_M2/2026-10-28/technical/SOURCE.json) SHA256：`c0fca9ff024329a384af1edf3b9dba52f95dbd2e95a840a06a8118b57def5a7b`。
- [原本62件](../../artifacts/research_c_pr_aware_df_M2/2026-10-28/run01/)／[要約](../../artifacts/research_c_pr_aware_df_M2/2026-10-28/execution_summary.json)／[postflight・原本hash](../../artifacts/research_c_pr_aware_df_M2/2026-10-28/execution_postflight.json)。

ユーザーはM2の技術準備から今回の2候補科学実行・数値正しさ・RPE/総資源比較まで一括して明示許可した。追加承認を求めず、人工試験後に新契約・SOURCE・input・call上限をcommit固定してfresh run01へ一度実行した。承認証明書、原文保存、approval packet、新grantは作成していない。

実行entry：`.venv/bin/python -B scripts/run_research_c_pr_aware_df_M2.py --contract docs/research/research_c_pr_aware_df_M2_same_JU_deterministic_contract_20261028.json --execute`。科学開始後のSOURCE/閾値/候補変更、再実行は0。

## 2．比較する物理条件・数値資格

保存直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ2／sector36、通常number CDF rank8。元g補正κ、核scalar一回、元frame順0..7、各DF自身のground-state oracle・状態準備費用除外。元P-H4 provenance SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。

| 入力 | SHA256 |
|---|---|
| 保存h/g/c snapshot | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| Baseline J/U | `f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02` |
| Proposed J/U | `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227` |

| 条件 | 今回の新決定論2候補 | 保存PR2候補 |
|---|---|---|
| J/U | Stage A Baseline/Proposedをそのまま使用 | 各同じStage A export |
| PF | 全frame ordered symmetric S2、PR splitなし | 元順prefix LD2、partial S2 |
| s/h/n | **s1、h=.1、n=2^m** | **s4、h=.025、n=4·2^m** |
| 時間・RPE | δ=.1、m0..12、CA=.00159／DF reserveCA/8、βtotalπ/6、βstatπ/12、candidate failure.01、axis.01/26 | 同じ時間窓・配分 |
| RTE/A/law | A=1、RTE補正/law誤差は決定論で非適用 | r16/K0、保存PF/RTE/A。ideal_exact_nativeとassumed_raw_signal_budgetの2仮定のみ |
| synthesis/reference | full512 Frobenius・両行列norm-growthを伝播するlevel B数値pilot | 保存C1の明示仮定。全law誤差資格ではない |
| native/compiler | 共通StrongBuilder全sweep相対融合、保存full-frame U、exact spin-lift κ、precision Gaussian、P0 MPFR160 phase | head内relative・tail隣接同一basis共有、cross-head/tail一般相対融合なし |
| 費用 | 固定gate metrics、fenced serial、head/readout各interrogation一回 | B2 run02各32独立cohortの平均、同じserialモデル。母平均/CI/ideal-law転送は未資格 |

両方式にP0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation_degree1/qubits_initially_zero=false、同じstate/oracle・状態準備除外政策を使う。ただし回路scope・合法融合の適用箇所は上表の通り異なる。**sも異なるため、PF方式以外が全て同じ純粋ablationではない。** 未実施のPR追加融合を費用から差し引かない。

元κはK1用に対称化せず、polar Uへ置換しない。既存`corrected_kappa`とgeneral-K/nativeの数値Hermitian扱いを維持し、独立raw CARと数値allowanceで確認した。F2-K1/F3-K1の理想unitary targetを今回のDF/PF targetへ移送していない。

## 3．実装・人工検証・再利用

追加SOURCEは[薄いM2 adapter](../../src/trottertracks/c_pr_aware_df_m2.py)、[専用runner](../../scripts/run_research_c_pr_aware_df_M2.py)、[人工test](../../tests/management/test_research_c_pr_aware_df_m2.py)の3ファイル。旧科学SOURCEの変更0。

- `prepare_arm`/一般Kとexact spin-lift κ、`StrongBuilder.ordered`/既存S2 schedule/relative Gaussian、precision/P0をそのまま呼ぶ。
- `raw_base/raw_frame`からfull256の独立CAR因子を作り、今回J/Uと同じB1保存sector generatorsの各因子とsumを照合する。Gaussian gate単体のsector射影でfull controlled回路を代用しない。
- `c1_molecule_extension.PF_rounds`の36次元norm2・反復二乗法、既存`reference_kernels`の独立full256 S2を再利用。旧PF値は移植しない。
- B1の同一J/U own-groundとphysical aliasを再利用し、残差・正規化・sector最低固有値を確認した。DF norm+guardは同一J/Uの保存値を再利用し、新DF norm計算・新fitなし。
- B2の共通head/X/Y raw/compiled QPY6件をhashと構造で再利用。新shell build/compileは0。解析的ancilla idealに対するfull512 segmentを評価し、誤差を0へ置換しない。
- `c1_h4_full_synthesis.segment/propagate`、`c1_h6_ca_resources.assess_saved_round`、`AccuracyTarget/assess_round/assess_energy`、既存Hoeffding/fixed-N、`serial_round_cost/aggregate_candidate/aggregate_totals`を再利用。新Gaussian compiler/RPE solver/aggregatorは実装していない。

人工・回帰は**68 PASS**（新21＋既存47、17.67秒）、skip0。旧fixed-N試験1件は人工tempにGit情報がないため原本をdeselectし、同じassertionを人工commit identityのpatch下で呼ぶ代替試験を追加した。旧SOURCEは変更していない。Phase/Gaussian方向/frame順/compile不一致、full512・adjoint/X/Y、PF13・shots/CA/serial、ineligible/UNRESOLVED/null、SOURCE/input不一致・partial STOP/fresh拒否を検証した。[最終log](../../artifacts/research_c_pr_aware_df_M2/2026-10-28/technical/tests_final.log)。

初回の人工debug（shell ideal dtype、sector mapping permutation、temp Git identity）と既存fixtureの問題は[技術証拠](../../artifacts/research_c_pr_aware_df_M2/2026-10-28/technical/)に残した。実H4を人工試験へ混ぜず、実科学開始前に修正・SOURCE固定を完了した。

## 4．数値正しさ・ground・全PF26とshots

SOURCE71/input32/environment/fresh、保存DF予算、physical/own-DF ground/alias、全18 raw factorのsector対応、独立full256とsector S2 kernel一致がPASS。両arm9 Gaussian bases・16 directed relative境界、元順・s1/h.1・scalar一回もPASSした。

| full512数値量 | Baseline | Proposed |
|---|---:|---:|
| raw/reference max-entry | 6.3932485e-15 | 5.7830259e-15 |
| compiled/raw max-entry | 6.2593609e-13 | 6.1954745e-13 |
| compiled/reference max-entry | 6.301723e-13 | 6.237733e-13 |
| compiled/reference Frobenius | 5.4649381e-12 | 5.4808211e-12 |
| compiled actual-adjoint recovery max-entry | 3.6104463e-13 | 3.6970444e-13 |
| compiled X+iY signal error（m0） | 3.4717976e-13 | 3.5496273e-13 |
| m12 synthesis complex-signal pilot | 6.3312624e-08 | 6.3496631e-08 |
| m12 PF+synthesis phase pilot | 0.069193038 | 0.069945131 |

全512入力×全512出力でcontrol0 identity/control1 evolution・相対位相・unitarity、raw/compiled actual inverse各2件、X/Yを確認した。Component/adjoint1e-10、signal2e-10、rtol0、global phase fitなしを変更していない。Frobeniusとmax-entryを混同せず、finite-probe検証や厳密rounding enclosureとも区別する。

以下は新決定論の無次元PF operator2誤差と各X/Y軸あたりの算定shots。A=1、RTE補正0は定義に基づく。未知合成誤差を0にした結果ではない。

| m | Baseline PF | Proposed PF | Baseline N/axis | Proposed N/axis |
|---:|---:|---:|---:|---:|
| 0 | 6.2529829e-05 | 6.2234868e-05 | 511 | 511 |
| 1 | 0.00012472293 | 0.00012413358 | 512 | 512 |
| 2 | 0.00024677932 | 0.00024560518 | 512 | 512 |
| 3 | 0.00047307637 | 0.00047076122 | 512 | 512 |
| 4 | 0.00080643151 | 0.00080198806 | 512 | 512 |
| 5 | 0.00097196914 | 0.00096410686 | 512 | 512 |
| 6 | 0.0013004969 | 0.0012877306 | 513 | 513 |
| 7 | 0.002428766 | 0.0024183792 | 514 | 514 |
| 8 | 0.0043725559 | 0.004407226 | 516 | 516 |
| 9 | 0.0086916613 | 0.008766891 | 520 | 521 |
| 10 | 0.017361718 | 0.017517048 | 530 | 530 |
| 11 | 0.034643657 | 0.034984609 | 549 | 549 |
| 12 | 0.069137776 | 0.06988805 | 590 | 591 |

両候補とも13/13合成＋PF位相条件、既存整数Hoeffding shots/fixed-N margin、energy十分条件がPASS。Statusは両方`NUMERICAL_PILOT_CA_PASS`。保存DF norm+guardはBaseline1.0297636237465247e-8/Proposed1.229561498918692e-8 Ha、energy誤差評価pilotは0.0012783276208742716/0.0012783296188530234 Ha。これは実測energy誤差・actual CA保証ではない。

DF[Ha]はenergy側で一回、PFとsynthesisはcomplex-signal側で各一回だけ計上した。Candidate failureは各.01で、4候補同時・事後選択の保証ではない。全roundは[26行CSV](../../artifacts/research_c_pr_aware_df_M2/2026-10-28/rounds26.csv)、各X/Yは[52行CSV](../../artifacts/research_c_pr_aware_df_M2/2026-10-28/axes52.csv)に残した。

## 5．Native価格と条件付き総資源

### 1 h=.1のcontrolled full-frame kernel

| 候補 | RZ count | RZ depth | CX count | CX depth | total depth | size |
|---|---:|---:|---:|---:|---:|---:|
| Det baseline | 11,026 | 5,568 | 7,000 | 6,276 | 12,098 | 20,919 |
| Det proposed | 11,026 | 5,568 | 7,000 | 6,276 | 12,098 | 20,919 |

**両armの登録6 native価格は一致した。** PRで得られたProposedのkernel平均価格低下は、今回の決定論kernelには現れなかった。これはこのcompiler・s1・分解の観測で、Gaussian価格一般の同一性を意味しない。

### 全13round・X/Yのshots加重費用

| 候補 | 総RZ | 総RZ depth | 総CX | 総CX depth | 総depth | 総size |
|---|---:|---:|---:|---:|---:|---:|
| Det Baseline s1 | 101,724,313,825 | 51,369,602,693 | 64,580,978,000 | 57,901,459,704 | 111,614,470,131 | 192,995,728,265 |
| Det Proposed s1 | 101,825,929,455 | 51,420,917,395 | 64,645,490,000 | 57,959,299,320 | 111,725,965,325 | 193,188,517,795 |
| PR Baseline s4・ideal仮定 | 3,194,333,274,946 | 1,268,642,747,401 | 1,601,103,148,210 | 1,240,683,073,505 | 2,768,469,723,089 | 6,209,963,917,154 |
| PR Proposed s4・ideal仮定 | 540,290,160,109 | 219,452,620,009 | 280,340,936,400 | 221,280,584,700 | 478,863,343,531 | 1,045,946,875,231 |

総shots/executionsは決定論Baseline **13606**、Proposed **13610**、保存PR ideal仮定Baseline **53136**、Proposed **19574**。同じJ/Uに対してPRはn=4·2^m、決定論はn=2^m。各interrogationのhead/readoutは一回、状態準備は除外、depthはfenced scheduled sumでglobal再compile最小depthではない。

| 登録比較（left/right） | RZ比 | CX比 | total-depth比 | 小さいRZ |
|---|---:|---:|---:|---|
| baseline:PR_ideal_exact_native/D_s1 | 31.401866 | 24.7921787 | 24.8038603 | right |
| baseline:PR_assumed_raw_signal_budget/D_s1 | 31.5523642 | 24.910999 | 24.9227365 | right |
| proposed:PR_ideal_exact_native/D_s1 | 5.30601747 | 4.33658924 | 4.28605242 | right |
| proposed:PR_assumed_raw_signal_budget/D_s1 | 5.33036536 | 4.35648869 | 4.30571996 | right |
| D_proposed/D_baseline | 1.00099893 | 1.00099893 | 1.00099893 | right |
| D_baseline/external_XDF8_B3 | 1.01453007 | 1.01725104 | 1.01201271 | right |
| D_proposed/external_XDF8_B3 | 1.01554352 | 1.01826721 | 1.01302364 | right |

保存PRの感度仮定でも決定論側が低費用で、PR/DのRZ比はBaseline31.552364/Proposed5.330365だった。[全7比較×6指標](../../artifacts/research_c_pr_aware_df_M2/2026-10-28/comparisons42.csv)。仮定付きPRの標本平均と決定論level B pilotは同一保証レベルではない。母平均費用保証・PR全law・厳密CA優位を認定せず、**今回の明示仮定と数値pilotの比較では決定論が有利**という結論に限定する。

決定論Proposed/Baselineの総RZ比は**1.0009989315845846**。同じnative価格でProposedのPFが僅かに大きく、保存したN/axisがm9で521対520、m12で591対590となり、総shotsは4多かった。ProposedのPR-aware最適化による利益は今回の決定論資源には観測されず、僅かに高費用だった。元目的項だけの厳密因果効果、fit一般の優劣・統計的有意差とは呼ばない。

## 6．強XDF8 B3外部対照

科学SOURCE `1aeedd96178a3f50a15bc2701c3c332f4f391b64`の[既報強baseline](research_c_c1_h4_strong_baseline_ca_resources_execution_20261026.md)をhash/SOURCE/86-call STOP照合の上で再利用した。XDF8のnew build/compile/Operator/PF/shots再計算は0。

保存XDF8 B3は全frame S2 s1・h.1、RZ/kernel10885、shots13600、条件付き総RZ **100,267,421,030**。M2決定論Baseline/XDFの総RZ比1.014530071、Proposed/XDF比1.015543518で、登録外部比較でもXDFの保存点値が低かった。

同一元P-H4・共通CA/配分/oracle・P0・serial費用で比較する一方、XDFはOpenFermion元順prefix8 square・exact spin基底＋relative、M2は新Stage Aの一般J/U CDF8であり、実DF誤差も異なる。XDF保存DF norm+guard9.533901455091894e-5 HaとM2の約1e-8 Haは完全一致しないが、いずれもCA/8内。数値資格はlevel Bで厳密enclosureではない。保存XDFを強い外部対照に残すことと、CDF/XDF一般の最適性・全条件の順位は別である。

## 7．Q1〜Q8への回答・次の最小検証

1. **同J/U決定論のPFと価格**：m12 PFはBaseline .06913777600000931/Proposed .06988804953189413、両native価格は全6指標で一致。保存PR s4のPFより大きいが、今回の数値phase room内だった。
2. **RPE/CA**：両13/13・energy十分条件が数値PASS、shots13606/13610。Actual CA・rounding enclosureは未資格。
3. **PR対決定論**：保存ideal仮定のPR/DはRZ31.40/5.31倍、CX24.79/4.34倍、depth24.80/4.29倍で、両armとも今回の決定論s1が低費用。Sとnative適用scopeの違いを含む比較である。
4. **PR Proposedの競争力**：PR内部の改善は保全されるが、今回の同J/U強決定論には達していない。これは固定H4/rank8/LD2/s4/r16/K0のPR政策に限る。PR全体の不可能性・研究の正式終了とは言わない。
5. **PRなしのProposedの利益**：今回native価格差はなく、PF/N/総資源はProposedが僅かに悪化。普遍的DF/native改善ではなく、現在のPR政策と結び付いた改善という解釈が適切な範囲である。
6. **外部XDF8**：保存強XDF8はM2決定論より点値で低い。一般J/Uを最適化したこと自体やrank8という一致だけでは最強baselineへの優位は生じていない。
7. **改善対象**：今はγをさらに下げるだけでなく、PRのmacrostep/head反復・finite-RTE減衰/shots・実kernel価格のtradeoffが主要課題。追加厳密保証や新fitを自動的に増やす根拠にはしない。PFはより大きくても決定論の十分条件が通るため、現s4政策の精度余裕が高価格になっていないかを次に問える。
8. **次に一つなら**：保存Proposed J/U・LD2/K0を固定し、**PR s1/r64、h=.1のPF/RTE・phase-room 13roundだけのfeasibility検証**を技術設計する案を推奨する。旧PR s4/r16/h.025とh/rおよびr·nが同じなので、K0の既存解析式では正規化Aと総random microstep数を保つ一方、決定論headのmacro反復数が変わる。これは代数的な設計理由で、新しいH4数値評価ではない。PF/補正RTEが許容範囲かは未取得で、改善を予測・保証しない。通過した場合だけ別許可でnative価格を確認する。反対に不適格ならhead反復を減らす案の精度tradeoffが分かる。今回この候補の契約・計算・samplingは追加していない。

研究方針の正式な継続／縮小／終了や、上の候補を採択するかは**ユーザー／研究C**が判断する。**Codex**はその具体的な次依頼と必要な科学許可後にだけ対応する。γ探索、決定論s4、H6移送、追加sampling、保証構築、重要GPTレビューへ自動進行しない。

## 8．Call内訳・資源・保全・mandatory STOP

| Logical call | 上限＝実績 |
|---|---:|
| evidence_load | 1 |
| input_load | 1 |
| fit_load | 2 |
| native_model | 2 |
| car_context | 1 |
| raw_factor | 18 |
| reference_load | 2 |
| sector_gate | 2 |
| ground_load | 2 |
| ground_gate | 2 |
| sector_eigenpair | 2 |
| PF_kernel | 2 |
| PF_round | 26 |
| full_reference | 2 |
| reference_sector_gate | 2 |
| gaussian_build | 18 |
| relative_gaussian | 32 |
| shell_load | 6 |
| shell_decode | 6 |
| shell_segment | 3 |
| shell_structure | 3 |
| kernel_build | 2 |
| fusion_gate | 2 |
| compile | 2 |
| operator_decode | 4 |
| segment_assessment | 2 |
| adjoint_build | 4 |
| adjoint_decode | 4 |
| adjoint_gate | 4 |
| XY_gate | 2 |
| metrics | 2 |
| round_synthesis | 26 |
| round_accuracy | 26 |
| energy | 2 |
| round_serial_cost | 26 |
| aggregate | 2 |
| comparison | 7 |
| result_summary | 1 |

253 attempt/253 COMPLETE、FAIL/UNRESOLVED候補0、retry/resume0。内部matrix columnを新候補に数えず、36次元PF norm、full256 reference、full512 segment/adjoint/X-Yをそれぞれ契約の単位で計上した。既存227/86-callやA/B/C/M1/F2/F3の枠・旧runnerは再使用していない。新2kernelのbuild/compile各2、actual inverse4、QPY8、operator/reference NPZ6、共通shell QPY6再利用、raw62 filesを保存した。

CPU affinity32、実行直前MemAvailable約56.5GB・memory pressure0を確認。支配負荷はfull512 Qiskit Operatorと行列比較で、2独立armを2workersで並列化した。小行列BLASは自job内1threadへ協調し、共有equivalence libraryに触るbuild/compile/inverse生成だけ逐次にした。任意のCPU/RAM/worker/wall capは追加していない。共有venv/package/cache/OS/scheduler/cgroup・他jobの変更なし。

Runner wall **81.25501689128578秒**、parent peak RSS **679,332 KiB**。Full512 SVDは行わず、既存Frobenius/両norm-growthを用いた。速度向上率は測定していない。Read-only /proc snapshotからは実jobを確認できなかったため、連続監視・瞬間CPU/RSS peakの追加証拠とは扱わない。Runnerのwall/RSS原値は保存済みである。

実行後もSOURCE71・input32・保護対象F2/F3/M1 raw44のhash不変。旧SOURCE/raw/grant/ledger/marker/STOPと旧強baseline/PR値は保全した。[Preflight](../../artifacts/research_c_pr_aware_df_M2/2026-10-28/execution_preflight.json)／[postflight](../../artifacts/research_c_pr_aware_df_M2/2026-10-28/execution_postflight.json)にidentity・新raw hashを保存。CSVは今回原本の転記・serializeで、科学評価関数の再実行ではない。

結果SOURCE・証拠・研究記録をcommit/pushし、remote SHA/Git tree・SOURCE/input/成果物を照合する。最終照合はチャットで報告する。**公開照合後mandatory STOP。** このM2は完了し、次の科学作業・重要レビューへ自動進行しない。
