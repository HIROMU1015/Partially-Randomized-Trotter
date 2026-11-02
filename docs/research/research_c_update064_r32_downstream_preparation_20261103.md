# 研究C・update064：r32 PR下流資源と同一J/U強決定論S2の技術準備

2026-11-03 JST。**TECHNICAL_READY_FOR_SEPARATE_SCIENTIFIC_PERMISSION**。新規実H4の配列decode、ground、PF/RTE、shots、native build/compile、強D評価は0。人工・回帰171 unique tests＋29 subtests PASS。未承認617-call契約を作成した。科学的改善は未判定であり、公開照合後mandatory STOP。

[未承認科学契約](research_c_update064_r32_downstream_contract_unapproved_20261103.json) ／ [SOURCE](../../artifacts/research_c_update064_r32_downstream/2026-11-03/technical/SOURCE.json) ／ [技術検証記録](../../artifacts/research_c_update064_r32_downstream/2026-11-03/technical/verification.json)。科学run01は作成していない。

## 1. 問いと比較単位

「追加64回のU更新で得たupdate064は、同じr32のupdate005より条件付き総量子資源を改善するか。同じupdate064の強決定論S2との資源tradeoffは何か」を将来の実行対象とする。λ低下からPF/RTE、shots、価格、総RZの改善を推定しない。

全比較は直鎖H4・隣接1.00 Å・STO-3G、通常number CDF8、signed対称J・実直交U、Nα=Nβ=2/sector36、full256/controlled512。元g補正κ、核scalar一回、元spin/frame順を固定する。PRは元順LD2・partial ordered symmetric S2、s1、δ=h=.1、r32/K0、m0..12。Dは同じJ/Uのκ＋全8 frame ordered symmetric S2/s1、同じδ/m。r/KはDに非適用。

CA=.00159 Ha、DF reserve=CA/8=.00019875 Ha、既存βtotal=π/6・βstat=π/12・failure=.01、各軸=.01/26。Own-DF ground-state oracle、状態準備費用除外。P0/serial構成・Gaussian融合・controlled phaseの政策を固定する。r32はupdate005での有望設定であり、update064の最適rとは認定しない。新しいr探索、rank/LD/γ/ε/PF/compilerの同時変更はない。

## 2. 固定identityと別branchの扱い

| 対象 | 固定値・役割 |
|---|---|
| 技術起点・update005 r再調整の結果commit | `f200e45afecc0a80b9f8a895b68c75ba4dda5dd5`、tree `676ba637305fdf6e0ddc9f56a92e65c5144cae7d` |
| 同科学SOURCE | `1cdeaa8bd55eb07a1d5b8cf2080259e929d405d1` |
| update005候補SHA256 | `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e` |
| update064結果commit | `ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6`、tree `326bfc65c8ca263b4da45a0e805678617a4890e0` |
| U最適化科学SOURCE | `025d495166311a58ec9c24c43f07ed6e40f65f42` |
| update064 terminal | `artifacts/research_c_u_step_control/2026-11-03/run01/terminal.npz` |
| update064候補SHA256 | `53d8f5027ce9e61bc72f6943c0268666bfcf87515154850a9a99326687b26ed3` |
| update064物理export SHA256 | `8591d0ca941137704b49914c2ca300a1c95391fdc8f97507db2b7e206ba2e6cd` |
| 元h/g/constant snapshot SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| 元P-H4来歴SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| 既存update005強D科学SOURCE | `5f27539ae0ab88181d9cfcd45badb54daf855c28` |

専用branch `codex/research-c-update064-r32-downstream-preparation-20261103`をf200から作成した。科学branchをmergeせず、U更新branchは既存peer worktree `/tmp/prt-c-u-step-control-execution-20261103`を読み取り専用で参照する。CLIの`--candidate-root`は配置だけを指定し、HEADがea54であること、必要8ファイルのbytes、JSON来歴を確認する。他環境では同じ固定commitのworktreeを渡せる。候補やrawのコピー、元SOURCEの編集はない。

保存結果は[U更新報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6/docs/research/research_c_u_step_control_execution_20261103.md)、[r再調整報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f200e45afecc0a80b9f8a895b68c75ba4dda5dd5/docs/research/research_c_rte_r_retuning_execution_20261103.md)、[同J/U強D報告](research_c_same_ju_strong_d_execution_20261102.md)を参照した。[Evidence Map](research_c_evidence_map_updated_20261102.md)と[既存A/B技術統合](research_c_pr_native_cost_improvement_feasibility_20261102.md)のclaim資格も保持する。「新selected」がupdate005を指す過去資料と、今回のupdate064を区別する。

## 3. 保存値の引用と新たに必要な評価

保存最適化のFγはupdate005 `.00661750979453`、update064 `.00603347984376`、exact tail λはそれぞれ`.568349854686`、`.542690658752` Ha。いずれもDF物理予算PASS・J一次MET・U一次NOT_METの有限endpointである。これは保存報告/JSONの引用で、今回の再最適化・数値再解析ではない。

update005/r32のideal shots `15202`、W `13979232`、平均kernel RZ `7636.5`、条件付き総RZ `106752458375`、同update005強D総RZ `105532782860`も保存値の引用だけとする。update064へ入力しない。

| 保存結果から再利用するもの | update064で新たに取得するもの |
|---|---|
| 正確なterminal identity、最適化SOURCE/契約/STOP、update064物理DF評価 | terminalと物理exportのJ/U完全一致、rank/直交性、元κ/scalar/General-K/CARの確認 |
| 同じ候補のDF sector norm＋engineering guard、保存独立CAR G/H | 新しい候補ground/alias。元physical groundも新規。新DF誤差normは0 |
| 元input、sector basisの規約、ancilla-only shell QPY/metrics | 新しいpartial PF・有限RTE・13roundの誤差/A/raw signal/RPE/shots/W |
| r32/P0/native32/RPE/会計の実装と固定政策 | update064/r32の32独立complete kernel、同update064強DのPF13と1 native kernel |
| 旧r32/旧D JSONの比較記録 | 新しい候補別のshell込み総量、6指標の条件付き比較 |

保存物理DF評価を使う条件はbytesとJ/U配列の完全一致である。不一致時に新DF normを実行して代替しない。旧ground・PF/RTE誤差・shots・価格・旧D資源は比較専用JSON以外の入口を設けない。

## 4. 再利用と最小差分

新規入口は[`run_research_c_update064_r32_downstream.py`](../../scripts/run_research_c_update064_r32_downstream.py)、接続は[`c_update064_r32_downstream.py`](../../src/trottertracks/c_update064_r32_downstream.py)、D adapterは[`c_update064_strong_d.py`](../../src/trottertracks/c_update064_strong_d.py)、追加testは[`test_research_c_update064_r32_downstream.py`](../../tests/management/test_research_c_update064_r32_downstream.py)。

| 処理 | 再利用した実装・差分 |
|---|---|
| identity/General-K/κ/scalar/sector | `c_pr_aware_df_m2.validate_arrays`、Stage B `prepare_arm/candidate_factor`、`SectorCAR`、`m2.sector_match`。新adapterはea54 peerとterminal/物理exportを結ぶ |
| 物理DF | `c_pr_integrated_evaluation.Evaluation.assess`が生成したupdate064保存評価を来歴付き再利用。その評価器を再実行しない |
| ground/alias | Stage B runner `ground/ground_gate`。新候補pairをAで一度取得し、同一H/J/UのDへメモリ内で渡す。旧pair読込・重複eigensolveなし |
| PF/RTE/13round | Stage B `kernels`、既存reference `reference_round`、`retuning.normalization`、既存feasibility。新しいHamiltonianで全13 exact/PF/RTE評価を行う |
| RPE/lane/shots | `retuning.assess_lanes`、Stage C qualification、既存`c1_stage3_accuracy`。未知誤差を0にしない |
| native32 | `c_pr_integrated_performance.native`の既存sampling引数を使用。r32、独立新cohortとcandidate/source IDのみ指定 |
| controlled512/adjoint/complex X/Y | Stage B2 raw/compiled/adjoint/XY gate、P0、既存StrongBuilder/General-K emitter、fullCAR reference。新engineなし |
| Gaussian/融合 | 既存head relative融合・tail exact basis_key共有、D全sweep relative融合。追加cross-boundary融合・全長再compileなし |
| 総費用/SE | Stage C calibration、M3 serial cost/weighted SE、accuracy aggregate。Wを共有校正SEへ掛ける |
| 強D | 旧`c_same_ju_strong_d`のPF/ideal/RPE/比較関数、M2 accuracyと既存native本体を再利用。旧runnerはupdate005と旧groundを必須とするため、そのnative本体を局所adapterへ移植し、Aの新J/U/pairを受け取る入口にした。旧ファイル無変更、production monkeypatchなし |
| SOURCE/Calls/STOP | 既存`Calls`とjournalをnamespace分離して使用。新SOURCEは既存2 manifestのunion＋必要差分91件。新grantや承認証明システムなし |

r再調整runner自体は保存update005 PF/groundを前提にするため、今回はその`prepare/evaluate_r`を直接呼ばない。統合performance runner全体もr128固定なので呼ばない。これが新しい接続層を必要とする理由である。

## 5. A〜Dと進行条件

| Stage | 登録処理 | 進行/停止 |
|---|---|---|
| A | fixed peer/bytes/STOP/契約/SOURCE、J/U/rank8/LD2、DF再利用、General-K9因子、新physical/new candidate ground/alias、signed tail/identity/K0 r32正規化 | すべてのidentity・representation・ground/alias/law gate PASSのみBへ。DF不一致・budget未資格・correctness FAILは全依存処理停止 |
| B | 新partial PF/RTE kernel、13 exact evolutions、PF errorとcorrected RTE errorを別項保存、A/raw complex signal、radius/phase room、3 laneのRPE十分shots | 全13がPOTENTIALLY_FEASIBLE、ideal energy same_accuracyのみCへ。未資格/UNRESOLVEDならC/Dと新総費用を省略しSTOP |
| C | 新candidate/r32の32独立trajectory、raw/compiled/actual adjoint/XY、full512全入力出力、6 native指標、mean/記述SE、serial shell費用 | すべてのcorrectness gate PASS後のみDへ。最初のFAILでD/比較を実行せずpartialを保全 |
| D | 同じupdate064 J/U・新ground、全frame S2/PF13、ideal RPE gate後に1 P0 kernel、独立fullCAR/sector・full512/adjoint/XY、数値合成伝播とideal lane、shell費用 | PF/ideal gate未資格ならD nativeを省略しD総量/比較null。D数値laneだけ未資格なら同laneの総量/比はnull、明示ideal反実laneのみ保持 |

Evidence laneのall-law synthesis/実装law未取得は想定されたnullであり、ideal/assumedをactual CAへ格上げしない。既存のunderflow/overflow/zero A・infeasible/unresolved処理を維持し、clipping・欠測0補完・新rへの切替はしない。

## 6. cohort・会計・比較資格

NativeはP0 Qiskit1.3.0、opt1、seed17、`rz/sx/x/cx`、approximation degree1、qubits_initially_zero=false、既存Gaussian/MPFR160 declared phase、global phase fitなし。新cohortは`update064_r32_fixed32_v1`、master `2026110302`、PCG64、32標本、各32 event、nonpaired/nonadaptive。seedはmaster/cohort/新SOURCE digest、arm/candidate、round=-1/axis/sample/occurrenceへ束縛する。旧cohortとpoolせず、結果を見て標本追加しない。

各量子shotのcoherent kernelは2^m個で、r32のmicroevent価格はcomplete kernel内に含まれる。

`W = Σ_m 2 N_m 2^m`、`C_total = W × mean(kernel) + S_shell`。

Ancilla head/X/Y shellは各interrogationで一回。32は古典的な価格校正標本数でありshotsやWの倍率ではない。共有meanの記述SEはW×SEkernelであり、13round独立として縮めない。sample extremaをsupport、記述SEを母平均保証・CIへ昇格しない。Codex-Aの追加価格統計コード/契約は変更しない。

比較1はupdate064/update005の同じr32をideal/ideal・assumed/assumedで比較。比較2は同update064のPR/Dをideal/ideal、PR assumed/D level-B数値laneで資格付き比較。補助としてnew/old D idealも保存する。全5比較は6指標の差と比、主指標はshell込み総RZ。旧D数値値が旧idealと同じ整数でも、その資格は交換しない。非pairedな有限cohortによる点推定比較で、J/U、law、PF、価格、shots、Wの連動を最適化目的だけの因果効果へ分解しない。

## 7. logical callsの積み上げ

| 部分 | 最大logical calls | 根拠 |
|---|---:|---|
| A | 23 | evidence/input/selected/reference/binding/sector/physical/model各1＝8、General-K9、ground2、target1、law prepare/gate/normalization各1 |
| B | 90 | kernel1、exact13、round reference13、classification13、summary1、qualification3、RPE39、energy3、shots3、admission1 |
| C | 357 | 共通native35、9種類×32標本＝288、calibration summary1、cost calibration1、serial26、aggregate3、weighted SE3 |
| D | 141 | 既存156から旧evidence/input/fit/reference/binding/ground-load/ground-gate/重複eigenpairの8、旧比較6、旧summary1を除外。新PF/native/合成/RPE/serial処理は保持 |
| 比較・全体summary | 6 | 上記5比較＋最終summary1 |
| 合計 | **617** | 同時にすべてが適格な経路の上限。未資格/early STOP/cache等では少なくなる |

共通native35＝CAR1/raw factors9/full-sector1/reference context1/full-law1/Gaussian9/relative4/shell-load6/shell-assessment3。標本ごとの9種類＝sample/build/reference/raw gate/compile/compiled gate/actual-adjoint pair gate/XY/metrics。Dの各kind上限を含む完全表は契約の`call_limits`に記載した。

新規eigensolve最大2、PR13＋D13round、RPE assessments最大65、PR32＋D1 kernel build/compile、QPY最大132。量子measurement実行0、DF norm再評価0。logical callは登録処理単位であり内部expm/iteration/量子shot数ではない。過去543/1042/1265/156-callの残枠は使用しない。

## 8. 人工・回帰検証

すべて新しく作った人工配列/回路またはmockを使い、保存実H4 NPZ/QPYはdecodeしていない。新規25 tests＋既存146＝**171 unique tests PASS、追加29 subtests PASS**。CI/他環境での再現認定ではなく、今回専用worktreeのローカル技術検証である。

| 検証 | 結果・範囲 |
|---|---|
| [新規接続24件](../../artifacts/research_c_update064_r32_downstream/2026-11-03/technical/new.log) | candidate/DF/STOP/SOURCE/input drift、peer/fresh/resume、J/U混在、候補固有ground、PF/RTE13、lanes、W/shell、C failure/依存STOP、D同J/U full512、actual native32全経路 |
| [D未資格追加1件](../../artifacts/research_c_update064_r32_downstream/2026-11-03/technical/new-D-stop.log) | PR保存後にD PF未資格：D compile0・同J/U比null |
| [再利用adapter等](../../artifacts/research_c_update064_r32_downstream/2026-11-03/technical/regression-adapters.log) | r再調整、同J/U D、統合native/pipeline：評価環境39 PASS、4件はJAX欠如で失敗 |
| [同4件を既存fit環境で再実行](../../artifacts/research_c_update064_r32_downstream/2026-11-03/technical/regression-fit.log) | 4 PASS。環境の切り分けで解消。旧SOURCE・test・共有venvを変更せず、追加installなし |
| [既存engine等](../../artifacts/research_c_update064_r32_downstream/2026-11-03/technical/regression-engines.log) | General-K、Stage B/B2/compat、M2/M4、feasibility：103 PASS＋29 subtests |

新人工complete testは実際の32 PR trajectory＋D1 kernelのP0/raw/compiled/actual adjoint/X/Y/full512を通し、PR128＋D4 QPY、全call kind上限、candidate ID、poolなしを検査した。人工PASSは実H4の資源改善・CA・全law保証を示さない。

評価は既存P0 venv、JAXを要する4件だけ既存fit venvで実施。独立test群を並列実行し、小行列BLASとnative taskのnested oversubscriptionを防いだ。確認時affinity32 CPU・MemAvailable約52 GiB。将来の重い13round/32 native tasksには既存resource_planとQiskit排他処理を用い、CPU/RAM/worker/wallの恣意的上限を追加しない。利用可能量と非干渉を実行前に再確認する。速度比の主張はない。

## 9. 未承認契約・passive確認・保全

契約は`UNAPPROVED_SCIENTIFIC_PROPOSAL`、retry/resume0、fresh `artifacts/research_c_update064_r32_downstream/2026-11-03/run01`。SOURCE91件、ローカルinput/記録/共通shell等14件、peer8件をbytes固定する。SOURCE SHA256は`ab9eb265211c83a3836b2322b00b58249e1492b4bd28dd2748e5fe09b4e56434`。`saved_inputs`と`peer_inputs`が入力manifestである。input取得後の科学結果を見て候補/r/誤差/標本政策を変更しない。

現在のrunnerは引数なしで科学実行しない。今回実施したのは以下のpassiveのみで、科学calls0、出力run01未作成を確認した。

```bash
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
  scripts/run_research_c_update064_r32_downstream.py \
  --contract docs/research/research_c_update064_r32_downstream_contract_unapproved_20261103.json \
  --candidate-root /tmp/prt-c-u-step-control-execution-20261103
```

将来の`--execute`にはこの固定617-call範囲への別のユーザー科学実行許可が必要である。承認原文の保存・新grant・認証を要求しない。root側の正式`research_project_instructions_unified_2026-10-25.md`現物の§8/8.4/10を確認した（f200 worktreeには同名ファイルがないため、存在したと主張しない）。適用AGENTS/PROJECT_MAPに従い主エージェントだけで作業した。

既存科学SOURCE/raw/契約/grant/ledger/marker/STOPは変更0。新規ファイルだけを追加し、ea54 peer HEAD/作業状態も保全した。共有索引、Evidence Map、研究概要、validation manifestの変更は依頼どおり後続統合担当へ渡す。新しい科学結果・evidence資格の変更はない。

## 10. 次の判断とSTOP

ユーザー／研究Cは、固定候補・条件・617-call契約を見て科学実行の可否を判断する。明示許可された場合だけ、Codexが同SOURCEを一度fresh実行し、候補別PF/RTE/shots/W/価格と同J/U Dを取得する。

取得後に判断できるのは、登録条件におけるupdate064対update005 r32の条件付き資源の点推定差と、新J/U内のPR対Dの6指標tradeoffである。厳密DF enclosure・実CA・PR全law合成/実装law・価格母平均/support保証・一般的な最適DF・最適r・目的関数の因果的優位は本契約で認定しない。Codex-Aの統計準備や追加校正を自動実行しない。**今回は技術準備と公開まででmandatory STOP。**
