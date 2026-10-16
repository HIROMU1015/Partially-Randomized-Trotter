# M1-S Gate 0：実行SOURCEと実験契約最終候補

2026-10-16 JST。**CANDIDATE / NOT ADOPTED / SCIENTIFIC EXECUTION NOT AUTHORIZED**。

SOURCEの分類は **IMPLEMENTED_NOT_SCIENTIFICALLY_EXECUTED**。今回の新規科学計算、Pauli係数生成、実データ・模擬データによる数値correctness testはいずれも0件。静的検査のPASSは数値的正しさや科学実行可能性の認定ではない。

[前段の草案][PREFLIGHT]を保存したまま、今回ユーザーが明示した技術実装条件を具体化した。科学判断の基準は[Checkpoint 2 GPT原本][GPT2]。採択済みT1a/T1bの限定的成果を変更・再審査せず、T1cや新規性判断を追加していない。指示原本・SOURCE hash・静的検査・未決事項は[証拠索引][EVIDENCE]にまとめた。

## 1. 科学的目的

保存されたDF Hamiltonianと数値stateに対し、非冗長Pauli-tailのsmall-step先導減衰に改善余地があるかを診断する。主量はW、v、定義域が解決した場合のρ。Oracle参照による診断であり、実用的predictor、有限step信号、RPE精度、native総資源の改善を認定する実験ではない。

## 2. 正式採択待ちの入力条件

技術実装はH4 linear chain、1.00 Å、STO-3G、8 spin orbitals、保存DF rank12、charge0/multiplicity1、保存development NPZだけを対象とする。

- 入力ref：`8b7666d0270f7fac01e430e195bef7b58ac0116c`。
- [入力NPZ][INPUT]：8,652 bytes、SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。
- 保存Hamiltonian hash：`de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。
- 保存full-state hashのmetadata：`31e63b0104126c85136ee173f1dce7642aee2d272924e70e8b120ac340ab45bd`。今回はstate数値配列を解析していない。

今回再確認したのはfile bytes/hashとmetadata_jsonのみ。新SCF、積分、DF因子生成、state生成は行わない。1.30 Åは独立確認候補として旧記録に保持し、初回評価には追加しない。技術実装条件の指定と科学的実験契約の正式採択は別であり、本書の入力は後者の採択待ちである。

## 3. 固定SOURCEとhash

新SOURCE固定commit：`4bd39ca68426f754111e47750534a7a56627107b`。基点は`05513eea807fbfb841c6012d2fe2133d70b5f1f9`。初回SOURCE commit `27a977081d3503a529e98ee7b93f34ab68bee409`を保持し、単位別許容値を分離し、親namespaceと数値import前の入力bytes照合を加えた最終版を追加commitした。

| SOURCE | 役割 |
|---|---|
| [control.py][CONTROL] | stdlibのみのCLI、設定構造検査、将来の契約・承認・SOURCE照合 |
| [science.py][SCIENCE] | 凍結入力adapter、独立再構成gate、W/v/ρ・誤差台帳・保存処理。未実行 |
| [runner][RUNNER] | 既存runnerと接続しない専用入口 |
| [static checker][CHECKER] | AST/構文・CLI・JSON・拒否経路のみ。数値import/NPZアクセスを禁止 |
| [package initializer][INIT] | 副作用のないnamespace |

新規5 SOURCEと再利用する親namespace initializer 2件のbytes/SHA256は[source identity][IDENTITY]が正本。実行依存の6ファイルは[設定候補JSON][CONFIG]にもhashをbindingする。checkerは実行依存ではないため別に監査する。既存trotterlibのeager importを避け、`trottertracks.algorithm_codesign.m1s_gate0`へ隔離した。共通library・旧runner・旧authorization/markerを変更していない。

## 4. G3／G6／G9

| ID | HDのDF block | HRのDF block |
|---|---|---|
| G3 | 0–2 | 3–11 |
| G6 | 0–5 | 6–11 |
| G9 | 0–8 | 9–11 |

Constantとone-bodyはHD。保存生成順indexを使い、重み順sortを呼ばない。既存`generation_partition`/`select_df_h_d`の規約との静的な矛盾は見つからなかった。追加split、rank変更、結果適応選択は拒否する。将来の結果は全3分割を報告し、global gate失敗で未評価の分割も行を残す。

## 5. Pauli-tail変換

保存演算子`H=c+Σh_pq a†_p a_q+Σ_l λ_l A_l²`、`A_l=Σg_l,pq a†_p a_q`を変換する。mode0が最上位bit、alpha偶数/beta奇数のoccupation basisを使う。Pauli wordはmode0からmode7の8文字、各文字はI/X/Y/Z、Python文字列辞書順で固定する。

JW ladderは`a†_p=Z_0…Z_(p−1)(X_p−iY_p)/2`、`a_p=Z_0…Z_(p−1)(X_p+iY_p)/2`。順序を保持してbilinearを作り、Pauli積としてA_l×A_lを展開するためfermionic contractionを落とさない。別にoccupation bitのcreation/annihilationからDF行列を作り、各ladderと分割後の演算子を照合するSOURCEを実装した。**この照合は未実行**。

同一wordの係数を符号付きで集約し、実部・虚部ごとに`math.fsum`を用いる。集約後の虚部L1を記録し、採択されたeps_imagを超えればtechnical STOP。許容内で実部へ射影した誤差も台帳と独立再構成gateに残す。実係数の近似cutoffは0という実装候補で、厳密な浮動小数点ゼロのみ削除する。小係数の自動削除、入力依存の閾値調整、暗黙のhermitizationは行わない。

Identity移動後の非零実係数c_Pから`λ_R=Σ|c_P|`、`p_P=|c_P|/λ_R`、`P_signed=sgn(c_P)P`。正負の同一wordを別eventとして残さない。DF-nativeの旧normalizerや旧価格は流用しない。λ_R=0なら空tailとしてW/v/ρをnullにする。Pauli表そのものは今回生成していない。

## 6. Identityとcontrolled relative phase

設定の`identity_policy="move_to_hd"`を明示的な候補とし、別policyを結果後に選び直さない。HRの集約後identity scalarをsとすると、`HD'=HD+sI`、`HR'=HR−sI`。保存constant、HD/HRのPauli scalar移動前後、s、全Hの再構成誤差を保存する。

T1bの正符号`exp(+i T_rel H)`規約で、HDへの追加phase/T_relは+s、HRからの除去は−s、和は0。deterministic half `exp(iδHD/4)`への追加phase/δはs/4と記録する。Scalar移動後は乱択law/λ_Rが変わるため、元のDF lawやidentityをtailに残すlawと有限step信号が同じとは主張しない。今回、controlled回路やphase信号を生成・検算していない。

## 7. State

保存`state_vector`の256成分を将来読み、36成分の`sector_state_vector`のN_alpha=N_beta=2 sectorへの埋込みと照合する。再正規化やphase修正をせず、norm誤差を検査した上で期待値は保存vectorの明示的なnorm²で割る。残差は全256次元の保存DF-Hに対し、Rayleigh値Eと`||Hψ−Eψ||_2/||ψ||_2`を記録する。

Labelは`NUMERICAL_SECTOR_GROUND_ORACLE_NOT_THEOREM_CERTIFIED`。小残差から全Fock ground、全Hのenergy非縮退性、(SP)、exact eigenstateへの距離を推定しない。gapのない残差はstate距離の証明にならず、本SOURCEは固有値計算へ自動拡張しない。

## 8. 数値精度と誤差予算

技術候補はcomplex128/float64。IEEE binary64の表現精度だけではDF展開の相殺、state誤差、小さいWの誤差は決まらないため、科学的許容値の数値は全て未決/nullとする。旧コードの1e−10等を転記していない。

| 設定 | Norm・単位・用途 |
|---|---|
| eps_H | full-HおよびHD/HR再構成の絶対Frobenius norm、Hartree。operator normを上回る技術指標 |
| eps_hermiticity_H、eps_hermiticity_factors | 入力h/full-H/HD/HRの`||M−M†||_F`（Hartree）とg因子の同norm（保存gの単位）を分離 |
| eps_phase | scalar移動に伴う全H差のFrobenius norm、Hartree。時間を掛けたphase誤差ではない |
| eps_state_norm | `abs(<ψ|ψ>−1)`、無次元 |
| eps_state_residual | 全H Rayleigh残差のEuclidean norm/||ψ||、Hartree |
| eps_basis | JW ladder行列・state埋込みの各F/2 norm、無次元 |
| eps_prob | p非負・総和1の誤差、無次元 |
| eps_imag_coeff、eps_imag_energy、eps_imag_expectation | Pauli係数の虚部L1（Hartree）、Rayleigh energyの虚部（Hartree）、Pauli期待値の虚部（無次元）を分離 |
| eps_W、eps_v | W/vの要求精度。独立のW_error_budget/v_error_budgetがそれぞれこの要求値以下であることを静的検査 |
| eps_rho | engineering error区間に対するρの最大許容不確かさ。未達はPRECISION_UNRESOLVED |
| W_error_budget、v_error_budget | 採択者が根拠を与えるW/vの絶対誤差予算。SOURCEはこれを自動証明しない |
| W_floor | Wの区間下端がこれ以下なら小分母UNDETERMINED。科学的headroom閾値とは別 |

概念上のeps_state/eps_imagは、normと残差、係数とenergyと期待値のように単位・用途別のkeyへ具体化した。別実験の許容値を科学的根拠なしに採択せず、それぞれの値を採択前に決める。

W/vのengineering区間は採択された予算e_W/e_vで`[W−e_W,W+e_W]∩[0,1]`、`[v−e_v,v+e_v]∩[0,1]`。W下端が正のときだけ`ρ_lo=(1+v_lo/W_hi)/2`、`ρ_hi=(1+v_hi/W_lo)/2`を計算する。区間整合性を検査し、生のW/v/ρをclampして隠さない。これは採択された予算への感度を示す**engineering区間**で、丸めを外向きに保証した厳密区間演算でもexact eigenstateの信頼区間でもない。

採択前に必要な根拠は、(a)要求精度の科学的意味、(b)DF→Pauli射影・再構成誤差のW/vへの配賦、(c)総和・行列積・差`1−<h²>`の丸め/相殺、(d)数値oracleとexact固有stateを区別する扱い、(e)小分母時のρ不確かさ。今回、これらの新規理論的error theoremや数値確認は行っていない。根拠不足なら数値を埋めず実行を保留する。

## 9. 主指標と分母分類

`h=HR/λ_R`、`W=1−||hψ||²/||ψ||²`、`a_P=<ψ|P_signed|ψ>/||ψ||²`、`v=Σp_P(a_P−Σp_Qa_Q)²`、`ρ=(W+v)/(2W)`。vは差二乗の中心化和を使う。W/v内部範囲、Pauli平均と行列平均の一致、虚部、finite値を独立に検査する。

λ=0はEMPTY_TAIL_NOT_APPLICABLE。浮動小数点W=0はW_ZERO_NUMERICAL_NOT_APPLICABLEで、数学的exact-zeroの証明ではない。W区間が0を含む場合、小分母の場合はUNDETERMINEDとしてρを補完しない。ρ精度不足も科学的STOPにしない。

## 10. 補助診断の優先順位

第一優先は入力・再構成・basis/law・state・定義域の監査。第二は全3分割の主量と誤差台帳。σ、|α|σ、gap、pair-sum/coupling、有限step信号は本runnerに実装・自動実行しない。必要性と範囲を別判断する。今回のSOURCE準備をT1c開始に読み替えない。

## 11. Baseline比較

GPT2の採択済み比較を仕様として保持する。同H、分割後law、state、理想相対時間、概念的総event位置Rを合わせ、shared r=R、ordinary r=R、independent各枝r=R/2（R偶数）。`λ_R²/R`で割った先導係数はshared `(W+v)/4`、ordinary/independent `W/2`。

本runnerはRやδの具体値を選ばず、有限stepやvisibilityを評価しない。(SP)未認証では実信号の単一rate比の等式とは呼ばない。全H非縮退性も未認証なのでJensen floor適用を認定しない。native単価・shots・control価格の等価性は仮定しない。

## 12. 適用域と限界

Gate 0数値oracle診断と、T1bのexact state・固定r小step・全H domain認証を分ける。対象energyの全空間非縮退性がないまま定理のfloorを数値結果へ無条件に適用しない。保存sector stateは全Fock groundの証明ではない。近接文献との厳密還元、新規性、一般DF event、Q4/Q5は未判断のまま。

## 13. 独立technical gate

| Gate | SOURCEでの検査と失敗の意味 |
|---|---|
| Input identity | raw SHA/bytes、member、shape/dtype、metadata、保存順。違えばglobal technical STOP |
| SOURCE/authority | cleanな専用checkout、各実行依存fileの固定commit/hash、採択済みcontract、別実施承認。数値importより前に拒否 |
| Basis ordering | 全16 ladderのJW/occupation行列差、sector indices/埋込み。未検証のbasis変換を許容しない |
| Reconstruction | occupation referenceとPauliのfull-H/HD/HRをfull256次元でF norm照合 |
| Hermiticity | 入力h/g、full-H、HD/HR、係数虚部L1を独立保存 |
| Identity/phase | scalar移動前後台帳とHD差+HR差の検査。回路phase実験ではない |
| Probability | signed aggregate後の非identity p、finite・非負・総和 |
| State norm/residual | norm²誤差、sector埋込み、全H残差。domain認証を代替しない |
| W/v内部整合 | finite、0≤v≤W≤1を誤差予算内で検査、平均の別表現と照合 |
| ρ定義域/精度 | λ、W区間、W_floor、ρ範囲/区間幅。NA/UNDETERMINEDを許容 |
| Domain applicability | oracle label、全H非縮退/(SP)未認証を必ず表示。無条件定理labelを発行しない |

technical STOPは当該技術的評価の停止で、研究全体のno-goではない。共通gate失敗でも全3行を保存する設計。分割固有の検査失敗はその行をSTOPとし、対象集合や閾値を変更しない。

## 14. 科学判断へ戻す結果

全3行のraw input/state/SOURCE identity、split、canonical表hash・実係数hex・p・signed event、identity台帳、λ/W/v/ρと定義域、全gate、誤差区間・根拠の区分、未評価事項を返す。改善余地が小さい/大きいという解釈も科学判断側へ戻す。有限step dressing、native価格・shot数が未確立であることを結果へ残す。

## 15. 未採択の判断閾値

ρ=0.7/0.85、|α|σ=0.1、必要な総資源削減率を採用しない。設定は`automatic_GO_STOP=false`、ρ/dressing閾値nullで固定。数値的研究STOPを正当化する根拠がないため、初回は全3分割を提示してユーザー/GPT判断へ戻す。将来の閾値案には使用する科学判断、理論的根拠、追加情報、閾値感度を結果前に示す必要がある。

## 16. 実行SOURCEの状態と未実行検証

SOURCEは実装済み。30件の静的/非科学的確認は、構文/AST、既定status、CLI help/schema、設定・JSON拒否経路、permit拒否、数値import/NPZ access禁止を対象にする。Python3.10.12と既存3.11.0rc1の両方で同じ30件を確認した。数値backendは一度も呼んでいない。

従ってadapterのcontraction/JW、Hamiltonian再構成、phase、state、W/v/ρ、数値精度・メモリ挙動は**correctness未検証**。実装の存在をRUN_READYや受理済み実験と呼ばない。将来の最初の数値検証も別実施承認に含める必要がある。単独のschema/hash検査ではこの未検証は解消しない。

## 17. CPU/GPU・メモリ準備案

Read-only capacity確認では32論理CPU、RAM 67,095,019,520 bytes（確認時available約54.7 GB）。`nvidia-smi`はdriver通信失敗で、GPUは利用確認できていない。benchmark、GPU kernel、科学的timingは実行していない。

静的な主要負荷候補はJW係数積の集約、12 DF blockの二乗、Pauli行列再構成。256×256 complex128行列は1 MiBで、12 block参照を共有保存し、bilinearとblock Pauli表を一度だけ構成する。Pauli作用はbitmaskで適用し、最大256種ずつのflip/phase配列をcacheする。全Pのdense行列をcacheせず、G3/6/9を単一プロセスで順次処理する。

初期backend候補はCPU、workers=1、BLAS/OpenMP/Numba/Rayon各1 thread。小さい256次元行列とPython辞書処理に多数processを投入せず、同じ参照dataを再利用する。科学的speedup倍率は主張しない。既存3.11環境のdistribution metadataはNumPy1.26.4等を示すが、import動作は今回未検証。将来の固定runtimeは採択待ち/null。

科学wall-time上限は既定なし。memory/output byte cap、Pauli表上限65536、積寄与のinstruction/call相当上限を設定してから実行する。memory/output/積寄与の数値上限は現状未決で、プロセスのRLIMIT_AS/FSIZEとコード内計数を使う案。出力capはgate台帳と表を含むaggregate resultに適用し、stdoutは結果の所在のみ。ログの別無制限生成はない。メモリ見積りは行列だけの下限であり、Python辞書・NumPy virtual memory・JSON展開を含む実測ではない。

## 18. 科学実行承認と起動防止

現在、実施承認なし、採択済みGate 0 contractなし、新authorization/runner markerなし。入力をCLIへ渡すだけでは引数エラーとなる。既定はstatus表示のみで、数値を計算するdry-runではない。

将来の`execute`は、(1) SOURCE固定commit、(2) それを参照し全精度・runtime/limitsを埋めた採択済み契約commit、(3) 契約採択とは別の明示科学実施承認commit、を要求する。cleanな専用checkoutのHEADを(3)と照合し、(1)→(2)→(3)の祖先関係、contract bytes/SHA、6 SOURCE bytes/SHA、Python/thread設定を検査した後に初めて数値moduleをimportする。

実施承認記録はscope、実ユーザー承認の引用、contract commit/path/hash、SOURCE commit、新出力directoryを含む。JSONを作れば人間の同意が成立するわけではなく、これは実際の別承認を取り違えないためのprovenance guardである。任意コード改変へのsecurity境界を主張しない。今回、承認記録を発行していない。

安全に利用できる非科学的コマンドは次の通り。

```text
python scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom.py --help
python scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom.py status
python scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom.py schema
python scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom.py validate-config artifacts/track_b_m1s_gate0_source_preparation/2026-10-16/v1/execution_contract_candidate.json
python scripts/tracks/algorithm_codesign/check_m1s_gate0_source_static.py
```

上記の設定確認はNPZを読まず、execution_ready=falseを返す。`schema`はJSON templateであり外部JSON Schema仕様そのものではない。検査規則はcontrol.pyに実装している。

## 19. 保存・再現性・監査

将来の結果は`artifacts/track_b_m1s_gate0_result/`以下の新directoryを排他的に作成し、result.jsonとSHA256を保存する。既存pathがあれば再実行・上書きを拒否する。新directoryは結果容器で、別のrunner markerではない。今回、結果directoryも作成していない。

JSONはnonfiniteを拒否し、入力/SOURCE/契約/承認、全設定、table、gate、失敗・未評価行、実際のruntime/RSS/elapsedを記録する。失敗時に閾値・分割・state・geometryを変えず、保存を超える異常（process kill、output cap、メモリ不足）ではdirectoryが残り得る。自動retryはしない。登録科学実行前の障害対策・出力上限は採択時に確認する。

今回の監査は[静的検査][STATIC]、[保存対象保全][PRESERVE]、[evidence index][EVIDENCE]を参照。既存index本文/manifest項目は保持して新状態を追記し、過去のSTOPや受理前labelを遡及変更していない。公開payload/receiptは専用branchの固定commitでremote bytes/SHA256を照合する。

## 20. Mandatory STOP

今回の範囲はSOURCE準備、静的検査、契約候補、commit/push・公開照合まで。**新規科学計算0、Pauli表生成0、数値correctness test0、科学条件の正式採択0、新規authorization/marker0。公開後mandatory STOP。**

次はユーザーがSOURCEと本候補を確認して科学条件を正式採択する段階。その後もGate 0科学実行にはさらに別の明示承認が必要で、自動継続しない。

[PREFLIGHT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/05513eea807fbfb841c6012d2fe2133d70b5f1f9/docs/tracks/algorithm_codesign/m1s_gate0_headroom_preflight_and_prereg_draft_20261016.md
[GPT2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62554ee127a2bec1206367cf1754df2ff49fa000/docs/research/track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md
[INPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz
[CONTROL]: ../../../src/trottertracks/algorithm_codesign/m1s_gate0/control.py
[SCIENCE]: ../../../src/trottertracks/algorithm_codesign/m1s_gate0/science.py
[RUNNER]: ../../../scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom.py
[CHECKER]: ../../../scripts/tracks/algorithm_codesign/check_m1s_gate0_source_static.py
[INIT]: ../../../src/trottertracks/algorithm_codesign/m1s_gate0/__init__.py
[CONFIG]: ../../../artifacts/track_b_m1s_gate0_source_preparation/2026-10-16/v1/execution_contract_candidate.json
[EVIDENCE]: ../../../artifacts/track_b_m1s_gate0_source_preparation/2026-10-16/v1/evidence_index.json
[IDENTITY]: ../../../artifacts/track_b_m1s_gate0_source_preparation/2026-10-16/v1/source_identity.json
[STATIC]: ../../../artifacts/track_b_m1s_gate0_source_preparation/2026-10-16/v1/static_checks.json
[PRESERVE]: ../../../artifacts/track_b_m1s_gate0_source_preparation/2026-10-16/v1/preservation_audit.json
