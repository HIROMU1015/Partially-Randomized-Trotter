# 研究C：H6独立検証の技術調査・実装準備

2026-11-06 JST。**技術準備のみ。新しい実H6/H2計算、H4保存値の数値再解析、重要科学レビューは実施していない。** [未承認契約案](research_c_h6_contract_unapproved_20261106.json)は条件・規模・検査資格を判断するための案で、`execution_ready=false`、`execution_entry=null`である。今回用意したCLIはpassiveで、実行オプションを持たない。

## 1. 固定資料・取得範囲

- H4科学結果はcommit `417fa459b823509be5c2010c5836e58955088282`の[報告](research_c_integrated_h4_100_results_20261106.md)と[科学SOURCE](../../artifacts/research_c_integrated/2026-11-06-picklefix/technical/SOURCE.json)。SOURCE SHA256は`8113f23e66c5eca3c09a8ed1c91fc58c01eeaa91d0c07850b685a20c816539eb`。主C0/64の共通r32で51.08%、公平retuningで45.47%低い条件付き総RZ点推定を得たが、U一次・actual CA・全law・一般化は未資格である。
- 論文準備commit `5a53d0569b70e4417765c4876616260db911e603`の[claim対応](research_c_h4_100_paper_evidence_20261106.md)、[Results](research_c_h4_100_results_draft_20261106.md)、[Discussion引渡し](research_c_h4_100_discussion_handoff_20261106.md)を再利用した。先行研究の2冊は`1a588f5491bce394ecf039ee4d2de57581e2e0b1`を起点とする既存整理であり、再調査していない。
- `research_c_publication_review_20261106.md`は作業ツリー・指定資料・当該pathのGit履歴から取得できず、**未読**である。GitHub code searchも0件だったがincomplete_results=trueなので不存在の証明とは扱わない。レビューの結論を代行・推測せず、今回のユーザー指示と取得済み固定科学資料に限って準備した。
- 正式運用指示`research_project_instructions_unified_2026-10-25.md`の§8.1/8.4/10/10.1/10.3を適用。承認証明・新grant発行・承認原文保存は追加していない。AI sub-agentは使っていない。

H6はH4とは異なるHamiltonianを用いる独立検証案である。H4 endpoint、ground、λ、native価格はH6へ移植しない。H4/1.15 Åの停止はそのまま保全し、修復・距離変更による代替移送を行わない。

## 2. 既存実装の適合性

調査基点は上記5a53d0569。下表の「再利用」はコード/手順の再利用であり、別runの実行許可や結果資格を移すことではない。

| 領域 | 既存の具体的実装 | H6への扱い |
|---|---|---|
| 物理入力 | `c1_molecule_extension.integrals_from_specification`、既存H6入力 | RHF/ao2mo生成処理は既存。今回の推奨案では保存H6入力をbyte固定で再利用し、新SCF/積分生成を省く。 |
| sector/物理DF | `c1_molecule_extension.SectorCAR(6)`、`rank_assessment`、`representation_gate`、`c_near_transfer.PhysicalContext` | sector400の疎CAR、元g補正κ、nuclear scalar once、norm+guardをそのまま再利用できる。H4専用`Evaluation.assess`のshape(8,4,4)/sector36を呼ばない。 |
| 中立初期値/codec | `c1_real_orthogonal_codec`、`c1_h6_corrected_fit`の保存CDF14 | 正則化なしの既存有限CDF14を共通C0へ。新Cholesky・中立fitは不要。旧の不忠実なreal logm codecは使わない。初期codec/物理/正規化は将来の新scopeで確認する。 |
| PR/ENTRY目的 | `c_pr_aware_df`、`c_near_objective`、`c_near_stable` | rank/norb/LDは引数化済み。既存のsigned対称J・frame別実直交U、同じtensor scale、各自の初期mass scaleを使える。 |
| 有限最適化 | `c_near_transfer.recipe_arm`、`c_integrated_robustness.recipe_cell/paired_specs`、`c_common_convergence` | 新optimizer不要。64受理更新/512 trial、16 checkpointを再利用。親と`PhysicalContext(6)`のRPC接続は未完。 |
| PF/RTE/ground | `c1_stage3_reference`、`c_pr_aware_df_stage_b.kernels/atom_operators`、`c_pr_aware_df_stage_b2.reference_context/trajectory_reference`、`c_pr_aware_df_m4`、`c1_h6_saved_pf.ground_alias` | 数値engineは再利用可。新helperはsector CARを既存random-path oracleへ渡し、full4096 CARを作らない。rank+1=15因子/LD4/r128への親の接続が必要。 |
| native/融合 | `c1_stage2_native`、`c1_h6_native_precision.PrecisionBuilder`、既存`StrongBuilder` | 一般K/number表現と既存の同等な相対Gaussian融合を両armへ。12-mode Gaussianの精度修正版を使う。新しい融合実験や一方だけの最適化は追加しない。 |
| P0/位相 | `c1_h6_compiler_precision.compiled_wrapper`、`pr_dictionary_validation` | Qiskit1.3.0/opt1/seed17/rz,sx,x,cxとdeclared phase集計を再利用。位相fitはしない。 |
| native正しさ | `c1_h6_state_action`、`c1_subspace_synthesis_bound`、`c1_h6_synthesis_bound`、`c_full_unitary` | 全800入力/full8192出力の列処理と局所gate bridgeを再利用し、raw/compiled/実逆回路/returnを新helperで接続。全標本か事前subsetかは新契約の科学条件。 |
| shots/会計/統計 | `c1_stage3_accuracy`、`c1_h6_ca_resources.serial_round_cost`、既存serial aggregator、`c_near_r32_statistics` | 式は再実装しない。固定N128・主97.5%を引数で与える。H4の64件/98.75%固定wrapperを直接使わない。 |
| 同J/U D | 既存全frame ordered S2・`c1_molecule_extension.PF_rounds` | 手順は再利用できるが`c_update064_strong_d.evaluate`は8frame/full256/9q固定なので、そのままH6へ起動不可。sector/新native検査への接続が必要。 |

Track Aの固定branch `track-a-h6-native-b0i-full-family-results-20261015`（commit `34efc914b2f485ec694c7d6016025d5c76829be6`）の`h6_b0i_family_bridge_v1.py`、`h6_b0i_family_parallel_compile_v1.py`も静的に確認した。入力/構築/compileの分離、identity/scalarの保持は参考になる。一方、そのbridgeが参照する`ax2b_native_df_v5`原本は同branchのGit treeで取得できず、AX engineを読了・移植したとは扱わない。別研究のprefix/q、T=.8、private grant/binding/control機構は研究Cへ持ち込まない。

既存`df_gpu_statevector.py`のcache/parameter template/XXPlusYY位相補正、`df_rpe_hadamard_compiled_cost.py`のworkloadと価格評価分離も確認した。Track B系の`h6_rte_fairness.py`（commit `00958ae1945c780718b2deef07d5dab1096a5cd8`）はPauli境界・別の十分shot条件を扱うため、研究Cのshot式と置換しない。現在の評価環境のAerはCPUのみで、GPU経路は今回採用していない。

## 3. H4から変わる条件と推奨案

| 項目 | H4既報 | 未承認H6案・理由 |
|---|---|---|
| 系 | linear H4/1.00 Å/STO-3G | linear H6/1.00 Å/STO-3G。距離は揃え、Hamiltonianサイズ/構造を変える。距離移送実験ではない。 |
| 空間/電子sector | 4軌道/8 modes、Nα=Nβ2、36次元 | 6軌道/12 modes、Nα=Nβ3、400次元。controlled full dimension8192。 |
| rank | 8 | **14を固定候補**。既存の中立CDF14でDF予算内の保存証拠がある。rank8を継承せず、新rank ladderも行わない。 |
| LD | 元順2 | **元順4を提案**。14frameの約1/4をheadとする事前規則に近く、tail10frameを残す。最適LDという証拠はない。LD2は同じhead個数を維持する代案で、同時に探索しない。 |
| J/U自由度 | 80+48=128 | 独立対称J294、実直交U210、計504。signed JをPSDへ制限しない。 |
| 初期値 | C0/P1/P2 | **既存の中立CDF14一つのC0**。まず独立Hamiltonianの問いへ絞る。H6の初期値分布・摂動頑健性は主張できない。 |
| 有限予算 | 16/64両endpointの下流 | 64だけ下流。16は物理/残差/λの保存診断。予算による資源頑健性は未検証となる。 |
| r | 16/32/64 | 共通r32と**32/64/128**の公平retuning。H6のtail規模を考え上側を含めるが、必要性/最適性は未測定。不適格でもr256を追加しない。 |

rank14の根拠は[旧の中立fit報告](research_c_c1_h6_corrected_fit_execution_20261025.md)の保存値引用である。CDF14のsector norm+guardは`1.0803058514e-4 Ha`、予算`1.9875e-4 Ha`以内だった。optimizer successは同時J/U一次収束を意味しない。この結果はPR/ENTRYのH6優位を示すものではなく、共通の正則化なし開始点を選ぶ根拠に限る。rank18は将来の別案として残し、rank14の結果を見て自動切替しない。

### 入力の所在（配列decodeなしのbyte確認）

| 用途 | サーバー/公開repository内のpath | SHA256 |
|---|---|---|
| H6原入力 | `artifacts/research_c_c1_h2_h6_science/2026-10-25/H6-run01/input.npz` | `b82ee9864d9c76071944834ae32d3d6eea7e95959708c2e19bca011936bce447` |
| 提案C0 | `artifacts/research_c_c1_h6_corrected_fit/2026-10-25/run01/fit/final.npz` | `cd4de687f99d32bd6f644ac33100fb52d4f7543e710e6c273723a0908b87dc45` |
| 初期化の来歴のみ | `artifacts/research_c_c1_h2_h6_science/2026-10-25/H6-run01/XDF_selected.npz` | `559dc43bf923a421ccc62d05c4cefa0bdf39a48ba68824abf3c8f770c04642b1` |

全3件が存在し、ハッシュは既存契約の値と一致した。今回NPZを開いていない。H6旧ground/native価格を新endpointへ転用しない。新実行時にはC0のfaithful codec、物理admission、正のtensor/mass scaleを同じ候補で確認し、不成立ならSTOPする。SCF・積分・neutral fitの自動再計算をfallbackにしない。

## 4. 公平比較・選択・統計

両armへ同一H6入力/C0、rank14/LD4、同じDF予算・工程的guard、同じfinite optimizer、初期正則化寄与.03/.1/.3、共通PF/RTE/RPE/P0/会計を与える。6 trajectoryを各64受理更新/512 trialまで評価する案である。J/U残差とraw optimizer状態を分けて記録し、U未収束を隠さない。同じ受理更新数は同じ古典計算量・到達誤差ではない。

各適格(candidate,r)のselection32のみから、共通r32では3強度、公平retuningでは同じ3強度×3rの最小条件付き総RZを選ぶ。tieは低いstrength、次にr32/64/128。最終データで選び直さない。新しいH6 gridは既存`select_training`の追加keyword引数で扱い、H4既定値は維持した。

独立final128を最大4個の`(arm,candidate,r)`へ与える。同armでcommon/tunedのペアが一致すれば同じcohortを再利用し、独立反復と数えない。**arm間は候補がbyte一致しても別cohort**にする。PR対ENTRYのWelch比較はarm間の独立性を仮定する。同J/U Dは同じ物理入力/候補/政策でだけ共有し、全決定論PFの最適解とは呼ばない。

主familyはH6 C0/64のcommon/tunedの2比較。family α=.05に対する各97.5%の保存前指定近似Welch差区間を提案する。H4の4比較family/98.75%は改訂しない。D副比較は個別nominal95%。固定N128で差区間半幅/ENTRY点推定≤5%、既存skewness/delete-one診断を確認し、未達ならUNRESOLVEDで終了する。必要Nを実H6データから検証した設計ではなく、効果や精度の達成は保証しない。任意追加・中間推論・bootstrap・新しい比CIは行わない。

費用は`C_total = W × mean_kernel_RZ + shell_RZ`。shots/W/kernel/shellを分離し、rや校正Nを再乗算しない。own-DF ground oracle・準備費用除外・ideal finite-RTE IID・exact-native等に条件付ける。全law、actual CA、一般的優位は未資格のまま。subset案では、さらに未検査価格回路の等価性という未検証条件が加わる。

## 5. Native正しさの4方式

| 方式 | 確認できる範囲・位相 | メモリ/時間の性質 | 採否とH4との差 |
|---|---|---|---|
| A 完全dense | 全8192入力/出力のraw/compiled/reference/実逆回路の数値比較。位相を除去しない。 | complex128一枚1 GiB、4枚4 GiBに参照/差/Gram等が追加。gate作用は概ねO(G D²)、dense Gram等はO(D³)。H4比で行列要素256倍、同次元cube4096倍（Gの増加は別）。 | 全入力範囲はH4と同じ種類。丸めenclosureや全law保証ではない。full4096 CAR/参照を無条件に作る案は推奨しない。 |
| B 構造的 | Gaussianの単粒子transfer、許可gateのideal unitarity、κ/identity/scalar、declared phase、inverse生成、正確なIR一致を利用。 | 小行列/IRで軽い。正確な同一IRと同一targetなら既存検査を再利用できる。 | 単粒子一致だけでは任意回路/controlled位相/P0最適化全体の等価性を証明できない。`Operator.equiv`はインストール版でもglobal phaseを無視するため不採用。完全なcompiler等価性証明を今回構築したとは主張しない。 |
| C 完全sector入力・full出力 | ancilla×sector400の全800基底を作用させ、各8192振幅を残す。raw/compiled、実際の両inverse、inverse returnを比較。漏出も残差に含む。 | O(G ×800×8192)型の作用、streamで常駐配列を抑える。局所gate変換/StateActions/ColumnStreamを再利用。 | 任意のsector入力に対する表現残差のFrobeniusによる数値上界（資格B）。他sectorの入力は未検査。5probeより強いが全8192入力と同等ではない。 |
| D 検査と価格標本の分離 | 事前固定した価格標本でCを実施し、全価格標本にはP0構造等を確認。 | 価格Nを維持したまま作用検査数を削減できる。 | 未検査標本の数値等価性は未確認。全標本PASSやH4同等資格と報告してはいけない。新契約での明示選択が必要。 |

新helperはCを実装準備した。`basis_batch`/`ColumnStream`を再実装せず、実際のraw/compiled逆回路を独立に生成・作用させる。reference daggerは逆回路の比較対象であり、実逆回路の評価の代替にしていない。Aerのdouble、zero_threshold0、truncation無効、max-component/norm-drift各1e-10、phase fitなしを使う。Frobenius計算の丸めやreferenceの誤差を囲っていないため、厳密certificateへ昇格しない。

### 推奨する未承認の検査配置

計算量を考慮し、**D+Cのpilot案**を契約の主案とした。各適格selection policyのsample0（最大18）、独立final pairのsample0（最大4）、同J/U D（最大4）の計最大26 kernelでCを行う。sample番号は費用を見て選ばない。残る最大1,066 kernelは`NUMERICAL_EQUIVALENCE_NOT_CHECKED`であり、構造検査のPASSを数値的等価性のPASSへ置換しない。

全1,092 kernelでCを行う代案も、異なるunit上限とともに契約内へ保存した。どちらもAのfull8192入力資格ではない。**主案の検査範囲の縮小を受け入れるかは、ユーザー/GPTが科学条件として判断する。** 今回の人工PASSは主案の科学的十分性の認定ではない。

shellはancilla-only構造を確認して全2×2を比較し、checked kernelには既存complex X/Y gateを接続する計画。scalar/nuclear/identityは一度だけ保持し、control0/1の相対位相を消さない。不変sector上の反復伝播には既存`repeat_bound`の前提を用い、compiledを途中でsectorへ射影しない。有限チェックからPR全law実装誤差を導かない。

## 6. 規模・時間・RAM・保存容量

| 登録作業 | 主案の最大 |
|---|---:|
| matched optimization trajectory / 保存checkpoint | 6 / 12 |
| downstream endpoint / candidate-r policy | 6 / 18 |
| selection32 / 独立final128 | 576 / 512 native標本 |
| stochastic native / 同J/U D | 1,088 / 4 kernel |
| 完全sector数値検査対象 | 26 kernel |
| backend state作用（入力列単位） | 26×6×800 = **124,800** |
| 最適化unit上限 | 19,278 |
| 全stage envelopeの合計 | **218,350 logical units** |
| 全kernelでCを行う代案 | 5,241,600 state作用、**5,335,150 logical units** |

unitは登録したmacro処理または入力列作用の数であり、CPU時間ではない。optimizer内部nfev/njev、gate/event数は別記録する。上限は`workload()`/契約のstage別内訳と一致し、未接続runnerでのcall kind割当・実測は未完である。scopeを結果後に増やす枠ではない。

sector400行列は一枚2.56 MB（約2.44 MiB）。全800列×8192振幅を常駐すると100 MiBだが、batch16なら一配列2 MiB、同時10配列のモデルは20 MiBである。これはprocess RSSの上限ではない。**mapped tail atom最大780個のdense cacheだけで約1.86 GiB/候補**、atom+3rのevent cacheを全保持すると約7.44 GiB/候補のモデルとなる。候補単位でatomを共有し、r別event cacheを解放/再利用して常駐を抑える。BLAS workspace、400行列の参照、Qiskit/Aer、JAX compilationを別に測る必要がある。

### 実測と予測を分ける

| 記録 | 処理/測定値 | 解釈 |
|---|---|---|
| 今回の[人工13q](../../artifacts/research_c_h6_validation_preparation/2026-11-06/technical/artificial_13q.json) | raw49/compiled121操作、4threads/batch32、800入力×6作用、**104.253秒**、peak RSS約690 MiB、最大成分残差1.60e-15 | 人工対角回路のみ。Gaussianや実H6のr128 kernelの速度ではない。 |
| 今回の[人工9q](../../artifacts/research_c_h6_validation_preparation/2026-11-06/technical/artificial_9q.json) | raw33/compiled81操作、72入力×6作用3.457秒、既存dense4行列構築0.877秒/16 MiB | 検査範囲/作用数が違うため高速化倍率として比較しない。streamはメモリ制御であり、常にdenseより速いとは限らない。 |
| [旧H6全800列](research_c_c1_h6_synthesis_bound_execution_20261026.md) | 4 unique circuitのforward計3,200列、runner wall5,173.3409秒（約86.2分） | 別SOURCE/旧D回路/別機材条件の既報値。今回再測定・再解析していない。 |

主案の26 checked kernelが人工回路と同じなら作用検査だけ約45分、全kernel案なら約31.6時間となる単純な規模換算にすぎない。旧H6 forwardの列throughputをそのまま6作用へ当てはめる別の感度例では、それぞれ約56時間/98日となる。**これらは実H6の予測区間でも上下界でもなく、どちらも外挿仮定が強い。** PRのr32〜128のgate数、Python batch組立、BLAS、共有負荷が異なるため、実分子全体の確定ETAは出せない。

fit6本・400次元PF/RTE・P0 compile・保存を加えた時間は未測定。主なボトルネック候補は、checked回路の多数入力作用、400行列のatom/event/reference積、r128の回路compileである。将来許可後の最初の**登録済み**batchのwall/CPU/RSS/QPYサイズから残件のETAを更新し、追加の未登録pilotを挟まない。

保存は**60 GiBの計画envelope**を置く。これは資源上限ではなく、sample当たりraw/compiled/両inverseのQPY計40 MiB以下なら全1,092件で約42.7 GiB、残りにevent/価格/参照/最適化rawを収める仮定である。単価未測定なので保証しない。checked以外の逆QPYを保存しない方針を選ぶならその差を後の実行SOURCEへ固定する。full8192 dense配列の標本別保存は行わず、checked列残差/順序・target identity・event・QPYを保全する。容量不足なら未投入分を止め、既存rawは削除しない。

## 7. 資源・環境・高速化

[環境/pins/観測](../../artifacts/research_c_h6_validation_preparation/2026-11-06/technical/environment.json)を保存した。64論理CPU affinity、RAM約503 GiB、観測中のavailable RAM約412〜417 GiB、保存先空き約351 GiB、A100 40GB×4が見えた。一方CPU loadは140〜160台、CPU PSI some約90%で、共有CPUは混雑していた。読めたcgroup階層ではcpu.max/memory.maxはmaxだったが、最上位の項目・disk quotaは未確認である。schedulerコマンドも未検出なので、無制限利用可能とは断定しない。

人工計測は小さな1 task/4 Aer threadsで行った。これは混雑と小規模処理に合わせた設定で、将来の恒常的上限ではない。実行時には候補並列とBLAS/OpenMP/Aerの積を協調させる。独立候補をprocessに分け、同一process内のP0/actual inverse生成には既存mutexを用い、参照計算のBLASを複数階層で過剰並列化しない。PSI上昇だけで既存科学runを強制再起動しない。

fit環境はPython3.12.3/NumPy1.26.4/SciPy1.14.1/ffsim0.0.83/JAX+jaxlib0.4.38、eval環境はNumPy1.26.4/SciPy1.14.1/Qiskit1.3.0/Aer0.15.1/Nature0.7.2/SymEngine0.13.0。fit環境のQiskit2.4.2をnative評価へ使わない。今回既存のプロジェクト専用fit/eval venvをread-onlyで使い、共有packageやGPU driverを変更していない。将来runnerは環境pathを明示引数で受け、ローカル絶対pathをportableと扱わない。

共有するものは同一物理input/context、候補内の15因子/ground/PF/exact-H、r間のPF参照、atom、basis/transition、shell、同J/U Dである。subset案ではunchecked価格標本の400次元trajectory referenceまで無駄に構築しない。正確なIR+phase+target identityの一致でだけ検査結果を再利用し、単にゲート数が同じという理由で共有しない。selection/finalの乱数と観測標本はpoolしない。

## 8. 実装差分・人工検証・再現方法

- 新`c_h6_validation_preparation.py`：metadata/workload/memory、既存sector-only random-path context接続、全sector raw/compiled/actual-inverse/returnチェック。
- `c_integrated_plan.select_training`：`r_values/common_r/tie_order`のkeyword引数だけ追加。H4の既定grid・tie・出力は維持。
- 新benchmarkとpassive preparation CLI、人工test2本。新optimizer、Hamiltonian再構成、native compiler、shots式、統計式は追加していない。

最初の関連人工/回帰105 testsがPASSし、その後の変更について最終の関連人工/回帰**126 PASS**を確認した。人工六軌道/rank14の既存PR/ENTRY目的は、codecのtensor fidelity、初期目的1.1、504成分gradientと有限差分を両armで確認し**2 PASS**。実分子のfitではない。最終件数と実行pathは[技術検証記録](../../artifacts/research_c_h6_validation_preparation/2026-11-06/technical/verification.json)を参照する。

位相だけの誤り、sector外へのleakage、実inverseの誤構成を拒否する人工反例を含む。sector外入力だけを変えた人工回路がsector検査を通る反例も明示し、full-Fock資格へ昇格できないことを確認する。これは作者自身の技術検査で、独立科学レビューではない。

```bash
# metadata/SOURCEのみ。科学実行機能なし。
python3 -B scripts/prepare_research_c_h6_validation.py

# Qiskit1.3/Aerの専用eval環境で、人工回路だけを測る例。
python -B scripts/benchmark_research_c_h6_validation.py --norb 6 \
  --threads 4 --batch-columns 32 --output /tmp/h6_artificial.json
```

## 9. 科学実行前に残す接続と判断

**技術的に未完**なのは、H6 PhysicalContextのfit RPCへの接続、rank14/LD4/sector400/r128を扱う親オーケストレーション、H6版D対照とshell/X-Y、subsetラベルとfinal128の会計/統計、stage counterとfresh保存先の接続である。個々のengineが存在することを、H6一括runnerの完成とは扱わない。今回のSOURCEは技術準備を再現するもので、接続後の科学SOURCEではない。承認後に旧H4 runnerをそのまま起動することはできない。

**ユーザー/GPTの科学条件判断**は、保存CDF14 C0の採用、rank14/LD4、C0単独/64下流という問いの範囲、r32/64/128とfinal128、D+Cの縮小した検査資格の許容、または全kernel C案への変更である。前者を採る場合でも「全価格標本を数値検証済み」とは書けない。lambdaだけ改善、総資源差なし/逆転、DF不適格、RPE不適格、精度未達、correctness未資格を区別して返す設計にした。

条件案は具体的な比較・上限・停止規則までレビュー可能だが、**現時点で科学実行を開始できる完成runner/SOURCEではない**。ユーザーが方針を選び、Codexが未接続部を同じ選択範囲で仕上げて人工E2Eを確認し、接続済みSOURCE/入力/契約を固定してから、H6科学実行への別の明示許可が必要である。中間の承認証明手続きやgrantは不要。

旧H4/H6のSOURCE・契約・input/raw・STOPは変更していない。新worktreeでselectorを拡張した一方、H4の固定科学コードはその歴史commit/元worktreeへ保全される。今回の差分を旧SOURCEへ上書きしていない。commit/push・公開照合後は**MANDATORY STOP**し、新H6計算、旧run resume、重要レビューへ自動進行しない。
