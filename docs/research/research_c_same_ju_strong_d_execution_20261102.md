# 研究C：同一新J/Uの強決定論S2・156-call科学実行

2026-11-02 JST。固定SOURCE `5f27539ae0ab88181d9cfcd45badb54daf855c28` と[156-call契約](research_c_same_ju_strong_d_contract_unapproved_20261102.json)を変更せず、fresh `artifacts/research_c_same_ju_strong_d/2026-11-02/run01`へ一度実行した。**156/156 COMPLETE、correctness FAIL0、retry/resume0。全13 round、full512 raw/compiled・actual adjoint・X/YがPASS。同じ新J/Uでも主RZの条件付き点比較では決定論が低費用だった。** 公開照合後mandatory STOPする。

[全結果](../../artifacts/research_c_same_ju_strong_d/2026-11-02/run01/result.json)／[集計](../../artifacts/research_c_same_ju_strong_d/2026-11-02/execution_summary.json)／[call/STOP](../../artifacts/research_c_same_ju_strong_d/2026-11-02/run01/STOP.json)／[準備・人工134 PASS](research_c_same_ju_strong_d_preparation_20261102.md)。新PRは保存された統合結果の値だけを使用し、新しいPR評価・DF fit・optimizerは実行していない。

## 1. 同一J/U・固定条件・Stage A

保存直鎖H4・隣接1.00Å・STO-3G、通常number CDF rank8、Nα=Nβ=2/sector36、full256＋last ancilla/full512。統合実行の未収束`update005`を、PRと決定論の両方で固定する。元g補正κ・nuclear scalar一回・spin/frame順0..7を保持した。

- selected J/U：`artifacts/research_c_pr_integrated/2026-11-02/run01/selected.npz`、SHA256 `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`。
- 元h/g/scalar snapshot：SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`。元P-H4来歴は`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。
- 契約SHA256 `f199202333a376edf7fd8a428fa77aa7b094155eb73ea33feea69b22e13e77bd`。
- SOURCE manifest SHA256 `1e416afe6737ad4b705ba8a273acd6d1147e0ba15c6401643a24fa02bbbdbb64`、SOURCE75/input27。

保存selectedと独立物理評価NPZ内のJ/Uをexact arrayで照合し、各raw CAR因子9件のfull256-sector36対応・sum、full/sector S2 kernel、ground残差/正規化/alias/最低固有値の照合がPASSした。一般K adapter/native modelは同じ新candidate hashを使用する。Gaussian gate単体のsector射影でfull controlled正しさを代用していない。

DF sector誤差5.23337351802e-09Ha＋元engineering guard3.22739270291e-09Ha、CA/8=.00019875Haの保存PASSを再利用した。今回はDF norm・tensor誤差・元physical Hamiltonianを再計算せず、原本SOURCE/候補hashに結びついた保存値として扱う。新しいsector eigenpair1件はDF HamiltonianのPF exact reference13 roundと最低固有値確認に共用した。

| 条件 | 新D | 保存新PR |
| --- | --- | --- |
| evolution | 全9因子ordered symmetric S2 | partial S2/元順LD2＋finite RTE |
| s/δ/h/round | s1/δ=h=.1/m0..12/n=2^m | 同じs/time/round、r128/K0 |
| RTE固有項 | r/K非適用、A=1/RTE=law誤差0は定義 | 保存PF/補正RTE/A・law資格を維持 |
| 共通精度 | CA.00159Ha/DF reserve1/8、βtotalπ/6、βstatπ/12、failure.01/axis.01/26 | 同じRPE/phase/failure政策 |
| state | 同じown-DF ground oracle、準備費用除外 | 同じ仮定 |
| native/P0 | precise spin-lift κ・元U・full sweep relative fusion、opt1/seed17 | head-relative/exact basis-key tail共有、同じP0 |
| 費用 | 固定kernel1件、fenced serial＋head/readout各interrogation1回 | 保存32標本平均/SE、同じserial会計 |


P0はQiskit1.3.0、rz/sx/x/cx、approximation degree1、qubits_initially_zero=false、MPFR160 declared phase。合法融合を方式の実装範囲に従って適用し、未実装のPR cross-head/tail融合や全round長回路の追加compile/境界融合を仮定しない。DへPRのr/Kや標本数32を機械的に移さない。

## 2. Native・位相・actual adjoint・X/Y

既存M2、StrongBuilder、一般K、CAR reference、P0、full-input Frobenius propagation、RPE/Hoeffding/fixed-N・serial aggregatorを再利用した。旧SOURCEと今回SOURCEの変更0。1件のD kernelをbuild/compileし、raw/compiledと各actual adjointの**4 QPY・4行列**を保存した。9 Gaussian basis、16 directed relative境界・元frame/S2順・時間係数・scalar一回がPASS。

| 数値量 | 実測 |
| --- | --- |
| raw/reference max-entry | 5.08048355127e-15 |
| compiled/raw max-entry | 7.8283873288e-13 |
| compiled/reference max-entry | 7.87147108581e-13 |
| compiled/reference Frobenius | 6.44328529886e-12 |
| compiled actual-adjoint return max-entry | 3.51067511941e-13 |
| raw complex X/Y error | 2.44264835122e-15 |
| compiled complex X/Y error | 3.41348708994e-13 |


登録component/adjoint allowance1e-10、signal allowance2e-10、rtol0、global phase fitなし。全512入力列×全512出力でcontrol0 identity/control1 evolutionと相対位相・unitarityを確認した。実際に生成したadjointの復帰を確認し、理論上のdaggerだけで代用していない。共通head/X/Y raw/compiled QPY6件は保存hashとancilla-only構造を照合して再利用した。

[Kernel誤差](../../artifacts/research_c_same_ju_strong_d/2026-11-02/run01/segment.json)はfull-input/output Frobeniusを含む**level B数値pilot**であり、CAR/expm・Operator再構成・norm評価の丸めenclosureではない。Kernel/head/readoutそれぞれの観測誤差と両行列のnorm growthを全13 roundへ伝播し、未知誤差を0で補完していない。

## 3. 全13 round・shots・energy十分条件

主laneは合成誤差を含むD数値pilot。`ideal_exact_native`は保存PR idealと宣言条件をそろえる**理想合成の反実条件**であり、実際の合成誤差0の認定ではない。

| m | PF operator2誤差 | 合成complex-signal pilot | 数値lane N/axis | ideal N/axis |
| --- | --- | --- | --- | --- |
| 0 | 6.31593296863e-05 | 1.82399286563e-11 | 511 | 511 |
| 1 | 0.000125935664162 | 3.64642915224e-11 | 512 | 512 |
| 2 | 0.000248851398938 | 7.29130173248e-11 | 512 | 512 |
| 3 | 0.000474896800671 | 1.45810468968e-10 | 512 | 512 |
| 4 | 0.000803767148813 | 2.91605372285e-10 | 512 | 512 |
| 5 | 0.00108029093334 | 5.83195178981e-10 | 513 | 513 |
| 6 | 0.00155434881723 | 1.16637479259e-09 | 513 | 513 |
| 7 | 0.00301385571505 | 2.33273402069e-09 | 515 | 515 |
| 8 | 0.00592607907446 | 4.66545248032e-09 | 518 | 518 |
| 9 | 0.0118091567283 | 9.3308894134e-09 | 524 | 524 |
| 10 | 0.0236031770846 | 1.86617633348e-08 | 536 | 536 |
| 11 | 0.0471818902113 | 3.73235113985e-08 | 563 | 563 |
| 12 | 0.0943270740487 | 7.46470084096e-08 | 623 | 623 |


13/13 roundが登録phase/accuracy十分条件と既存の整数fixed-N marginをPASSした。m12 PF=0.0943270740487、合成signal pilot=7.46470084096e-08、合計phase pilot=0.094467592799<π/12。Physical DF誤差[Ha]はenergy側で一回、PF/合成誤差は同じsignal誤差尺度で一回ずつ計上した。

両laneの総shots/executionsは**13728**、energy十分評価pilotは0.001278325784Ha、candidate failure upper .01。測定したenergy誤差ではなく、own-DF ground oracle・宣言した誤差モデルに条件付く。Hoeffdingの整数十分shotsであり、普遍的最小shots・実量子samplingではない。合成誤差を除いた反実laneも、整数shotsと費用は今回の数値laneと一致した。両laneのphase pilot自体は異なり、観測合成誤差を消去したわけではない。

## 4. 同じ新J/Uのnative価格・総費用

| metric | 新D kernel固定値 | 保存新PR kernel平均 |
| --- | --- | --- |
| rz_count | 11026 | 19825.625 |
| rz_depth | 5568 | 5632.1875 |
| cx_count | 7000 | 7399.3125 |
| cx_depth | 6276 | 4152.59375 |
| total_depth | 12098 | 12393.1875 |
| circuit_size | 20919 | 39603.9375 |


| metric | 新D数値lane総費用 | 新PR ideal総費用 | PR/D ideal比 |
| --- | --- | --- | --- |
| rz_count | 105532782860 | 207919150188 | 1.97018542062 |
| rz_depth | 53292834864 | 59067007298 | 1.1083480068 |
| cx_count | 66998834000 | 77599490718 | 1.15822151051 |
| cx_depth | 60069240312 | 43549878473 | 0.724994660275 |
| total_depth | 115793216908 | 129972306754 | 1.12245181734 |
| circuit_size | 200221319010 | 415342108618 | 2.07441500571 |


主RZは新D **105532782860**、保存新PR ideal **207919150188**で、同じselected J/Uの**PR/D比1.9701854206178373**。PR assumedは208893540201.5、D数値laneに対して比1.9794184758552098。理想合成を両方に宣言した比較も主RZ比は1.9701854206178373で、今回の約2倍という点差をD数値合成/PR idealという資格差だけへ帰すことはできない。

一方、CX depthのPR/D比は0.724994660275でPR側が小さい。主RZで決定論が低費用という結果であり、すべての副指標でDが支配するとは主張しない。Depthはfenced scheduled sumで、全長回路を再compileした最小depthではない。

PR価格は32件の有限bit/PCG64標本平均で、母平均・厳密support・CIを認定しない。32をshots乗数にしない。D価格の等しい上下区間は固定されたserial gate countsの会計上の確定性であり、hardware費用や実CAの確率保証ではない。全law PR synthesis/実装law誤差・actual CA/保証付き総資源優位はnullのまま。

## 5. 旧/新J/Uを分けた結果

| 保存/方法 | kernel RZ | 総shots | 条件付き総RZ |
| --- | --- | --- | --- |
| 旧J/U D（M2 proposed） | 11026 | 13610 | 101825929455 |
| 旧J/U PR s1/r128 ideal | 14076.4375 | 16056 | 234117788135 |
| 新J/U D numerical | 11026 | 13728 | 105532782860 |
| 新J/U PR s1/r128 ideal | 19825.625 | 14048 | 207919150188 |


| 登録比較 left/right | RZ比 | CX比 | total-depth比 |
| --- | --- | --- | --- |
| new_PR_ideal/new_D_numerical | 1.97018542062 | 1.15822151051 | 1.12245181734 |
| new_PR_assumed/new_D_numerical | 1.97941847586 | 1.16364938722 | 1.1277120628 |
| new_PR_ideal/new_D_ideal | 1.97018542062 | 1.15822151051 | 1.12245181734 |
| old_PR_ideal/old_D | 2.29919618105 | 1.43546873267 | 1.42529657545 |
| old_PR_assumed/old_D | 2.30938780422 | 1.44183172025 | 1.4316144726 |
| new_D_numerical/old_D | 1.03640382587 | 1.03640383884 | 1.03640381689 |


旧J/U内のPR/D主RZ比は2.2991961810519377、新J/U内は1.9701854206178373。新PR自身は統合結果の保存比0.888096337507285（約11.19%減）だが、新D/旧Dは1.0364038258706803（約3.64%増）。決定論kernelの登録6価格は旧/新で一致し、旧M2 m12 PF .06988804953189413から新 .09432707404873293へ増えた。旧N/axis m12=591→新623、総shots13610→13728となり、後半roundの反復重みを含む費用が増えた。

この固定H4・rank8・PF/会計条件では、PR-aware表現更新による利益を、全決定論にも共通するnative削減としては観測しなかった。PR内部の条件付き改善と、同J/Uの強決定論を上回ることは別であり、後者は主RZで支持されない。これは登録した有限endpointの比較で、目的関数だけの因果効果、最適化済み決定論DF一般の最適性、PR全体の不可能性・研究の正式終了を証明するものではない。

## 6. Calls・資源・保全

| logical unit | 実績 |
| --- | --- |
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
| ideal_round_accuracy | 13 |
| ideal_energy | 1 |
| full_reference | 1 |
| reference_sector_gate | 1 |
| shell_load | 6 |
| shell_decode | 6 |
| shell_structure | 3 |
| shell_segment | 3 |
| gaussian_build | 9 |
| kernel_build | 1 |
| relative_gaussian | 16 |
| fusion_gate | 1 |
| compile | 1 |
| adjoint_build | 2 |
| operator_decode | 2 |
| adjoint_decode | 2 |
| segment_assessment | 1 |
| adjoint_gate | 2 |
| XY_gate | 1 |
| metrics | 1 |
| round_synthesis | 13 |
| round_accuracy | 13 |
| energy | 1 |
| round_serial_cost | 13 |
| aggregate | 1 |
| ideal_aggregate | 1 |
| comparison | 6 |
| result_summary | 1 |


最大156＝実績156 COMPLETE、FAIL/inflight0。新fit/optimizer/rank探索/DF誤差再計算/PR再計算/量子sampling/retry/resume0。別候補・s4・law/compiler探索・旧543/1759-call残枠の流用はない。

[Preflight](../../artifacts/research_c_same_ju_strong_d/2026-11-02/execution_preflight.json)で公開5f27539aeのcommit/tree・契約本文、SOURCE75/input27、10依存とfresh outputを照合した。134人工/回帰PASSは同じSOURCEの準備証拠として再利用し、今回新たにCI/外部再現や人工suiteの全再実行を行ったとは主張しない。

CPU affinity32、MemAvailable約54GB、load約1、CPU/memory pressure avg10=0を確認した。準備済みresource_planは4独立full512 decode/BLAS1を選び、Qiskit build/compile/inverse作成は逐次、raw/compiled/両actual-adjointのOperator計算を並列化した。4は独立task数で、CPU/RAM/worker/wallへ任意の低いcapを追加していない。利用不能なcgroup quota情報を無制限と読み替えない。

[実績](../../artifacts/research_c_same_ju_strong_d/2026-11-02/run01/resources.json)：wall31.5469345246秒、process CPU66.418104086秒、peak RSS609764KiB。Speedup比較は行っていない。Matplotlibの既定configが書込不能というwarningはtask-local TMPDIR内の自動cacheへ退避され、科学処理は正常終了した。共有config/venv/cacheや他ユーザーjobは変更せず、warningを理由にSOURCE変更・再実行していない。

[Postflight](../../artifacts/research_c_same_ju_strong_d/2026-11-02/execution_postflight.json)でSOURCE75/input27、旧SOURCE20組＋今回75、旧raw3079件を照合する。新raw39件（4QPY含む）とhash、journal、STOP・13round・全比較を保存した。旧SOURCE/raw/contract/grant/ledger/marker/STOPは不変、旧138科学recordsを維持して今回1件を追記、当日noteはappend-only。科学STOP後は既存JSON/JSONLの報告集計・hash照合・公開だけで、追加NPZ数値decode・科学queryは0。別Codex-Bの資源寄与分析ファイルを編集していない。

## 7. 結論の範囲・次の担当

同じ新selected J/Uでも、共通CA・RPE政策と準備除外・既存強native/serialモデルの主RZ点比較ではDが低費用だった。PR内部の改善は保全されるが、今回の表現更新でDも同様に改善したという結果は得られなかった。主RZとCX depthのtradeoffを区別し、未資格の全law/actual CA/母平均優位を認定しない。

次は**ユーザー／研究C**が、この同一J/U結果と別Bの資源寄与分析を踏まえ、H4でkernel価格を考慮する改良を続けるか、既存の条件付き成果を整理するかを判断する。採択された場合の**Codex**の候補作業は、同じ誤差/会計規約を維持した新しい最小技術設計と固定契約の準備であり、本156-call scopeに未完了科学計算はない。新fit・PR探索・H6・厳密保証の追加計算・重要GPT科学レビューへ自動進行しない。

Branch `codex/research-c-same-ju-strong-d-execution-20261102`。SOURCE/結果・call/STOP/QPY・本報告をcommit/pushし、remote SHA/tree・公開本文・SOURCE/inputを照合する。**公開照合後mandatory STOP。**
