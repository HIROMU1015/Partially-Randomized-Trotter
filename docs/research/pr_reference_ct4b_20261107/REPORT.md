# 新A CT-4B：q128/r1の有限strict Clifford+T比較

2026-11-07 JST、SERVER 2。基準commit **`69e2c0ae4e79febe54df59b6ea7704238467a968`**、専用branch `codex/pr-reference-ct4b-20261107`。[固定scope](SCOPE.md)、[SOURCE・入力・raw・STOP](../../../artifacts/pr_reference_ct4/2026-11-07/README.md)。

## 結論

**保存q128/r1・rare/nonrare各j0・一様epsilon=10^-11では、Referenceの1shot T+T†はB′比でそれぞれ1.865933%／1.881674%少なかった。** X/Yとも同じT数だった。判定は`FINITE_TRAJECTORY_T_REDUCTION`。これは固定2 trajectoryと実採用word policyでの数値的削減であり、全support期待T、全18round総T、最小T、T-depth/FT計算全体の優越ではない。

8 native再compile、299 word（再利用111・新規188）、299 wordのstrict/±emitter検査、25局所対象、誤差/保存shot条件の確認を完了した。旧SOURCE・結果・STOPを保全し、**mandatory STOP**。著者Codexの技術検証であり、独立科学レビュー・第三者再実行ではない。

## 固定条件

Linear H4・1.00 Å・STO-3G・DF12、元の物理Hの厳密groundをshot毎に供給し準備費除外。Partial S2・LD3・h=.02・finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、保存18round scheduleのm7/q128/r1のみ。energy精度1.5936e-4 Ha、全失敗率.05、phase cap .4、alpha1/720、保存199 shot/axis。Qiskit1.3.0/compiler A/rz,sx,x,cx/全結合。

B′は4-CX controlled-RZZ。ReferenceのF/G・actual SOURCE inverse・vacuum P・2 register・8-CX CSWAP、D融合・basis共有・scalar・event符号・X/Y wrapper、追加8 qubit/reset/測定を維持。新gadget・新fusion・sampling law・precision/alpha/shot最適化は行っていない。

入力は保存transfer archiveの `raw/q128_r1_rare0_0.json`、`raw/q128_r1_rare1_0.json`。seedは14787950215150025216／9772911353196937292。各128 eventをそのまま用い、保存B′側events_sha256との一致も確認した。[PREDECLARED](../../../artifacts/pr_reference_ct4/2026-11-07/PREDECLARED.json)にevent全配列、hash、literal角・多重度、精度、reuse/missing、local対象を合成前保存した。SCF/DF/ground solve、新samplingは0。

## 実費比較

表は**各axis・各shot**のT+T†と、そのうちT†。X/Yは別々に検証・会計し、結果のT数が同じだったため併記する。stratum平均・law期待値は計算していない。

| stratum | axis | B′ T+T† | Reference T+T† | B′ T† | Reference T† | T削減数 | 相対削減 |
|---|---|---:|---:|---:|---:|---:|---:|
| nonrare | X/Y各々 | 9,307,406 | 9,133,736 | 1,690,742 | 1,586,726 | 173,670 | 1.865933% |
| rare | X/Y各々 | 9,230,078 | 9,056,398 | 1,690,746 | 1,586,606 | 173,680 | 1.881674% |

exact削減率はnonrare `86835/4653703`、rare `6680/355003`。全8行の[実会計](../../../artifacts/pr_reference_ct4/2026-11-07/PILOT_T_COUNTS.json)、[比較](../../../artifacts/pr_reference_ct4/2026-11-07/COMPARISONS.json)を正本とする。T/T†は実wordの文字数とactual adjointに従い、各物理出現へ課金する。key共有を物理gateの共有と数えず、未取得価格の0埋めもしない。

### Native指標と追加qubit

新native8構成は保存価格と一致した。以下のnative metricはTと合算しない。Referenceは全構成で追加8 reference qubit、reset8、source CSWAP16、最終measurement1を含む。

| stratum | arm | RZ X/Y | CX（各軸） | native depth X/Y | qubit |
|---|---|---:|---:|---:|---:|
| nonrare | B′ | 917709 / 917710 | 153846 | 311616 / 311616 | 9 |
| nonrare | Reference | 916901 / 916902 | 140222 | 140631 / 140632 | 17 |
| rare | B′ | 916806 / 916807 | 153746 | 311892 / 311892 | 9 |
| rare | Reference | 915985 / 915986 | 140118 | 140896 / 140897 | 17 |

## Role別の差

固定8構成に対するV3 role projectionと、再compileしたnativeのliteral signed RZ histogramはすべて一致した。各roleを同じ採用word価格で会計し、role合計が直接native会計と一致することも確認した。[ROLE_T_COUNTS](../../../artifacts/pr_reference_ct4/2026-11-07/ROLE_T_COUNTS.json)。

| T+T†寄与（X/Y各々） | B′ nonrare | Ref nonrare | B′ rare | Ref rare |
|---|---:|---:|---:|---:|
| D未融合部分 | 8,235,520 | 8,174,080 | 8,235,520 | 8,174,080 |
| D境界融合のsigned会計補正 | −420,624 | −536,760 | −420,624 | −536,760 |
| RTE frame/inverse | 1,463,720 | 1,463,720 | 1,386,384 | 1,386,384 |
| RTE中央回転 | 28,672 | 15,104 | 28,680 | 15,104 |
| product | 0 | 0 | 0 | 0 |
| event phase | 0 | 0 | 0 | 0 |
| scalar／vacuum P | 118 | 120 | 118 | 118 |
| CSWAP | 0 | 17,472 | 0 | 17,472 |
| X/Y wrapper | 0 | 0 | 0 | 0 |

D部分と中央回転の差が、ReferenceのCSWAP寄与を上回ったという**この有限会計の内訳**である。D融合は既存SOURCEの構成であり、今回新しいfusionを探索した結果ではない。signed補正は加法台帳からの差分で、負の物理gateや負の誤差上界を意味しない。

T=0のroleも、literal RZのstrict recordを含む実wordを通じて会計した。pi由来角をexact Cliffordと認定して無料化したのではなく、epsilon内で受理されたwordの価格である。今回の価格はcompiler後のRZごとのstrict置換policyに対応し、CSWAP等のblockを別のClifford+T gadgetへ置換する最適化とは比較していない。

## Precision・correctness・誤差

一様epsilon_G=10^-11はCT-0のm7/B′/Reference×X/Yの保存十分上端を満たす、既存仕様内の最も緩い候補として事前固定した。CT-1のq1用10^-9を移送していない。同一e・backend2.0.0・strict・seed17のC2-B record111件を再利用し、固定missing188件だけを独立cacheへ取得した。CT-1 e=-9 recordの再ラベル、legacy e=-13の未資格metadataの補完、cross-precision価格選択は0。

| 検証 | 結果と保証範囲 |
|---|---|
| 299 word | exact cyclotomic W付き行列とexact binary targetの外向きFrobenius guard PASS。±emitter598検査、actual word adjointと順序、実T数を確認 |
| 25 local対象 | D0 frame/inverse、D端/融合と実SOURCE inverse、RTE order0/2 Z/ZZ、CZ/Z、CSWAP、H/Sdg、2 vacuum Pを全局所作用空間で検査。phase fitなし。最大観測spectral差1.70070209583656e-11、各telescoping上界内 |
| 人工検査 | CT-1の3-qubit matrix-action対Qiskit検査を再利用。両CX方向・wire順・非ゼロglobal phaseを確認 |
| whole対応 | 全8 nativeのliteral RZ出現、qubit/global phase/reset/measurement・保存metricを保持して会計。全299 keyが閉包を覆う |
| 誤差/shot | 有限8構成と全support一様精度の**条件式**について計16行PASS。保存N=199/axisを維持 |

CT-1のGaussian単体代表1件は、固定wholeにないkeyを要するため合成前に除外した。新keyを足さず、対応Gaussianを含むD0 frame/inverse全体で局所検証した。ほかのlocal対象もwhole key閉包内に限定した。

wordのWはexactな数学的演算子として保持し、Qiskitの浮動global_phaseは別の数値診断とする。戻りSOURCE inverseのnative targetと、positive合成wordのactual adjointを混同しない。full joint relative phaseを保持し、新control付加・branch別phase fit・粒子数sector投影を行わない。

各物理RZのstrict guardを加算し、同じjoint unitary/reset/observableのoperator差をtelescopingで覆う。`d_syn≤2sqrt(2)sum_j epsilon_j` を旧compiler/vacuum算術等のraw diskへ一度だけ加え、半径/位相へ変換した。期待gate数をworst-case誤差数にしていない。

有限検査の連続shot必要数の保守上端は最大198.63828487867713。全supportで一様誤差を実装できた場合の条件式では最大198.68357185052454で、いずれも保存整数199以下だった。真の最小shot数を求めたものではない。[ERROR_AND_SHOTS](../../../artifacts/pr_reference_ct4/2026-11-07/ERROR_AND_SHOTS.json)に全16行のdisk・半径・位相・Nを保存した。

**全supportのword policyは実装/資格されていない。** 上の条件式PASSをcanonical law全体のCT RPE成功保証とせず、2本の有限trajectoryからcoherent mean、期待価格、sampling分散を推定しない。whole state simulation、17-qubit dense全operator、全expanded CT whole QPYは作らず、厳密誤差保証と有限数値診断を分けた。

## 再利用・実行・保全

SOURCEは[pr_reference_ct4](../../../src/trottertracks/pr_reference_ct4.py)。B′ build_saved/reconstructed、transfer.assemble、pilot.reference_core/wrapper、compiler A、track_c_h4_synthesis.acquire/strict_guard/exact_word_matrix、CT-0 validate_record、CT-1のrecord/word/local/人工検査、track_c2_b_verify.lower_native、V3 saved_path_partsを再利用した。q1固定入出力・endpointだけでは適合しないため、q128入力と有限会計のadapterを追加した。旧runnerを起動していない。

native8 task、missing key単位、local25 taskを独立processで並列化した。実CPU affinity64に対してnative8 worker、合成64 worker、local25 workerを使用し、BLAS thread数は各段階の独立task数に応じて設定した。恒常的なCPU/RAM上限を新設せず、共有設定・他ユーザー・他Trackを変更しない。実行環境は[ENVIRONMENT](../../../artifacts/pr_reference_ct4/2026-11-07/ENVIRONMENT.json)。

段階ごとのSOURCE snapshot、固定入力、PREDECLARED、実word/guard、native QPY/metadata、log、結果とSTOPを独立領域へ保存した。roleのsigned会計補正にoperator norm上界という意味を与えないよう、検証段階のSOURCEではrole価格から当該fieldを除いた。準備/合成段階の実行SOURCEもそのまま保全し、合成・compileの再実行はしていない。

## 旧CT-1との関係・未確定事項・STOP

旧CT-1のq1/e=-9ではB′/Referenceがnonrare64046/77882、rare63930/77766 T/axis/shotで、Referenceは増加していた。今回q128/e=-11では減少した。ただしprecision・trajectory・反復境界が異なるので、**qだけによる反転の因果証明ではない**。今回同一条件のB′/Referenceを比較した有限結果として読む。

未確定なのは、全supportの取得/停止/価格上界、canonical期待T、全18round総T、他q/r・precision・trajectoryへの移送、最小T/最適policy、T-depth/FT depth/space-time、一般的な新規性/優越である。既存のCX削減≥4.681782%、native depth削減≥40.995952%をTの保証へ統合しない。追加8qubitと旧RZ増加のtradeoffも保持する。

承認範囲完了で**mandatory STOP**。CT-3B/SK、全support、全18round、他候補・precision最適化へ進まない。次担当は**GPT／ユーザー**：この有限T削減と未資格域を踏まえ、全support価格保証へ研究資源を投入する価値・必要な次の証拠を判断する。Codexによる追加実行には、別途の限定scopeと明示承認が必要。
