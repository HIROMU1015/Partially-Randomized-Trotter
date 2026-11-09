# 新A Q0：保存m17の有限区間・shot境界診断

2026-11-09 JST／SERVER 2。**Q0固定scope完了。両armで、理想feedback下の単一方向340 shotsが保存誤差域を含む幅条件を満たした。339は同じ十分判定を満たさない。** これは測定方向・区間decoderを実装した結果ではなく、保存保証から作る許容集合に対する限定診断である。実RPE成功、新回路資格、Reference固有の改善、全18round総資源／総T改善は未認定。

承認対象は[固定計画1b795c25 §2][PLAN]。同じ研究系列の枝`codex/algo-improve-new-a-total-cost-20261109`を継続し、既存3計画文書とV6枝・結果・STOPは変更していない。今回はQ0のSOURCE、人工test、結果、検算、報告だけを追加した。Q回路・I接続案へ進まない。

## 1. 条件・入力・再利用

Linear H4、1.00 Å、STO-3G、DF12、元H厳密ground供給・準備費除外、Partial S2／LD3／h=.02／K2、canonical independent finite bits、ordinary diag(I,V)、Compiler A／Qiskit1.3の保存保証。energy精度1.5936e-4 Ha、failure .05、phase cap .4。対象はB′／Reference-B1のm17、q131072/r32/bits88だけ。PF・law・物理条件・precision・旧SOURCEを変更していない。

保存入力は計画の5ファイルに限定した。

- `5f672ce6884bd49231366875a9e24a539fcaec7a`のV6-P `RESULTS.json`／`SOURCE.json`／`CHECKS.json`。
- `f2638a7845685a119619382b850441f5d8d33713`のB′ `ERROR_RPE.json`と`GROUND_ALIAS.json`。

元energyのRayleigh値・真signal・gap・state vectorはpolicyにも診断値にも使用しない。aliasは保存されたGershgorin条件／branchの成功fieldのみを使う。m0〜16の保証は保存済みのprefix条件として保持し、shot／signalを再計算していない。Referenceの旧630ではなくB1後626/axisを対照にする。

再利用は、旧`track_c_compiler_a_18round.I`のAST抽出による同じmpmath区間constructor、既存`_hoeffding_shots`の式・人工反転確認、B′／V6-Pのraw radius／beta／disk規約。量子依存をimportする旧entry pointは呼ばない。旧二項関数はscipyの点tail／ppfであり、未知p全域の外向き区間反転としての資格はないため、計画のfallbackどおり**両armと全対照にHoeffdingを使用**した。二項による改善を0や推定値で補わない。

| 保存m17 field | B′ | Reference-B1 |
|---|---:|---:|
| raw radius lower | 0.5726873297263416 | 0.5726872813784947 |
| statistical phase lower β | 0.38598699711850315 | 0.3859869126962799 |
| total raw disk upper | 3.6648120444602677e-06 | 3.713159891479787e-06 |
| 旧十分shots/axis | 626 | 626 |

各binary64保存端点をそのまま区間へ入れる。`B=2/5−β`にはPF・SOURCE・finite-bit・native等の既存位相chargeが既に含まれる。diskを再加算せず、ρLも保存raw radius lowerのまま。無重みphysical coherenceの外側制約ρU=1を用い、真の半径とは置かない。Closed P3等のweighted信号へは適用していない。

## 2. 診断の意味と全域の扱い

prefix成功event上で前段のenergy半幅を`.4/t16`とし、m17の残差位相`u=−t17(E−Ec)`の半幅は`U=.8`。目標は最終集合のphase径≤`.8`、すなわちenergy半幅≤`.4/t17`。未知のEcを真値と置かず、この全区間を保持する。元のenergy alias範囲との交差は集合を広げないため、全域保証にはその外側集合を使用できる。

Qの方向を前段中心に対する接線方向としたときの物理meanは`μ=ρ sin(u+b+eφ)`。`ρ∈[ρL,1]`、`|b|≤B`、`|eφ|≤δ`。δ=0は楽観的screenであり、feedbackの未評価誤差を0と認定していない。δの探索域全体[0,.4]でも`U+B+δ<π/2`を192-bit区間で確認した。

固定nでは`c=2k/n−1`、`ε=sqrt(2 log(720)/n)`、`[c−ε,c+ε]`を反転する。最終roundの失敗予算は1/360。prefixの34/720と合わせて1/20となる。各回はfresh independent trialという旧モデルに条件付け、実samplingや適応停止は行わない。

**全countを数点gridで代用していない。** 単調sin枝では、正のcにおけるhull幅は上端がUへ到達する`c*=ρL sin(U−B−δ)−ε`まで非減少、その後は非増加。未飽和部分の導関数比較と、飽和後の下端の単調性から従う。したがってc*の両隣の整数count、中心・端点と負側の反射を検査すれば、全k=0..nの最大幅を囲める。c*の整数境界は両方向の区間端点から取得した。n=1〜1252の全候補を評価し、不確定なnはなかった。nを増やせば常に幅が減るとは仮定していない。

別のcheckerはsinの**両方の逆枝とperiod**からpreimageを作る。最初のPASSとその直前、1252について全kを実際に列挙し、上の包絡法と照合した。空集合は全モデル点と不整合なcountとして記録上区別し、幅超過を捨てない。分裂した枝のhullも確認する。

## 3. 数値結果

以下の幅はexact rational区間を12桁へ外向き表示したもの。表示が一致しても証明端点を丸めて同一視していない。完全な192-bit端点はraw結果に保存している。

| arm | n | 最悪phase径の包含区間 (rad) | `.8`判定 |
|---|---:|---:|---|
| B′ | 339 | [0.801590010863, 0.801590010864] | FAIL |
| B′ | 340 | [0.798974641702, 0.798974641703] | PASS |
| Reference-B1 | 339 | [0.801590095285, 0.801590095286] | FAIL |
| Reference-B1 | 340 | [0.798974726124, 0.798974726125] | PASS |

両armで、この固定annulus＋Hoeffding＋hull径判定を満たす有限候補中の最小nは340。物理的に必要最小のshot数や、あらゆるconfidence／decoderに対する最適値ではない。

未知半径だけによる無限標本時の最悪径はB′約0.39682693368336、Reference-B1約0.39682709289392 radだった。`.8`未満であり、今回の条件では半径不確かさだけを理由とするSTOPには至らなかった。半径が既知になったという意味ではない。

### feedback誤差に残る条件

| arm | n=340で許されるδ境界の近傍 (rad) | 安全側に切り下げた十分値 |
|---|---:|---:|
| B′ | 0.00074890040744192886776 | 0.000748900407 |
| Reference-B1 | 0.00074877425786144305756 | 0.000748774257 |

境界は[0,.4]の単調包含判定を80回二分し、下側PASS／上側FAILの有理bracketとして保存した。別checkerの全count反転でも双方を確認した。これは**将来のfeedback生成・lowering等が満たすべき条件**であり、実providerがこの誤差内にある証拠ではない。

1251・1252 shotsでの境界もscope内で保存した。離散count格子のため1251の境界が1252よりわずかに大きい例があり、単調なn最適化を仮定していない。これらを採用policyや別precisionへ読み替えない。

## 4. 強い対照と公平性

### 通常X/Y：十分証明内の整数最適化

同じm17予算`A=1/360`を使い、`nX,nY≥1`、`αX+αY=A`を認める。既存座標HoeffdingによるL2十分条件は

`2 log(2/αX)/nX + 2 log(2/αY)/nY ≤ (ρL sinβ)^2`。

固定整数に対する最良αは`αX=A nY/(nX+nY)`、`αY=A nX/(nX+nY)`。連続relaxation下界は`8 log(4/A)/(ρL sinβ)^2`で、B′は[1251.5932507407667,1251.5932507407668]、Reference-B1は[1251.5939820949366,1251.5939820949367]となる。この式の整数総数1251以下は不可能、1252で全配分を対称性込みで確認し、626/626・α各1/720が実行可能だった。

これは**このL2十分証明familyでの最適値**。relaxationを実行可能な価格や一般的shot下界とみなさない。前段情報を使うより強い集合decoderまで1252が最適とは主張しない。

### 前段情報を使うPauli方向・より強いX/Y集合対照

単一のXまたはYを中心phaseに応じて選ぶ対照には、中心π/4で同じ投影meanを持つ二つの許容点がある。両Pauli選択は反射により同じ障害を持つ。bias込みの残差径はB′約0.98904208120601、Reference-B1約0.98904230902751 radで`.8`を超える。**無限標本でも、この保存許容集合の一様hull条件は満たせない。** 実H4がその反例点を取るという観測ではない。

さらに、X/Y両方を使い前段区間・未知ρを利用する強い集合対照も、総数340以下では同じHoeffding CI-hull条件を一様に満たせないことを検算した。中心π/4、等半径ρL、二つの残差`u=±.41`を取る。投影信号の両座標は共通中点から`Δ=ρL sin(.4−B+.01)/sqrt(2)`だけ離れる。各軸に**総m17予算を丸ごと与える楽観的relaxation**でも、全n=1..340で`ε(n)−1/n−Δ>0`。その最小余裕は両armとも約0.0375884だった。中点に最も近い整数countは誤差≤1/nなので、同じX/Y count区間に両点が入り、径`.82>.8`となる。配分が中心phaseに依存しても、この中心における全整数配分を覆う。

この検算により、旧1252だけを弱い対照として比較する問題は避けた。ただし排除したのは、**同じ保存許容集合とHoeffding CI-hull保証を使う固定方向X/Y**の総数≤340である。既存Hamiltonianの到達可能signal全体、直接の二項検定、任意の測定・統計方式に対する情報論的下界ではない。強い集合対照の厳密な最小総数は341〜1252の間で未確定で、追加の最適化はしていない。

B′にもReferenceにも同じQ改善機会を与え、両者とも340という同じ判定になった。Reference固有の利益を示した結果ではない。強Full／Truncated PFにはQ0を新たに適用しておらず、それらに対するPR優越も未認定。

## 5. 検証・SOURCE・完全結果

- [診断SOURCE](../../../scripts/diagnose_pr_reference_q0.py)：固定2 armの入力射影、192-bit区間、全n包絡、feedback境界、整数X/Y十分条件。
- [人工test](../../../tests/test_pr_reference_q0.py)：8件＋必要な追加2件、計10件PASS。追加2件の際に旧8件は不要に再実行していない。
- [別の反転checker](../../../scripts/check_pr_reference_q0.py)：全count／全feedback bracket、同投影反例、強X/Y境界、failure和、入力不変を検算。
- [SOURCE.json](../../../artifacts/pr_reference_q0/2026-11-09/run01/SOURCE.json)／[INPUTS.json](../../../artifacts/pr_reference_q0/2026-11-09/run01/INPUTS.json)：実行script hash、固定commit／blob／hash、再利用SOURCE、保存端点と条件。
- [完全RESULTS.json.xz](../../../artifacts/pr_reference_q0/2026-11-09/run01/RESULTS.json.xz)：両armの全n=1..1252、最大幅・count witness・整数端点・feedback bracket・X/Y会計。未公開の抜粋を結果本体にしていない。
- [CHECKS.json](../../../artifacts/pr_reference_q0/2026-11-09/run01/CHECKS.json)／[STOP.json](../../../artifacts/pr_reference_q0/2026-11-09/run01/STOP.json)。検算は著者Codexによるもので、独立科学レビューではない。

raw結果は4,233,562 bytes、SHA256 `d338fb14201ec23274687a7acabe704007a53521eced2ad8c8c6c79b21259ec8`。公開xzは396,748 bytes、SHA256 `6e81eb2b8357257e5a751d3a1db930f419961554d36bb58df8ccd0b2bda2323e`。解凍後の全bytesがrawと一致することを確認した。元rawは同じlocal run01に保全し、新namespace内の.gitignoreで重複公開だけを避ける。

再実行には別途その範囲の許可が必要。今回のcommandは`python -B scripts/diagnose_pr_reference_q0.py --output artifacts/pr_reference_q0/2026-11-09/run01`、検算は`python -B scripts/check_pr_reference_q0.py artifacts/pr_reference_q0/2026-11-09/run01`。公開xzを別のfresh出力領域へ解凍すればcheckerの入力形式になる。保全済みrun01を上書きしない。

診断CLIは1回、exit0、科学診断部分12.735秒（一般的性能比較ではない）。2独立armを2 processで処理し、人工的なCPU/RAM上限は設けなかった。起動時のMemAvailableは約254 GiB、CPU／memory PSI avg10=0を観測。既存mpmath1.3.0 runtimeを読取利用し、共有package・OS設定・他processへ変更なし。最初のPython探索で標準venvのensurepip不足が分かったため、システムを変更せず既存runtimeを利用した。

## 6. 費用接続・未資格・mandatory STOP

Q0はnative価格を取得せず、期待RZ/CX/depthや全roundを再積算していない。資源jでの次の判定は

`n(μevolution,j + δcreadout,j) + Aj < NX μX,j + NY μY,j`。

新readout角のSOURCE／lowering／feedback latency／strict word等、追加半径校正が必要ならその全shot費も残る。今回Aj=0、δc=0と価格会計した結果ではない。340をそのまま全18roundへ掛けたり、任意角RZを一定Tへ換算しない。

**次担当はGPT／ユーザー。** 保存誤差域の下でのQ0結果を踏まえ、Qの成功保証・実feedback構成を次に扱う価値があるか判断する。採択する場合は、δ境界を満たす角度生成・wrapper・confidence decoderとその限定correctness／compile scopeを別途固定・承認する。今回その実装・重要レビューを開始していない。

旧SOURCE／raw／result／grant／ledger／marker／STOP、V6-P/Q、CT、D3、43-key、他Trackは不変更。Q0以外の新科学計算、sampling、回路build/compile/simulation、RPE全round、価格／総T評価は0。

**結果・検算を公開しremote SHA・内容を照合後mandatory STOP。追加探索・回路実装へ自動進行しない。**

[PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1b795c250c8c2fc0a463695b472aa6877951c620/docs/research/pr_reference_algorithm_options_20261109/NEXT_VALIDATION_PLAN.md
