# Q4-0後：強X/Y・matched二項CI比較の限定実行案

2026-11-10 JST。**未実行・科学数値実行は未承認。** 本書を承認済みcampaignや準備済みの実行runnerと読み替えない。今回完成させたのは対象・推定器・幾何・SOURCE再利用・検算・判定規則の設計。下記H/E/Cを実行するにはユーザーの明示的な実行承認が必要。外部Bayesian全protocol、別q、物理sampling/compile、CTはこの案にも含めない。

## 1. 目的・先に固定する条件

問いは、同じprefix情報・未知ρ/B・成功基準で強X/Y集合推定器を最適化すると、保存340-shot Qとの差がどこまで残るか。加えて、CIを強めた場合はQにも同じ機会を与える。

固定caseは[BASELINE_COMPARISON §1](BASELINE_COMPARISON.md)のH4／Partial S2/LD3/h=.02/K2、m17/q131072/r32/bits88、全failure .05、phase cap .4、精度1.5936e−4 Ha、既存prefix・compiler・law・192-bit directed interval。B′とReference-B1を両方独立に扱う。真のE0/ρ/gap/state、校正shotは追加しない。

主要taskは連続ψ全域のrobust許容集合。`U=4/5`、hull径上限`4/5`、A=1/360、prefix34/720、同じρL/β/diskを使用する。実H4の到達信号に対する情報論的最適値とは主張しない。有限実count中心だけへdomainを縮めたり、prefix条件を変更して利益を作らない。

### 入力正本と再利用SOURCE

すべての既存入力は`08f9cbe77456ec29d50a992d7165c4d726df7bf7`から読む。元の科学入力commitはf2638a78、Q0は34327b90、Q3-Bは1795d72b。同じ公開blobを別版と混ぜない。

| 保存path／SOURCE | 使用範囲 |
|---|---|
| `artifacts/pr_reference_q0/2026-11-09/run01/INPUTS.json` | 両armのρL、β、disk、target、失敗率規約。binary floatはexact Fractionへ |
| 同`SOURCE.json`／`RESULTS.json.xz`／`CHECKS.json`／`STOP.json` | 旧Qの340/339、旧L2 1252、強X/Y witnessの条件を**保存結果として**参照。旧scan／bisection／witness runnerは呼ばない |
| `artifacts/pr_reference_feedback_q3b/2026-11-10/run01/B1.json`／`B2.json`／`CHECKS.json` | fixed-prefix受理・δarith・中心κ・energy restorationの保存証拠。587,815 count監査を再走査しない |
| `artifacts/pr_reference_feedback_q3cd/2026-11-10/run01/C0.json`／`D.json`／`COST.json`／`CHECKS.json` | 同core・P/H/Sdg・旧Q成功保証・保存35/36 typeのnative単価/全prefix費用 |
| `pr_reference_feedback_q1.py`のcount型、`_feedback_phase`、`saved_q0_model`とQ3-B selector | API／丸め／bindingを継承。旧`decode_counts`はn340・旧Hoeffding限定なので、新n/CIへ無断改造しない |
| `scripts/diagnose_pr_reference_q0.py`の`setup/I/iv/exact/endpoints/pack/asin/verdict/epsilon` | 純粋helperの必要定義をAST再利用。`run/arm_run/worst/margin/xy_certificate`の旧科学campaignは呼ばない |
| `scripts/check_pr_reference_q0.py:sine_preimage` | branch/periodを残す単軸独立照合の設計を再利用。両軸を独立消去して交差するだけでは強joint集合を実装したことにならない |
| `src/trotterlib/rpe_hadamard_failure_validation.py:_clopper_pearson_interval`等 | API・端点caseの参考。scipy beta.ppf/点tailのままでは厳密coverageの数値資格が不足。import経由で科学builderを起動しない |
| `scripts/run_pr_reference_feedback_q3cd.py:pair/add/scale/subtract` | 科学runnerではなく純粋Fraction会計helperだけ必要なら再利用。`main/cost_stage`の旧campaignを再実行しない |

必要な新実装はjoint幾何、配分certifier、外向き二項tail反転と、その独立checker。将来の新SOURCEはこの4文書と上記固定入力に対応させ、実行前に新namespaceへ固定する。SOURCE/input hash・実行範囲・結果・STOPに必要十分な来歴だけを保存し、承認証明書／新grant／管理基盤は作らない。

## 2. 数値実行前に完成させる最小実装・人工検証

### Joint幾何

共通zのCI可行領域を解く。理想PauliならCI rectangleとannulus `[ρL,1]`、登録native軸(0,a)なら

`lX≤x≤hX, lY≤cos(a)x+sin(a)y≤hY`

というparallelogramとannulusの交差になる。aは保存binary literalのexact値、sin/cosは192-bit区間。sin(a)>0のSOURCE域を保ち、aをπ/2へ置換しない。[静的共有信号補題](BASELINE_COMPARISON.md#3-joint信号を共有するためのsource条件静的補題)のcore/observable/norm前提を人工fixtureと既存metadataで確認する。新H4回路compileは不要。

角度の極値候補はpolygon vertex、edgeと内外circleの交点、座標軸との境界点。angle branchを分割し、各候補間のradial可行性を調べれば角度集合を区間のunionとして構成できる。circle-edge tangent、円周全包含、外側／内側接触、空集合、分裂、π切断を含める。最後にBで角度集合を膨張し、ψを引き、[−U,U]と交差してu集合を得る。各軸のρを別に選ぶ近似は主certifierにしない。空集合は成功幅0として数えず`INCONSISTENT_COUNTS`にする。正しいprefixとCI coverage上では真値が残るという包含証明により、その拒否を末尾Aのfailureに含める。全count走査はこの非空集合の幅を囲み、空集合を理由に追加のfailureやretryを無料にしない。

人工caseは①一つの共通ρが存在しないのに各軸は別々に通る例、②内円で分裂するrectangle、③tangent/端点、④π切断、⑤未測定n0、⑥axis signとnative a、⑦失敗率合算、⑧binary中心区間を含むlookup境界。小さい有理数／symbolic例で処理し、保存H4の値で人工testを代用しない。

**実装上の不足は明記する：** repoにこのjoint optimizer／外向き二項inverseの資格済み実装はまだない。既存`q0.hull`はQの単調sin一軸に特化しており、関数名を変えて二軸用へ流用できない。不要なBayesian backendや回路providerの再実装はしない。

## 3. H：同じHoeffding familyでの強X/Y最適化

固定するCIはtwo-sided Hoeffding。Q側は既存n340・同じ成功基準の資格をそのまま使い、旧Qのn走査を繰り返さない。Q0のδ=0でn≤339がFAILという結果はこの固定Q familyの限定下界である。

### 最適化変数と量化順序

各prefix中心ψについて、整数nX,nY≥0、総数N≤1252、αX,αY≥0、和≤Aを最適化する。未測定軸はCI=[−1,1]として扱い、division by zeroを避ける。測定する軸のαは正。連続αを単なる有限gridやL2用のα式へ固定しない。

uniform-budget問題は概念的に

`N* = min{N : ∀ψ ∃prefix-only policy(ψ), nX+nY≤N, ∀(kX,kY) with S≠∅, diam hull S≤4/5}`。

`min_allocation sup_ψ`だけを解くとψ依存配分を禁じた弱い問題になる。反対に真のu/ρ/bを見てallocationを変えてはいけない。ψをinterval cellで覆い、**そのcell全体**に通用する配分・αのlookupを返す。入力中心区間が境界を跨ぐ場合の選択規則も証明し、未観測の真中心を選ばせない。

### 有限call範囲と計算方法

- 2 arm×理想Pauli／登録nativeの2幾何を対象とする**新H campaign 1回**。理想版は旧witnessとの定義照合、native版を主要な保証付き比較とする。
- Nの有限候補は1〜1252、nX=0〜N。既存の下界・可行対照を再利用して排除済み領域をpruneする。1252は旧L2の保存可行policyをfallbackとして保持するが、別joint decoderの全count幅まで自動的にPASSとはしない。
- Q0のπ/4 witnessは理想幾何のanchorとして読み、native a版には同じ二点構成の包含条件を**新幾何で**照合する。有限実履歴／実H4への下界に格上げしない。
- ψとαはinterval branch-and-bound、count indexはinteger boxを分割する。可行な2つのモデル点が同じcount CIに入りu差>4/5ならFAIL witness。PASSには全ψ/candidate count boxの外向き幅上界を用いる。
- 一軸Qの`candidate_counts` turning-point定理は二軸に無条件適用しない。必要なら全count latticeを覆う。continuous-centerの有限grid PASSだけでは不十分。
- count boxの上界にはpolygon/annulusの外側enclosure、下界には同時に実現可能な二点と正確なcount-lattice条件を使う。α→0境界はuninformative CIの極限として扱い、不都合な小αを任意cutoffで除かない。

numerical stopping ruleは192-bitを保つ。ψ/2π又はα/Aのbox幅が2^-128以下でも判定できないleafはUNRESOLVEDとして残し、scope完了時にgapへ算入する。この尺度はoptimizerの証明分解能であり、物理tolerance／成功条件を緩めるものではない。最適性が不要な可行policyが得られた後に余分なgap縮小を自動追加しない。

出力は`N_lower≤N*≤N_feasible`、可行lookup、排除witness、未解決leaf、n/αの自由度を含む。exact optimumは上下界が一致した場合だけ。最初に得た可行解を「最適」としない。全Nに対する上界が閉じない場合も、保存1252対照と新しく証明できた可行解／下界／gapを区別して保存する。

数学的幅と実装保証を区別する。可行候補はQ3-Bのκを保った`Ec−hull(u)/t17`を外向きに復元し、丸めを含む**energy区間径≤(4/5)/t17**まで確認する。新geometryの端点誤差を旧Q専用ωQ0で代用しない。数学的条件だけを通る境界候補は`MATHEMATICAL_WIDTH_ONLY`であり、Cへ渡さない。192-bit実装で閉じない場合は未資格として記録し、toleranceを緩めない。

## 4. E：等尾二項CIで両側を同時強化する閾値比較

Hと独立して、CI改善でQの優位が消える可能性を調べる。**新E campaign 1回、2 arm、登録native geometryのみ**。物理信号・precision・prefix・lawは不変更。

Bernoulli tailは有限和多項式として192-bit outward intervalで計算し、p∈[0,1]の単調tailを区間反転する。k=0/n、p=0/1、丸め端点を明示する。`scipy.stats.beta.ppf`は初期guessに使うとしても、最終上下界を必ずtailの有理／interval評価でcertifyする。点tailやWilks近似を証明としない。

等尾規約は双方同じ。X/YにはαX+αY=Aの連続配分を許し、Qには一つのCIへA全部を与える。非対称tailやjoint LRをQだけ／対照だけに追加しない。

最初の有限範囲は総数N=1〜340のA-E全整数配分と、Q-Eのn=1〜340・全k。Hの下界をEへ流用しない。この範囲で得られるのは、旧340を強X/Yが同じfinite-CI保証で達成／改善できるか、及び**Q-Eにも同じ強化を与えたとき**の比較である。A-Eが全てFAILなら`N_AE>340`というこのfamily・domainの下界だけを報告し、341以上の最適値を勝手に探索しない。

Q-Eの方向は保存Qと同じprefix後固定方向。δarithはQ3-B保存上界を保持し、新nで0にしない。n340専用の旧`decode_counts`を改変せず、新CI inverseを専用adapterで接続する。energy中心κ／丸めchargeを含む幅を評価し、Q3-Bの旧Hoeffding inverse専用ωQ0を二項root反転の誤差証明へ無条件流用しない。新inverse自身の外向きenclosureを使う。

可行Q-Eの最小nと下界が一致しなければgapを報告する。Q-Eが得られなければ既存Q-H340を有効なfallbackとして残す。CPが常に全countでHoeffdingより狭い、又はnのPASSが単調であるとは仮定しない。

## 5. C：保証成立後だけの保存価格接続・判定

H/Eの有効なpolicy／下界を対象に、新しい回路価格取得をせず保存COSTのµX/µY/µQとprefix費を再加重する**限定会計1回**。RZ/CX/size/shot加重depthを別々に扱い、peak qubitsは別欄。強baselineのn・αの可行解と緩和下界を同じ価格として扱わない。shot下界からは、例えば`保存prefix下界＋N_lower×min(µX下界,µY下界)`という安全側費用下界を使えるが、それを実行可能価格や最適価格とは呼ばない。N最適policyが費用最適とはしない。既に証明できた複数配分は同じ価格機会で比較し、この粗い下界と可行価格で判別できなければ費用gapを残す。新しい価格目的の探索へ自動拡張しない。

history依存policyに対する主出力は、履歴の分布を仮定しないconditional worst-case native費用境界。期待費用を報告する場合はhistoryの確率分布と全nuisanceモデルが別途必要で、この初手では行わない。追加calibrationを必要とする候補は今回実行しない。古典optimizerの所要時間／memoryとonline lookup／feedback費はnative gate costへ混ぜない。

µ自身は保存lawのfresh trajectoryに関する期待資源の境界であり、その平均は既存価格SOURCEに既に含まれる。上のworst-caseはprefix履歴に関するもの。trajectoryを含むpathwise最大費用や実機時間を保証したとはしない。

| 得られた証拠 | 許される結論 |
|---|---|
| Qの資格済み費用上界 < 指定baseline familyの資格済み最小費用下界 | **A：そのfamily／robust taskでQ優位**。世界最良推定器への優越ではない |
| 共通測定family・同じ保証付きpolicyに還元 | **B：原理／policy同等**。追加の実装保証・資源評価として位置付ける |
| baselineの資格済み可行費用上界 < Qの対応費用下界 | **C：そのbaselineが有利**。Qの優越を主張しない |
| 可行解しかなく最適性gapが残る／価格境界が重なる | **最適性又は費用優劣未確定**。小さい点推定だけでA/B/Cとしない |
| shared nuisance、native、数値coverage等が閉じない | **D：UNQUALIFIED**。不足した前提・seamを特定する |

H/E結果の大小だけから新規性や主要claimの正式採否を決めない。全方向・全CI・sequential policyの情報論的下界を得たとはしない。

## 6. 保存・STOP・実行しないもの

新SOURCE／入力blob／型・数値規約／scope／lookup／witness／結果／checks／STOPは新しいQ4数値領域へ保存し、旧ファイルを変更しない。保存結果checkerはcampaignを再呼出しせず、tail/root包含、joint二点の同時実現性、全域coverage、成功確率和、費用の同端点相関、最適性gapを検査する。

独立arm・割当・count boxは利用可能資源に応じて並列化し、既存CI tableと幾何cacheを共有する。CPU/RAM/workerへの独自恒常上限を設けず、共有設定・他process・fileには触れない。ここで固定した候補数・数値分解能は科学scopeであり資源制限ではない。

固定SOURCE/input不一致、geometry/coverageのcounterexample、誤差二重計上、未資格runtime／nativeを要する場合、旧one-shot/STOPへ抵触する場合は停止。判定不能をPASSにせず、別CI・n範囲・precision・tolerance・center domainへ事後変更しない。Hのoptimizer gapだけなら未確定として保存し、独立したEを同じ承認範囲内で完了してよい。共通モデルの保証破綻なら後続Cは実施しない。

H/E/Cの承認範囲完了・公開・remote照合後mandatory STOP。EのN>340、Bayesian particle simulation、逐次方向変更、adaptive停止、time-adaptive、別PF/law、Full/Truncated新比較、回路build/compile/sampling、CT合成／価格／総Tへ進まない。

**次担当：GPT／ユーザー**がこのrobust比較domainとH/E/C bundleを採否判断し、必要な新科学数値診断を明示承認する。**Codex**はその承認後、必要最小限の実装・人工検証・SOURCE固定と計画内の科学callを実施する。現在Q4-0は静的調査だけでmandatory STOPする。
