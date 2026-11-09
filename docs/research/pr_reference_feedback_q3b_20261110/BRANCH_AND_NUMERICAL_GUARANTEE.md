# Q3-B：区間branchと数値余裕の保証

2026-11-10 JST。基準はQ3-A `c624114d99056373c7cbf5e933fb8ee786259c42`。[REPORT](REPORT.md)／[次段階](NEXT_VALIDATION_PLAN.md)。対象は固定H4・既存prefix成功event・既存192-bit directed-intervalモデル。library／CPUの形式検証や実機誤差の新しいcertificateではない。研究上の正式な採用はGPT／ユーザーの判断に戻す。

## 1. 再利用・変更境界

新しいopt-in [pr_reference_feedback_q3b.py](../../../src/trottertracks/pr_reference_feedback_q3b.py)だけを追加した。Q1のexact count型、`_count_phase`、Q0区間constructor、Prefix型、feedback／readout／confidence decoderを再利用する。Q1／旧RPE decoderは不変更。旧float decoderはinterval選択後のdiagnosticであり、その失敗・値を保証の根拠にしない。

実科学利用では`saved_arm='Bprime'`または`'Reference_B1'`による保存Nへのbindingが必要。省略した人工／外部historyを、旧RPEの確率保証に自動的にbindingしない。保存Q0モデルを使うQ1 decoderも同じarmのbindingを要求する。

保存Nsはf2638a7845685a119619382b850441f5d8d33713のB′ `ERROR_RPE.json`／Reference `STAGE4_RPE.json`、m0〜16のまま。両armとも

`192,192,193,193,193,194,196,199,199,206,206,206,221,222,256,258,347`。

物理条件・h・law・失敗配分・phase capは変更しない。新しい数値guardは、既存192-bit演算の十分余裕としてB1実行前に固定したもので、科学的なprecision候補探索ではない。

## 2. 数学的規則とm0開始

c=2/5、qm=2^m。旧座標Hoeffding成功event Sでは測定vectorは非ゼロ、各roundのcount arg pmは真のqmθから円周距離c以内である。これは旧ρm>0・座標誤差≤ρm sinβm/√2・βm<cから従う。[Q3-Aの論証][A]を再利用する。

固定Gershgorin alias証拠から`|θ|<1/8`。従ってm0の正しいpmは`|p0|<1/8+c<1`で、±πのcutをまたがない。m0はQ1のcanonical arg区間そのものを中心C0とし、branch integerを0とする。

帰納的に直前の正しい数学的中心θhatが誤差≤c/qm−1を持てば、

`R=(qm θhat−pm)/(2π),  |R−j*|≤3c/(2π)<1/5`。

従って正しい整数cellには半整数境界から3/10より大きい余裕がある。正しいliftの中心誤差は再び≤c/qmとなる。これは既存nearest-branch規則と同じ選択であり、新しい位相推定器ではない。

## 3. 実装の整数cell判定

実装はRのintervalを計算し、その端点をexact Fractionとして読む。`floor(lo+1/2)`と`floor(hi+1/2)`が同じ整数jで、さらに**両端がstrictに(j−1/2,j+1/2)内**にある場合だけ選ぶ。float floor、argmin、任意tie-breakを使わない。曖昧・非finiteは拒否する。

次の中心は`C=(pm+2πj)/qm`。signedなunwrapped jを保持し、branch_mod_qもdiagnosticに保存する。各roundでmodを掛け直さないため、±π付近のpmも同じlinear base branchへliftできる。既存`_period_near`の点による代表選択はprefix rankingには使わず、Q1 feedbackの周期代表を選ぶ箇所にだけ残る。

S上では`|j|≤qm`。固定aliasから最終数学的中心は絶対値1/4未満であり、実装の粗い`abs(C16).upper<1` guardを通る。これは固定H4の既存alias証拠を利用する条件で、旧一般RPEのaliasを変更するものではない。

## 4. 192-bitの一様受理条件

事前固定guardは

- arg区間幅 Wp≤2⁻¹⁶⁰ rad。
- base中心区間幅 Wc≤2⁻¹⁵⁰ rad。
- energy中心区間幅 κ≤2⁻¹⁴⁰ Ha。
- basic interval演算の粗いslackは、phase chainのAθ=2⁻¹⁶⁰ radとcell chainのAR=2⁻¹⁶⁰（dimensionless）。SOURCEの`BASIC_SLACK`はこの同じ数値上界を別々のchainへ使用する。

directed roundingの単位をu=2⁻¹⁹¹とする。成功event上のpm、C、jとqm≤65536では、branch演算の中間量の絶対値は2²²未満。64 operation未満の端点chainの丸めは粗くても数値上`64·2²²u=2⁻¹⁶³<2⁻¹⁶⁰`。既存Fraction→interval変換はrelative boundとして先に処理する。qmのpower-of-two scalingはexactである。

π区間幅をWπとすると、各m≥1で

`Wc,m ≤ Wp,m/qm + 2Wπ + Aθ`

となる。jの整数値は既に一意なので、前段の区間幅をCへ加算し続ける必要はない。一方、cell判定に使うRの幅は

`WR,m ≤ (qm·2⁻¹⁵⁰ + Wp,m)/(6 rad) + AR < 2⁻¹³⁰`。

分母2π rad>6 radと、その微小な区間幅による寄与もARに含める。RとWRはdimensionless。正しいRから半整数境界までの余裕>3/10に対し、この幅は十分小さく、R包含の両端は必ず同じstrict cell内になる。従ってS上の正しいintegerを拒否しない。

既存binary hはexact rationalで渡す。Ec=−C/hの区間幅には、上式Wc/hと、除算の端点丸め≤2⁻¹⁷⁰ Haを加える。m16のκはB1でその値を数値化し、2⁻¹⁴⁰より小さいことを確認した。最終energy hullの復元には別の丸めchargeを使い、center除算と同じ操作を二度数えたわけではない。

以上の帰納とB1のWp coverageにより、判定は**`PROVED_ACCEPTANCE`（固定成功event・192-bit区間モデル内）**。S外のcountや不一致scheduleまで成功とするものではない。

## 5. B1の全域coverageと包含の意味

監査は11種類の保存Nについて、全`0≤kX,kY≤N`を扱った。合計587,815位置、うち非ゼロ587,808、ゼロ7。両arm・全17roundへ対応させると1,652,068位置、非ゼロ1,652,048、ゼロ20。これはcount位置のcoverageであり、生成した実測shot数ではない。

同じNではQ1のcount→arg処理がarm・mに依存しないため共有する。exact Fraction平均を192-bit区間へ入れ、同じ方向の整数座標`(2kX−N,2kY−N)`から得たarg区間との整合を各非ゼロ位置で確認した。負の実軸上の−π規約も維持する。

包含は**旧directed-interval primitive契約とexact入力からの構成**による。整数座標とのoverlapだけを独立したlibmpの厳密証明とは呼ばない。全件の幅・period・feedback端点誤差を有限監査し、単なるgrid sampleにはしていない。

最大Wpはexactに2⁻¹⁸⁹ rad（約1.27448e−57）。m16のN347に対する121,104位置すべてでQ1 `_feedback_phase`を評価し、最大δarithは約2.22034e−16 rad。各戻り値が、実binary parameterと理想代表intervalの全端点の差を囲むことをexact Fractionで検算した。代表periodは固定の有限範囲内で、±πで誤差を捨てていない。

結果は[B1.json](../../../artifacts/pr_reference_feedback_q3b/2026-11-10/run01/B1.json)、全2,495 rowのstatus範囲・極値・witnessは[B1_ROWS.jsonl.xz](../../../artifacts/pr_reference_feedback_q3b/2026-11-10/run01/B1_ROWS.jsonl.xz)。各rowは全kYのPASSとゼロ位置を範囲で表し、case stream hashを持つ。各位置の全angle表を重複保存したものではなく、全域判定に必要な極値と完全coverageを保存している。

## 6. B2：新しいhull丸めchargeの導出

Q0の旧全n探索・反転・二分探索は呼ばない。保存下側PASS δloと共通δsafeだけを使う。旧radius・bias・PF／finite-bit／native diskは変更・二重加算しない。

逆正弦の±1付近ではLipschitz定数が発散するため、「全てO(u)」とは置かない。固定入力ではρL≥1/2、B∈[0,1/4]、n340、|c|≤1、ε<1。regularな入力処理では、例えばcの誤差<4u、epsilonとその入力chain<64u、radius除算と符号caseを含めたclip前の誤差<512uという粗いboundが使える。α=1/360、2/α=720、logの引数が0へ近付かず、radius除算の増幅≤2であることを用いる。

SOURCEではこれより大きい一様入力slack

`e0=64·2¹⁶u=2⁻¹⁶⁹`

を採った。clipは非拡大。`|sqrt(a)−sqrt(b)|≤sqrt(|a−b|)`と、clip後のasinのHölder bound `|asin(x)−asin(y)|≤4sqrt(|x−y|)`を使う。asinを計算する`atan2(v,sqrt(1−v²))`の理想vectorのnormは1で、端点丸め後も1/2より大きい。従ってatan2のregularな角度誤差も囲める。±1のspecial branchは保存π区間を使う。

`s0=sqrt(2e0)=2⁻⁸⁴`とし、両endpoint・min/max・phase径までの粗い合成slackを`64(s0+e0)`で囲む。これは事前に固定した

`ωQ0=2⁻⁶⁴ rad`

より小さい。ここでも正しくdirectedに丸める192-bit elementary primitiveという既存モデルを使用する。新しいlibraryの形式検証ではない。

energy復元は受理された|C16|<1、h>1/64より|Ec|<64、t17>2048。phase endpointも有限で、energyの減算・除算・径には16 operation未満、量の絶対値は2⁸未満という粗いdomainを使える。`16·2⁸u=2⁻¹⁷⁹`より大きい

`ωE=2⁻¹⁷⁰ Ha`

を事前に固定した。hと2^17hはbinary rationalのpower-of-two scalingなので時刻自体の新しい幅は0だが、energy計算の丸めは上記の正の量として残す。

## 7. 保存余裕との接続

[Q3-Aのclipped-affine幅補題][A]を使い、保存δloで全count径≤.8なら、δsafeにおける真の径には少なくともδlo−δsafeの余裕がある（空集合／幅0の扱いを含む）。その余裕へ新κとωをchargeする。

両armでm16の一様κ boundは約3.42121e−47 Ha。判定はexact rationalで行い、

`κ t17 + ωE t17 + ωQ0 ≤ δlo−δsafe`

がPASSした。lhsは約5.42102e−20 rad。保存余裕はB′約1.26151e−7 rad、Reference-B1約8.61444e−13 rad。完全な有理数と差は[B2.json](../../../artifacts/pr_reference_feedback_q3b/2026-11-10/run01/B2.json)。丸めた表示値を判定に使っていない。

従って**算術・区間中心・hull丸めの不足は、固定モデル内では解消**した。新しいfailure budgetは必要なく、全prefix S上での受理を無料のpost-selectionとして扱っていない。

## 8. Nativeと最終成功保証の残余

Q1の生成する共通feedback objectをhistoryごとに一度生成し、同じbinary parameterを全trajectoryで使う条件は変わらない。B2後、科学入力を使わない1qubitの自由symbol φに対するP(−φ)→H→MeasureをCompiler Aで1回lowerした。φ依存のnative instructionはRZ(−φ)のみで、global phaseのφ係数は−1/2だった。[POST_B2_TECHNICAL.json](../../../artifacts/pr_reference_feedback_q3b/2026-11-10/run01/POST_B2_TECHNICAL.json)。これは孤立symbolic readoutの条件を強める証拠であり、H4全contextの資格ではない。

残るのは、m17の全core／trajectoryで同じparameter経路と既存body誤差chargeが保持され、trajectory依存の追加signal誤差が未処理で残らないこと。一般operator／signal diskを単一のrad誤差へ読み替えない。Q2の16回路・q1価格や孤立symbolic fixtureだけでは、この条件は閉じない。

そのnative条件も満たされれば、fresh independent m17 trialsに条件付けたHoeffdingと旧Q0 certificateを使い、`34/720+1/360=.05`と最終energy精度へ接続できる。現時点では実装済みm17の340 shotsを無条件に採用せず、全18round総費用・Tも未評価のまま。研究上の正式な保証採択はGPT／ユーザーへ戻す。

[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c624114d99056373c7cbf5e933fb8ee786259c42/docs/research/pr_reference_feedback_q3a_20261110/GUARANTEE_CONNECTION.md
