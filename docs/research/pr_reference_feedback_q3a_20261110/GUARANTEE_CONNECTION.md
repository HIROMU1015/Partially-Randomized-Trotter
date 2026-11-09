# Q3-A：prefix・feedback・Q0十分条件の保証接続

2026-11-10 JST。基準commit `b6b74a3f0a73bb5984d8ee978705eb0e498b86c2`。本書は固定SOURCEの静的解析と条件付き数学的論証であり、新しい科学保証の正式採用・独立科学レビューではない。[REPORT](REPORT.md)／[次の限定検証案](NEXT_VALIDATION_PLAN.md)。Q0／Q1／Q2のSOURCE・結果・STOPは変更しない。

## 1. 対象と成功event

Linear H4／1.00 Å／STO-3G／DF12、元H exact ground供給・準備費除外、Partial S2／LD3／h=.02／K2、canonical independent finite bits、既存18round schedule。m17はq131072/r32/bits88。精度1.5936e−4 Ha、failure .05、phase cap c=2/5。今回変更・再計算していない。

`pr_reference_bprime_accounting.error_and_shots`は各roundについて、保存raw radius lower ρm>0、残余統計角βm>0を外向き端点で保存し、

`εm = ρm sin βm / √2,  Nm ≥ 2 log(1440)/εm²`

という座標Hoeffding十分条件を使用する。Referenceは保存`STAGE4_RPE`のprefix、m17はV6-Pを経たQ0のReference-B1条件を用いる。新しいshot配分を作らない。

前段成功event Sは、m0〜16の各X/Y座標で標本平均と当該physical meanの差がεm以内となる34個のeventの共通部分。各失敗予算は1/720。独立fresh trialと旧SOURCEのPF・law・native disk条件を前提に、`P(Sᶜ)≤34/720`。ここで平均は無重みcanonicalの複素physical信号を表す。weighted Closed P3やchannel平均の一致へ読み替えない。

S上では、測定vectorと保存meanとのL2差≤√2εm≤ρm sinβm<ρm。従って測定vectorは非ゼロであり、その位相誤差はβm以内。既存のsystematic位相charge Bmと合わせて、

`d_circle(arg zobs,m, qm θ) ≤ Bm+βm ≤ c,  θ=−h E0`

となる。arg(0)に便宜的な0を与えて成功とする定義は使わない。真のSが起きたかをcountから観測できる、という主張でもない。

## 2. 数学的branchの一意性：証明

qm=2qm−1。直前中心の誤差≤c/qm−1、今回の測定argの誤差≤cなら、正しい候補と直前中心の円周距離は≤3c/qm。別候補は2π/qm以上離れているため、その距離は≥(2π−3c)/qm。

従って正しい候補と他候補の距離gapは

`γm ≥ (2π−6c)/qm > 0`。

これはc<π/3から従う。qm≤65536、π>3を用いれば**全prefixでγm>2⁻¹⁵ rad**という粗い数学的下界も得られる。新しい科学データの評価値ではない。m0を起点として帰納すれば、正しいnearest branchの誤差≤c/qmが続く。数学的tieはS上で起きない。

固定alias証拠はf2638a78の`A_guarantee/verified/GROUND_ALIAS.json`。保存されたGershgorin spectral upper `3.0552911673129084` は4より小さく、保存hのbinary64値は1/32より小さい。従って真の基準位相は`|θ|<1/8`。E0のRayleigh値をdecoderやfeedbackに与えていない。

正しいm16中心は`|θhat16|<1/8+c/65536<1/4`。一方Q1のalias guardは`π−c+c/65536>2`。従ってこの固定系では、正しい中心に十分狭い包含区間が付けばalias拒否は不要となるほど余裕がある。一般的な「±πからc以上離れる」という等号付き前提だけではguardのstrict判定まで導けないが、固定SOURCEのより強いalias証拠でその問題を避けられる。alias条件そのものは緩和していない。

## 3. Q1がそのbranchを受理する条件

[Q1実装](../../../src/trottertracks/pr_reference_feedback_q1.py)は、exact rational count→192-bit atan2区間を作る一方、branch integerは旧`reconstruct_rpe_phase`から得る。旧実装はbinary64の`math.atan2`、全候補のwrap、`numpy.exp/angle`、`argmin`を使う。Q1はその候補をintervalへ復元し、両隣とのstrictな距離比較を行う。

次の区別が必要である。

1. **数学的branch一意性：** §2でS上の全対象について論証できた。
2. **旧binary選択の一致：** 各候補の計算距離と対応する数学的距離の誤差をηmで囲み、`2ηm<γm`ならrankingは変わらない。これは候補生成・前段中心・atan2・exp/angleの誤差を含む条件であり、binary64という型だけからηmを0としない。
3. **区間検査の受理：** 正しい候補・隣接候補の距離包含の端点slackをνmで囲み、`2νm<γm`ならstrict比較は必ず通る。区間の包含が正しいことだけでは幅の上界は導けない。

固定aliasでは、正しい中心の包含上端が絶対値1未満に収まるという粗い追加条件でもalias guardを通る。S上で正しい候補と旧canonical代表が十分近ければ、`_period_near`の整数periodも同じlinear energy branchを選ぶ。保存N・h・armと正しくbindingされたhistoryであることも必要で、異なるscheduleの拒否は数値失敗と区別する。

**総合判定は`CONDITIONAL_ACCEPTANCE`。** 上の数値誤差・区間幅条件の下ではS上の受理を論証できる。しかし現固定SOURCEに、全対象count／全旧binary距離演算を覆うηm・νmの一様certificateはない。今回それを新しく数値評価していない。`PROVED_ACCEPTANCE`への格上げは行わない。

人工fault injectionでは、良いcount履歴でも旧branchを誤らせる、または真のargを含むだけの広い区間に置き換えると、Q1が拒否することを確認した。これは必要な数値契約を示す反例であり、**実際の固定mpmath／NumPyでS上の拒否を発見したという`COUNTEREXAMPLE`ではない**。そのような実反例は今回確認していない。不要な拒否が固定SOURCE上で実証されたわけではないため、Q1の受理条件・成功event・precisionは変更しなかった。

## 4. Feedback算術：閉じる部分と残る条件

最後の数学的count argをp16とする。正しいbranch中心について`q16 θhat16=p16+2πj`なので、

`φ*=−t17 Ec+π/2 = 2p16+π/2 (mod 2π)`。

旧decoderの出力する基準位相を単に2倍する式ではない。Q1はこの小さい角の式を使い、丸めたEcを大きいt17で再乗算しない。

Count平均は`Fraction(2k,n)−1`でexact。Q1は旧Q0のinterval constructorを再利用し、atan2、π、2倍、加算、period減算の包含Jを保持する。既存のdirected-interval算術モデルでJが真の代表を含むなら、

`δarith = upper(abs(I(φbin)−J))`

という戻り値は、**その呼出しが実際に返したbinary64角度**の誤差上界になる。period整数がcutの反対側でもwhole 2πだけ代表が変わるため、Pの同じ周期類であり、差を落として過小評価していない。

ただし、このper-call包含と、**全対象countでδarithがQ0 cap以内**であることは別。例えばJの幅≤2⁻¹⁶⁰、代表とmidpoint算術が通常のbinary64／192-bit丸めを満たすなら、`|φbin−φ*|≤2⁻⁵²+width(J)+O(2⁻¹⁹⁰)<2⁻⁵⁰`という十分条件になる。これは幅条件付きの丸め補題であり、保存科学count全体について幅条件を計算した結果ではない。

mpmath1.3.0の`mpi_atan2`は象限の端点を選び、`mpf_atan2`へfloor／ceilingを渡す実装である。Q0と同じintervalモデルを保つ。これを新しい形式検証済み算術libraryと主張せず、全countへの包含・幅の移送条件を残す。人工oracleは384-bitの点参照を使ったが、productionのmp／iv precisionは192-bitのまま変更していない。

## 5. Parameter・loweringと追加誤差の単位

有限binary64の符号反転はexact。Qiskit1.3の`Gate.validate_parameter`はPython floatをそのまま返すため、Q1の`P(−φbin)`へのparameter受渡しでは別の角度丸めを導入しない（このSOURCE経路・型が前提）。

Qiskit1.3の等価libraryには全実λについて

`P(λ) → U1(λ) → exp(iλ/2) RZ(λ)`

が登録されている。従って**この経路で同じλがnative RZへ渡り、既存core loweringと誤差chargeが保持される場合**、追加の純粋角度誤差は0でよい。λ/2の共通global phaseはwhole joint信号に影響しない。lower後に新controlを付ける場合の保証ではない。

閉じたのはsymbolic identityとparameterの部分である。実`BasisTranslator.run`はaccelerated `base_run`へ渡すため、等価経路の存在だけから、全m17 context・全feedback角でその経路と既存core保証が維持されることまでは認定しない。Compiler AにRZの数値fusion／角度cutoff passがないことと、Q2の16 native尾部で同じRZが残ったことは確認済みだが、一様context保証とは区別する。

さらに、Q0へ純粋なeφとして渡す角度差は、historyで条件付けた後、trajectoryに依存しない共通readout軸の差である必要がある。各trajectoryのcompile誤差が小さいという上界だけでは、誤差がtrajectoryの複素信号と相関し得る。平均を一つの角だけ回転させたモデルへ自動的には還元できず、その場合は追加raw signal diskとして扱う。共通parameterを全shotへ渡すSOURCE上の条件と、この一般摂動を区別する。

量を次のように分ける。

- prefixの統計的不確かさ：残差`u=−t17(E−Ec)∈[−.8,.8]`へ既に入る。
- 旧PF・SOURCE・finite-bit・native disk：保存B・ρLへ既に反映。再加算しない。
- 新しい純粋な角度差：`δarith+δparameter/lowering`としてradで加算可能。Q0の共通十分条件は≤`.000748774257`。
- 一般joint operator誤差η：期待値の摂動は一般に≤2ηなどのsignal単位になる。純粋な回転とは限らず、radへ直接足さない。旧diskと同じ単位で合算してradius／phaseを再接続するか、追加mean誤差として統計集合を再資格化する必要がある。
- 将来のClifford+T・hardware誤差：今回未評価。0として採用しない。

従って**全feedback実装誤差を共通cap内へ保証したという判定は未資格**。Q1の追加誤差`None`での拒否は、現状では必要なfail-closed動作である。

## 6. Confidence hullと有限precisionの残余

正しいprefix中心をEc、理想coherenceを`ρ exp(−it17 E+ib)`とすれば、readoutは`ρ sin(u+b+eφ)`となる。実軸角の差をΔφ=φbin−φ*と定義するとeφ=−Δφ。Q0は絶対値boundを使うため符号の取り違えは生じない。

未知ρ∈[ρL,1]、|b|≤B、|eφ|≤δを保ち、fresh固定340 trialsのcount kから既存`Q0.hull`をそのまま使う。空集合を成功にせず、非単調枝を捨てず、n・モデルを事後変更しない。Q0の全count十分certificate自体は再計算していない。

追加で確認すべき実装残余は、Q0の数学的な点中心Ecに対して、Q1がEcの包含区間Cを持つ点である。`width(C)=κ`、その他の外向き丸め余裕をωEとすれば、新energy hullの径には最大

`WQ0/t17 + κ + ωE`

が必要。Q0の`WQ0≤.8`だけでは、Q1のstrictな`.8/t17`判定へ移す残余余裕まで自動的には閉じない。

この確認は旧全count反転を繰り返さずに準備できる。Q0に保存されたδlo（下側PASS bracket）と共通δsafeについて、`σ=δlo−δsafe>0`の余裕を使う案がある。単調枝では各countのhull径はD=B+δの関数として、0・1・2の傾きを持つclipped affine関数。0より大きく全prior径2U未満なら傾き≥1。従ってδloで径≤.8が証明済みなら、δsafeでの真の径は≤`max(0,.8−σ)`。このため十分な移送条件は、追加のhull数値slackも含め`κ t17+ωE t17+ωQ0≤σ`となる。

今回はσや新κ・ωを保存科学値から数値評価していない。既存の安全側capを変更したり、小さい余裕を0と見なしたりせず、次の最小診断候補とする。

## 7. 全体成功確率：条件付き接続定理

次の全前提が満たされる場合、固定340 shotsの十分結果を接続できる。

1. 旧prefix十分条件・law・独立性が適用され、S上で正しいbranchを必ず受理する。
2. S上の全feedbackについて、算術・parameter・loweringの追加誤差がcap内、追加の非角度signal誤差が未処理で残らない。
3. 過去historyで条件付けたm17のmeanが、同じB・ρLによるQ0モデルを満たす。角度選択が過去だけに依存し、fresh bits／shotsを再利用しない。
4. Q0の全count hull保証と、§6のenergy-hull数値余裕が適用できる。

各受理historyでHoeffding失敗≤1/360なので、条件付き確率を平均してもこの上界は保たれる。従ってunion boundは`34/720+1/360=1/20`。独立した「受理失敗」を無料で削る定理ではない。受理／実装に新たな失敗確率αextraが残るなら上界は`.05+αextra`となり、既存予算をそのまま満たさない。受理後だけのpost-selectionで隠してはならない。新failure配分を今回は決定しない。

最終hullの径≤`.8/t17`ならexact rational中点の誤差≤`.4/t17`。保存binary hは1/51より大きく、`.4·51 < 131072·(1.5936e−4)`という定数不等式により、この誤差は目標energy精度内となる。SOURCEの既存時刻誤差chargeは保持する。この定理は条件の論理接続であり、現在の実m17へ前提が全て資格化されたとの認定ではない。

## 8. 費用接続の境界

旧m17価格の入口は、f2638a78のB′`TOTALS_18ROUND`／`pr_reference_bprime_accounting.price_bounds`、Reference `pr_reference_transfer_accounting.expected_rz_bounds`／`pr_reference_transfer_cost_bounds.expected`、depthの`pr_reference_depth_v5.round_bounds`。V6-Pの保存`P_run01`（5f672ce6）と、V6-Q producer／schedule／consumer（1cd1fd29）のphase生成・reset条件を別にbindingする必要がある。旧X/Y consumerへaxis名だけQを渡す変更はしない。

Q2のq1/r1・4角の価格は、このm17接口と全lawへの移送certificateではない。古いaxis別wrapperの相殺、共通coreのlowering、readout RZの保持、phase-providerの生成本数／domain、全supportの誤差と期待単価が必要。今回は価格関数を呼んでいない。

資源jごとの将来条件は`340 μQ,j+Aj < 626 μX,j+626 μY,j`。prefix費を固定すれば`G18,j=Gprefix,j+G17,j`と記号的に接続する。AjもμQ,jも0と置かず、RZ/CX/depth/qubits/classical/Tを合算しない。両armに同じQ機会を与える。
