# Q1：feedback readoutとcount decoderの接続条件

2026-11-10 JST。実装は[pr_reference_feedback_q1.py](../../../src/trottertracks/pr_reference_feedback_q1.py)。[REPORT](REPORT.md)／[Q2案](NEXT_VALIDATION_PLAN.md)。本書はSOURCE対応と条件付きの数理整理であり、実H4の新しい成功保証の正式採用ではない。

## 1. 入力と旧SOURCEの保全

Q1はopt-inの追加module。既存X/Y wrapper、V6-Q caller／consumer／producer、B′、F/G、actual inverse、vacuum phase、sampling law、Compiler Aを変更しない。旧`consume_canonical_axis`のaxisをQへ偽装せず、既存builderが返す**未測定core**に新wrapperを付ける接口とした。実H4 coreの生成は今回行わない。

主なAPIは次の順序で使う。実科学データで呼ぶにはその範囲の実行承認が必要で、APIの存在は承認を意味しない。

1. `prefix_from_counts(history, saved_arm=...)`：m0〜16のX/Y countから旧branch decoderへの対応を取る。
2. `feedback_from_prefix(prefix)`：同じprefixから角度と算術誤差包含を生成。
3. `feedback_readout(core, feedback, family=...)`：B′／Reference-B1へ同じ機構を適用。
4. `decode_counts(prefix, feedback, plus, model, additional_feedback_error=...)`：固定340回のcountをQ0の集合へ反転。

`AxisCounts(n,plus)`のplusは**測定bit0＝符号+1**の件数。nとplusは整数、0≤plus≤n。`RoundCounts`はm0〜16を重複・欠損なく順に持つ。未来のcount、E0、gap、ground state、未知の真signalをfeedback入力にしない。

## 2. Prefixの中心とbranch

旧`reconstruct_rpe_phase`の`final_phase_estimate`は基準時刻の位相θ̂であり、m16の累積位相ψ̂16ではない。旧SOURCEをASTからそのまま再利用し、近傍branchを選ぶ規則を再実装していない。floatの点推定を、そのまま保証付き区間へ格上げもしない。

各roundのcount平均`x=2kX/nX−1`、`y=2kY/nY−1`をexact rationalとして扱い、192-bit区間atan2で**数学的なarg(x+iy)**を囲む。ゼロvectorは拒否し、負の実軸上では旧[-π,π)規約の−πを使う。旧decoderが選んだbranch integerを用い、periodの整数差だけを補正して、同じ候補の正確な位相区間を復元する。

さらに、その候補が前段から見た最近傍であることを、隣接する二候補との円周距離の区間で確認する。格子は等間隔なので両隣の確認で足りる。tie・区間の重なり・不一致では`UnqualifiedPrefix`を返し、別branchを都合よく選ばない。旧float出力はdiagnosticとして残すが、energy中心の定義には丸めたfloatを使わない。

数学的な接続は次の条件付きである。

- 保存されたprefix成功event：m0〜16の各測定位相と元H energyの位相の誤差がcap=.4以内。
- fresh independent trial、元のPF／finite-bit／native誤差、十分Nが既存SOURCEどおりである。
- 旧alias promise：真の基準位相が±πから少なくともcapだけ離れている。
- 新しい区間検査が旧branchと数学的nearest branchの一致を確認する。

cap<π/3の既存RPE規則では、正しい候補と直前中心の距離≤3cap/q、隣の候補との距離≥(2π−3cap)/qである。従って成功event上の数学的branchは一意で、m16の中心誤差≤cap/q16。alias条件により円周上の結果を同じlinear energy branchへ移せる。

実装は`Ec=−θ̂/h`を区間で保持し、`w=cap/(q16 h)`のprefix区間を返す。hは旧energy変換のbinary64 literal `.02`のexact rational imageで、旧SOURCEの時刻・算術誤差chargeを変更しない。`t17/t16=2`なのでQ0の残差半幅U=.8はこの表現差に依存しない。

`saved_arm`を指定すると、旧固定commit f2638a78のB′ `ERROR_RPE.json`またはReference `STAGE4_RPE.json`から**保存Nだけ**を読んで実count件数を照合する。N・shot十分式を再計算しない。省略した人工／外部prefixは未bindingと表示する。保存Q0モデルを使う場合は対応するarmのbindingを必要とする。

**残る未資格：** countから真の成功eventが起きたかは観測できない。元SOURCE・lawの仮定、実測系の適合、および区間検査が成功event上の全実行で受理するというnumerical acceptance coverageは、今回の人工例だけで一様に資格化していない。従ってprefix出力のstatusは条件付きであり、Q1だけで全成功確率≥.95と認定しない。alias端・ゼロvector・branch tieの拒否を無償の成功や無視できる事象としない。

## 3. Feedback角と算術誤差

理想中心について`ψ̂16=q16 θ̂=−t16 Ec`。最後のcount argをp16とすると、ψ̂16=p16+2πjであるため、

`φ = 2ψ̂16 + π/2 = 2p16 + π/2 (mod 2π)`。

新adapterはこの**branch不変な小さい角**を使う。丸めたEcや小さいbase phaseを2倍する方法、あるいは丸めたenergyを大きいt17で再乗算する方法を使わない。branchはenergy区間には必要だが、feedbackのmod 2πには整数倍として消える。

区間atan2、π、2倍、加算、period分の減算を含めた理想角の区間Jφを保持する。代表を選ぶ整数がcutの反対側になっても、差はwhole 2πであり、Pゲートの同じ周期類となる。±π付近を広い一つの区間に潰して誤差を過小評価しない。

Qiskitへ渡すfinite binary64値φbinを生成した後、`sup |φbin−Jφ|`の上端をexact rationalで保存する。πやatan2の区間幅も含まれ、未評価算術を0としない。round-to-nearestの精度を仮定するだけでなく、**実際に返したbinary値との差**を囲む方式である。192-bit contextが変更された場合は拒否する。旧mpmathのglobal contextを使うため、同一processでの外部からのprecision変更を許容する接口ではない。

この算術包含を実H4のm16 countへ適用する新しい数値評価は今回行っていない。実装が返す値でQ0上限を満たすかは、将来の対象データと実行範囲で確認する。

## 4. Readout observableとcore

旧と同じancilla-last、target-firstの配置。新wrapperは`H → core → P(−φbin) → H → Measure`。よって測るobservableは`Xφ=cosφ X+sinφ Y`、coherence zに対して`Re(e^(−iφ)z)`である。φ=0はX、π/2はY、−π/2は−Y。初期H、coreのglobal／relative phase、scalar／vacuum補正を保持する。

coreと元のregister幅は不変更、出力clbitは従来同様1個、measurementは最後。Referenceの初期resetはwrapperが追加しない。既存consumerの**compile後にresetをprefixするが、物理実行はresetから始める**規約をそのまま使える。Q1ではその順序を既存関数＋mock compilerで検査した。本物のcompilerは呼んでいない。

人工2-system-mode fixtureでは、旧B′ controlled-RZZ kernel、旧Reference core／actual inverse／vacuum位相関数を用い、非自明なGaussian frameと符号付き対角wordを共用した。測定後の誘導observableを全4次元system入力について行列で比較した。phase fitは使わず、core全体のcommon global phaseと、ancilla branchのrelative phaseを区別した。実H4 coreの資格へ拡張しない。

## 5. Q0のcount反転と出力

保存モデルはcanonicalの無重みphysical meanに限定。ρ∈[ρL,1]、|b|≤B、B=cap−保存βである。旧diskは既にbとρLへ反映されており、もう一度加算しない。weighted Closed P3やchannel平均には流用しない。

固定n=340、c=2k/n−1、ε=sqrt(2 log720/n)。feedbackの追加角度誤差δextraと算術包含δarithを**異なるものとして**保持し、δ=δarith+δextraを使う。追加誤差が未評価なら`UNQUALIFIED_FEEDBACK_ERROR`。明示された0は人工ideal-provider条件としてのみ今回のtestで用いた。

保存モデルの共通安全側条件はδ≤0.000748774257 rad。これはモデルへの条件であり実providerのcertificateではない。一般のoperator/raw signal diskをradのδextraへそのまま代入してはならない。その場合は別の誤差接続とQ0モデルの再資格が必要である。

`Q0.hull`をそのまま用いて、u∈[−.8,.8]の可行集合を囲む。単調枝の前提を満たさない場合、枝を削除せず未資格で終了する。空集合は`INCONSISTENT_COUNTS`、幅不足は`INSUFFICIENT_OR_UNRESOLVED_WIDTH`。nやρ・bをcountを見て変更しない。

energyは`E=Ec−u/t17`へ戻し、理想中心の区間幅も一度だけhullへ含める。出力はprior、feedback角／誤差、count、confidence区間、phase／energy hull、径、条件statusを持つ。点推定が必要な場合は外向きenergy hullの端点からexact rationalの中点を返し、float表示の丸めを新たな保証へ混ぜない。

形式上の失敗配分は`34/720+1/360=.05`。これは旧prefix十分条件・独立性・numerical acceptance・新readout誤差条件が適用できる場合の接続式である。コードは`formal_RPE_or_resource_qualification=False`を返し、人工PASSを340 shotsの正式採用にしない。

## 6. Compiler・費用・未証明事項

Qiskit1.3のSOURCEにはP(λ)→U1(λ)→`exp(iλ/2) RZ(λ)`のsymbolic等価経路がある。Compiler Aは全joint circuitを既に制御した状態でlowerし、一つのcommon global phaseを許す。**新controlをlower後に付けない。** このidentityは追加phase instructionをnativeで扱える根拠だが、実contextの選択経路・数値誤差・gate cancellation・最終価格を新角度へ無条件に移す根拠ではない。

raw SOURCEとして増えるのはancilla上のP instruction。qubit・reference register・CXや追加量子queryを設計上要求しない。実nativeのRZ／CX／DAG depth／size差、classical feedback latency、将来のstrict T／T-depthは未取得。Pがfree、Clifford扱い、一定Tという仮定はしない。

残る資格は、実prefix data／schedule／lawへの接続、実角度・P parameter／Compiler A contextの誤差、numerical acceptanceの扱い、追加価格、全round／強PFとの公平な比較。これらを今回のreadout実装だけで解決済みとせず、[Q2計画](NEXT_VALIDATION_PLAN.md)の範囲選択をGPT／ユーザーへ戻す。
