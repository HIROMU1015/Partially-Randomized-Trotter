# Track D C7-0：長q価格モデルの構造解析・追加検証計画

2026-11-07 JST。**CX差の `208−224q` はq=2・4・8の保存有限標本と整合する構造候補であり、全pathのP0価格公式としては未資格である。次の最小案は、価格のestimandと構造モデルを先に固定し、別途許可されたfresh q=16で限定的に検証すること。q=4096までの直接compileや一括外挿は前提にしない。**

本書はSOURCEの静的読解、保存欄の転記、条件付き構造式と未採択の計画である。新sampling、分子回路生成、gate/depth計数、compile/postcheck、テスト、保存値の再集計・fit・統計再解析、長q予測値・N・全RPE総費用の算定、詳細科学レビューは実施していない。[次段階条件案](C7_1_EXECUTION_DRAFT.md)は実行許可ではない。

主張の区分：**[保存]** 既存保存結果で確認済み、**[構造]** SOURCEの組立規則から確認済み、**[条件]** 明示した条件の下で導出可能、**[仮説]** 技術的候補、**[未資格]** 未確認または保証不足。本書内の「推奨」は正式採択と区別する。

## 1. 固定条件・確認資料・保全

linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、8 system modes、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1。元Hamiltonian Hの厳密基底状態を仮定し、状態準備費を除外する。保存own-DF groundの信号・Nは転用しない。

|identity|固定SHA256|
|---|---|
|元H packet|`1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`|
|CDF endpoint|`a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`|
|保存law記述|`c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5`|

価格境界は `single_hadamard_interrogation_without_state_preparation`。H→whole core→YのみSdg→H→ancilla測定を含むcompiled full wrapper。Reference register、CSWAP、actual A†、χ、cut損失も含む。P0はQiskit1.3.0、optimization_level=1、seed=17、basis=rz/sx/x/cx、approximation_degree=1.0、qubits_initially_zero=false、topology-free、位相保持。RZ/CX count、各depth、total depth、sizeを別資源、peak qubitsを幅制約として扱う。RZ→T、routing、noise、実機時間へ換算しない。

共通・repository AGENTS.md、PROJECT_MAP、研究概要の全文、研究目的、関連VALIDATION_STATUS/manifest、正式共通指示2026-10-25の保全・承認・再利用規約を確認した。C7-0専用branch `codex/track-d-c7-0-structural-preparation-20261107` とworktreeをC6b最終公開commitから分離した。正式共通指示は共通原本で読み、未収録の公開リンクを作らない。

指定した以下12 commitはローカルGit objectで取得し、GitHub接続のcommit/treeとも一致を確認した。資料・コードは固定版を読んだ。全依存closureの再監査、raw配列展開、QPY decode、既存チェックの再実行はしていない。

|固定資料・commit|今回確認した範囲|
|---|---|
|[C6b最終結果][C6B] `fed5f458be2f625de4e1b8c98b8cc5da6e551b14`|RESULT、最終SOURCE/result/STOP、価格recordの来歴、保存paired統計・凍結モデル・探索的診断。tree `b1a4e5a954e32dd44e9490b376cd61081fe5f39c`|
|[C6b当初SOURCE/PLAN][OLDPLAN] `23725fc6f6d6d0c2bec93e31e32074711804e250`|samplingと最初の処理、固定条件、旧partial result/STOP|
|[承認済み引継ぎSOURCE/PLAN][HANDOFF] `44a58b9fba5d3aba7a31d0a6b12e3aa25be9def4`|未着手taskだけの引継ぎ、旧新生成証拠の区別、最終STOP|
|[C6b-0][C6B0] `c93f3c2373b158f8eff3800a62fe1a94cce5c3b0`|paired/絶対価格、校正分散、CX証明義務、旧提案の資格|
|[C6a結果][C6A] `e9d4e2f602d5f9e014507ca7e391719bab017b88`|q2/q4各16、価格・統計・係数/共分散・SOURCE/STOP|
|[C6a科学SOURCE][C6ASRC] `6b6667026ec8b670f610e41141d195883a9eb513`|fresh request、canonical sampling接続、共有組立、統計・fitter入口|
|[C5][C5] `9a6920b8355643701ab7436a4fd3b34862e9e3bd`|共通価格入口、固定q2資格、strong公平性、既存C2との一致|
|[C4][C4] `a5784c4bd3e3fb9779849a1e1fc1759559e2caea`|full-wrapperと旧fenced会計、boundary/lane候補、全law/長q不足|
|[C3完了][C3] `cc554cf6daea08df6c20ea8e1e92405259ce49b5`|COMPLETION_RESULTと固定fixture/cut/融合政策。旧失敗RESULTとは区別|
|[Stage A][A] `3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0`|保存q1の32 path、event16 cut、有限cohort価格・SOURCE/STOP|
|[B4-2][B42] `07d18bfe1d5145d6f8de026c0d994d5cb1c783c8`|13-round ideal条件付き十分Nと前提を参照のみ|
|[B5][B5] `0a8b953aea1be20b110784960763d7f36ebccac8`|finite-bit law、joint-law、全path実装誤差、共通actual Nの不足を参照のみ|

**[保存] C6bの旧新SOURCEを混同しない。** 旧23725は256 occurrenceのsampling、raw16、compile/postcheck48を生成した。旧controllerのpartial集約はpostcheck32だが、drain中に完成した16件を含む保存PASSは48。新44a58はそれらを再利用し、未着手raw16・compile/postcheck80だけを初回実行した。最終はraw32・compile/postcheck128・pairedχ32 PASS、sampling追加0、完了task再実行0。旧STOP_FAILURE、旧source_input_preserved=falseを改変せず、新側の前後照合一致と併記した来歴を維持する。旧新の役割は[最終result][FINALJSON]と[引継ぎPLAN][HANDOFF]にある。

C6bの高速化SOURCE切替許可は当該移管だけのもの。今回C7-0から旧runをresumeしたり、次の長q実行許可へ読み替えたりしない。C6a/C6bのraw event/QPY/NPZは従来どおりローカル保全のみ。新規公開対象は本書・次段階条件案と最小索引だけである。

## 2. 保存観測の整理

**[保存]** 次表は[C6a保存paired統計][ASTATS]と[C6b保存paired統計][BSTATS]のmean欄の転記である。差はReference−strong、各qのX/Yは保存価格で同値だった。再集計・新fit・外挿はしていない。

|資源|q2：16 whole-path|q4：16 whole-path|q8：32 whole-path|
|---|---:|---:|---:|
|RZ count|+338.125|+219.6875|−16.90625|
|CX count|−240|−688|−1584|
|RZ depth|−1310.375|−2555.5625|−5232.0625|
|CX depth|−1160.9375|−2342.4375|−4791.875|
|Total depth|−3177.25|−6230.0625|−12747.9375|
|Circuit size|+290.125|−276.3125|−1408.90625|
|Peak qubits|+8|+8|+8|

各qのCX差は全標本で同じで標本SD=0。RZ差の保存標本SDはq2=1.8211717839530313、q4=2.9825883613622133、q8=3.8718119039223535。q8の絶対RZはstrong平均42628.65625、Reference平均42611.75で、差−16.90625は大きな二つの価格の小さな差である。RZの実用的優位やT費削減を認定しない。

保存された価格、有限標本平均/分散、q2/q4校正モデル、独立q8診断、未確認の長q母平均/モデル誤差は別の層である。seed分離は擬似独立設計であり数学的IID証明ではない。X/Yを独立標本として数えず、nは16/16/32のまま。観測min/maxはlaw supportではない。

**[未資格] q-domainの境界。** Stage Aのq1は保存cohort・event cut、C5のq2とC3のq4/q8は既知の固定連結fixture、C6a/C6bはfresh設計の有限価格標本である。q1のCX差は保存32本で変動し、q≥2のcomplete step cut候補と同じ公式を適用できない。C2にはC3以降の追加手動cross-step最適化がなく、C5固定q2で同じ政策へ揃えた際はP0後7資源がC2と一致した。この一fixtureの一致を全pathの最適化不要の証拠にはしない。q16〜4096のReference full-wrapper価格は未検証である。

## 3. CX paired差の構造

### 3.1 SOURCEから直接確認できる組立規則

以下は[C3/C6a共有builder][BUILDER]、[H4 adapter][ADAPTER]、[元StrongBuilder][STRONG]、[tail emitter][EMITTER]、[directional emitter][DIRECTION]を静的に読んだ範囲である。

|要素|[構造] 確認できること|compiled価格へ残る条件|
|---|---|---|
|h-step/head|最初のforwardはuncontrolled、最後のreverseはdirectional。Gaussian basisはsystem上uncontrolled、相対Gaussian transitionにanalytic vacuum位相補正がある|Gaussian全制御・共有除去による弱い対照へ置換しない|
|directional中心|`append_directional_diagonal`はsystem parityとancilla-targetのCX、符号付きRZを用いる。RZZのzero-angle分岐とglobal phaseのancilla RZがある|support・係数の零/非零と特殊角、P0での隣接相殺を覆う必要|
|head relative fusion|`mixed_head`はcontrol modeと同rotationのrunをまとめ、diagonalを合成し、異basisは`transition`で接続する|同じbasisかどうかのSOURCE規則と、P0後costの一定性は別|
|step境界|`head_recipe('boundary')`のblock0はdirectional half＋uncontrolled half→ordinary controlled full。strongにq−1境界|directional fullと誤解しない。head-tail横断の新手動融合は両方式とも追加されていない|
|tail|同basisの連続runを共有。固定H4 Providerはsingleton時も元full frameを使う。support・event順序・signed角度を保持|event数が同じでもbasis run・support・隣接符号の並びはpath依存|
|control-one|Referenceの物理segmentは同じcontrolled tailをancilla=1へ制限。対象ancillaの制御ゲートはbase gate、CXはsystemのXとなり、宣言位相も保持|これはP0後の単一無制御rotation価格を先に仮定する処理ではない。残るRZ/Xの単純化はcompiler依存|
|A/B cut|Bは前半、Aは後半。各segment内部を融合するため合計q−2境界、global cutで1境界の融合を失う|cut端のbasis/phaseとwhole compileの相殺が必要。cutは価格を見て動かさない|
|gadget|前後二つのCSWAP cascade、systemへA.inverse、referenceへB各一回、最後にP(χ)。保存H4では16 logical CSWAP・追加8 qubits|AとA†を両方走らせる二重計数をしない。単体CSWAP価格×16はfull-wrapper価格と未同定|
|scalar/identity/χ|strongのscalarはpath単位のancilla P、Referenceは各segment global phaseとwhole χを保持。`path_phase`はsigned event角を含む|これらの明示位相ゲート自体は一qubit/宣言位相だが、周辺のRZ/CX削除機会への影響までゼロとは認定しない|

`unsigned_rotation_angle`というAPI名でも、ここで使う保存値には係数の符号が含まれる。絶対値へ置換しない。normalization/attenuationはlaw・信号会計の値で、unitaryへ重ねて掛けない。

### 3.2 条件付き構造モデル候補

**[仮説]** ある固定の局所展開規約を定義できたとき、次の分解を候補にする。今回primitive価格や係数は計数していない。

\[
\Delta C_{CX}^{P0}(\omega,q)=
F_{\rm gadget}+F_{\rm ends}
+q\,d_{\rm step}
+(q-2)b_R-(q-1)b_S
+T(\omega,q)+\epsilon_{P0}(\omega,q).
\]

Fはgadget/shellと端部の差、dは同じ局所規約でのstep差、bは合法な内部boundary融合補正、Tはその規約に残るtail/run/support/符号依存項、εP0は局所規約と一体P0との差と定義する。cutの失われた1境界はq−2対q−1に含め、同じ損失を別途加算しない。A†の順序反転に伴う効果も端部/T/εに一回だけ含める。分解は会計候補であり、各項の独立性・一定性を主張するものではない。

**[条件]** b、d、端部が固定され、TとεP0の合計が全対象pathで同じaffine関係を持つなら、差を定数項＋q比例項へ整理できる。逆に、この残差にpath依存や長q依存が残れば、native構造の単純さだけではcompiled差の公式にならない。qとq−1/q−2は代数的に従属するので、三つのq平均からstep効果・boundary効果・固定費を独立にfitして識別できるとも限らない。

**[保存]** 指定された観測関係 `ΔCX(q)=208−224q` は既存q2/q4/q8の有限結果と一致する。**[未資格]** 208をCSWAPだけの価格、−224を全pathの1 step節約と認定するSOURCE上のP0 bridgeは得ていない。今回この二係数をゲート列から新しく算定せず、q16以降の値も算定していない。

### 3.3 全path公式へ進む前の不足

|分類|不足する証拠|
|---|---|
|局所case|保存atomのsupport/符号/identity、同basis/異basis run、零角・合成角・逆回路の全該当caseで局所差が同じか|
|境界case|head/tail、step bridge、A/B閉包、CSWAP/lane端、readoutとχのinterfaceでの相殺規則|
|P0 bridge|[phase compiler][PHASE]→[unroll/transpile][P0]の全対象rewriteが差を保存すること、または有効な残差包絡。seed固定だけでは足りない|
|任意列・長さ|短いcaseの対応が任意ordered列へ閉じる条件。compilerの最適化が未知の長runをまたいでもcoverageが保たれるか|
|適用域|同endpoint/h/r/K、basis policy、cut政策、P0/precision/price scope。q1 event cut、別topology/compilerを含めない|

本調査はproject compilerの呼出しとphase処理までを確認した。Qiskit内部全passの一般cost不変性を証明したものではない。追加random標本で差が一定でも全support定理にはならず、逆に定理が未完成でも、許可されたqにおける有限paired観測には意味がある。

## 4. RZ countとcircuit size

|要因|[構造] SOURCEから説明できること|[仮説]/[未資格] 残ること|
|---|---|---|
|固定gadget負担|CSWAP、χ補正、二つのlane閉包とreadoutがwhole pathに一組|P0後のRZ固定項は角度・phaseの集約/削除に依存し、常に一定とは未確認|
|A†とB|両laneを各一回実行。A†は逆順・逆角度・宣言位相のactual inverse|逆回路が同じ一体P0 RZ数になる全path根拠は未取得|
|制御除去|tail中心のancilla CXをXへ制限し、Gaussianは元からuncontrolled。directional中心とphysical中心は異なる列|RZ/X共役とone-qubit合成により削減し得るが、その量をSOURCEのRZ命令数だけでcompiled値としない|
|basis/run共有|同basis tail run、head relative transition、内部half-sweep融合を両方式で維持|basis runとsupport/符号の配列がcompiled差に影響し得る。q増加単独の因果効果ではない|
|global cut|complete step cutが内部融合を一つ失わせ、別registerに分ける|固定負担とみなすモデルでもP0端部残差は確認が必要|
|scalar/identity/sign|scalar・extracted identityを一回ずつ会計し、χは全path metadataから求める|χの角度、合成された正負rotation、phase相殺によるRZ生成/削除はpath依存し得る|

**[仮説]** 固定gadget/閉包の負担を、長いpathで積み重なる制御除去・内部融合の差が上回ることは、保存された符号変化を説明する候補になる。ただし、寄与を分離したcompileはなく、RZの正→負やsizeの正→負をこの一因に確定できない。`qが大きければ必ずRZも減る`とは結論しない。

**[条件]** P0後が指定basisとterminal measurementだけなら、size差はRZ/SX/X/CX/測定の各命令差を合わせたものになる。RZだけの差やRZ+CXだけでsizeを代用しない。[metrics][METRICS]は`count_ops`、filtered depth、`size`を別々に保存する。今回SX/Xの新しい計数や、size差からの逆算は行っていない。RZとsizeの保存SDが似ていることも、全path同一式の証明にはしない。

差のモデル `d_g(q)` と方式別絶対価格 `μ_R,g(q), μ_S,g(q)` は別に保持する。小さなpaired分散は方式間の相関で起こり得るが、絶対価格の校正精度まで保証しない。q8の微小RZ差には、その符号を論文等で主張するのに必要な精度と、価格モデルの用途上の精度を別途選ぶ必要がある。

## 5. Depth：lane・DAG・critical path

**[構造]** strongのdirectional/tail制御は同じancillaを使い、同じsystem wire上のhead/basis/中心がstep間の依存を作る。異wireの操作には並列機会があるので、controlが共通というだけで全命令を直列と数えない。

Referenceの中央A†とBは別registerに置かれる。SOURCEでcomposeが順番でも、wireを共有しない命令間にはDAG上の並列機会がある。一方、前後のCSWAP cascadeは共通ancillaを使い、laneへの入口/出口とχ/readoutが依存を作る。これらを無償の同時操作にしない。global cutはlane分配・融合損失の双方を変え、実際のpathでは片laneがcriticalになることがある。

**[仮説]** 中央のlane深さのmaxとgadget/shellを組み合わせるモデルは候補になる。ただしcascadeがwireごとに異なる時点でlaneを開放/回収すること、P0後の命令列・相殺・commutation、A†の逆順を含むため、`D_full=定数+max(D_A,D_B)`を無条件の恒等式にはしない。深さを半分のstep数だけで決めず、必要ならlane入口/出口ごとの依存時刻を特徴にする。

RZ depth/CX depthは保存実装の `circuit.depth(filter_function=...)`、total depthは `circuit.depth()`。物理durationの重み付きscheduleではなく、三者は加算できない。DAG候補を作る場合も、対象Qiskitのfiltered-depth定義とterminal measurementの扱いへ一致させる必要がある。全qで同じcritical pathが支配する証拠はない。

|モデル候補|構造・必要パラメータ|既存q2/4/8での識別・追加情報|利点と限界|
|---|---|---|---|
|affine `a q+b`|資源/方式/axis別の傾き・切片|q2/4で既に校正、q8で独立診断。校正残差自由度0。未使用長qでの検証が必要|最小で比較しやすいが、長q biasやcritical-path切替を覆わない|
|step＋boundary補正|step数、内部boundary、端部/cut、run/support特徴、compiler残差|q−1/q−2とqは従属。q平均だけでは内訳を識別できず、SOURCEで係数を拘束するか別特徴が必要|countの説明・残差位置の限定に有用。depthの加法性は得られない|
|Reference lane|max型lane負荷、入口/出口、左右のjoint変動、cut損失、shell|保存whole-wrapper価格だけではlane別価格・依存時刻が不足。許可後の最小観測拡張と独立qが必要|半path並列性を表せる。E[max]をmax(E)で代用しない。lane間covarianceも保持|
|critical-path/DAG|wireごとの依存、gate/filtered weight、basis/CSWAP interface、P0残差|現在の3種類のq平均だけでは高次元パラメータを識別できない。未取得のDAG特徴と検証が必要|full-wrapperに近い説明。P0後DAGが必要ならcompile自体の削減にはならず、native DAGならP0 bridgeが別途必要|

**[未資格]** 説明変数を増やすこと自体を改善としない。既存の3qで高次元fitを行わず、まずCXの構造残差とRZ/sizeの固定項・bulk項、depthの最小lane候補に絞る。精度用途に対してaffineで十分なら、追加モデルの開発を必須にしない。

## 6. 不確かさ：何を増やせば何が解決するか

|要因|対象・現状|次の対処候補と限界|
|---|---|---|
|同qのpath変動|[保存] 各qのsample SD/共分散。whole-pathが単位|そのqの独立標本で推定精度を改善し得る。観測SD0を母分散0にしない|
|calibration uncertainty|[保存] C6a q2/q4各16の係数/共分散。全予測qに共有|必要なら追加calibrationを新scopeで設計。holdout追加だけでは係数の不確かさを消せない|
|構造bias|[未資格] affineと真の母平均のずれ|独立qで診断し、構造仮定・残差domainを検証。標本数増だけでゼロにならない|
|q-domain移送|[未資格] q16〜4096、長run/phase/critical-path変化|適用域を覆う構造/P0 bridgeか、必要domainの別証拠。q16一箇所から全域を認定しない|
|finite-bit/ideal law|[未資格] PCG64/choiceとideal積lawの距離、joint-law|implemented lawでの価格とideal期待価格を分ける。正当なlaw bridgeと費用support等なしに誤差0としない|
|compiler依存|[構造] P0とSOURCEは固定、価格はpath依存で決定|固定seedは全列costの定理にならない。モデル残差と科学的実装誤差を別扱い|
|相関|[保存] 方式・資源・axisの共分散、係数covariance|paired分散を保つ。X/Yをn増に使わず、特異共分散の逆行列を仮定しない。cross-q独立も設計条件|

### C6bの二つの診断

**[保存]** 一次は凍結予測そのものに条件付け、`|e|+z s8/√32≤τ`。別population-line欄は `|e|+z√(s8²/32+vcal)≤τ`。eは同じ保存された観測−予測、τとzも同じ。追加された校正分散がmarginを変える。以下は[保存診断][DIAG]からの転記であり、再算定していない。X/Yそれぞれが該当する。

|資源|保存e|保存τ|一次margin|別population margin|既存判定|
|---|---:|---:|---:|---:|---|
|RZ depth|−186.125|499.778125|323.2342322|565.984122|一次幅内／別欄幅外|
|Total depth|−412.25|1163.753125|710.9963196|1244.288129|一次幅内／別欄幅外|

一次12/12幅内、別欄は上記2資源×X/Yの4/12幅外。校正分散はC6aの線形誤差伝播を保存したもので、q間擬似独立・有限sample SD等を前提とする。追加marginは新しいq8のcorrectness不一致でも、真のmodel biasの確定値でもない。normal/tail/IID/ideal-law/bias coverageが未資格なので、一次も別欄も探索的・名目的であり、確認的95%保証はない。

**[条件]** 新qの直接paired母平均を推定する目的なら、モデル係数を使わない推定へcalibration varianceを足す理由はない。凍結予測値の誤差評価でも一次と同じconditioningを守る。一方、population校正線や共有係数を通した長q評価なら係数不確かさを無視できない。biasはさらに別項であり、独立性不明の全誤差を一律二乗和にしない。

**推奨判断順序：** estimand/用途の選択→SOURCEに基づく低次元モデル固定→必要な独立長q証拠→なお係数精度が用途を妨げる場合だけ追加calibrationを検討する。C6bの別診断未達だけからq2/q4追加を必須にせず、未達を通すための幅拡張・同holdoutへの再fit・都合のよい資源選択もしない。C6bは既に観測済みなので、新モデルの設計材料にした場合、新モデルの独立holdoutには戻せない。

## 7. 再利用SOURCEと必要最小差分

ここでの再利用は将来の許可後の候補であり、今回はimport・実行・変更していない。origin側部品はC6b SOURCEが固定した元C science `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7` の版を参照する。bootstrapは元C側を優先し、Track D追加adapter/AX2B helperを固定Track D側から補う。StrongBuilder、tail emitter、P0 gates、宣言位相compilerは元CとC6b checkoutのGit blobが一致した。一方、`resource_applicability/ax2b_native_df_v5.py`は元C commitに存在しないため、[C6b固定版][DIRECTION]とSOURCEの`new_and_reused`に記録されたSHA256 `7b8b13483066bb39cc4aad1c36f6e5da6fbc679ecb8bae9c08a0b326648a36fd`を出典とする。同名fileをすべて元C由来と扱わない。

|資産|再利用できること|長qでそのまま使えない部分／最小差分案|
|---|---|---|
|[C3共有builder][BUILDER] `prepare_steps/physical_segment/mixed_head`|明示steps、q≥2のcomplete cut、強い融合、ordered CAR積、actual inverse|固定`prepare`はq4/8保存fixtureのみ。共有関数がq≥2を受けることは長q科学資格ではない。新requestから既存組立へ渡す|
|[C5価格入口][PRICE] `wrapper_for/_checked_price`|同requestのmethod/axis、P0/精度/scope、full-wrapper6資源と幅の資格検査|旧公開`price_record`は保存00→01/q2限定。guardを緩めず新scopeの価格入口を追加。全law/long_qは実証なしにtrueへ変更しない|
|[C6a][C6ASRC] `binding/sample_record/car_context/prepare_fresh`|canonical sampler、event保存binding、既存CAR/head/atom cache、共通組立への接続|q2/4、n16固定。q16には新domain/request/固定標本数と未使用seedの定義だけを分離。samplerを再実装しない|
|[C6a fitter][C6ASRC]・[共分散後処理][CALCODE]|generic affine fitter、method/axis/資源covariance、共有係数の線形伝播|旧係数・frozen予測は変更しない。新モデルのfit/再解析が必要なら別途許可し、設計データと新holdoutを分離|
|[C6b][HOLDOUT] `HoldoutRequest/paired_statistics/diagnose`|同pathの両方式/axis、完全標本の集計、frozen条件付きとpopulation-lineの区別|q8/cut4/n32/family/幅は固定。将来のq/nを旧guard書換えで偽装せず、新scopeで最小接続|
|[adapter][ADAPTER]・[Stage A共通処理][PAIRED]・[StrongBuilder][STRONG]|χ、signed/scalar/identity、actual inverse、Gaussian/relative basis、対照公平性|条件/位相/順序を保つ。旧SOURCEをhot-editしない。別cut/新最適化は別の検証対象|
|[repeated builder][REPEATED]・repeated cost|half-sweep融合/位相集約、matched whole回路と加法診断の区別|legacy DF型・ordinary制御の実装をnative CDF strongへ丸ごと差し替えない。節約値を別回路に単純加算しない|
|[P0][P0]・[宣言位相compiler][PHASE]・[checker][CHECKER]|公式定義unroll、位相保持、CAR/raw/observable/χ/leakageの既存precision|長qでの数値誤差は未確認。MPFR160は宣言位相蓄積であり、T合成精度や全round丸め保証ではない|
|[高速化部品][FAST] `BatchedStateActions`|同Aer double/native変換、copy削減、入力batch、declared phase復元|q8の適合証拠を長q性能保証にしない。private native callbackへの版依存と長回路変換/copy/出力memoryを確認。全256入力/全振幅は省略しない|
|[人工テスト][TESTS]、C5/C3/高速化tests|request/phase/AB、統計semantics、旧guard、高速化の回帰資産|今回は未実行。必要な新差分の人工回帰を将来scopeに含め、分子性能証拠と区別|
|[旧resource accounting][ACCOUNT]|axis別期待価格・scope/fingerprint・coverageの設計、共分散会計の考え方|旧fenced kernel＋readoutやown-DF targetをそのままReferenceの価格/N providerへ使わない。N別途、全round資格別途|

最小実装候補は新scopeのrequest/seed/価格資格と、必要なら保存scalarのlane/boundary特徴を追加する程度。新Gaussian合成器、sampler、compiler、承認証明・grant発行基盤は不要。既存のq制限は旧研究の科学的資格を守るため残す。

## 8. q16〜4096への方法比較と最小案

|方法|検証できる範囲・必要入力/SOURCE|sampling/compile・ボトルネック|条件と限界・STOP|
|---|---|---|---|
|A 構造モデル先行|同endpoint/law/P0でCX固定/bulk/boundaryとRZ/size/depth候補を定義。上記builder/emitterを再利用|静的説明にはsampling不要。局所caseやP0 bridgeの数値確認は別許可。compiler全域証明は難度未確定|条件付き式を公式へ格上げしない。閉包/残差domain不足なら全path claimをSTOP|
|B 必要な長qの直接paired標本|新qのimplemented whole-path lawでの有限価格、条件が整えばそのqの期待価格|新q/nのcanonical sampling、raw n、PASS後最大4n wrapperとpostcheck。長いIR/逆回路、P0 DAG、CAR積、全入力state-actionが負担|強い対照/phase/精度を維持。新qの結果だけで未観測qへ移送しない。不一致/欠測で依存処理STOP|
|C 段階的検証|まず未検証q16を候補に、モデル残差・数値精度・計算実現性を確認|採択されたq16だけの4n上限。追加qはその後の別scope。n/幅/信頼度は未採択|q16をq4096の資格にしない。結果後のmodel修正は探索として別版を作り、新holdoutへ進む許可を戻す|
|D 直接compile削減|局所正規形/合法境界/primitive/依存interfaceからfull P0との同値または残差包絡を得る|特徴抽出・bridge検証の最小実装が将来必要。P0後DAGが必要ならcompileを省略できない|fenced kernelへ無断置換しない。証拠のない局所和・単体shell価格でfull-wrapperを代用しない|

**推奨はAの小さな構造仕様固定とCのq16限定検証を組み合わせ、Dはbridgeが説明できる範囲だけ採ること。** Bの全q大量直接compileを初手にしない。現在のq2/4/8を新モデルの設計に使う場合、そのモデルの検証には新しい独立domainが必要である。少数長qを追加しても、全域の一様性は構造/残差条件なしに得られない。

n・価格幅・古典信頼度は本書で新たに固定しない。探索的なら有限観測/名目診断まで、確認的ならestimand・family・有効なlaw/独立性/tailまたはsupport・精度用途から数を設計する。独立長q標本だけで消えない校正floorと構造biasを別に扱う。標本設計の新数値算定も次の許可対象に含める。

未来の実行ではraw PASSごとのcompile pipeline、独立path並列、CAR/head/atom cache、既存state-action高速化を再利用する。長qではqubit幅が同じでも回路・変換・検証の負担が増え得るため、q8の実測memoryを長q上界にせず、起動時の空きCPU/RAM等と実task観測で配分する。恣意的な低い上限を置かず、共有設定・他ユーザー・他jobを変更しない。SOURCE切替やretry/resumeが必要なら旧C6b移管許可を流用せず、新scopeの判断へ戻す。

## 9. Stage Bと全RPE会計への接続

**[条件]** 将来の会計は資源gごとに

\[
G_{a,g}=\sum_{m=0}^{12}\sum_{b\in\{X,Y\}}
N_{a,m,b}\,\mathbb E_{\omega\sim p_m}
[C^{full,no\text{-}prep}_{a,g,m,b}(\omega)].
\]

同じlawと同じNを使えるなら差は同じNで重み付けしたpaired期待価格へ接続できる。方式別Nが必要ならこの簡約は使えず、方式別絶対価格も必要になる。共通Nには全使用domainでの測定信号・誤差資格とshot条件が必要で、q8の有限correctnessだけでは足りない。

**[保存]/[未資格]** B4-2の14,738は元H厳密ground、ideal finite-RTE IID、exact native等を前提とした13-round十分shot数。B5が示したfinite-bit law、shot/occurrence joint-law、全path/全round実装誤差、actual N共通化は未解決として保持する。価格側の古典統計誤差とenergy/failure保証を分け、保存Nとの乗算や再算定はしない。B側branch/SOURCE/STOPへ介入していない。

共有モデル係数θを使う将来の価格予測では、条件付きcovarianceは異なるqでも共有される。線形特徴x(q)なら `Cov(f(q),f(q'))=x(q)^T Σθ x(q')` が条件付き候補となる。roundごとに独立な誤差として縮めず、model bias/law差は別に評価する。qubitsはpeak幅でありroundやshotへ加算しない。全13-roundの数値価格・総費用は本書では未評価のままである。

## 10. 終了記録と次の担当

**C7-0静的調査・計画作成を終了し、成果公開・remote SHA/tree/公開bytes照合後mandatory STOP。** 今回の変更はMarkdown報告・条件案と最小索引だけ。科学SOURCE、input/raw、既存価格/統計/モデル、旧grant/ledger/marker/STOPは変更しない。テストを含む科学処理0、新規回路/計数/fit0、raw payload追加公開0。

次担当は**ユーザー／GPT**。選ぶ事項は、paired差と絶対価格のどちらを何の精度で使うか、探索的か確認的か、構造候補・q16限定scope、n決定法・family/幅/信頼度、必要なら追加calibrationの独立scopeである。**Codex**は採択と明示的な新科学実行許可後にだけ、[条件案](C7_1_EXECUTION_DRAFT.md)の採択範囲を実装・検証する。本書はq16以降、追加校正、確認的保証、全RPE費用、actual成功保証、詳細科学レビューの実行を開始しない。

[C6B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fed5f458be2f625de4e1b8c98b8cc5da6e551b14/docs/research/track_d_c6b_q8_20261107/RESULT.md
[OLDPLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23725fc6f6d6d0c2bec93e31e32074711804e250/docs/research/track_d_c6b_q8_20261107/PLAN.md
[HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/44a58b9fba5d3aba7a31d0a6b12e3aa25be9def4/docs/research/track_d_c6b_q8_20261107/ACCELERATED_COMPLETION_PLAN.md
[C6B0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c93f3c2373b158f8eff3800a62fe1a94cce5c3b0/docs/research/track_d_c6b0_preparation_20261107/REPORT.md
[C6A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e9d4e2f602d5f9e014507ca7e391719bab017b88/docs/research/track_d_c6a_paired_price_20261106/RESULT.md
[C6ASRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6b6667026ec8b670f610e41141d195883a9eb513/src/trottertracks/track_d_c6a_pilot.py
[C5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a6920b8355643701ab7436a4fd3b34862e9e3bd/docs/research/track_d_c5_q2_20261106/RESULT.md
[C4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a5784c4bd3e3fb9779849a1e1fc1759559e2caea/docs/research/track_d_c4_price_model_preparation_20261106/REPORT.md
[C3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc554cf6daea08df6c20ea8e1e92405259ce49b5/docs/research/track_d_c3_q4_q8_20261106/COMPLETION_RESULT.md
[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0/docs/research/track_d_h4_paired32_reference_20261106/RESULT.md
[B42]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/07d18bfe1d5145d6f8de026c0d994d5cb1c783c8/docs/research/track_d_b4_2_rpe_shots_20261106/RESULT.md
[B5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0a8b953aea1be20b110784960763d7f36ebccac8/docs/research/track_d_b5_actual_rpe_static_survey_20261106/REPORT.md
[BUILDER]: ../../../src/trottertracks/track_d_c3_whole_path.py
[ADAPTER]: ../../../src/trottertracks/track_d_h4_reference.py
[PAIRED]: ../../../src/trottertracks/track_d_h4_paired_reference.py
[PRICE]: ../../../src/trottertracks/track_d_whole_path_price.py
[HOLDOUT]: ../../../src/trottertracks/track_d_c6b_holdout.py
[FAST]: ../../../src/trottertracks/track_d_state_action_acceleration.py
[CALCODE]: ../../../scripts/summarize_track_d_c6a.py
[STRONG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7/src/trottertracks/c1_h4_strong_baseline.py
[EMITTER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7/src/trottertracks/pr_dictionary_validation/circuits.py
[DIRECTION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fed5f458be2f625de4e1b8c98b8cc5da6e551b14/src/trottertracks/resource_applicability/ax2b_native_df_v5.py
[P0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7/src/trottertracks/pr_dictionary_validation/gates.py
[PHASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7/src/trottertracks/c1_h6_compiler_precision.py
[REPEATED]: ../../../src/trotterlib/df_partial_s2_repeated.py
[ACCOUNT]: ../../../src/trotterlib/rpe_resource_accounting.py
[METRICS]: ../../../src/trotterlib/track_d_minimal_reference.py
[CHECKER]: ../../../tests/track_d_h4_reference_checks.py
[TESTS]: ../../../tests/check_track_d_c6b.py
[ASTATS]: ../../../artifacts/track_d_c6a_paired_price/2026-11-06/run01/paired_statistics.json
[BSTATS]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/paired_statistics.json
[DIAG]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/exploratory_diagnostics.json
[FINALJSON]: ../../../artifacts/track_d_c6b_q8/2026-11-07/run01_accelerated_completion/result.json
