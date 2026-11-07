# Track D C7-2：CX構造・人工P0検証・長qモデル準備

2026-11-07 JST。**固定54人工回路のraw検査98項目、P0 compile/postcheck54本、whole-pathのobservable/coherence/leakage検査36項目はPASS。既存人工回帰12件もPASS。** 局所CX節約の機構は説明できるが、実H4の`208−224q`を全path公式とは認定しない。加法性と単純lane depth式への人工反例を保存した。実H4の新sampling・回路生成・compile・価格再解析/fitはすべて0。

[固定PLAN](PLAN.md)／[固定case数値](CASES.json)／[人工結果summary][SUMMARY]／[全資源CSV][METRICS]／[compact命令・critical path][TRACE]／[STOP][STOP]／[q32未採択案](C7_3_EXECUTION_DRAFT.md)。本書は許可された技術検証であり、詳細科学レビューではない。公開照合後mandatory STOP。

## 1. SOURCE・保全・適用域

実行SOURCEは **`34929f8a517afbf70604b0ba007f8cdd9deac4c2`**、tree `a6fae653aecef30f2291b385fc229f9fbe8112ef`。初回SOURCE `72afe45f3201c59a3d3a0acf98f1d8c8a750d956` は、固定exportをGit worktreeと誤認した起動前確認で停止した。回路生成/compileには到達せず、CASES/精度/compilerを変えずにexport bytes照合へ修正した。[停止経緯](PRELAUNCH_NOTE.md)を保持する。科学的な不一致後のretryはない。

元科学SOURCEは `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`。指示中の38桁prefixはこのcommitへ一意に解決された。固定exportの関連SOURCEと指定commitのbytesを照合し、実行前後も一致した。[SOURCE索引][SOURCE]。既存SOURCE・入力・結果・grant/ledger/marker/STOP、旧scope guardは変更していない。

|固定参照|commit|今回の扱い|
|---|---|---|
|C7-1最終|`86c5c9d48dbddc085f388b6521be8cd92ee49d0d`|基点、q16保存結果/STOPの転記|
|C7-0|`8b422c28c22666a32103fdae33a3c9b9512ef248`|条件付きモデル・不足整理|
|C6b|`fed5f458be2f625de4e1b8c98b8cc5da6e551b14`|q8保存観測。承認済み高速化SOURCE44a58の来歴を維持|
|C6a|`e9d4e2f602d5f9e014507ca7e391719bab017b88`|凍結affineとq2/q4保存観測、再fitしない|
|C5|`9a6920b8355643701ab7436a4fd3b34862e9e3bd`|共通whole-path入口・guard・人工回帰|
|C3補完|`cc554cf6daea08df6c20ea8e1e92405259ce49b5`|既存step境界政策・保存証拠|
|B4-2 / B5|`07d18bfe1d5145d6f8de026c0d994d5cb1c783c8` / `0a8b953aea1be20b110784960763d7f36ebccac8`|統計資格の境界のみ、再計算しない|

実H4への接続条件はlinear H4/隣接1.00Å/STO-3G/Nα=Nβ=2、number-CDF8/元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1。元Hの厳密groundを仮定し状態準備費除外。packet `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、law `c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5` は旧結果のidentityとして維持し、今回そのpayloadは読み込んでいない。

人工fixtureは**2 mode、三つの人工deterministic block、2 events/step、h=.1、τ=.2、scalar .17/identity .04**。分子geometry/AO basis/CDF rank/実H4 lawは非該当。q2/q4のcomplete half-step cutのみで、r32のH4標本とは異なる。測定wrapperの価格境界は`single_hadamard_interrogation_without_state_preparation`と同じ定義、局所kernel/lane/shellの価格は別の診断値として扱う。

P0はQiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation1、qubits_initially_zero=false、topology-free、既存MPFR160宣言位相保持。component max-entry 1e−10、observable Frobenius 2e−10、rtol0、global phase fittingなし。MPFR160は宣言位相の精度でありT合成精度ではない。

## 2. 既存の実H4有限観測

次表は[C7-0保存整理][C70]と[C7-1最終報告][C71]の転記。差はReference−strong、再集計・新fitなし。X/Yは対応する同pathであり独立標本数を倍にしない。

|資源|q2 n16|q4 n16|q8 n32|q16 n16|
|---|---:|---:|---:|---:|
|CX paired平均差|−240|−688|−1584|−3376|
|RZ paired平均差|+338.125|+219.6875|−16.90625|−488.75|
|RZ depth差|−1310.375|−2555.5625|−5232.0625|−10523.9375|
|CX depth差|−1160.9375|−2342.4375|−4791.875|−9673.9375|
|Total depth差|−3177.25|−6230.0625|−12747.9375|−25648.4375|
|Size差|+290.125|−276.3125|−1408.90625|−3672.75|
|幅差|+8|+8|+8|+8|

CX差は各保存qの全観測pathで一定。これは`208−224q`と一致する有限証拠であり全support定理ではない。q8/q16の凍結予測への一次12量は探索的幅内、校正不確かさを含む別population-line診断ではRZ depth/total depthのX/Y計4量が幅外。correctness失敗とは異なる。q1/event cut・C2の手動cross-step未適用回路を同じモデルへ無条件に混ぜない。q8/q16を今回のモデル設計に使ったため、新モデルの独立holdoutとして再使用しない。

## 3. CX差の構造的分解

### 3.1 SOURCEで直接確認した組立

[C3 prepare_steps/mixed_head/physical_segment][BUILDER]、[StrongBuilder][STRONG]、[directional emitter][DIRECTION]、[tail emitter][TAIL]、[Reference adapter][ADAPTER]を確認した。

|寄与|SOURCEから確認済み|価格の残条件|
|---|---|---|
|deterministic中心|非零RZ：directionalはancilla CX2、physicalはRZのみ。非零RZZ：directionalはparity込みCX4、physicalはRZZ|特定native展開でphysical RZZがCX2なら差2。P0の境界相殺・特殊角は別|
|ordinary中心|RZはCRZ、RZZはcontrolled-RZZの既存gate定義|directionalと同価格とは限らない。人工の混合ordinaryで実際に異なった|
|Gaussian|uncontrolled basis、相対Gaussian transition、analytic RZ vacuum位相補正|局所Gaussian価格がfull-wrapperでも保存されることは未証明|
|head相対融合|同modeかつ厳密同rotationをgroup化、diagonalをfsum、異basisはrelative transition|同basisで角度が零になればentryをprune。近似同値として勝手にgroup化しない|
|tail|同basis runを共有し、signed angleとsupportを保つ。H4 providerはsingletonでもfull frame|basis/run/support/符号で絶対価格は変わる。手動の新head-tail横断融合はない|
|step境界|directional half＋uncontrolled halfのblock0をordinary controlled fullにする。strongはq−1境界|directional fullへ置き換えない。普通のonebody block0はRZ中心であり、一般controlled-RZZの場合と分ける|
|global cut|ReferenceのA/B内部は合計q−2境界、完全step cutで1境界の閉包が増える|同じ損失を「cut費」とq−2対q−1の両方で二重計数しない|
|gadget|二つのCSWAP cascade、A†をsystem/Bをreferenceへ各一回、最後にP(χ)|AとA†を二重加算しない。logical CSWAP数とcompiled CXは別|
|scalar/identity/χ|strongのpath位相集約、Referenceのsegment global phaseと全pathχを保持|明示位相は一qubitだが、周辺P0書換えまで無影響とは証明していない|

`unsigned_rotation_angle`というAPI名でも実際には係数符号を含む。normalization/attenuationは別会計で、unitaryへ重ね掛けしない。

### 3.2 条件付きの数式と未識別項

一般の会計候補は

\[
\Delta C^{P0}_{CX}=F_{gadget}+F_{ends}+q d_{step}
 +(q-2)b_R-(q-1)b_S+T(q,\omega)+\epsilon_{P0}(q,\omega).
\]

boundaryを含めた実組立と同じ展開規約で各項を一回だけ定義する。Tは残るevent/run/support/符号依存、εは局所展開の和と一体P0との差。A†による反転・隣接相殺は該当端部/T/εに一回だけ含む。

より狭い**条件付き説明候補**は、reverse halfのprune後中心数をPdet、tail回転数r、単体Gaussian閉包の局所CXをG0、単体Fredkin CXをfと置き、

\[
\Delta C^{P0}_{CX}(q,\omega)=2Nf+2G_0-2q(P_{det}+r)+R_{CX}(q,\omega).
\]

これは、各中心のcontrol premiumが2CX、parity/basis共有の差がその規約で相殺し、唯一のcut閉包が2G0で表せる場合の候補。Rはfull P0残差を含み、0と証明したものではない。この表示ではcut閉包を2G0へ移したため、上のboundary項をさらに足さない。

number-CDF8/LD2でonebody8、二つのdiagonalが各8 RZ＋28 RZZをすべて非零で保つならPdet=80。r32なら`−2(Pdet+r)=−224`が得られる。**非零entry数・group化・特殊角・全P0界面を全supportで覆うことが条件**であり、今回H4入力から新しくprimitive数を計数していない。単体Fredkinは人工P0でf=8だったためN=8なら局所和のshell候補は128。208を導くにはcut/端部/P0残差を独立に同定する必要がある。

**G0=40は今回独立に測定していない。** `208−128=80`からG0を逆算して証明扱いしない。GaussianのSOURCEと保存観測だけで、局所価格およびそのfull-P0保存性を同時に同定できていない。従ってA=208/B=−224は今も`P0_GLOBAL_BRIDGE_MISSING`。人工のf=8も16個のCSWAPを含むH4全体の固定費128という定理ではない。

## 4. 人工caseと反例

CASESはP0実行前に固定した。[raw記録][RAW]、[postcheck][POST]、[各caseの分類と資源][SUMMARY]、[命令列][TRACE]。raw CXは明示`cx`だけで、CRZ/RZZ/CSWAP/XXPlusYY等の未展開価格ではない。書換えはraw/P0列と既知のゲート恒等式から説明し、未観測の特定passへ帰属させない。

|family/case|raw明示CX→P0 CX|成立範囲・原因・分類|
|---|---|---|
|正RZ：U / ordinary / directional|0→0 / 0→2 / 2→2|非零単RZの局所control premium2を確認。P0_LOCAL_VALIDATED|
|明示零：U / ordinary / directional|0→0 / 0→0 / 2→0|零RZのCX対が消える。通常H4 constructorはexact zeroをpruneするため、そのまま合法H4反例ではない|
|signed RZ/RZZ＋phase：U / ordinary / directional|0→2 / 0→18 / 8→8|ordinary−Uは16で「全control modeが中心毎+2」は不成立。controlled-RZZの既存定義展開を含む局所反例。H4のblock0 ordinary onebodyと区別|
|同Z正連続：controlled / physical|4→4 / 0→0|physicalのRZ/Xは1 RZへ合成。controlledの4 RZ/4 CXは残る|
|同Z正負：controlled / physical|4→4 / 0→0|両unitaryはIに一致するが、controlled P0は4CXを残す。数学的な相殺可能性とP0の実施は別|
|同ZZ正連続：controlled / physical|8→6 / 4→2|隣接parity CXが界面で消える。無補正のnative局所和への反例|
|support変更：controlled / physical|4→4 / 0→0|physicalは別wire上の2 RZ、total depth1。support依存を保存|
|隣接CX対|2→0|CX²=I。境界を跨ぐ価格加法性への反例|
|CX–RZ(control)–CX|2→2|unitaryはRZ(control)だがP0はCXを残す。「合法な相殺は全て実施」への反例|
|CX–RZ(target)–CX|2→2|RZZに一致し、2CXは残る。可換control caseと意味が異なる|
|単体CSWAP|0→8|exact bit置換と位相付き一致。P0_LOCAL_VALIDATED、任意全体への加法性は未資格|
|4 CSWAP＋χ/readoutのscalar-only shell、全8本|0→28|孤立価格の和4×8=32と不一致。二cascadeの界面とP0の相殺/再配置を含む。COUNTEREXAMPLE_FOUND|
|実A→A†|0→0|rawには72 RZなど未展開basisを含むがP0はsize0。q2_sameではA=Bで単体A/BとA†が各28CX、連結は0。無条件加法性への反例|

正負同supportの人工caseは、実H4の同atomに都合よく符号を振った標本ではない。H4 law内に同等の組が存在するかは未確認。価格候補への反例はcorrectness不成立ではなく、全54回路のunitaryは精度内で一致した。fixtureを除外・変更せず全部保存した。

## 5. whole-path：強い対照・資源・correctness

既存StrongBuilderをそのまま使い、相対Gaussian/analytic vacuum補正、uncontrolled basis、directional中心、tail full-frame共有、strong q−1/Reference q−2境界融合、scalar/identity集約を維持した。toy用にN=2とdescriptionだけを局所適応し、H4 guardは緩めていない。元C5の簡易人工fixtureはGaussian transitionとtail emitterを置換しているため、今回はそれを構造証拠にせず既存emitterを直接呼んだ。

independent exterior-minor liftとoccupation対角からordered Vを作り、AB=V、actual adjoint、vacuumχ、strong枝間coherence/direct-V対応、X/Yを検査した。P0は全unitaryのglobal phase込み比較とobservable/coherence/leakage比較。小規模最大5qubitのみで、新state simulatorやGaussian/compilerは実装していない。

|検査|件数/最大残差|
|---|---|
|raw独立unitary・basis・AB/adjoint/χ・observable等|98項目、max-entry 3.219646771412954e−15|
|54 compile後のraw/P0 full unitary|54 PASS、max-entry 4.834571936074951e−14|
|whole-path observable/coherence/leakage|36項目、最大Frobenius 1.4398351969241058e−14|
|既存回帰|C5 8＋C3 4、12 PASS|
|SOURCE前後|一致、CASE/compiler/thresholdの事後変更0|

各whole-pathでX/Yの価格は同じだった。次表は各axisの実full-wrapper P0値で、RZ/CX depth、total、size、qubitsを別々に保持する。単位を合算して主指標は決めない。

|fixture/方式|RZ|CX|RZ depth|CX depth|Total depth|size|qubits|
|---|---:|---:|---:|---:|---:|---:|---:|
|q2 same / strong|178|92|113|92|262|369|3|
|q2 same / Reference|212|88|73|54|162|419|5|
|q2 mixed / strong|182|94|114|94|265|377|3|
|q2 mixed / Reference|217|90|77|56|170|428|5|
|q4 same / strong|336|180|215|180|502|699|3|
|q4 same / Reference|344|136|115|78|254|683|5|
|q4 mixed / strong|344|184|217|184|508|715|3|
|q4 mixed / Reference|354|140|119|80|262|701|5|

CX差はq2両fixture−4、q4両fixture−44。実H4の数値候補をこの人工domainへ適用しない。これらも有限4人工fixtureの証拠で、任意人工列/任意qの定理ではない。実H4のSOURCE/input/loadは0、ground solve/DF fit/sampling/新しいRPE Nは0。初回起動前の実装誤認は前節の履歴どおりで、科学処理のretry/resume0。

## 6. RZ・sizeの構造要因

ReferenceはCSWAPの固定RZ/SX/CX、actual inverse、二lane、χ/readoutとcut閉包を含む。一方、control_oneでCXがXへ変わり、`RZ(θ) X RZ(−θ) X`がP0で合成される。符号/零角/同wire隣接がRZ残数を変える。scalar/identityとχは明示位相なのでglobal phaseとancilla-relative位相を分ける。

人工の単体FredkinはRZ10/CX8/SX2、scalar-only shellはRZ38/CX28/SX6/measure1、size73。sizeはRZ＋CXではない。q2_sameのRZ差+34/size差+50に対し、q4_sameは+8/−16。mixedでは+35/+51から+10/−14。固定負担と制御除去・融合の競合を示す限定証拠で、RZが必ずq一次/いずれ必ず負になるとの保証ではない。

raw/P0のsizeはそれぞれの命令表現で数えており、raw sizeをnative展開後価格と混同しない。位相保持MPFR160をRZ→T換算へ使わない。routing/T synthesis精度/実機noiseとコストは未評価であり0ではない。

## 7. Lane・DAG・filtered depth

SOURCEの中央A†/Bは異なるregisterなので相互に独立だが、前後のCSWAPは共通ancillaを使いwire毎に入口/出口時刻が異なる。strongは制御中心が同ancillaを使うため依存が連鎖する。scalar集約とreadoutもancilla依存を持つ。

Qiskit1.3.0の`QuantumCircuit.depth(filter_function)`実装も静的確認した。各gateの対象wire時刻の最大を取り、対象gateなら1加え、対象外でもそのgateの全対象wireへ時刻を伝播する。**Filtered-out gateを削除した回路のdepthとは限らない。** このmax-plus recurrenceを人工のraw/P0全108回路で既存metricsと照合し、RZ/CX/totalすべて一致した。古典条件/変数/directiveを含まない今回domainに限定したfeature extractorで、simulatorではない。

|fixture|P0 Reference total|shell total|max standalone A†/B total|full−shell−max|同RZ/CX depth残差|
|---|---:|---:|---:|---:|---|
|q2 same|162|42|115|+5|+3 / +3|
|q2 mixed|170|42|123|+5|+3 / +3|
|q4 same|254|42|207|+5|+3 / +3|
|q4 mixed|262|42|215|+5|+3 / +3|

X/Yとも同じ反例。shellはA_shell=e^{iχ}I、B_shell=I、同じχを使う妥当なgadgetである。A=B=Iへχだけを残した不正なnegative fixtureではない。それでも「shell＋max(lane)」は恒等式にならない。+5/+3がH4でも一定とは主張しない。

必要なのは、wire別arrival/exit時刻、CSWAP順、control依存、basis transition、cut前後のinterfaceを含むmax-plus/DAG記述とP0後の補正である。compiled DAGからのcritical path抽出は正確な価格の再記述だが、compile不要の安価な長q predictorを得たことにはならない。[命令列・critical path][TRACE]と[wire終端feature][SUMMARY]を保存した。

## 8. モデル候補と採択前の不足

|候補|入力・正当化される部分|人工確認と実H4の不足|負荷・affineに対する利点・限界|
|---|---|---|---|
|凍結affine|q、C6a係数/共分散、method/axis/resource|q8/q16一次探索的診断は保存済み。別population-line depth不一致|予測は軽い。高qのbias/全supportは無保証、今回再fit0|
|CX構造|N/r、非零primitives、basis/cut/interface、P0 profile|局所premium/CSWAPを確認、非加法反例あり。H4 G0・全P0 bridge未取得|長さ比例計数は軽いがP0残差の証明/包絡が必要。208−224qは保証式に昇格しない|
|RZ/size固定＋bulk|scalar/χ/shell、basis run、signed angle、zero、SX/X/CX|人工で固定負担とbulk節約が競合。H4での残差設計/独立検証が必要|event列featureはO(qr)、共分散保持。単純q一次以上の精度は未認定|
|二lane|A†/Bのwire別負荷、入口/出口、CSWAP順|単純max式へ全人工fixtureで反例。H4全体interface/P0校正は未取得|中央の並列性を表せる。単体compile和への代替ではない|
|DAG/critical path|gate-wire依存、重み1/0、basis/ancilla/phase、P0補正|recurrenceは人工で一致。長q pre-P0→P0移送が未検証|compiled DAGならO(size)の再計数。native template型の予測精度は別途必要|

最小仕様案は、凍結C6a affineを比較用に保持し、CX候補の**整数残差**、countsのfixed/bulk/boundary feature、depthのwire interface featureを別々に記録するもの。未校正の係数を追加して少数q平均へfitしない。価格代理をC5 `_checked_price`へ登録せず、q32以降の実compileを置換しない。

q8/q16はモデル設計資料になったため、新モデルを評価するdomainは未使用のfresh q32などへ分離する。モデル仕様/feature/係数/価格幅/判定familyはsampling前に採択・固定する必要がある。任意列への局所規則の閉包、P0全体残差の包絡、長さとcutの依存を得るまでは`UNQUALIFIED`。

## 9. 実装再利用・公開・STOP

追加したのは[人工adapter][NEW]、[明示runner][RUNNER]、[toy結果projection][PROJECT]、固定case/文書。C3/C5/StrongBuilder、既存Gaussian・tail・directional・Reference・actual inverse、phase compiler、小規模Operator/observable checker、metrics、exterior liftを再利用した。H4専用のrequest/price guardは2modeへ対応しないため、人工local metricsへ分離した。C6b高速state-actionとcanonical sampling/統計pipelineは静的参照に留め、不要な17qubit処理やsamplerを呼んでいない。

実空きCPU/RAM/負荷を確認し、54独立compileをtask数とCPU affinityに応じてprocess並列化した。小行列BLASとnested transpileの重複並列だけを抑えた。環境が提供する資源を恣意的に小さく制限せず、共有設定/他job/他userへ変更なし。内部観測はローカルに留め、速度の最適性や比較未実施のspeedupは主張しない。

人工QPY108原本とverbose記録はlocal/・cases/で保全。公開は54 caseのcompact scalar投影、約0.4MBの純人工命令trace、SOURCE/case/検査/hash索引。実H4のraw event/QPY/NPZ、CPU/RAM/絶対path/資格情報は公開しない。[公開区分][PUB]。SOURCE/CASESの前後一致とgit差分で旧結果/STOP不変を確認する。

分類は、組立規則 `STRUCTURALLY_EXPLAINED`、人工局所 `P0_LOCAL_VALIDATED`、無条件加法/単純lane式 `COUNTEREXAMPLE_FOUND`、実H4任意列の価格接続 `P0_GLOBAL_BRIDGE_MISSING`、q≥32/確認的価格保証 `UNQUALIFIED`。反例を修正して消す実行は行わない。

B4-2の保存14,738 shotsは元H厳密ground/ideal finite-RTE IID/exact native等の条件付き結果。B5のfinite-bit law、joint-law独立性、全support/全round実装誤差、actual共通Nの資格は本検証で解決しない。同じNを使うには同law・同path分布と全該当roundの測定/実装誤差資格が必要であり、方式別Nが必要な場合の全RPE会計は別問題。今回N/全13-round価格/actual成功保証は算定しない。

**技術処理を終了し、公開SHA/tree/bytes照合後mandatory STOP。** 次の担当はユーザー/GPT：q32案の目的・標本数・価格幅/古典信頼度・モデル仕様を採択するか判断し、必要なら別途実行指示を与える。Codexはq32sampling/compile、追加fit、全RPE会計、詳細科学レビューへ自動進行しない。

[SOURCE]: ../../../artifacts/track_d_c7_2_toy/2026-11-07/run01/SOURCE.json
[SUMMARY]: ../../../artifacts/track_d_c7_2_toy/2026-11-07/run01/summary.json
[METRICS]: ../../../artifacts/track_d_c7_2_toy/2026-11-07/run01/case_metrics.csv
[TRACE]: ../../../artifacts/track_d_c7_2_toy/2026-11-07/run01/operation_traces.json
[RAW]: ../../../artifacts/track_d_c7_2_toy/2026-11-07/run01/raw_checks.json
[POST]: ../../../artifacts/track_d_c7_2_toy/2026-11-07/run01/postchecks.json
[STOP]: ../../../artifacts/track_d_c7_2_toy/2026-11-07/run01/STOP.json
[PUB]: ../../../artifacts/track_d_c7_2_toy/2026-11-07/run01/publication_scope.json
[C70]: ../track_d_c7_0_structural_preparation_20261107/REPORT.md
[C71]: ../track_d_c7_1_q16_20261107/RESULT.md
[BUILDER]: ../../../src/trottertracks/track_d_c3_whole_path.py
[STRONG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7/src/trottertracks/c1_h4_strong_baseline.py
[TAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7/src/trottertracks/pr_dictionary_validation/circuits.py
[DIRECTION]: ../../../src/trottertracks/resource_applicability/ax2b_native_df_v5.py
[ADAPTER]: ../../../src/trottertracks/track_d_h4_reference.py
[NEW]: ../../../src/trottertracks/track_d_c7_2_toy.py
[RUNNER]: ../../../scripts/run_track_d_c7_2_toy.py
[PROJECT]: ../../../scripts/summarize_track_d_c7_2_toy.py
