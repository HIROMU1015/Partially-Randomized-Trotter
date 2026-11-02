# C1 Stage 3：統合実行経路の技術準備

2026-10-25 JST。起点 `df3d53904ba00dde72e9e9f1a8e55b2757812935`。
Branch `codex/research-c-c1-stage3-integrated-20261025`。

**保存入力から最終比較・partial STOPまでの統合runnerを実装し、人工検証を完了した。実H4の科学実行は0。全lawに有効な有用な合成誤差上界は未資格であり、H4の同CA達成・保証付き資源優位を認定していない。**

[新未承認契約](research_c_c1_stage3_integrated_contract_unapproved_20261025.json)／[SOURCE・人工結果・STOP](../../artifacts/research_c_c1_stage3_integrated_technical/2026-10-25/)。旧準備SOURCE・契約・科学結果は変更していない。正式共通指示2026-10-25版は前作業からSHA256 `e1e3e2625e2b728977402bded6583658e7fb762160f5e787fd45572af0095720`で同一であることを確認した。許可の証明書・承認記録・新grantは不要であり、追加していない。

## 1. 採用されたcore条件と固定した実行条件

保存P-H4（直鎖1.00 Å、STO-3G、Nα=Nβ=2、sector36／system full256／controlled512）、通常CDF number・元g補正κ・rank8／保存XDF rank10を使用する。新fit0。元outer frame順を維持する。

ユーザーが採用した初回条件は、CA=0.00159 Ha、DF reserve CA/8、δ=0.1、m=0..12、決定論ordered symmetric S2のs∈{1,4}、PR L_D=2・s∈{1,4}・r∈{4,16}・K∈{0,2}、計20候補である。各DF Hamiltonianのsector ground-state oracleを仮定し、状態準備費用を除外する。初回参照モデルはideal finite-RTE IID law、主指標は期待総RZ数、科学call上限は5,637 logical units。

残る実行詳細は旧準備案を継承して新契約へ固定した。βtotal=π/6、βstat=π/12、alias energy窓[-10,10] Ha、量子統計failure配分0.04／4選択protocol／13round／2軸。P0はQiskit1.3.0、opt1、seed17、rz/sx/x/cx、approximation_degree1、qubits_initially_zero=False。correctnessは最大成分差1e-10、unitarity1e-10、X/Y絶対差2e-10、位相fitなし。PR cost calibration32標本／候補、選択後virtual RPE1000試行／protocol。これらは次回の科学実行許可の確認対象であり、今回実行していない。

状態準備を除く**条件付きinterrogation資源**を比較する。CDFとXDFの物理DF誤差は別であり、近似精度の完全一致や無条件の総資源優位は要求・主張しない。CA/100への拡張、登録外探索、literal/sharedの未観測追加利得はない。

## 2. 完成した一体entryと再利用

```bash
# byte／JSON／環境metadataのみ。科学実行しない。
python scripts/run_research_c_c1_stage3_integrated.py \
  --contract docs/research/research_c_c1_stage3_integrated_contract_unapproved_20261025.json \
  --input /path/to/saved/P-H4.npz

# 次回、別途明示科学許可を得た後だけ使用する。
python scripts/run_research_c_c1_stage3_integrated.py \
  --contract docs/research/research_c_c1_stage3_integrated_contract_unapproved_20261025.json \
  --input /path/to/saved/P-H4.npz --execute \
  --output artifacts/research_c_c1_stage3_science/<date>/<fresh-run-id>
```

既存Stage 2 task専用venvのPythonをread-onlyで再利用する。Python3.11・数値／回路に必要な5 libraryを検査し、既存52-package lockを再現用に参照する。依存追加、共有venvや他ユーザー環境の変更はない。cache／temp／Python bytecodeは当該worktreeに分離する。

| SOURCE | 実装・再利用範囲 |
|---|---|
| `c1_stage3_integrated.py` | 固定20候補、byte preflight、科学call、並列実行、候補状態、旧evidence評価内部への接続、選択／比較、fresh output／partial STOP |
| `c1_stage3_molecular.py` | 既存保存fit loader・一般K adapter・Stage 2 native model／Gaussian・CAR・compiler・metric・RPEを呼ぶ実backend。保存H4を使う手動の追加driverは不要 |
| `c1_stage3_native_bridge.py` | 旧bridgeの負係数角度検査だけを新versionで修正。native D-forward／RTE／D-reverse／scalar emissionを再利用 |
| `c1_stage3_qualification.py` | 有限supportの事前計数／小supportの網羅、費用の全support範囲、有用な合成誤差enclosureの有無を区別 |
| 旧`c1_stage3_accuracy/reference/connection` | PF／有限RTE平均／減衰、horizon・signal→phase→shot→energy、直列費用、比較、virtual binomial/RPE復元を再利用。旧SOURCEは無変更 |

`evaluate_evidence_packet`を再度呼んで計算を重ねる代わりに、その内部と同じ`assess_round`・`assess_energy`・`aggregate_totals`を直接接続し、各callを一度だけ記録する。人工packetについて旧entryとの出力一致を検査した。旧entryへ外部から科学evidenceを手動で渡すだけだった準備段階の未完成部分は解消した。

新fitter、Gaussian合成器、線形代数／RTE／RPEアルゴリズムは作成していない。Stage 2のcorrectness helperは診断用spectral SVDも計算するため、新backendの同じ最大成分差判定では不要なfull512 SVDを省き、Frobenius診断を保存する。合成誤差資格には使わない。PF／補正RTE誤差に必要なoperator normは旧referenceで別々に測る。

## 3. 実行順と科学call

1. 固定SOURCE／契約／元入力／保存fit・行列・QPYのbyte確認と環境metadata確認。元fit SOURCE、Stage 1評価SOURCE、新統合SOURCEを独立に保持する。
2. 保存XDF whitelist・保存J/U/κ/constantを読込。保存sector3＋CDF full base／8frame／totalの13行列を再利用する。CDF物理normの反復測定や新Hamiltonian生成／fitは行わない。未保存XDF full generator11だけ、既存CARで構成する。
3. 3sectorのground eigenpairと各armのfull状態埋込／aliasを確認する。ground状態をNPZへ保存する。DF上界入力は旧Stage 1の物理normに元のengineering allowanceを加えた値とし、厳密区間証明とは区別する。
4. arm×roundのexact evolution26件を共有し、20候補×13roundのPF・補正RTE operator誤差とattenuationを評価する。全中間evolutionをfull256で扱い、総Hamiltonianのsector保存だけを理由に中間を36次元へ射影しない。
5. 登録の十分条件が既に満たせない候補ではnative費用を省略する。その他はbasisを初回だけbuild/cacheし、独立CAR densityから各eventのtargetを作る。raw controlled、actual inverse、P0 compile、compiled operator／unitarity、oracle状態でX/Yを確認して費用を測る。
6. 合成誤差・law・費用supportの資格を処理し、旧evidence評価へ接続。全roundと共通CA条件を満たした候補だけNと総費用を出す。各armの決定論／PRを別に選び、独立seedのvirtual RPEとCDF/XDF比較を行う。
7. 正常・不適格・未資格・FAILのいずれも証拠を保存してmandatory STOP。条件や閾値を事後調整しない。

5,637の既存34種類のlogical unit枠を維持し、今回の実装に合わせて内部処理の説明だけを整合させた。call表の合計とrunnerのper-kind／total上限をpreflightで照合する。fit／optimizer／retry／resume0。参考：round reference260、feasibility260、final round260、PR draw最大512、新build／compile／raw／adjoint／compiled／metrics各最大517、X/Y最大514、serial cost展開最大520。1 composite unit内部のmatmulごとに追加課金する形式ではない。候補不適格、trajectory重複、保存kernel再利用で実数は上限より少なくなる。

compile上限517はPR候補16×最大32 unique kernels＋新deterministic s=4の2件＋head／X readout／Y readoutの3件。小supportを網羅する場合も先にsizeを確認し、32件以内に収める。網羅後の32 calibration drawsはそのcatalogを再利用し、追加compileしない。s=1 deterministicはStage 2の**HF wrapperを含まない**同じcoherent kernelと費用を再利用する。旧raw／compiled QPY各arm2件のidentityを固定し、時間／順序／P0 profileは同じである。

## 4. 全lawの上界：実装済みの方法と未資格

**Sampling law。** 初回ideal finite-RTE IID lawを参照targetとするので、そのモデルに対するlaw誤差は定義上0である。実NumPy finite-bit samplerとの誤差は未知のまま別記する。実samplerを理想法則と同一視した証明はない。calibrationの標本平均も実装lawによる点推定であり、ideal-law期待値への厳密な移送を主張しない。

**費用support。** Kで許された全micro-event数とr回の全word数を、列挙前に上限付き整数計数する。全wordが登録の最大32 build枠に入る場合だけ、既存finite-event列挙を使って**全support**の実kernelをcompileする。そのmetricの最小／最大は、任意のsupported lawの期待費用を包含する決定論的範囲になる。これは標本最大値や標準誤差ではない。理想法則と実samplerの平均差を0と置かなくても有効だが、範囲が広ければ優位は未判定となる。

全wordが32枠に入らない場合は32 IID drawsの点推定だけを残し、cost support／intervalはUNRESOLVED。P0 whole-kernel最適化はwordに依存し、単一event費用を足しただけでは全lawのcompiled費用上界を保証できないためである。数学的に正のideal weightが浮動小数点で0へunderflowした場合も、そのwordを勝手にsupportから除かず未資格にする。

**合成誤差。** 完全supportと各reference／compilerの有効なnorm enclosureがある場合の合成・直列反復接続を実装した。unitary間の普遍的上界2は有効でも、このpilotの位相予算を資格付けするには粗すぎる。H4 backendのSciPy／CAR／Qiskit数値比較はcorrectness診断であり、全lawの有用な厳密enclosureではない。従ってH4 backendでは**有用な合成誤差上界をUNRESOLVED**とする。完全supportを数値的に検査できた場合でも、reference自体の誤差enclosureがないまま小さい上界へ昇格させない。

人工のexactな例だけはanalytic enclosureを持つbackendで成功経路を検証した。実H4へのenclosure注入用file／承認証明／外部権限は用意していない。通常のH4実行では、PF/RTEで既に不適格ならINFEASIBLE、そうでなければ合成誤差未資格でUNRESOLVEDになり得る。**現SOURCEだけで有用な合成誤差保証が得られるとは約束しない。**

## 5. 精度・費用・判定の意味

既存のsignal単位の接続を保持する。

```text
δraw = A(εPF,op + εRTE,corrected,op) + εsynthesis,raw + εlaw,raw
ρlower = A - δraw > 0
asin(δraw/A) + βstat <= βtotal < π/3
Naxis = ceil(2 log(2/αaxis) / [ρlower sin(βstat)/sqrt(2)]²)
physical ground-energy bound = dDF + βtotal/tmax <= CA
Gmetric = Σm (NX,m CX,m + NY,m CY,m)
```

DFをHa単位のenergy biasとして一度だけ数え、同じDF誤差のsignal biasを重ねない。未知synthesis／lawならN・総費用を出さない。十分条件の上界が予算を超えたINFEASIBLEは、登録の評価方式で資格を満たさない意味であり、実際のエネルギー精度が必ず悪いという必要条件ではない。

費用方式はnative kernelを明示的なfenceで直列に繰り返すもの。PRのoccurrenceごとにfresh trajectoryを用いる。depthは直列scheduleの和であり、長qをglobal compileしたdepthではない。head HとX/Y readoutを別segmentにして、空のX-shellのH/H cancellationによる無料化を防ぐ。classical calibration回数をquantum shotsへ掛けない。実行回数はshotごとのinterrogation、backend job数は未指定である。

CAを満たした候補のpoint totalと、資格付きintervalに基づく比較を分ける。費用supportが未資格ならPOINT_ESTIMATES_ONLY、区間重複ならINDETERMINATE_OVERLAP。同CA未達・unknown synthesisの候補を資源比較へ入れない。最良候補の選択を、登録外global optimumや状態準備込みの優位とは呼ばない。virtual RPEはmarginal binomial診断であって解析的誤差保証の代用ではない。NumPy int64を超えるNを黙ってclipしない。

## 6. 高速化・STOP・証拠

空きaffinity CPU、実cgroup CPU／RAM quota、MemAvailableとdense workloadのmemory見積もりから候補worker数を決める。恣意的なCPU／RAM／worker／wall上限は追加しない。26 exact evolutionと独立candidate／compilerをthread poolで並列化し、BLAS threadsとQiskit内部parallelismを当該process内で調整する。immutable input／CAR／basisを共有し、arm別exact exponential cacheを使う。matrix powerと有限平均を使い、長q回路・全trajectory・全shotの列挙を避ける。GPUの速度向上は未測定で主張しない。

最初のcorrectness FAIL／input不一致／非有限値／call超過で新規依存callを止める。既に開始した独立callは安全に完了し、poolをdrainする。候補INFEASIBLE／UNRESOLVEDはFAILと区別し、登録済み残候補の評価だけを続ける。retry／resume0、別namespaceでの回避なし。

`science_calls.jsonl`にattemptと完了／FAIL、候補別JSONとprogress JSONLに完了済みround値・trajectory／metrics、raw／compiled QPYに失敗対象、ground NPZにoracle根拠を保存する。`failure.json`・`STOP.json`とwall／CPU／peak RSSも保存する。科学evidence／SOURCE／条件の記録であり、許可を証明する台帳ではない。既存STOPやgrantは読替え・再発行しない。

## 7. 人工検証と完了範囲

新規19件、既存Stage 3／Stage 2の52件、一般Kの13件＋29 subtestsがPASS（計84 tests＋29 subtests）。[最終log／JUnit](../../artifacts/research_c_c1_stage3_integrated_technical/2026-10-25/)参照。

- 20候補×13roundの人工end-to-endで、成功・全不適格・全未資格・混在・費用point-only・科学的FAIL相当・途中STOP・fresh output・call上限・provenance出力を確認。
- 実保存backendに人工2mode／full4次元のNPZを渡し、whitelist読込、CAR、3ground、sector埋込・alias、26 exact evolutionを確認。fit／optimizerは遮断し、全読込先が人工temp配下であることを検査。
- 人工2mode一般KのPR sample／全support／新deterministic／保存費用再利用を、実Qiskit P0で技術検証。計19回の**人工回路compile**、raw controlled・actual adjoint・compiled unitarity・X/Yを確認し、意図的なcompiled phase異常で停止・QPY保全を確認。
- 旧bridgeが負係数RTE eventを誤拒否することを人工例で再現し、新bridgeの独立controlled operator一致を確認した。旧SOURCEは保全。

初回の結合testでは、保存kernel診断のpath文字列と数値diagnosticのkey衝突を発見して修正した。また実Qiskitを使う新testと旧mock testを同一processへ混在すると旧stubのmodule cache前提が崩れたため、旧回帰群はfresh processで独立実行した。SOURCEの科学条件を変更してPASSさせていない。

**実H4 decode／新operator・PF/RTE/RPE・shots・総資源評価／H4回路build・compile／新fitはすべて0。** 人工PASSはH4での科学的GOではない。

次はユーザー／研究CがこのSOURCE・条件・未資格事項を確認し、限定科学実行をチャットで別途明示許可するか判断する。現SOURCEはINFEASIBLE／UNRESOLVEDを含む限定pilotを一体実行できるが、保証付き総資源比較には有用な全law合成誤差上界等が必要である。Codexはその許可後だけ実行する。今回のremote照合後は**mandatory STOP**。
