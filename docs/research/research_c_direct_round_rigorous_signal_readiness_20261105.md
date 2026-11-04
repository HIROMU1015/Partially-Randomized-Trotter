# 研究C：Direct roundのrigorous信号評価・技術準備

2026-11-05 JST。**人工回路・人工状態のArb192信号包囲を実装。実H4の科学実行は0、SCIENCE_NOT_READY、MANDATORY STOP。**

Branch：`codex/research-c-direct-round-rigorous-signal-readiness-20261105`。
独立worktree：`/tmp/prt-c-direct-round-rigorous-signal-20261105`。
起点commit `325f07ae02d8872d4fe322d6ad6e08c256e0bef7`、tree `d4cc880dfe8b4b8e747e5732ed474fabd96e8574`。
この統合treeに必要な公開A/B/C実装とStage P/R/Head資料があるため、第一候補を採用した。
別Codexの未公開変更は使わず、既存Python module、科学SOURCE、契約、raw、STOPを編集していない。固定7-callの報告は統合treeへ未収録なので、指定commitのgit blobを直接読んだ。
正式共通指示2026-10-25の§8.1・§8.4・§10を実読し適用。承認証明、承認原文保存、新grant、認証基盤は追加しない。

## 対象と固定参照

将来対象は直鎖H4・隣接1.00 Å/STO-3G、number CDF8、元順LD2、partial ordered symmetric S2/s1/r32/K0、h=δ=.1、RPE m0..12。
Nα=Nβ=2、sector36/system256/controlled512、own-DF ground oracle、状態準備費用除外。
Endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
P0=Qiskit1.3.0/opt1/seed17/rz,sx,x,cx、既存Gaussian/relative fusion、MPFR160 declared phase、独立53-bit IID-gridモデル、既存RPE/CA/confidence政策を維持する。
本報告の新数値はすべて人工問題であり、この科学条件の新評価ではない。

| 参照 | 固定commit | 調査した役割 |
|---|---|---|
| 統合 | `325f07ae02d8872d4fe322d6ad6e08c256e0bef7` | 作業起点、A/B/Cの既存資格 |
| 固定7-call | `c0213176181b0f84eca0cf96464246d6a14e63a7` | 保存契約・報告の所在。再実行しない |
| Stage P | `c01bd2742ca88071d17db1fc3e3aec6cb9f3cb9e` | inverse、ancilla head/X/Y shell、有限wordの保証 |
| Fresh管理 | `9098de49f1bd816dbaf810eb63815c841169236d` | 標本・失敗・STOPの保存 |
| 高速Arb | `ce3185513578a8329efe364c748b5ce2aacdda45` | native局所作用、layout、全列法 |
| Population | `1c0e1d1b4967efecb08dd3d8e7a747eba6c83039` | 非unitary literalとnorm/rangeの意味 |
| Stage A | `ae9a4e13d1ab6f48fc4d38cbbcd0fddca30e9c53` | IID-gridと保存referenceの相違 |
| Head bridge | `369e2dfad9a4251ba5eb2fda28f183d559a791cc` | full-CAR headとB2 sector head間の接続 |

現状の科学資格は[Stage P報告](research_c_stage_p_completion_execution_20261104.md)、
[Stage R報告](research_c_population_range_schema_execution_20261104.md)、
[Head報告](research_c_head_bridge_execution_20261104.md)、`VALIDATION_STATUS.md` と既存manifestから確認した。
過去の科学数値は再計算せず、旧失敗・未完了・STOPを保持する。

## Direct roundの数学的対象

既存[期待bias仕様](research_c_iid_grid_expected_bias_readiness_20261104.md)と
`c_pr_aware_df_stage_b2.XY_gate`、`rpe_hadamard_interrogation`に従う。
Systemのq0はLSB、ancillaは最後のqubitで最高bit。初期ketは `v0=(ψ,0)`。
Head ancilla shellはH、X readoutはH、Y readoutはH S†（S†が先）。最後にancilla Zを測る。

一つのword wは32 event。Literal atomの数学的積T(w)、raw gate R(w)、compiled native C(w)を区別し、literal controlled mapは `D(w)=diag(I,T(w))`。
Literal head、signed atom、tail scalar、保存角度のexact dyadic意味を保持する。
Raw gateとliteral CARの差は一般に0ではない。

Round mのn=2^m個のword列に対して `V_C=C(w_n)…C(w_1)`、`V_D=D(w_n)…D(w_1)`。
Compiled shell P_C、S_C,X、S_C,Yを実際に作用させ、

`x_C=<S_C,X V_C P_C v0, Z S_C,X V_C P_C v0>`、同様にy_C、`z_C=x_C+i y_C`。

Literal側はideal shellを用い、`z_T=ψ†T(w_n)…T(w_1)ψ`。
非unitary Tでもこのalgebraic signalとideal-shell二次形式は一致する。Normで割らず、測定確率へ改名しない。
今回の評価器は二次形式を包囲し、`d_X=x_C−Re z_T`、`d_Y=y_C−Im z_T`をsigned区間で返す。

既存第一estimandは **`|E_q z_C−E_q z_T|`**。
`||C(w)−D(w)||`、一roundの信号差、`E|z_C−z_T|`、`|E(z_C−z_T)|`、有限標本誤差は別である。
新しいestimand、attenuationによる追加の乗除、kernel信号からのround外挿は採用しない。

Fresh IID-gridでは各occurrenceごとに新しいword、各eventにorder/atom用53-bit整数を消費する。
X/Y校正評価には**同じ実現word列**を使う。これはclassical couplingで、実測shotの共有ではない。
単一wordの反復とIID occurrenceの積を混同しない。
独立なliteral occurrenceなら `E[T_n…T_1]=(E T)^n` だが、一般compiled Cの平均channelは `E[CρC†]` であり、`(E C)ρ(E C)†`とは異なる。
各sampleのpure stateを伝播する本方式に、dense平均channelを構成する必要はない。
混合初期状態、noise、reset、途中測定・分岐は今回未対応。適切な分解またはdensity operatorが別途必要。

Actual inverseは同じforwardから生成されたrecordの復帰診断である。
現在のforward Hadamard wrapperに逆回路を追加しない。Born式のbra側の†は逆回路の追加実行ではない。
Gaussian内部のinverseは既にCのgate列に含まれる。
新adapterは各actual inverseのparent/word/SOURCEを照合し、その伝播入力に対するactual inverseとexact C†の作用差を包囲する。
この診断は状態限定であり、全operatorのinverse一致や非unitary Dの逆行列を保証しない。

## 方法の比較と採用実装

| 方法 | 保証する量・丸め | 1 occurrence / 1 round | m12・追加条件 |
|---|---|---|---|
| 1. Arb statevector | 指定状態の全出力、native局所作用、phase、shell、Z期待値の外向き包囲 | 局所arity一定ならO(GD) / O(nGD) | 採用。D=512でも一列。4096回の逐次伝播とreference生成費は残る |
| 2. Rigorous state-action合成 | `||v−c||≤E`を持ち、nativeでE'=E+δ、literalでE'=aE+δ | 方法1と同じ次数。再中心化でwrapping抑制 | 採用。literal aは包囲から計算し、1へ置換しない |
| 3. 必要な内積のみ | Exact block controlとideal shellならψ†T_n…T_1ψ。複素内積も実装 | system dでO(nGd)またはblock作用 | Literal側に適合。Compiledのexact block構造は未保証なのでfull controlled状態を使う |
| 4. Operator bound併用 | 既存shell/reference bridgeを一回ずつ加算 | 保存済みboundの再利用で軽減可 | 比較対象が異なる時だけbridge。今回はshellを直接作用させ、Stage P shell誤差を再加算しない |

全方式で、初期状態の資格、reference target、controlled相対位相、actual inverseの対象を固定する必要がある。
Direct法は「任意入力に同じ上界」の保証を捨て、指定状態とその不確かさ集合に対する信号保証を得る。

実装：[c_direct_round_signal.py](../../src/trottertracks/c_direct_round_signal.py)。
人工入力：[c_direct_round_artificial.py](../../src/trottertracks/c_direct_round_artificial.py)。

再利用した処理は、`fixed._local/_validate/_midpoint_vector/adjoint`、`z._real/_fraction/_norm/_scale`、
`c_arb_column_batch.layout`、`fixed_car.pack/unpack`、`integration._reference`、
freshのsample/context検査・exclusive保存・outcome/STOP、および既存統計engine。
`c1_h6_state_action`はAer double/complex128による有限probe観測で、rigorous radiusを返さないため数学的包囲には流用しない。
CARの既存producerはliteral block referenceを作れるが、fresh wordの生成は今回行わない。

Nativeはexact stored binary64 gate/phaseをArb192で評価する。各local unitaryを全出力状態へ作用させ、再中心化で失ったvector2半径δを加算する。
Global phaseをfitせず、controlled相対位相とregister順序を保持する。
Literalは既存block enclosureをそのまま作用させる。Block Bのnormに
`min(||B||F, sqrt(1+||B†B−I||F))`を使い、全blockとinactive identityの最大をaとする。各段の半径も同じArb192で外向きdyadicへ包み直し、exact Fractionの分母の指数的な表現増大を防ぐ。
これはunitarityの仮定ではなく、三角不等式とFrobenius上界からの保証。Nonunitary、non-Hermitian headを排除・置換しない。
最終Z期待値には `2||c||E+E²` のpadを加え、norm²の包囲も保存する。
区間endpointとradiusはexact rationalで記録し、精度は192から緩和しない。

既存per-word bindingを各occurrenceで呼ぶと、fresh.validate_sampleがround全体を繰り返し再生してO(n²)になる。
新adapterだけで全sample検査を一回へhoistし、既存と同じbinding schemaをO(n)で照合する。
この小さなmetadata組立の再実装は性能上必要で、既存moduleを変更しない。回帰テストで全round再生が一回であることも確認した。
Gate/phase/layoutとliteral blockは同じtask内の同一内容でcacheできる。異なるSOURCE、candidate、word、angle、targetの結果を転用しない。
入力・partial記録とunique-word reference cacheはO(nG)およびO(UΣb²)のメモリも使う。全メモリをO(D)とは主張しない。
Production m12ではreferenceを逐次消費・解放するcache方針がなお必要である。

## 初期状態、reference、shellの資格

初期状態をexact rational complex centreとvector2距離ηでbindする。η=0ならnorm²=1をexact検査する。
η>0は**供給されたstate-distance前提に条件付き**。SOURCE/target/state ID、normとの整合、provenanceを保持し、状態を黙って正規化しない。
入力の関連付けはproducerの正しさや承認の証明ではない。

既存own-DF oracleは自身のDF Hamiltonianのground stateを使う政策で、保存sector vectorのnorm drift/固有残差/最低固有対の確認はbinary64工程検査である。
真の正規化ground oracleとのrigorous距離をこれだけでは認定できない。
距離を要するなら、正規化と固有対の包囲、groundの一意性または対象subspace、検証されたspectral gap/isolation等を使う必要がある。
数値残差をη=0へ置換しない。**状態準備費用除外は数学的状態誤差0を意味しない。**

既存full/sector接続はsaved interleaved sector indicesをbit reverseしてsystem computational basisへ置く。
System256の全出力を保持し、Gaussian途中をsector36へ射影しない。
今回のcoreは明示されたsystem amplitude配列を要求し、実H4用sector embedding/ground読込runnerは未実装・未資格である。

Stage Pのancilla head shellと分子HamiltonianのHead bridgeを区別する。
前者はH/X/Y readoutのgate誤差、後者はfull-CAR headとB2 sector headという異なる数学targetの接続である。
今回のdirect C→literalでは、compiled shellとideal shellを直接評価するため既存shell誤差もC→raw→literalのoperator上界も足さない。
Literal block内のArb丸めは既にstate包囲へ伝播され、再加算しない。

保存B2 signalまで比較する場合の会計は、
`native→q literal（新項） + categorical q→p + full-head→sector-head + B2 atom→CAR + mathematical CAR→binary64 reference`。
Head/B2の保存合計を用いる場合はその構成項を再度足さない。
Stage Rのnorm/rangeはsupportとgrowthの資料で、別のsignal誤差項ではない。
全transportのtarget/state適合を確認するまでは統合科学資格を返さない。

## 人工・回帰検証

[テスト](../../tests/management/test_research_c_direct_round_signal.py)と
[technical artifact](../../artifacts/research_c_direct_round_signal/2026-11-05/technical)を参照。
局所gateのexact rational行列解、Arb384の独立なsin/cosをArb192区間が包含すること、複素内積、Z/norm²、X/Y符号、global/controlled位相、
同じtrajectoryのactual inverse、noncommuting occurrence順序、head/X/Y shell、非unitary literal、状態ηの伝播を検査する。
Nonunitary scale4ではliteral X=4、norm²=17/2、差X=−3を包含し、cap2/正規化をしていない。
Wrong SOURCE/candidate/word/angle/phase/register/target/precision、block欠落・重複、compiler失敗/未完了、kernel/round混同、missing enclosure、confidenceとの混同を拒否する。
例外・割込みは元標本とpartialを保存して既存STOPを閉じ、再抽出・retry/resumeをしない。

新規46テストと関連回帰の正本は`technical/final_tests.txt`、`technical/tests.json`。関連回帰246 PASSの後、dimension表示だけを加えた最終新規46 PASSを`technical/new_final_tests.txt`へ保存した（重複を含む）。初回の241 PASSログ`technical/tests.txt`も保全した。
旧全体suiteを科学評価込みで走らせず、既存integration/fresh/期待bias/CARと新pathに対象を絞った。
ローカル人工検証であり、immutable CIや外部独立再現ではない。

## 人工benchmarkと費用の限界

[runner](../../scripts/benchmark_research_c_direct_round_signal.py)は人工入力だけを作る。実データのpathや科学実行flagを受け取らない。
同じnative/literal pairを、既存`compare_literal`全列法とdirect信号法で比較する。
Directはshellとinverse state witnessも含み、operator側は全列operator保証を計算するがshell/inverseは含まない。資格と仕事量は同一ではない。
Runtime importを両方式でwarm-upし、24独立process task、各case/方式3反復。
32 CPU affinity、空きRAM約52 GiB、GPU driver利用不能を確認した。Arb/FLINT CPU経路を使用し、BLAS/OpenMPはprocess並列と競合しない1 thread/task。
小問題の全列法は各worker内で直列、case間を並列化した。CPU/RAM/wallの人工capは置かず、共有環境・他job/cache/lockを操作しない。
時刻・CPU秒・process peak RSS・gate数・dimension・occurrence・round長は各measurementに保存する。
Peak RSSはimportを含むworker全体のhigh-waterで、coreのstateメモリやworker合計peakとは異なる。

正本は[artificial_final_v2](../../artifacts/research_c_direct_round_signal/2026-11-05/artificial_final_v2)。
初回の非対称cold-import測定`artificial`、hoist前の`artificial_warm`、半径dyadic化前の`artificial_final`も保全した。最終比較に混ぜない。
最終値（各3反復）は次の通り。測定中の他の人工taskとの負荷も含み、単調なdimension scalingや統計的speedupを証明しない。

| system / controlled | occurrences | forward gates/occ | method | wall秒 中央値 [min,max] | CPU秒 中央値 | worker peak MiB 最大 |
|---|---:|---:|---|---:|---:|---:|
| 4 / 8 | 1 | 11 | direct | 0.007139 [0.005744,0.011909] | 0.007140 | 102.11 |
| 4 / 8 | 1 | 11 | operator | 0.004212 [0.004119,0.004523] | 0.004122 | 101.05 |
| 16 / 32 | 1 | 23 | direct | 0.012812 [0.012398,0.012927] | 0.012650 | 102.11 |
| 16 / 32 | 1 | 23 | operator | 0.143861 [0.082836,0.177057] | 0.143839 | 99.62 |
| 16 / 32 | 4 | 23 | direct | 0.083995 [0.083267,0.084729] | 0.083924 | 101.61 |
| 16 / 32 | 4 | 23 | operator | 0.237593 [0.235001,0.237738] | 0.237562 | 100.11 |
| 32 / 64 | 4 | 29 | direct | 0.064983 [0.064919,0.167046] | 0.064911 | 102.37 |
| 32 / 64 | 4 | 29 | operator | 1.125016 [1.099140,1.208927] | 1.124672 | 101.55 |

controlled64・4 occurrenceの観測operator/direct wall中央値比は約17.31。controlled8ではoperatorの方が速かった。
小dimensionでdirect overheadが勝つ場合があり、比較の観測速度比を実H4や一般回路の速度保証へ移送しない。

人工m12はsystem2/controlled4・4096 occurrence・各1 gateの独立技術検証で、実H4の負荷ではない。
結果とcos/sin(1024)のArb384包含確認は`technical/artificial_m12_dyadic.json`。4096/4096 occurrenceを完了し、wall 3.02348秒、CPU 3.02149秒、process peak RSS 425324 KiBだった。
初回はexact Fractionの分母が増大し、信号endpointの文字列化がPythonの4300桁上限で失敗した。初回の数値partial自体はprocess終了で残っておらず、失敗位置・例外・この欠落を`technical/artificial_m12_initial_failure.json`へ記録した。
各literal段の半径をArb192で外向きdyadicへ包み直して修正し、別の人工出力へ検証した。精度低下やinterpreter全体の桁数制限変更はしていない。新しい人工検証は科学retryではなく、旧科学STOP/契約を消費していない。

## m12までの条件付き計算量

D=512、system d=256、native gate数G、literal block寸法b_j、n=2^m、unique word数Uとする。

| 費用 | Direct | 既存全operator列法 |
|---|---|---|
| Native 1 occurrence | O(GD)。現実装のinverse診断を含め約3本のgate伝播 | O(GD²)、全D列 |
| Literal state作用 | O(Σb_j²)。prepared block cacheから一状態に作用 | 各列に作用し、operator上界へ集計 |
| Reference準備 | 既存word producerのmatrix積/指数など＋新norm準備O(Σb_j³)。U件 | 原理的には同じproducerを再利用可 |
| Round/sample | n個の順序付き作用＋head/readout各一回、native raw/P0準備は別費用 | 全列のn作用、またはkernel証明後の保守的な伝播 |
| 全round校正 | Σ_m N_m 2^m のoccurrence、paired X/Yは同じ列をclassical再利用 | Kernel法はN個のkernel証明を全roundへ移送できるが保証の緩さが異なる |

m12は4096 occurrence、131072 event、262144個のuint53消費/sample。
全13roundを各1 sampleなら8191 occurrenceであり、実際には未採択のN_m倍が必要。
Raw build/P0 compileは既存fenced kernel政策なら原則occurrenceごとに一回ずつ、shellは同じSOURCE/recordを共有できる。
Literal head/atom、layout、gate値、shell、referenceは同一identity条件で再利用できる。別標本のword出現回数は保持し、cache hitを標本削除にしない。

既存Stage P後続は32 workersでwall約1667.16秒・child CPU約51607.27秒と[保存報告](research_c_stage_p_completion_execution_20261104.md)に記録される。
これは全512列inverse等を含む特定の15-call workloadで、direct一sample秒数へ割り算して流用しない。
実H4のG、fresh compiler費、unique-word reference生成、oracle資格、long-round interval幅の支配を新測定していないため、実秒数の確定見積りは出せない。
数学的なD列から一状態への削減はあるが、512倍の実速度向上を主張しない。

正規化oracle・native unitary・||T(w)||≤aの人口上界があれば、各round軸差のsupportは `±(1+a^n)`。
Stage Rのkernel range `1+a`を全round共通のrangeへ転用しない。非unitary literalへ無条件の±2は使わない。
W_m=2(1+a^n)、軸平均の許容幅ε_mとδ_mを固定するなら、既存paired Hoeffdingの一例は
`N_m ≥ W_m² log(4/δ_m)/(2 ε_m²)`。Complex目標は二軸rectangleから扱い、全roundのδ配分は別途固定する。
これは計画式で、N、δ_cal、δ_compile、failure/abort許容値を採択していない。
科学logical callsはsample countだけでなくstate/reference/build/compile/評価/summaryの数え方を実runnerへ固定する必要があり、現時点はnull。

## Fresh校正への接続と未資格事項

`fresh sample → 全occurrenceのword/SOURCE/整数記録 → raw/compiled/actual-inverse/target照合 → state/shell伝播 → X/Y signed区間 → paired_round_upper`を人工データで接続した。
`raw_signals`は点値へ潰さず、real/imagそれぞれexact rational endpoint pairを持つ新schema。
旧fresh direct入口はper-word Full512証明とscalar raw signalを要求するため、そのままでは使えない。
専用`calibration_payload/evaluate_calibration`が新資格を確認して**統計だけ**既存`c_iid_grid_expected_bias.paired_round_upper/failure_accounting`へ委譲する。
旧統合adapterのbinding/target語彙とBの標本保存を再利用し、Full512証明を捏造しない。

全planned sample、全occurrence、同じstate/target/round/paired X-Y、Arb192/丸め、shellと二脚差の整合を要求する。
数値包囲のconfidenceはnull。統計confidenceは別に条件付きで生成する。
実PCG64 replayはIID保証ではなく、`PCG64_joint_stream_qualified=false`のまま。
全word compiler成功、実H4 state/reference、人口support、failure/abort政策、N/confidenceは未資格・未採択。
新入口もall-q compiler-domain前提または別途採択/実装するfailure-aware estimatorを必要とし、後者を勝手に補完しない。
Compiler失敗wordの削除・再抽出、成功部分集合のみの推定、未訪問標本の0扱い、Stage Cへの昇格を拒否する。

数値primitiveの意味は[python-flint公式acb資料](https://python-flint.readthedocs.io/en/latest/acb.html)にも照合した。Webの最新版は0.9.0であり、今回実行した固定版0.8.0/FLINT3.3.1の代替証拠には使わない。

## 次の最小科学検証案とSTOP

[未承認・非実行の契約案](research_c_direct_round_signal_contract_unapproved_20261105.json)を準備した。
候補は**Stage Pに既にある固定1-word、m0、一つの明示状態**のsaved-record direct比較。
新sampling/build/compile、Stage P・7-call再実行、期待bias校正ではない。
まずoracle状態の距離/参照政策とsector embeddingを資格付け、同じliteral target・compiled shell・forward/inverseを使う。
Arb192、fresh output、failure/partial保存、retry/resume0、mandatory STOPを固定する。
状態資格取得の方法・logical calls・新scientific SOURCE/input inventoryは未確定のためnullで、実行可能な契約ではない。
旧SOURCEを新計算のSOURCEへ付け替えず、旧grant/残callを転用しない。

次担当は**ユーザー／研究C**：この最小検証のoracle/target条件と実施可否を決める。
その具体的依頼後、**Codex**：実runner、入力path/hash、必要calls、資源計画を固定して提示する。
新科学実行と重要科学レビューにはそれぞれ別の明示許可が必要で、今回は進まない。

今回のtechnical SOURCE、人工raw、テスト、保全確認とSTOPをcommit/pushし、remote SHA/tree・公開本文・主要blobを照合する。
公開記録は同branchの別技術commitへ保存する。
**実H4期待bias、全law保証、RPE accuracy、actual CAは認定せず、技術作業と公開照合の完了でMANDATORY STOP。**
