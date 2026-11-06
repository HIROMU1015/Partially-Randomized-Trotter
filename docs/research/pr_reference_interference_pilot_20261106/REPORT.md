# 新A：PR × reference干渉によるcontrolled evolution削減 — 限定pilot

**理想的なpath単位の信号復元は成立した。強baseline後も短roundのCX・depth削減は観測されたが、RZ・sizeは増え、全18roundの費用改善は未確定。**

A-1／A-2を通過し、A-3の登録6セル・48 paired realizationを完了した。A-4は保存shot scheduleと取得済み3roundの費用を接続した。全18round価格と新回路の全support誤差上界は未取得。科学作業を完了し[mandatory STOP](../../../artifacts/pr_reference_interference_pilot/2026-11-06/STOP.json)を記録した。結果に応じたPF・law・配置・分子の追加探索は行っていない。

専用branch：`codex/pr-reference-interference-pilot-20261106`。2026-11-06、SERVER 2 `GPUA100-gleap02`。ユーザーのA-1〜A-4の条件付き実行許可の範囲で実施した。正式な科学レビュー・新規性認定ではない。

## 条件・SOURCE・実行範囲

[固定scope](SCOPE.md)。基準は新A/B静的調査[2397c5ac][static]、科学SOURCE[602ca4b][fixed]。Linear H4、1.00 Å、STO-3G、DF12、Partial S2、LD3、h=.02、finite RTE K=2を再利用した。元H groundを各shotで供給する仮定で、その準備費を除外する。追加referenceの準備費は除外しない。

保存common preparationを独立領域へsnapshotし、固定commitの公表済みcommon arrays、元DF入力、保存groundと一致させた。[INPUTS.json](../../../artifacts/pr_reference_interference_pilot/2026-11-06/INPUTS.json)。SCF、DF fit、ground solveは0件。元Hの連続積分表現からの新しい厳密包含は認定していない。

primaryは既存Track C SOURCEのordinary `diag(I,V)`であり、独立な2 random branchの異なるfinite meanへ置換していない。独立2分枝については、同じjoint realizationの`V0† V1`と両reference位相差を人工例で検証した。片方を同じ乱数に変更していない。

## A-1：reference作用と位相

登録physical basisはRZとXXPlusYYで、粒子数を保存する。Z/ZZとそのrotation、scalar、actual inverseもvacuumを位相まで保存する。このfamilyでN=0 sectorは1次元であり、任意の順序付き有限積に同じ性質がある。genericなDF registry／Pauli辞書全般への主張ではない。

- diagonal primitiveのvacuum位相：`global_phase − Σθ_Z/2 − Σθ_ZZ/2`。
- `P=s B Z/ZZ B†`のvacuum固有値はs。paired Taylor eventの位相は、`π(n/2) + π×負product数 − signed_rotation_angle`（mod 2π）。
- 完全trajectoryではD、event、constant、extracted identityの位相を加える。未知のE₀、gap、ground vectorは位相計算に使わない。物理Hのvacuum energyだけで代用しない。
- actual adjointは同じgate列をreverseして各gateをinverseにする。inverse側の独立再samplingはしない。

[family／trajectory 19チェック](../../../artifacts/pr_reference_interference_pilot/2026-11-06/A1_REFERENCE_PHASE.json)の最大vacuum作用差は8.83e-15。さらに[符号・identity・負時間の16ケース](../../../artifacts/pr_reference_interference_pilot/2026-11-06/A1_SIGN_IDENTITY_NEGATIVE_TIME.json)を確認した。負のRTE時間は人工factor検査であり、物理PF条件を変更する探索ではない。

## A-2：実回路・X/Y・strong baselineのcorrectness

[SE-QPE v2][se] Fig.1の**2つのCSWAP cascade**を実装した。n=8ではCSWAPは合計16個。F、Gの共通固有基底はfresh-shotの単一干渉blockの計算に使わない。canonical QPEの連続gadget保証とは区別する。

targetをlow n bit、referenceを次のn bit、ancillaを最後に配置し、F†とGを別laneで実行する。YはSdg→H測定である。2分枝の内積は

\[
z_{\rm ref}=e^{i\vartheta}\langle\psi|FG|\psi\rangle
                 \overline{\langle0|FG|0\rangle}.
\]

`ϑ=φ(F)+φ(G)`でpathごとの信号を復元する。scalarをuncontrolled回路のglobal phaseとして落としたままにはしない。この位相はreferenceとの積で一度相殺されるため、ancillaで一度だけ復元する。

[人工例](../../../artifacts/pr_reference_interference_pilot/2026-11-06/A2_SYNTHETIC.json)では、非可換F/G、非固有target、正負angle・scalar、actual adjoint、独立2 branch、X/Y、5qubit全operatorとnative loweringを確認した。全operatorの比較で許すのはjoint全体の共通phaseだけで、branch別phaseはfitしない。reference条件を破るRX反例ではoverlap 0.93937、phase-only復元の信号差0.05695を検出し、family guardが拒否した。

実H4ではq1の通常・rare各1 realizationを9qubit直接制御／17qubitreferenceで検証した。新referenceのnative X/Y信号差は最大6.97e-14。全2-register巨大operatorは作らず、statevectorを用いた。加えて既存の[repeated-S2 builder](../../../src/trotterlib/df_partial_s2_repeated.py)の境界融合を再利用し、q2/q8・両stratumの9qubit全operator、F/Gの積、vacuum作用を確認した。最大のnative全operator差は共通phase補正後1.49e-11だった。

許容値は人工1e-11、H4診断1e-9。これらは**局所数値correctnessの診断**であり、全RPEのdeterministic disk上界ではない。新CSWAP/register構成へ旧compiler Aの誤差保証を無条件に移していない。

## A-3：強baselineと限定native価格

同じsampled events、gate set `rz/sx/x/cx`、compiler Aのdefinition-based lowering、全結合profile、X/Y wrapperでpaired比較した。旧opt2価格関数は呼ばない。

| 方式 | 実装・資格 |
|---|---|
| 既存直接制御 | basis非制御、diagonal中央制御、RTE内のbasis共有、既存equal-boundary D融合、scalar集約 |
| Simon–Love型の強baseline | 左D非制御、右D方向制御。control0では左Dのactual inverseとなって相殺し、control1では同じRTEを挟むS2。既存境界融合も適用。全operatorで既存直接制御との一致を検証 |
| reference | 全trajectoryを固定の時系列境界でF/Gへ分割。両lane内に同じ既存D融合を適用。追加8qubit、16 CSWAP、path位相補正、8 reference resetを含む |

[Simon–Love][sl]の対称Trotter制御を強対照に取り入れたもので、この最適化の発明は主張しない。適用できないPauli／random sequenceへ対称性を仮定しない。独立PR 2 branchのmodeをordinary信号に仮想適用した価格とも比較しない。PFはS2で固定し、Full Y4等へ探索を広げていない。

q=1,2,8、r=1、K=2。通常／少なくとも1つのorder2 eventを含むrare stratumを各8標本、計48 realizationとした。canonical lawのstratum確率で期待費用を加重し、両軸の対応・paired covarianceを保持した。条件付き価格samplingはNumPyの保存seed規約であり、実際の量子shotのfinite-bit samplerとは区別する。

以下は**X軸、1 shotのMC平均**（表示を丸めた値）。Y軸を含む全値・paired MC SEは[A3_PRICES.json](../../../artifacts/pr_reference_interference_pilot/2026-11-06/A3_PRICES.json)。

| q | 方式 | RZ | CX | depth | size |
|---:|---|---:|---:|---:|---:|
| 1 | Simon–Love | 7,829.535 | 1,280.504 | 2,676.385 | 11,590.551 |
| 1 | reference | 7,978.534 | 1,290.504 | 1,444.259 | 11,785.549 |
| 2 | Simon–Love | 15,012.568 | 2,491.508 | 5,131.269 | 22,253.599 |
| 2 | reference | 15,798.566 | 2,455.507 | 2,514.267 | 23,251.595 |
| 8 | Simon–Love | 58,034.452 | 9,749.051 | 19,863.485 | 86,121.652 |
| 8 | reference | 58,610.444 | 9,005.048 | 9,091.218 | 86,177.638 |

q8のreference−Simon–Love差は、RZ **+575.993**、CX **−744.002**、depth **−10,772.268**、size **+55.986**。depthのpaired MC SEは5.821。いずれの削減も既存直接制御との比較だけに依存せず、Simon–Love後に残った。q1ではCXも増えており、全資源での一様な改善ではない。

referenceは17qubit、対照は9qubit。referenceの8 resetはnative unitary/readoutのcompile後にprefixし、実際の合成回路のdepth/sizeへ計上した。理想vacuum準備に追加rotationは不要。追加reference測定やpostselection、fan-outは用いていない。実接続topologyを指定しない元profileなのでrouting費の実機保証はない。Tは合成しておらず**未取得**。RZ数を総Tへ変換していない。

## A-4：RPEへの接続と未資格

[18round表・再会計](../../../artifacts/pr_reference_interference_pilot/2026-11-06/A4_RPE_ACCOUNTING.json)、[TSV](../../../artifacts/pr_reference_interference_pilot/2026-11-06/A4_ROUNDS.tsv)。

元の18round、energy精度1.5936e-4 Ha、全失敗率.05、位相予算.4 radを維持する。path単位の理想信号等価性と同じsampling lawの条件下では、保存された**8,598 shots**のscheduleを参照できる。成功規約は旧報告の固定保存Hermitian行列・exact ground・独立finite bitsモデルに限定する。保存raw diskはそのまま一度だけ参照し、normalizationを変えたり同じ誤差を二重加算したりしていない。

新native回路・位相補正の全support誤差上界は未取得で、各roundの新native disk、資格付きshot数は`null`。数値signal差をその上界として採用しない。新誤差が異なる場合には、同じraw信号単位でdiskを加算後、半径・残余phase・整数shotを再評価する必要がある。今回はそのための保証入力が不足しており、旧shot数を新回路の無条件保証へ格上げしていない。

価格が取得できたのはm=0,1,3（q1,2,8、r1）の3roundである。保存shotを掛けた**3roundだけの条件付きsubtotal**は次のとおり。

| 資源 | Simon–Love | reference |
|---|---:|---:|
| RZ | 31,173,243.023 | 31,754,615.136 |
| CX | 5,211,586.039 | 4,914,416.886 |
| shot加重depth | 10,665,444.235 | 5,029,860.851 |
| size | 46,239,688.027 | 46,719,408.679 |

これを全費用や実行可能な18round baseline価格とは扱わない。残る15roundにはq4とr>1・長qがあり、価格を外挿・0補完していない。全18roundのRZ/CX/depth/size、総Tは`null`。全roundを同じgadgetに置換する理想scheduleなら追加reference resetは68,784回となるが、これだけからnative総費用は決まらない。

判定は**全RPE費用改善：未確定（UNDETERMINED_ALL18_NATIVE_COST）**。新構成の全support native成功保証は未資格。短roundの数値的なCX/depth利益、理想信号等価性の成立条件、保証未取得を区別する。

## 成果物・再利用・検査

- [SOURCE](../../../src/trottertracks/pr_reference_interference_pilot.py)、[factor検査](../../../src/trottertracks/pr_reference_interference_factor_checks.py)、[再会計](../../../src/trottertracks/pr_reference_interference_accounting.py)、[runner](../../../scripts/run_pr_reference_interference_pilot.py)。
- [出力・provenance入口](../../../artifacts/pr_reference_interference_pilot/2026-11-06/README.md)。A1/A2、raw price、代表native QPY、入力snapshot、実行時SOURCE、環境、STOPを保存。
- 既存rte._make_event、DF builder、conditional sampler、repeated-S2の融合、compiler Aを再利用。新差分はreference gadget・vacuum phase追跡、適用可能な強制御lift、有限比較と再会計。既存pipelineに2register gadget／この対照がなかったため追加した。
- 当初のA1/A2 SOURCEと、既存境界融合を追加したA2/A3 SOURCEを保存。融合追加後には新しいq2/q8 correctnessを済ませてから価格を取得した。初回の不正な結果を採用したものではない。
- H4 correctnessは2独立task、融合確認は4、価格は48をprocess並列化した。worker数はtask数とOS affinityの小さい方。BLASの重複並列を避け、利用資源へ恣意的な恒常上限を置いていない。旧環境・他ユーザーを変更していない。
- これはprimary agentによるローカル検証で、第三者の独立レビュー／CI成功ではない。公開前の検査はJSON・リンク・diff・archive・保存byteの技術照合に限る。

既存Track C/C2、新B、P-D、BF/BM、PR-3等のSOURCE/raw/result/grant/ledger/marker/STOPは変更していない。既存研究の保留・STOPを解除していない。

## 終了と次の重要な判断

**承認された有限pilotは完了しmandatory STOP。** 追加合成、長round価格、最適化、別配置、別PF、分子、Pauli辞書、sampling lawへ自動進行しない。

**GPT**：このCX/depthとRZ/size/qubitのtrade-offに、追加の全support誤差資格・長round価格を調べる科学的価値があるかを判断する。ordinary信号での結果と独立2 branchのRPE modeを混同せず、より強い適用可能な制御対照が結論を変え得るかも判断対象とする。

**ユーザー**：継続／保留と、必要なら次の限定対象・資源指標・検証範囲・実行許可を決める。Codexの自動後続作業はない。

[static]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/2397c5ac511148423ae89f25f74f1de967d1d39a
[fixed]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/602ca4b67b1037e221e910a2733444630c5add62
[se]: https://arxiv.org/abs/2604.14921v2
[sl]: https://arxiv.org/html/2511.13855v1
