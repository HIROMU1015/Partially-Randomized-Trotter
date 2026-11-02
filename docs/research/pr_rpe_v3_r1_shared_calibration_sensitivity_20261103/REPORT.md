# RPE V3-R1：共有較正・長q移送の固定18round費用再解析

2026-11-03 JST / LOCAL / 主エージェントのみ。観測開始2026-11-02 20:58 UTC（11-03 05:58 JST）。ユーザーが明示許可したR1の保存価格再解析を実施した。

**元のPartial／Full費用を完全一致で再現した。約4.858%のPartial点選好は変わらない。共有較正を集約したsampling評価では、元SOURCEの扱いや明示的独立性の参考条件なら条件付き分離するが、共分散未知を許す幅では0をまたぐ。短q残差とsamplingを合わせる場合、保存5%／25%の場合も未確定。真の長q systematic幅が未保証のため、総合結論は `POINT_PREFERENCE: Partial` と `UNDETERMINED` である。**

全価格でPartial上振れ・Full下振れが各約2.490%、または後半3roundだけで各約2.858%あれば点順位は反転する。これは固定planの条件付き感度で、新しい実測ではない。R3局所補完の寄与は小さく、R2中間q holdoutにも最終長qの誤差幅を保証する根拠がないため、**今はR2/R3を費用順位決定目的で実行しないことを推奨する。**

## 成果物と固定条件

- [R1_FIXED_PLAN_RESULTS.csv](R1_FIXED_PLAN_RESULTS.csv)：Partial／Full各18round×両軸の72行。q/r/K/N、保存単価と再現値、共有group、個別・集約係数、価格SE、固定結果selector。
- [DEPENDENCE_AND_SENSITIVITY.json](DEPENDENCE_AND_SENSITIVITY.json)：14軸group、28anchor変数、既知／未知共分散、仮定別sampling幅、systematic帯、反転条件、追加価格の情報価値。
- [MINIMAL_NEXT_ACTION.md](MINIMAL_NEXT_ACTION.md)：R2/R3の非推奨理由、将来取得する情報の役割、STOPと必要な別承認。
- [VALIDATION.json](VALIDATION.json)：50固定入力、9定義の変更なしAST再利用、再現精度、13検査、原本保全、実行scope。
- [解析コード](analyze_fixed_plan.py)、[人工・整合性検査](test_fixed_plan.py)。

対象はlinear H4／隣接1.0 Å／STO-3G／legacy DF rank12／8 system qubit。Partial L_D3、Full L_D12、δ=.02、m0–17、q1–131072。保存coherent opt2のr/K、N/α/βを固定した。controlled partial-S2、cosine/sine full Hadamard interrogation、状態準備なしcompiled RZ数。Qiskit1.3.0、opt2、basis rz/sx/x/cx、transpiler seed17、backend/coupling/layout/routingの保存nullを保持する。新しい候補・compiler・物理条件は追加していない。[元M06-F][OPT2JSON]、[V3 preflight][V3REPORT]

V1/V2の物理平均・入力状態・信号半径・外部alias・IIDに関する条件付き解釈はそのまま。物理信号、branch、成功率、energyの再計算はしていない。状態準備は未測定で、shot数Partial13588／Full11162が異なるため共通準備費用も一般には相殺しない。最終総資源優位、実backend/noise、他分子/rank、一般最適性は今回認定していない。[V1][V1REPORT]、[V2][V2REPORT]

## 1. 保存点費用と共有較正の再現

各回路族・軸の保存q1/2平均から、元の `affine_prediction_with_standard_error`、`AffineAxisCostModel`、`_axis_model`／`_pair_model` を変更せず用いた。科学module全体やfull runnerはimportせず、必要な固定AST定義だけを取り出した。

\[
\widehat C(q)=(2-q)\overline C_1+(q-1)\overline C_2,\quad
\widehat G=\sum_{m,a}N_{m,a}\widehat C_{m,a}(q_m).
\]

| 結果 | Partial | Full |
|---|---:|---:|
| 保存値 | 1263314146772.9375 | 1327822299806.0 |
| 固定N・較正値による再現 | 1263314146772.9375 | 1327822299806.0 |
| Decimalによる別の点和 | 1263314146772.93750 | 1327822299806.0 |

全72軸のモデル単価と単価SEの差は0、元 `_comparison` のsampling／model幅／保存4scenarioも一致した。D=G_P−G_Fは **−64508153033.0625 RZ**、点比は `.9514180827942961`。新最適化は行っていない。[元SOURCE][OPT2SRC]

選択価格モデルと保存直接証拠の新たな重要不整合は見つからず、`MODEL_INCONSISTENT` の新判定は出していない。共分散・長q幅の不足は `INSUFFICIENT_EVIDENCE` として区別し、既存scientific statusや正式claimを変更していない。

共有groupごとに `A1=ΣN(2−q)`、`A2=ΣN(q−1)` を整数で集約し、同じ二つの較正平均へ係数をまとめた。7回路族×2軸＝14groupで、q1/q2のanchor変数は28。28の独立実験という意味ではない。以下の係数は**各軸ごと**で、今回の保存配分ではcos/sinで一致する。

| 回路族 | 最終round q | A1／軸 | A2／軸 |
|---|---|---:|---:|
| Partial r1/K2 | 1–128 | -95320 | 98886 |
| Partial r2/K2 | 256,512 | -275034 | 275757 |
| Partial r4/K2 | 1024 | -342370 | 342705 |
| Partial r8/K2 | 2048,4096 | -1907492 | 1908114 |
| Partial r16/K2 | 8192,16384 | -7158650 | 7159229 |
| Partial r32/K2 | 32768,65536,131072 | -81885294 | 81886263 |
| Full r0/K0 | 1–131072 | -42810680 | 42816261 |

group別 `A1*C1+A2*C2` の和も両候補の保存費用へ完全一致。CSVのgroup集約値と元round両軸費用は複数行に反復するので、列名で重複加算を禁止している。

## 2. 確認できた依存とUNKNOWN

| 関係 | 保存証拠で確認できること | 未確認部分 |
|---|---|---|
| 同じanchorを複数roundで再使用 | 同じ回路族・軸・policyのq1/q2が各roundへ線形に入る。集約係数は上表 | roundごとに独立価格を取得したとは扱えない |
| q1/q2 | 異なるtask seedとSOURCEの独立較正前提。manifestのseed/n/cohortをgroup欄へ対応 | 数値共分散はUNKNOWN。異なるseedだけからすべての相関を実測済みとしない |
| cos/sin | 同trajectoryのpaired価格。保存mean/SEも両軸で一致 | 個別価格vectorがなく、共分散値や完全相関を確定しない |
| selected／full-basis policy | 同trajectory上のpaired比較。初期とfresh置換の区分を維持 | 二つの独立標本集合ではない。Partialのfull-basis policyはL_D12のFull候補とは別 |
| r・cohort | r1/2/4は較正各n8、r8/16/32はfresh各n32。initial失敗groupは置換でn40にpoolしない | group間の数値共分散はUNKNOWN |
| Partial／Full | H/compiler/wrapperの条件を共有。Fullは固定決定論価格で保存sampling SE0 | compiler/model/systematicの候補間相関はUNKNOWN。共通条件から完全相殺を仮定しない |

既知LOCALのinitial／fresh32 runごとにtasks/checkpoints/worker_resultsの計6directoryだけを確認し、すべて未発見。固定aggregateはtask/result fingerprint索引で、必要なpaired個別価格vectorを保持していない。legacy、M08、A0の再引用を追加独立標本として数えず、raw全探索・再生成・他SERVER照会は行わなかった。[保存監査][AUDIT]、[較正の静的定義][PRESRC]

## 3. Samplingだけを切り出した評価

費用差Dのanchor重みをw、共分散をΣとするとVar(D)=wᵀΣw。同じanchorへの重みを集約してから評価した。表の幅は**保存SEをplug-inして1.96倍する診断**であり、選抜biasやmodel誤差を含む厳密95% CIではない。systematicを外した比較は成分診断に限り、未知systematicを0として総費用優位を証明するものではない。

| Samplingの扱い | SEまたはscale | Dの1.96幅による診断区間 | R1判定 |
|---|---:|---|---|
| 元SOURCE：q1/q2独立SE、N×SEをround/軸で保守加算 | 29863541590.16444 | [-123040694549.7848,-5975611516.340195] | CONDITIONAL_SEPARATION |
| 共有roundを集約し全anchorを独立とする参考条件 | 19616387096.71714 | [-102956271742.6281,-26060034323.496902] | CONDITIONAL_SEPARATION、未採択の独立性 |
| cos/sin完全相関、q1/q2独立、回路族独立の参考条件 | 27741760676.937958 | [-118882003959.8609,-10134302106.264107] | CONDITIONAL_SEPARATION、共分散は未推定 |
| 保存対角のみ固定、任意PSD共分散を許す上限 | 41031613084.75388 | [-144930114679.1801,15913808613.055115] | UNDETERMINED |

元SOURCEは既にround/軸横断の保守和を用いている。共有較正を明示したことは、元処理がround独立を誤採用していた、あるいはSOURCEが不整合だったという発見ではない。

PSD範囲はb_i=w_i*SE_iについて `max(0,2*max|b_i|−Σ|b_i|) ≤ SD ≤ Σ|b_i|` というGram-vectorの三角不等式による対角固定の感度範囲。今回下端0、上端41031613084.75388。下端0は完全打消しを観測した結果ではなく、許した行列集合の数学的端点。任意PSDのすべてが実際のcompiler標本に実現すると主張していないし、保存SE自体の母分散保証でもない。

cos/sin相関とq1/q2相関をそれぞれ−1/0/+1に置く9つのPSD Kronecker参考scenarioもJSONに分けた。例えばcos/sin=+1・q1/q2=−1ではDの幅が0をまたぐ。これは未知相関を推定した値ではなく、符号付きanchor係数と依存仮定により結論が変わる例である。

Fullの保存sampling対角が0なので、これを固定するPSD sampling modelではFullとの交差共分散も0となる。一方、Full長qのmodel errorや候補間systematic相関は0と認定していない。元q1/q2独立前提を保持した条件と、その前提を緩める感度を混同しない。

## 4. 長q systematicを加えた結果

Partial各rのq1/2/8、Fullのq1/2/16/32、A0 r32のq16/32という直接域はV3から変わらない。最終q16/32はr1であり、A0 r32の直接価格をそこへ流用していない。q>32は未検証のまま。[A0][A0]、[V3価格対応][V3MAP]

| Systematicの参考帯 | 元SOURCE sampling幅も加えたD区間 | 判定 |
|---|---|---|
| 元per-r selected q8残差 | [-176299478307.3865,47283172241.261505] | UNDETERMINED |
| 元per-r full-basis補助残差 | [-183677443353.19266,54661137287.06766] | UNDETERMINED |
| r32だけq8/A0 selected最大残差へ置換する新参考scenario | [-181165627089.8523,52149321023.727295] | UNDETERMINED |
| 保存local5% | [-252597516878.7317,123581210812.60669] | UNDETERMINED |
| 保存transfer25% | [-770824806194.5192,641808500128.3942] | UNDETERMINED |
| 真の長q幅を未知とする | 有限の保証幅なし、JSON null | INSUFFICIENT_EVIDENCE／総合UNDETERMINED |

統計SE、経験的holdout残差、想定5%／25%移送幅は別の量。表は保存手順に沿いsampling診断幅へ仮定したmodel帯を加えたもので、各行を同じ信頼水準のCIとして比較しない。全anchor独立の参考samplingへ変えても、これら全model帯との組合せは未確定だった。

A0を使う行は**R1の追加感度**として隔離し、元coherent結果を更新していない。A0の誤差は `|prediction−direct|/direct`、保存per-r scenarioはその値をモデル費用へのfractionとして移す規約である。この規約を同じく使った参考帯で、直接価格に対する厳密な上下変換や一様長q上界ではない。A0 selected q32の誤差約4.911%、標準化残差約2.012、full-basis q32約5.598%の非gating FAILも変更していない。

## 5. 点順位が維持・反転する条件

新候補を探さず、保存価格に対して指定subsetだけに仮想的fractionを加えた。Partial上振れ・Full下振れというadverse方向でDが0となる点は次のとおり。他subsetの価格は固定。

| 変化する価格 | 各候補に置くfractionの点反転閾値 |
|---|---:|
| 全roundでPartial↑・Full↓ | 2.489569899656664%ずつ |
| Partial全roundだけ↑ | 5.106263806025194% |
| Full全roundだけ↓ | 4.858191720570395% |
| 後半3roundでPartial↑・Full↓ | 2.8581115607456794%ずつ |
| Partial後半3roundだけ↑ | 5.587134848911367% |
| Full後半3roundだけ↓ | 5.851417528156006% |

閾値で同点、それより大きいadverse変化でFull点選好、それより小さければPartial点選好。これらはsamplingを加えない固定点のbreak-evenである。例えば全round各2.5%adverseなら反転し得るが、後半3roundだけ各2.5%ならその点閾値未満。JSONに0/1/2.5/5/25%の例を保存した。

両候補全価格に**完全に同一fraction**eが掛かる仮定ならD_true=(1+e)D_modelなのでe>−1で符号は維持される。しかし同じcompilerという事実はこの完全共有fractionを保証しない。

元SOURCE sampling幅を残し、同じ対称adverse帯でもなお診断が0から離れる条件は、全round各 **0.23061740049350778%未満**、後半3roundだけなら各 **0.26475667887473717%未満**。後者は他roundのsystematicを0と置く参考条件であり、新価格取得のSE要件や実現済み一様上界ではない。共分散未知を許すPSD上限ではsamplingだけで0をまたぐため、同じ形式の正の許容帯は得られずnullと記録した。

## 6. R2/R3の情報価値

**R3は現在の費用比較目的では非推奨。** q16/32・r1の二つのround費用は185863281 RZ、Partial総費用の約0.0147124%。5%変化は9293164.05 RZで、点差約6.45e10 RZには届かない。両点だけでなくr1族全体（937506875.5 RZ）へ変化を波及させる参考評価も行い、点反転にはfraction68.80819193849474という大きな上乗せが必要だった。無上限の未知誤差に対する保証ではないが、5%／25%程度の局所補完を次の順位判別へ選ぶ根拠は弱い。

**R2も順位決定の次実行としては現時点で非推奨。** r32／Full後半価格が重要なのは確かだが、中間q64/128等の平均を取得しても最終q32768–131072の誤差が0.2648%以内という一様な条件は検証できない。A0 q32の保存予測較正相対SEだけでも約2.3054%で、単に新holdoutの直接SEを小さくして未知systematicを解消したとはしない。

R2には、選んだqでaffine式が破れるかを調べる**反証目的**の情報価値はある。その目的をユーザーが別途選ぶ場合だけ限定候補を再提案できるが、順位決定や全長q保証とは別である。最大qまでの直接compile、51task再compile、機械的n増加は推奨していない。必要な追加情報と再利用コード・既存STOP変更の条件は[MINIMAL_NEXT_ACTION](MINIMAL_NEXT_ACTION.md)を参照。

## 7. 実装・検査・数値精度

固定PRESRCのaffine関数、ABLATIONSRCの2 dataclass、DECISIONSRCの3定義、OPT2SRCの3定義、計9定義をAST変更なしで限定再利用した。元SOURCEの `_measured_model_half_width`／`_comparison` は保存scenario再現に使用。追加したのは共有係数、wᵀΣwの仮定別評価、対角固定PSD感度、指定subsetのbreak-evenだけの薄い処理である。既存full analysis／`_grid_search`／Hamiltonian準備／runnerは起動しなかった。[PRESRC][PRESRC]、[ABLATIONSRC][ABLATIONSRC]、[DECISIONSRC][DECISIONSRC]、[OPT2SRC][OPT2SRC]

Python標準ライブラリ、float64のfsum/hypot、整数anchor係数、別のDecimal点和を用いた。数値処理は72軸・28anchorの小さい代数演算であり、metadata読取を並列化し、重い科学workload・恣意的資源capは置いていない。GPUや共有環境の変更は不要だった。1.96幅を外向き保証する区間計算や、選抜bias補正を実施したとはしない。

13検査PASS。人工の共有係数、独立round誤認、PSD端点、paired符号、同点境界、保存点とDecimal、全軸、SEの別精度和、元scenario一致、反転前後、UNKNOWN保持、fresh置換／A0条件、科学runner非importを確認した。主エージェント自身の技術検査であり、独立した科学レビューではない。

今回実行した科学的処理は許可済み固定保存価格の再解析だけ。新sampling／RNG replay、回路build/compile/transpile、H/state/operator/reference/signal生成、新しいaffine fit、N/α/β/schedule再最適化、V1/V2再実行、詳細科学レビュー、主要claim採択は0。

## 8. 固定入力・原本保全・公開とSTOP

V3固定commitは `cc60beb51930d4311d2ded84440682e583ccafc7`。M06-F／A0／SOURCEは固定 `4591b8d75a2a6a55c74a51225f0c61648023c8ee`。V1 `e520e6e396f7e0def9e522ea1cff58eb10ddfa8f`、V2 `78c49d0b5f1d71e63b5959e78813237923bbc648` の後続条件を維持。50固定ファイルのbyte/blob/SHA-256を照合した。carrier commitを元実行のclean SOURCE commitとは呼ばず、initial/fresh32のdirty provenance、raw未発見、外部再現未確認も維持する。

LOCALのAGENTS.md／PROJECT_MAP.md、出力親のAGENTS.md／PROJECT_MAPの入口・M06-F部分、正式共通運用指示2026-10-25の§8.1・§8.4・§10を確認し、前段の読了済み固定研究概要・規範資料も再利用した。出力親の旧AGENTSとLOCAL現行指示は同一版ではなく、未取得資料を読了と扱っていない。sub-agentは使用していない。

出力は独立Git作業領域・新branch `codex/pr-rpe-v3-r1-shared-calibration-sensitivity-20261103`、親V3固定commitに限定した。元treeではGIT_OPTIONAL_LOCKS=0／GIT_NO_LAZY_FETCH=1で固定objectとmetadataだけを読み、原SOURCE/input/raw/QPY/result/grant/ledger/marker/STOP、他研究branch/worktree/archiveを変更しない。SSH・他SERVER探索・fetch/prune・cleanup、他jobや共有設定の操作は行っていない。21:07 UTCの観測では元HEAD、dirty/untracked status全行、研究refsは開始時と一致した。その後21:11 UTCに別の整理結果TSV一件の追加を観測し、元HEAD・開始時status全行・研究refsの保持を確認したため、元status全体が不変とは報告しない。この追加に変更・停止・復元操作は行っていない。最終観測はVALIDATIONと最終チャットに記録する。

新成果物に内部host/IP/絶対path/秘密情報/非公開rawは含めない。固定公開URL、相対path、hashと今回の匿名化解析結果だけを新規公開する。専用branchへ7成果物をcommit/pushし、remote SHAと公開全本文を照合する。結果を未照合で公開済みと報告しない。

今回のR1は共分散・長q幅が未解決でも完了できる。**次はユーザーが、現状の条件付き点選好とundeterminedで止めるか、別目的で追加情報を追究するかを判断する。Codexは新指示がある範囲だけ担当する。既存M06-F STOPを維持し、公開・照合後mandatory STOP。R2/R3、旧STOP変更、モデル再較正、詳細科学レビュー、中心claim採択へ自動移行しない。**

## 固定参照

[A0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/m06f_a0_proxy_lineage_reconciliation_v1.json
[ABLATIONSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_ablation.py
[AUDIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_fresh32_audit_20260925_065827.json
[DECISIONSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_decision_cost.py
[OPT2JSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json
[OPT2SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_opt2.py
[PRESRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_prevalidation.py
[V1REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md
[V2REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md
[V3MAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc60beb51930d4311d2ded84440682e583ccafc7/docs/research/pr_rpe_v3_cost_transfer_preflight_20261103/PRICE_EVIDENCE_MAP.csv
[V3REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc60beb51930d4311d2ded84440682e583ccafc7/docs/research/pr_rpe_v3_cost_transfer_preflight_20261103/REPORT.md
