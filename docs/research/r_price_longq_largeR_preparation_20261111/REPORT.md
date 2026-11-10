# AのP1結果を反映した長q・大R価格検証の準備（2026-11-11）

**第一案は、AがNを取得したH4 m6・H6 m8のbaselineとR128、計4セルの有限価格検証（40 full-wrapper compile）とする。** 全6中時間候補や最終roundを一律に直接compileせず、R256は保留する。q/Rの誤差原因を分離する必要がある場合だけ、低q/R128の2セル（別16 compile）を追加検討する。これは未承認・未実行の提案であり、価格精度やRの採択を決めたものではない。

今回は既存コード/固定成果の読取、候補metadataと整数allocationの生成、保存Nの無変更copy、8件の技術テストだけ。新sampling、回路build/compile、実分子S計数、fit、実価格予測・残差/保存科学値の数値再解析、N/signal/phase/radius、実N×C/G・順位、詳細科学レビューは**全て0**。希少eventに関する追加調査・設計・検証は対象外で、専用実験を計画へ含めていない。

[EXECUTION_PLAN.json](EXECUTION_PLAN.json)／[HANDOFF](HANDOFF.md)／[metadata生成器](../../../scripts/plan_r_price_longq.py)／[テスト](../../../tests/check_r_price_longq_plan.py)／[8件の記録](TESTS.txt)／[保全・技術照合](CHECKS.json)。metadataコードSOURCEは`f0daeb96bf53b43ce4e6e1d61ed351a4f7c688b0`。既存C branch `codex/r-variable-compiled-price-design-20261111`を継続し、基点`2e6a0cdf5`の科学成果・STOPを保全した。

## 固定SOURCEと科学条件

| 成果/実装 | 固定commit・参照先 | 再利用 |
|---|---|---|
| A P1結果 | `71ba49bb28b295aa28e51101bc3b1bf07d500767`、[REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/71ba49bb28b295aa28e51101bc3b1bf07d500767/docs/research/pr_r_detailed_n_p1_results_20261111/REPORT.md)、`run01/N_HANDOFF.json` | 公開12候補のidentity・条件付きstatus・policyと整数Nをcopy。A科学SOURCE `a1a79213a9b5331145e20e8778a3be9d9fc4e497` |
| A固定計画 | `d33380a8c2fe68d0be1f2c5faaf7fc3b3a956b58`、`pr_r_detailed_n_preparation_20261111/EXECUTION_PLAN.json` | P1の候補条件・比較policy。未実行のP2/P3を完了扱いしない |
| C1結果 | `2e6a0cdf5a647058cc7afeeb5741bfafa24c974b`、[REPORT](../r_price_phase1_results_20261111/REPORT.md) | 8セルの役割・保存価格coverage・既存検証結果。今回残差を再評価しない |
| C1係数凍結 | `7eeabf0ae482ee610d667e28470692d209f2bb21`、[FROZEN_MODELS](../r_price_phase1_results_20261111/run01/FROZEN_MODELS.json) | 値を変更せず、将来の反証対象モデルとする。C1科学SOURCE `1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7` |
| B最新受入API | `724d313b9c8445763f0fe9c41feb573df28ba5d5`、[REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/724d313b9c8445763f0fe9c41feb573df28ba5d5/docs/research/rpe_r_price_intake_20261111/REPORT.md)、`rpe_r_price_intake.py` | canonical menu/request、固定R scope、有限cohort/未採用modelの区別。BのC1実結果取込みには介入しない |

Linear H4/H6、隣接1.00 Å、STO-3G、保存tol-only DF10/DF19・weighted Hermitian completion・元fragment順、保存H/state/snapshot/sectorとscalar一回会計を維持する。対象は保存DF-Hの同sector ground energy、保存数値stateの条件付き設定、P=0。H4はS2/δ=.1/PR prefix1/K2、H6は4次m5_best/δ1.6/PR prefix7/K2。PF対照H4 B0/p7・H6 Full19、H6 m0..3の別δ/p5は変更・拡張しない。

Qiskit1.3.0、`rz/sx/x/cx`、opt1、transpiler seed17、symmetric-directional制御、X/Y測定付きfull Hadamard wrapper。資源unitは**完成Hadamard回路1回当たりcompiled RZ count**。h=.1、13round、εE=.00159360010199040 Ha、各方式αtotal=.05、βtotal/stat=.4/.2、保存prior/decoderは転記のみ。新科学実行SOURCEはnull、将来のbounded runnerは別途固定が必要である。

## C1の範囲とA P1のcoverage

C1のH4はqPF8/16×R4/8、H6はqPF1/2×R16/32の各4セル。各系で既存Rの2セル＋新R/低qの1セルを校正、残る高q/新Rの1セルを独立検証した。保存の独立path相対RMSEはH4 q16/R8で1.039841%、H6 q2/R32で0.229959%（各n8）。今回この指標を再計算していない。3校正セル・3係数でlack-of-fit DOF=0、採用閾値なし、`qualified_for_candidate=false`を保持する。

以下のN_X=N_YはAの保存値のcopy。全12候補が`R_AWARE_EMPIRICAL_ACTION_COUNT_V1`によるideal条件付きPASSであり、最小N・actual実装の成功保証ではない。新しいNの比・費用比・価格許容幅は算出していない。

| 系/m | qPF | R | 保存N（各軸） | 同一条件の保存C | C1との関係・今回の案 |
|---|---:|---:|---:|---|---|
| H4/0 | 1 | 2 | 764 | 確認catalogに無し | q下側・R下側、未取得のまま保留 |
| H4/0 | 1 | 4 | 734 | 同上 | q下側。SHORT任意案 |
| H4/0 | 1 | 8 | 719 | 同上 | q下側、未取得のまま保留 |
| H4/6 | 64 | 4 | 9638 | 同上 | 固定Rの長q、第一案Q |
| H4/6 | 64 | 128 | 766 | 同上 | q/Rとも外側、第一案QR |
| H4/6 | 64 | 256 | 735 | 同上 | q/Rとも外側、保留 |
| H6/4 | 1 | 11 | 729 | 同上 | C1より小R・最低整数配分、SHORT任意案 |
| H6/4 | 1 | 16 | 721 | C1保存探索cohort | C1の保存校正点。direct/S/model予測が保存済み |
| H6/4 | 1 | 32 | 712 | C1新校正cohort | C1の新校正点。direct/S/model予測が保存済み |
| H6/8 | 16 | 16 | 1015 | 確認catalogに無し | 固定Rの長q、第一案Q |
| H6/8 | 16 | 128 | 735 | 同上 | q/Rとも外側、第一案QR |
| H6/8 | 16 | 256 | 720 | 同上 | q/Rとも外側、保留 |

「確認catalog」はC1と元H4/H6固定providerが読むV1/V2成果を指す。全履歴に他の価格が存在しないという断定ではない。H4 V1の保存PRはq8/R4、q16/R4は旧独立q結果、H6 V2のq1/2/4・R16も今回のq16価格ではない。別DF/旧DF12/Track Dのq価格を混ぜない。

P1とC1のexact候補重複はH6 m4/R16・R32の2点で、**どちらも校正点**。P1にC1の独立検証点は無い。保存価格が存在すること、式を計算できること、科学的採用資格があることは別である。旧H4 Mq/H6 Affineの数値APIやC1式を計算できても、新q/Rの保存予測や資格が生じるわけではない。C1 evaluator自体は固定8セル外を拒否する。将来の明示許可で評価範囲を限定した別SOURCEが必要で、今回は呼んでいない。

## なぜ長q/Rの検証が必要か

固定δではT=qPFδなので、T/q/δを独立なfit変数にしない。q増加はPF outer/piece/head-tail境界の数、R増加は各pieceのevent数とsigned short-time/finite sampling lawを変える。Rだけの比例補正や、qRが等しい別候補の同一価格扱いは不適切である。

H6は元順11 signed piece・負時間・min-one/largest-remainder整数allocationを保持する。R≥11で、2冪/11の倍数は必須ではない。R16は`[1,2,1,2,1,2,1,2,1,2,1]`、R128は`[10,11,10,12,10,22,10,12,10,11,10]`、R11は全1。配分の階段変化で各τ_j=λtail δ w_j/r_jが変わるため、全eventを同一IID列として平坦化しない。Σr_j=R、event数=qPF Rであり、11qPF Rではない。

C1のSは隣接inverse basis、literal Z-fold、head/tail・relative phase・scalarを数える。Qiskitの1Q再合成、CX界面等まで同一に再現する保証は無い。θ0+θq q+θL qRが長q/Rでも残差を表すか、整数allocation・piece/outer境界で別の項が必要かは未検証である。C1の小q RMSEから新domainの誤差上限や許容率を決めない。

## 既存コード・方法の再利用判断

以下のcode参照は、別記しない限りC1公開`2e6a0cdf5`の固定treeで読む。これは静的調査で、分子入力/科学backendはloadしていない。

| 実装・関数 | 再利用する部分 | 直接流用しない部分 |
|---|---|---|
| `r_price_phase1_science.structural`、H4 `h4_df10_price_v1.Accounting`、H6 `h6_df19_price_v1.Accounting`（H6は`ae44283d`内） | 同条件のnative展開・basis cancellation・wire Z-fold、全wrapper Sの意味、S-before-C | C1の8セルguardを黙って変更しない。Sをcompiled実値へ昇格しない |
| `track_d_native_resources.NativeCounter/NativeTemplate` | 確定native列の一回走査・countとwire transfer、固定wordの合成 | Track DのStrong/Reference価格係数、CDF8条件やP0 bridgeの資格を移植しない |
| `rte_boundary_cost_validation`、`hierarchical_cost_validation.validate_controlled_repetition_extension` | 2/3-event補正、校正と未使用反復数のholdout、cohort照合という方法 | 旧DF12等の係数・別制御/δ/prefixの価格。単純event和を採用しない |
| `rte_connected_cluster_cost_validation.connected_value`（`_compile_calibration_length`内）、`_sequence_form/_iid_sequence_form`、`predict_connected_cluster_cost` | inclusion-exclusion、window多重度、独立校正/transfer、chunk統計の増分合成 | 現行local compilerは`controlled=False`の中央RTE。full controlled head/tail/wrapperへ係数・独立SE式を移植しない |
| `rte_compiled_cost.TranspiledCircuitCostCache/transpile_and_measure_cost`、`canonical_qiskit_circuit_fingerprint` | 数値回路・compiler・target bindingによる保存値再利用、count取得 | 角度/τ/Rを省いたcache key。未compile回路のfingerprint計算は無料ではない |
| 元`h46_long_time_native.sample_requests/compose`（科学carrier `155529e2`）・provider `c31ef424` | signed schedule・canonical seed・sampling・native full wrapper・元compiler。将来の有限直接価格oracle | 巨大event列を自動実行する入口。旧grant/STOPの再利用 |
| `r_variable_compiled_price.describe_request`、B `build_menu/request_context` | canonical ID、qRPE/qPF、R・allocation・候補/法則・scope記述 | B共通pipeline、N、optimizerの再実装 |

connected incrementの設計はκ1(e1)=C(e1)、κ2=C(e1e2)−C(e1)−C(e2)、κ3=C(e1e2e3)−C(e1e2)−C(e2e3)+C(e2)。この局所構造は方法として使えるが、長range相殺を打ち切った誤差が0とは言えない。現在のlocal係数をそのまま移す計画ではない。

## 巨大回路を直接compileしない方法と計算量

E=qR、J=piece数、p=prefix、W=wire数、H=一組のdeterministic head/tail展開量、D=event当たりの展開量、g=完成回路のinstruction数とする。式はcode loopに基づく構造的な評価で、新benchmark・wall予測ではない。

| 方法 | q/R/J依存・主要処理 | memory/metadata | 科学的な制約と用途 |
|---|---|---|---|
| 1 全wrapper直接compile | sampling/signed request→composeはO(qJH+ED)相当の出力走査。Qiskit transpileの実負荷はpassとgに依存し、線形時間保証は無い | requests/event/application記録O(E+qJ)、raw QC・fingerprint用再帰payload・compiled QC/DAGがgに応じて増える。通常path/axisの並列数で常駐量も増える | 固定compilerの直接整数価格を得るoracle。最終roundを一律に回す方針にはしない |
| 2 S＋C1残差 | AccountingはO(qJH+ED)のnative feed、補正式だけなら定数規模。ただしS未取得なら価格は出ない | counterの作業状態はO(W＋pending basis)、固定head metadataはO(Jp)。**現C1 backendは全requests/eventsを保持するため全体memoryはO(E)** | compilerなしのS経路とincremental eventsを分離すればQCを省ける候補。長q/R補正と構造Sの双方を検証する必要がある |
| 3 局所window/境界/繰返し | 実pathを長さk≤3で走査するならO(kE+qJH)。同一law内window期待多重度を使う式はpieceごとにO(J×class数)、q/Rは多重度に入る | streamingならwindow/pending stateのみ。κ・境界class・nominal covarianceを小さい固定metadataとして保持できる可能性。全event原本を保存するならstorageはO(E)のまま | **候補法**。現controlled構成の局所interface・非局所compiler残差の検証前に実価格を保証しない。期待多重度によるlaw評価は別の科学実行 |

元`sample_requests`はqJ requestとevent配列を作り、`compose`は各pieceのtail/partial circuitを全QCへcomposeする。`_circuit_payload`も全instructionをlistへ保存してJSON化する。したがって大Rではsampling metadata・QC生成/複製・fingerprintとtranspileの常駐量が問題になり得る。どれがwallの支配要因かは未測定で、既存C1のwallやevent比から未実行条件の時間・高速化率を断定しない。

Accounting自体はpending basisとwire-local Zを持ち、全historyを保存しない。将来は既存の同一sampler stream/順序を保持したiterator入力と、元counterの継続stateを接続すれば、科学的Sを変えずmemoryを減らせる可能性がある。chunk末で毎回`finish()`すると境界の相殺を失うので、stateを跨いで保持する必要がある。現samplerの一括draw順序をstream化で変える場合は、新しいSOURCEと人工比較・許可された有限実条件での一致確認が先に必要となる。

`NativeTemplate.then`は**既に確定したnative word**のcount/transferを合成できるが、random outerのwordは一般に同一ではない。deterministic headを単独で計数・q倍してtail界面を捨てる方法や、native列の厳密countをP0後価格と呼ぶ方法は採用しない。RZだけならdepth用の大きいtransfer機構を追加する必要もない。必要な境界stateを閉じた最小count経路を先に検討する。

方法3の当面の設計は、同じ構造Sを基準に局所的なcompiled残差をpiece内部、head↔tail、piece間、outer間、wrapper端部へ分けること。内部κだけでhead/tailを再加算せず、重複するwindowをinclusion-exclusionで会計する。H6の正負時間・r_j別のlawを保ち、局所端部の人工相殺と全wrapperの残差を分離する。境界classが有限に閉じることも未証明なので、現段階でκ表の新compileを要求する一括科学packageは作らない。まず人工wordでcounter/chunk/boundary仕様を固定する技術候補とし、今回これを新規実装・計数したとは扱わない。

cacheは数値τ、ordered application、preparation、compiler/full wrapperの一致が必要。元snapshot/固定δのhead metadataは再利用可能だが、R変更時のtail・allocationや異なるpathの完成価格を同一視しない。whole-path並列、context共有、重複fingerprint回避はC1の実装方法を再利用できる。将来のworker数は仕事量/実メモリ/affinityに合わせ、独自の恒常CPU/RAM上限や共有環境の変更は導入しない。

## 次の有限科学計画

全て**提案・未承認・未実行**。第一案はP1_DIRECT=Q＋QRで4候補・4cohort・最大40 wrapper。wrapper数はX/Y価格取得枠と新compile上限であり、元fingerprint/compilerが一致するcache再利用があれば新compileは減り得る。Sは各新pathについて元counterでCより先に取得する。X/Yは同じpathを共有し、独立標本数を2倍に数えない。

| component | 系/m | T | qPF / qRPE | R | canonical candidate ID | event数 | n path | 新wrapper数 |
|---|---|---:|---|---:|---|---:|---:|---:|
| Q（第一案） | H4/6 | 6.4 | 64 / 64 | 4 | `65c51c5cb3e39a2266e3` | 256 | 4 | 8 |
| Q（第一案） | H6/8 | 25.6 | 16 / 256 | 16 | `53797d9e5a2eeec31a13` | 256 | 4 | 8 |
| QR（第一案） | H4/6 | 6.4 | 64 / 64 | 128 | `2a8c6718220eec59d419` | 8192 | 6 | 12 |
| QR（第一案） | H6/8 | 25.6 | 16 / 256 | 128 | `55a8eb0ae50ec3f7c713` | 2048 | 6 | 12 |
| R（追加原因診断） | H4/3 | .8 | 8 / 8 | 128 | `e8294c5247446ff2ff67` | 1024 | 4 | 8 |
| R（追加原因診断） | H6/4 | 1.6 | 1 / 16 | 128 | `bd40c810c315915c4136` | 128 | 4 | 8 |
| SHORT（任意） | H4/0 | .1 | 1 / 1 | 4 | `03a51c8d13a91a403742` | 4 | 4 | 8 |
| SHORT（任意） | H6/4 | 1.6 | 1 / 16 | 11 | `5e2581e9c4f25bdf8529` | 11 | 4 | 8 |

formula/δ/prefix/Kは固定条件表のまま。H4 allocationは[R]、H6は前述またはPLANの元11piece配分。全行にfull request/weights/allocation/compiler/価格unit/role/source/nullを保存した。metadataのevent数やhead訪問回数はcompiled RZの見積値ではない。

**選定理由**：第一案は、公開Nが同policyで揃う中時間のbaselineと一つの大R帯をまず比較可能にする価格証拠を狙う。R128はAの既定候補のうち小さい大R帯であり、R256より経済的・最適と判定した選択ではない。q-only baselineを同時に持つことで、単に両方を変えた2点だけを取る計画にしない。新点の直接CとS/凍結C1残差を同じcohortで得れば、式が不適合かと、有限scenario価格を得られるかを分けて判断できる。

**原因の分離**：第一案だけでは、R移送誤差とq×R相互作用を完全には分離できない。必要な場合、C1の低q/既存R anchor（H4 q8/R4、H6 q1/R16）とQ・QRに、R-onlyの低q/R128を加えて矩形を閉じる。q-only、R-only、mixedの差を、同一の凍結モデルに対するe=C−Ĉとして記述する。例えばmixed contrastはe(qH,RH)−e(qH,R0)−e(q0,RH)+e(q0,R0)。今回はこの数値を計算していない。

**nと精度**：n4は小さい有限pilotの平均/分散・path散らばりを観測する入口、QRのn6は両方向移送を調べる点へ標本を多めに置く有限budget案である。3または5の標本分散自由度が得られるだけで、統計power計算や必要精度の充足根拠ではない。C1 n8の踏襲・1%保証・95%同時区間と扱わない。ユーザーがこのpilot目的/budgetを採用するか判断し、不確かさが大きければ追加標本は別の具体的範囲として提案する。自動増標本・retry・閾値変更はしない。

全新点は**C1を変更しない独立transfer検証**であり、今回提案に新fitは含めない。C1の校正/検証役割を保持し、独立検証点を後付け校正へ変えない。将来新しい局所残差モデルや再fitを選ぶ場合は、使ったC2点を開発/校正へ役割変更したことを明示し、別cohort・未使用点を新たな独立検証へ予約する。現在のC1 runnerを直接新点へ流用・guard解除しない。

指標はpaired path C/S、bias/MAE/RMSE/path相対RMSE/最大相対誤差、有限cohort平均・名目分散、X/Y・残差共分散、凍結係数のnominal covarianceの別保存。法則の距離、長q/R systematic、cross-R/round joint covariance、total CIは未知ならnull。実行前に対象package・SOURCE・seed/cohort・役割・指標・fresh出力を固定し、C1旧STOPを残す。SOURCE/入力/候補/法則/時刻/allocation/compiler不一致、欠測axis/S/C、非finite・負予測、budget超過、backend異常では証拠を保全してSTOPし、無断の再実行や別候補fallbackは行わない。

## 保留領域・価格精度・Bへの受渡し

R256の2候補、H4短時間R2/R8はmissingのまま残す。R128から補間・比例移送して埋めない。SHORTは短時間候補の一部を対象にする独立の任意案であり、中時間の原因診断用R packageとも別である。全候補を覆う計画ではないことをPLANに明示した。

H4 m12/q4096/R8192の33,554,432 event、H6 m12/q256/R2048の524,288 eventは、構造metadataとして保持するだけで新直接compile案は0。AのP2/P3も未実行として扱う。streaming S/局所境界法の人工仕様と、有限実条件でのcompiled対応を確認する範囲を先に固定し、その結果を見ずに最終時間を科学対象へ追加しない。

必要な価格精度は一律%で決めず、将来Bが保存Nを固定し、例えば |ΔG|≤ΣN|ΔC|、区間を用いた候補間の識別可能性で判断する。今回は実N×C、価格誤差の加重和、ratio/順位を計算していない。paired X/Y・shared calibration/source依存を残し、独立SEを無条件に足し合わせない。RPEのαtotalは価格CIの信頼水準ではない。

現在の出力`new_price_records=[]`、全新C/S/予測/残差/Gはnull。Bの`c_provider(task, requested_rows)`とexact identity/law/axis/cohort/compiler/source形式を維持し、C1のdirect有限cohort・保存model・校正/独立検証を分ける。B `C1PriceIntake`は固定8セル専用なので、C2結果をそのままC1へ混入させない。将来は新publicationの有限domainを受けるB側入口が必要で、Cは共通APIやoptimizerを変更していない。[HANDOFF](HANDOFF.md)に追加証拠・不足と受入案を記載した。

## 技術確認・保全・次の担当

新8件のmetadata/保存fixtureテストはPASS。candidate/時刻/qRPE-qPF/R/11pieceのbinding、A公開12件のstatus/同policy、C1との2重複、q/R/QRの有限budget、未選定価格null、旧validation役割、巨大round未提案、欠測・未資格保持を確認した。既存Cのdescriptor・B menuをそのまま使い、共通pipelineはB724dと同じblobであることも確認した。作者自身の技術確認であり、独立科学レビューや長q価格精度の検証ではない。

以前のC1 SOURCE/PLAN/frozen/result/marker/STOPと旧科学moduleは変更していない。A/Bは固定公開Git objectだけを読み、実行中worktree/process/cache/非公開raw・環境を操作していない。公開A結果のNは整数のままcopyし、N solverや分子runtimeを呼んでいない。

**次：ユーザー**がP1_DIRECTの40 wrapper（または明示的な有限subset）のpilotを次の科学範囲とするか判断する。**Codex C**は選択範囲のbounded runner/SOURCE・seed等を技術的に固定してから、具体的な科学実行許可の範囲だけを実施する。**Codex B**は公開N/Cを許可scope内で取り込み、将来の価格精度/費用感度に必要な証拠を保持する。今回の計画だけでN×C/GやR採択へ進まない。commit/push・remote SHA/tree/主要成果物照合後mandatory STOP。
