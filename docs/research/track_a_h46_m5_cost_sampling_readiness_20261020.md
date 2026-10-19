# H4/H6 高次PF：回路費用取得・PR標本設計の実行準備

2026-10-20。**保存signal88候補のmetadataを照合し、費用取得案、12保存軸の受入証拠、native adapter、並列launcher、4つの独立した無効契約を準備した。新規の実分子signal・compile・sampling・fit・shot再計算・費用再ランキングはいずれも0件。有効grantは発行していない。mandatory STOPする。**

基点・signal結果commit：`a960a41da5ebc564b0306e798572257091c8509c`。新しい費用取得用SOURCE：`635ba563766875d688ad6912499b7f90397d0cc5`。専用branch：`track-a-h46-m5-cost-sampling-readiness-20261020`。成果物は[公開bundle](../../artifacts/resource_applicability/track_a_h46_m5_cost_sampling_readiness/2026-10-20/)に置く。人工検証は作成者自身の検査であり、独立レビューではない。詳細科学レビュー・最終evidence freezeは実施していない。

初版SOURCE`3e24d4dbc108146f9ba0e7c9e005463ceb2bb9f1`の未承認DRAFTはprivateに保持した。最終構造確認でadapterが読む保存signal一覧の明示封印を追加し、新SOURCEと`closure_v2`の4無効契約へ更新した。過去の実行結果・承認・STOPは変更していない。

## 1. 費用取得対象案

[signal run02](track_a_h46_m5_signal_run02_results_20261020.md)の保存880軸行を読み、**同じεにおけるreal/cosineとimag/sineの両方が保存ラベルELIGIBLE**であることだけで対象案を作った。bias、normalization、allowance、shotを再計算していない。片軸だけの適格性、異なるεの片軸同士を結合していない。

| 対象 | 登録候補 | 両軸適格なεが存在 | 現在の費用対象にしない候補 | 対象内の保存pair | 新規compile案 |
| --- | ---: | ---: | ---: | ---: | ---: |
| H4 m5決定論 | 28 | 20 | 8 | 1 | 19 pair・38軸 |
| H6 m5決定論 | 36 | 28 | 8 | 1 | 27 pair・54軸 |
| H4 PR：S2/Yoshida4/m5 | 12 | 12 | 0 | 0 | n_E採択後に固定 |
| H6 PR：同3公式 | 12 | 12 | 0 | 0 | n_E採択後に固定 |

決定論64候補中48候補を取得案へ、16候補は保存したまま今回の費用取得案から外す。PR24群は全て保持する。実データの未確定ラベルは0だが、コードはUNDETERMINEDを未解決のまま残す。保存表を[候補88行](../../artifacts/resource_applicability/track_a_h46_m5_cost_sampling_readiness/2026-10-20/candidate_cost_targets_88.csv)、[同精度両軸join440行](../../artifacts/resource_applicability/track_a_h46_m5_cost_sampling_readiness/2026-10-20/candidate_endpoint_join_440.csv)へ転記した。CSVの空欄n_Eは未確定で、0標本を意味しない。

H4のprefixはPR3、m5 B0/B0I3・8・9、full10。H6はPR5、m5 B0/B0I5・7・11・15、full19。各qは1/2/4/8。PRはK4、`ABS_TIME_PER_OUTER=32`を維持する。各群の費用は一度取得し、将来の5精度会計で同じpaired trajectoryを使う。5精度ごとに新しいtrajectoryや回路を作る計画ではない。

PRの保存両軸適格εはH4/H6共通で、S2はq1で.05、q2で.05/.01/.005、q4でさらに.001、q8で全5精度。Yoshida4はq1で.05/.01/.005、q2/4/8で全5精度。m5は全qで全5精度。これは保存ラベルの転記であり、適格なqの費用順位は未評価である。

## 2. 保存12軸の再利用範囲

入力ham/state/snapshot SHA、preparation hash、native builder、scalar/control政策、compilerの設定・実体、runtimeと元receiptを照合した。12軸とも**同じm5 native系列内で再利用可能な単価の証拠**がある。ただし、12軸すべてが新しい対象案に入るわけではない。

| 系 | 保存構成 | 保存RZ/軸（cosine・sine） | 現在の対象案 |
| --- | --- | ---: | --- |
| H4 | B0 p3/q2 m5 | 51,501 | 両軸適格εなし、保存のみ |
| H4 | B0I p3/q2 m5 | 51,501 | 同上 |
| H4 | full p10/q2 m5 | 171,533 | 保存2軸を再利用 |
| H6 | B0 p5/q2 m5 | 228,570 | 両軸適格εなし、保存のみ |
| H6 | B0I p5/q2 m5 | 228,570 | 同上 |
| H6 | full p19/q2 m5 | 831,662 | 保存2軸を再利用 |

したがって、対象案内で重複を省けるのは4軸であり、48決定論候補の残り46 pair＝92軸（H4 38/H6 54）を追加取得する案になる。[保存12軸の証拠表](../../artifacts/resource_applicability/track_a_h46_m5_cost_sampling_readiness/2026-10-20/reuse_cost_axes_12.csv)には軸別receipt/PASS result SHA・SOURCE・compiler hash・preparation hashを記載した。

通常のpilot費用SOURCEは`c7d53f04c3be131078ee91e8bef3c78ca434ed3e`。H6 fullの元cosine SOURCEは`f743a88225f7cb2c8d2c91d7e19813a5780f803b`で、旧1e-9判定のSTOPを保持している。新CT01 SOURCE`885bbd7199c9ada26704b7db75b99767734e92e7`の独立したcosine受入receiptとsine PASS・終了証拠に結合した。旧STOPをPASSに変更していない。詳細は[CT01実行結果](track_a_h6_m5_ct01_sine_run01_results_20261019.md)。

declared profile `4e32bb9b…`は元契約でも`profile_is_reference_only=true`である。文字列の一致を旧主比較のruntime全体との同一性証明に用いていない。今回確認したのは保存入力・preparation・実native/composition・compiler/runtimeの同一m5系列内の照合である。

## 3. 維持する実験条件と技術検査

- 線形H4/H6、隣接1.0 Å、STO-3G、tol-only1e-8・rank自動決定、保存観測rank10/19。入力を再生成しない。全生成政策の完全同一性やサイズの因果効果は認定しない。
- T=.8、P=0、通常signed bias・経験的allowance倍率1、保存5精度を対象案の根拠とする。
- PF係数は既存S2/Yoshida4と`PF_M5_BEST_20D64C2_BINARY64_V1`。m5の11 signed S2 piece・負時間・整数step配分を保持する。再最適化・公式交換なし。
- full Hadamard wrapper、cosine/sine、symmetric-directional control、scalar once。native policyは`M5_TOL_ONLY_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1`。
- Python3.12.3、NumPy1.26.4、Qiskit1.3.0、rz/sx/x/cx、opt1、seed17。compiler settings SHAは`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。元presetを変更しない。
- compiledとuncompiledの状態vector差はCT01の**1e-7未満**。native control・independent occupation oracle・Hadamard期待値・保存log_B照合は従来の**1e-9未満**。有限値を要求し、位相合わせ・補正を行わない。CT01は全PF/PR・両軸共通で、m5固有の免除ではない。

各candidate/trajectoryで低controlの恒等性、高controlの漏れと独立occupation oracle、両軸Hadamard対応、compiled状態差を検査する。**各pairではcosine検査にPASSしてからsine compileへ進む**。同じevolutionとpreparationを両軸へ再利用する。失敗時にも未検証QPY・metrics・vectorをquarantineとして保存し、受理費用へ格上げしない。

## 4. PR24群の標本設計案

**n_Eは24群すべて未確定。今回新しいseed registryは作っていない。** 同じ入力・directional control・outer32 lawの費用分散が未取得なので、以前のpiece-r2 pilot、ordinary-control探索n8、高次PF独立確認13/45/45を標本数の根拠へ流用しない。保存分散・SEの新規集計も行っていない。

提案は次の2経路であり、いずれも採択待ちである。

1. **固定標本の探索**：ユーザーが有限標本の探索目的を採択し、n_Eを先に固定する。同一公式の4つのqに等しいn_Eを割り当てる案は比較の透明性を保つが、数値自体の根拠にはならない。標本平均・paired SEを報告し、所定のpower/MCSEや母平均順位の保証としない。
2. **同lawの設計pilotを別承認**：outer32の設計用標本を独立契約で取得し、その後に分散・希少eventの限界を踏まえてn_Eを固定する。pilotの標本数も今回恣意的に固定しない。設計pilotを本探索・確認にpoolしない。

いずれも同一trajectoryのcosine/sineをpairedにし、全5精度で標本を共用する。候補間のpaired構造を仮定しない。nは取得前に固定し、結果を見て自動top-upしない。探索後に対象・εを固定し、新しいCONFIRMATION cohortとn_Cを別途設計する。原則端点.05/.0005を検討するが、不適格候補を確認対象へ無理に入れない。n_Cも未確定で、探索・pilot・旧確認標本とpoolしない。主比較の採用AD03は保留のまま。

**希少event**：全K4無条件lawを保持し、s4を強制したり、通常標本から棄却したりしない。各pieceのseed・order列・event digest・normalizationを記録する。N4≥1ならs4、それ以外でN2≥2ならs2_multi、N2=1ならs2_single、残りをs0とする。未観測層の費用・分散・tail上界はnull。0補完しない。例えば理想独立step法則の4次観測確率は記号的に `1−∏piece(1−p4_piece)^r_piece` と記述できるが、signed pieceごとのlawは異なり得る。今回この確率や必要nを実データから計算していない。条件付きrare pilotを後日使う場合も、層重みと母集団を結合する別契約が必要になる。

seedはSOURCE・新campaign ID・EXPLORATION/CONFIRMATION・候補sample・outer・pieceを含むSHA256から導出する実装を用意した。現在照合した既知除外seedは4,603個。これは照合済みregistryの閉包で、別研究の全seedを網羅する認定ではない。実行前に最新の除外manifestを再照合し、新規registry内も含め衝突すればSTOPする。再hashによる自動retryはしない。

## 5. call範囲・契約・並列化

`H4_DET`、`H6_DET`、`H4_PR`、`H6_PR`を独立config・grant・output・ledger・消費marker・外部終了witnessに分けた。1 invocationは1 stageだけで、次stageへ自動進行しない。公開した4つのgrantは全て`valid=false`、private承認receiptも`execution_authorized=false`である。

| stage | task数 | sampling factory呼出し | native piece合成 | compile軸 | oracle/low/high作用（各） |
| --- | ---: | ---: | ---: | ---: | ---: |
| H4_DET | 19 | 0 | 803 | 38 | 19 |
| H6_DET | 27 | 0 | 1,133 | 54 | 27 |
| H4_PR | n_E採択待ち | 同左 | 同左 | 同左 | 同左 |
| H6_PR | n_E採択待ち | 同左 | 同左 | 同左 | 同左 |

DETのpiece合成は候補のq×11の総和で、compileは候補ごとに両軸。PRは各trajectoryにつきq×piece数のsampling requestと32qのRTE eventを登録する。両wrapperのuncompiled/compiled作用はそれぞれ2回。各論理callは専用台帳で非返金予約し、完了は結果receiptで別確認する。実行前に確定した上限へbindingし、失敗しても自動で予約を回復しない。

各系で公式内の4つのqに共通n_S2/n_Y4/n_m5を採用する場合、trajectory数は `4(n_S2+n_Y4+n_m5)`、compile軸数はその2倍、request呼出しは `15n_S2+45n_Y4+165n_m5`、RTE event数は `480(n_S2+n_Y4+n_m5)`。これは登録格子からの記号的予算であり、n未採択の現在のPR総call数は**null**である。

task単位のspawn process poolを使い、worker内のBLAS/transpilerを1threadにして二重並列を避ける。process内で入力context・prefix preparation・fingerprintをcacheし、同じevolutionをpaired軸へ使う。特定のCPU数・worker数を恒常上限にせず、fresh idle CPU、既存cgroup内の実効空きRAM、task数でadmissionする。H4/H6とDET/PRは承認済みのもの同士だけを並行起動できる。複数stageを起動する場合は、担当Codexが先行stageのadmission/実使用を確認して残り資源を割り当て、一斉起動時の同じ空きRAMの二重計上を避ける。全stageを無条件に一括起動する実装ではない。

RAMの初期admission proxyは過去の同入力q2観測×最大q比4で、H4 25,412,015,444 bytes、H6 92,394,930,176 bytes。**上界ではなく、未取得q8・rare回路の実メモリは未確定**。disk proxyも入力byte数に基づく最低限の観測で、全QPY/vectorの出力容量保証ではない。ユーザーがresource policyを採択し、実行直前にscratchの十分性と既存jobを確認する必要がある。runtimeの資源監視は残すが、host PSIや他worktree変更だけで正常workerをkillしない。OS・共有環境・他利用者の設定やプロセスは変更しない。

SOURCE/input/runtime/compiler・承認binding不一致、既知seed衝突、数値検査FAIL、worker異常・実OOM、出力/台帳不整合でそのstageをSTOPする。自動retry・再fit・追加compileは禁止。別stageの正常jobは停止しない。自stage worker回収・結果SHA・native exitを保存し、成功/失敗ともmandatory STOPする。

## 6. 強化S2/S4対照との公平な接続

[対照の互換性表](../../artifacts/resource_applicability/track_a_h46_m5_cost_sampling_readiness/2026-10-20/retained_S2_S4_baseline_compatibility.json)に証拠範囲を分けた。

- **H4 legacy DF12**：commit`4e8ad193…`の強化104候補を独立成果として保全する。DF10と入力政策が違い、単価・順位をそのまま転送しない。
- **H4 tol-only DF10 F/FP**：commit`97d25cc4…`のinput hashとcompilerは今回の系列と一致した。ただしnative/preparation/profile全体の結合は未完了。固定q4のB0I p8 S2/S4・full10 S2/S4・PR p3 panelであり、強化全格子の代用ではない。
- **H6 DF19 V1**：commit`f89c40af…`のS2細格子・打切りS4と保存直接実測/model区分を保持する。旧主profileから新m5 nativeへ単価を移せる証拠は未確立。Yoshida4のmodelをm5へ流用しない。
- **別のPF直接compile788軸研究**：本ユーザー指示による別系列として記録。対応するSOURCE・input・profileの個別manifestは今回未取得で、NOT_VERIFIED。grant/call範囲/結果を輸入しない。将来同じidentityのreceiptが提供された場合は、別契約で静的照合してから重複dispatchを防ぐ。

今回のm5単価取得だけでは、公平な主論文の最良対照を確定できない。旧強化対照を弱めず、共通の入力・preparation/control・compiler・shot政策で結合できる範囲を次の会計契約前に確認する。欠測や未認定profileを0費用や「m5優位」に置き換えない。実測/model/欠測を常に区別する。

## 7. allowanceとshot規則の採択を分ける

保存signalでは`M5_SIGNAL_EMPIRICAL_OBSERVATIONS_V1_PROPOSED`を使っている。uはclosure等からの経験式、guardはu/headroom≤.01、log margin floor1e-12。referenceの厳密誤差certificateはnullで、保存適格性はreference条件付きの経験的診断である。

今回の単価取得案を採択する場合は、この限界を持つ保存適格性を**取得対象選定に使うこと**への承認が必要になる。ただしunit-cost取得はshot計算を行わないので、先に正式shot規則を採択したことにする必要はない。configの`formal_shot_policy_adopted=false`は単価取得の範囲を示し、shot政策の将来の採否を禁止する意味ではない。

将来の会計前に、軸別ε/√2、α、normalization、bias・u・log margin、headroom guard、shot整数式、経験的allowanceの扱いを原SOURCEと照合して別途正式採択し、保存再現gateと新しい数値会計の実行範囲を固定する。今回はshot再計算・平均単価集計・SE・費用比・q選択・rankingをいずれも実施していない。

## 8. 検証・保全・成果物

新人工18件、既存signal回帰18件が固定Python3.12.3でPASS。新人工には小さな人工DFで3公式PR/決定論のnative→wrapper→compile経路、paired receipt/call台帳、cosine compiled不一致のquarantine保存とsine未実施、spawn poolの成功/失敗/別stage成功が含まれる。実分子compileではない。保存候補・契約・seed除外・改変拒否・CLI起動拒否・保全のmetadata監査も通過した。metadata監査84項目（48改変拒否・8 CLI起動拒否を含む）と、公開bundle構造159項目がPASS。4契約のsealは各2,320ファイルで、正確な内訳はbundleの`metadata_audit.json`を正とする。

privateに既存SOURCE/runtime/原raw/grant/ledger/marker/STOPの既知3,137ファイルの事前hashを保存し、終了時に変更0を確認した。他worktreeのHEAD/statusを凍結していない。入力array・QPYを復元せず、科学値の算術・統計集計をしていない。4つのdraftは`--execute`/`--supervise`とも入力を読む前にexit2で拒否され、新しいrun root・marker・有効grantは存在しない。

実行用SOURCEは上記`635ba563…`、metadata監査器と報告・公開bundleは後続資料commitに記録する。公開するのはコード・hash/metadata・候補表・無効契約・人工結果・報告だけ。raw入力/実回路/有効承認/seed値一覧はprivateに残す。新しい科学承認は受けていない。

実装入口：

- [対象・call・seed設計](../../scripts/h46_m5_cost_design.py)、[独立認可契約](../../scripts/h46_m5_cost_contract.py)
- [native費用adapter](../../scripts/h46_m5_cost_native.py)、[spawn並列launcher](../../scripts/run_h46_m5_cost.py)
- [metadata準備・封印](../../scripts/prepare_h46_m5_cost_readiness.py)、[private契約のmetadata監査](../../scripts/audit_h46_m5_cost_readiness.py)、[公開bundle構造検査](../../scripts/check_h46_m5_cost_public_bundle.py)
- [人工18件](../../tests/test_h46_m5_cost_readiness.py)

## 9. 次の担当と必要な承認

| 担当 | 次に判断・実施すること | 現状 |
| --- | --- | --- |
| ユーザー | 保存両軸適格性を使う取得領域と経験的限界、資源admission政策を採択するか判断 | 未採択 |
| ユーザー | 決定論H4 38軸・H6 54軸を各独立契約で一回取得するか明示承認 | 未承認。PRのn決定を待たず個別判断可能 |
| ユーザー | PRの固定探索n_Eか別設計pilot経路を選択し、n・有限標本/希少層の解釈を採択 | 24群全てn_E未確定 |
| Codex | 採択後、新しいSOURCE/config/approval/最新除外seed・registry/予算を封印し、承認されたstageだけfresh gate後に実行 | 今回は実行しない |
| ユーザー・Codex | 単価取得後、正式shot政策・強化対照のidentity接続と、別の会計/選択/独立確認範囲を判断 | 今回の承認対象外 |

m5主比較採用AD03、追加PF・候補・q・H8拡張、母平均順位保証、詳細科学レビュー、最終freezeへ自動進行しない。ここでmandatory STOPする。
