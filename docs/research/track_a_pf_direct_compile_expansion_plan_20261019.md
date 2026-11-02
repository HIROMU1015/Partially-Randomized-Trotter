# PF費用モデルの直接コンパイル拡充計画

2026-10-19。**PLAN_ONLY / NOT_EXECUTION_AUTHORIZED / mandatory STOP**。
ユーザーの計画作成指示に基づく固定資料の棚卸し・技術計画・構造確認である。新規科学計算、分子回路生成、compile、sampling、fit、モデル予測、費用再会計は行っていない。候補の登録・保存単価・保存選択結果の転記とmetadataの件数だけを扱った。

[計画packet](../../artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19/README.md)、[候補一覧](../../artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19/candidate_inventory.csv)、[軸別一覧](../../artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19/axis_inventory.csv)、[出典付き測定・モデル値](../../artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19/axis_evidence.csv)。

## 基点と比較単位

公開済み採用プロトコルv2のH2完了追記commit `3826f566b04ad9c2ead9abf1b80c864f7b34b785` を文書基点とする。古いmain `0babed07` の状態を最新実験状態とは扱わない。H2の独立確認の正式受入はこの基点へ反映済みだが、今回その科学値や論文主表を変更しない。

対象はlinear H2/H4/H6/H8、隣接1 Å、STO-3G、T=.8、P=0。新tol-only threshold1e-8のDF3/10/19/28を区別し、legacy固定H4 DF12を別系列にする。状態・Hamiltonian・DF/snapshot・fragment順・scalar・preparation・control・compiler/runtime/profileまで一致を必要とし、rankや候補名だけでは再利用しない。

費用指標は**状態準備を除くfull cosine/sine Hadamard wrapperのcompiled RZ、軸別整数ゲート数**。決定論prefixとB0Iのtail identity／元scalarを含む元の意味論を維持し、symmetric_directional、reference無回転とする。保存shot×RZの総費用・PR比較・最終energy/RPE/FT資源は今回再計算しない。CX/depth/RZ_depth/CX_depth/size等も取得・保存する計画である。

登録集合はv2保存台帳と最終技術監査の既存domainから転記した。

- H2：B0/B0I p0..2、B1 full p3、q1/2/4/8/16、S2/S4の70候補。
- H4 DF10：B0/B0I p1..9、B1 full p10、同q、S2/S4の190候補。
- H6：B0/B0I S2 p5..15、S4 p11..15、B1 full p19、q1/2/4/8の136候補。
- H8：B0/B0I S2 p7..21、S4 p15..21、B1 full p28、q1/2/4/8の184候補。
- H4 legacy DF12：B0/B0I p2/3/4/6/9/11、B1 full p12、q1/2/4/8、S2/S4の104候補。

全684候補・1368軸。B0/B0I・両軸を別単位として数え、full-rank aliasを増やさない。H2 p0のideal同値でもnative分解を同一視しない。高次PFの別研究、登録外q/T/geometry、ordinary-controlのH4別geometry地図は加算しない。

## 棚卸し結果

| 入力系列 | 登録候補 | 両軸直接実測あり | 直接実測を確認できない候補 | モデルのみの軸 | 保存単価もない軸 | 新規取得候補軸 |
|---|---:|---:|---:|---:|---:|---:|
| H2 DF3 | 70 | 34 | 36 | 0 | 72 | 72 |
| H4 DF10 | 190 | 129 | 61 | 0 | 122 | 122 |
| H6 DF19 | 136 | 29 | 107 | 192 | 22 | 214 |
| H8 DF28 | 184 | 27 | 157 | 258 | 56 | 314 |
| H4 legacy DF12（別系列） | 104 | 71 | 33 | 0 | 66 | 66 |

固定公開資料で確認した軸数であり、未知の非公開実測が存在しないことの証明ではない。片軸だけの候補は今回の固定資料では0。重複receipt、モデルのtraining anchor、後続転記を新しいcompile回数として重複計上しない。

H2はD新28候補と明示再利用FP6候補、H4 DF10はD新125候補とFP4候補である。過去Dの「全候補評価完了」は、全登録回路のcompile完了とは異なる。保存の精度不適格により単価が取得されなかった候補も、このモデル比較用データ取得の計画では残す。

H6は旧full8・B0 S2実測・B0I実測・登録S4検証・p7追加実測・p11/q4 S4追加実測を結合した。H8は旧full8、B1の観測anchor（fit実行そのものをcompileと数えない）、B2検証、主要6軸、追加D4軸を結合した。元runがSTOPした保存測定は、後続の受入記録と元STOPを併記し、元runをCOMPLETEへ書き換えない。

## H6 ε=.005と実測・モデル混在

保存の通常signed・倍率1・強化136比較では、ε=.005の選択PFは `B0_p11_q4_2nd`。両軸とも保存単価 `433317/5 RZ`、status `VALIDATED_MODEL_ESTIMATE`、model SHA `406c6f36f93dac317f9709e68b042c4563edf5ec4b06219e1b60de660b695211` であり、直接実測ではない。新しい予測値は算出していない。

この候補を**最初の2軸**に推奨する。p11/q1 S4やB0I p11/q4 S4の実測を、p11/q4 S2へ移して実測扱いにしてはいけない。

同じ通常signed強化H6のε=.01（B0 p11/q2 S2）、.001（B0I p11/q2 S4）も保存選択PFがモデル。両端.05/.0005のPFには直接実測がある。一方、旧32集合・非相殺比較では選択PFも証拠区分も異なる。条件を混合せず、保存比較を[saved_mixed_comparisons.csv](../../artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19/saved_mixed_comparisons.csv)へ転記した。勝者の再選択はしていない。

H8の強化集合でも直接実測とモデルが候補単価に混在する。例えば通常signed .01/.005は実測B0 p15/q4、非相殺 .01/.005はモデルB0I p15/q4である。同じp/qでもmethodとbias条件を省略しない。H2/H4 DF10の現在の適格PF費用は直接測定に基づくが、全登録回路に単価があるわけではない。

## 再利用の判定

`axis_inventory.csv` のdirect statusは**既存観測の存在**であり、新しい実行環境での再利用認可ではない。`axis_evidence.csv` は各整数値・モデル有理数表記、fingerprint、SOURCE、compiler/profile、出典locatorを保持する。省略されたfieldは空欄のまま未知を示す。

H2/H4には保存binding一致と受入証拠がある。H6の旧B0/full系列32軸とH8のlegacy full16軸、計48軸は、旧profileの原本等の照合が未解決で、[条件付き再取得一覧](../../artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19/conditional_reacquisition_axes.csv)に分けた。旧profileを現profileで補完したり、同じQiskit optionsだけでbinary同値と認定しない。

48軸以外にも、全既存軸で予定targetの入力・preparation・wrapper/control・軸・scalar・compiler/runtime/profile・fingerprintを実行前に再照合する。未生成のfuture回路fingerprintを確認済みとはしない。照合できた保存観測は流用し、照合不能は保留または別承認の新しいprofile層の観測とする。候補の直接実測済み数を減らすことと、再利用不可を同義にしない。

## 実行可能性と推奨範囲

既存記録から、H2/H4とH6の多くは取得可能性が高い。H8の大きいS4/full回路は資源を見ながら投入する必要がある。これは技術的見通しで、将来実行の成功保証や所要時間の新規推定ではない。

- H4 DF10：125候補250軸を含む既存D stage wallは1017.5871014744043秒。signal/control等も含み、compileのみの時間とはしない。
- H6：p15/q8 S4はcompile各530.6741248946637／534.8988664690405秒。登録4軸stageのaggregate peak RSSは9545846784 bytes。別p11/q4 S4のnative stage peak RSSは3815890944 bytes。
- H8：p15/q4 S4の初回cosineは668636 RZ、worker peak RSS 10561605632 bytes。p16/q4 S4は713432 RZ、compile各144.41190458647907／144.56327833794057秒。p21/q8 S4の1852640 RZ/軸という直接測定実績もある。driverのRSSを全workerのpeakやaggregateと読み替えない。
- H4 legacy：55新規候補110軸の既存D wallは536.0692663248628秒。残33候補は過去endpointでの不適格であり、物理的にcompile不能な証拠ではない。

実行時にはaffinity・scheduler/cgroupの許可・実効空きRAM・FSと、保存された候補別の回路規模・worker/aggregate RSSを確認する。[保存resource receipt](../../artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19/saved_resource_receipts.json)と[実行可能性評価](../../artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19/feasibility_assessment.json)を参照。今回、新規benchmark・分子入力decode・実際の資源admission検査はしていない。大きい未測定回路の数値的な時間/RAM予測を費用modelから生成していない。

推奨は、H6指定2軸と保存選択PFのモデル依存箇所を先行し、続いて**全登録qを含む未測定集合**を完了させること。
現行4系は361候補722軸、legacy DF12は別契約33候補66軸、合計394候補788軸である。48軸の条件付き再取得はこの788軸に含まず、照合後に必要な部分だけ別の明示scopeとして決める。小qだけで打切らず、H2/H4のq16、H6/H8のq8/S4/fullも計画に残す。順序と実行対象の除外を混同しない。

## 並列化・高速化・必要な実行契約

既存のH2 recovery、H4 main/legacy、H6 DBA native、H8 DBA directのコードを最大限再利用する。固定SOURCEとpath/hashは[source_inventory.json](../../artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19/source_inventory.json)にある。現行runnerをそのまま全候補へ起動すると、signal/shot会計や旧call制限が接続される場合があるため、将来は**compile取得専用の新契約・adapter**へ最小限切り出す。今回はadapterの実装・SOURCE seal・有効grant発行はしていない。

保存full preparationからprefix view、cosine/sine間のevolution共有、検証付きfingerprint cache、QPYの一軸ずつのload、候補間parallelを流用する。実測のある同一回路の重複compileを避ける。CPU/RAM/worker数に独自の恒常上限を設けず、実際の許可・空き資源とtaskごとのfootprintで投入数を決める。自分のprocess内threadと候補並列を調整し、共有OS・scheduler・共有venv・他ユーザーprocessを変更しない。PSIや一時的なCPU競合だけで正常workerをkill/restartしない。

[未認可契約案](../../artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19/execution_contract_DRAFT.json)は、missing-onlyでevolution394、wrapper build/compile各788、control最大1576、必要ならQPY dump/load各最大788を提案する。これらcontrolも科学callであり承認前には実行しない。事前に入力load/preparation viewのclosureと上限、使用する既存oracle/primitiveの範囲、candidate/axisごとの予約を最終sealする。追加oracle取得・reference/signal再計算が必要なら自動拡張しない。

新契約のdata取得では、モデルとの誤差を記録しても、モデル不一致だけでは正しい直接測定を捨てず、無関係な正常taskを停止しない方針を提案する。再fitはしない。入力/control/回路意味論/fingerprint/compiler不一致、native異常、実OOM等は技術STOPを維持する。**旧契約の2%判定や元STOPはそのまま保全し、提案を遡及適用しない。**

実行前にユーザーが対象・再利用/再取得範囲・call範囲を確認し、将来SOURCE/input/runtime/profile・task一覧・output・one-shot識別子を固定した新しい科学実行を明示承認する必要がある。計画公開の承認と実compileの承認は別である。

## 保存証拠と将来のモデル比較

取得する各軸について、入力/DF/state/snapshotとfragment順、preparationとscalar、PF係数・時間・wrapper/control、compiler/runtime/profile hash、compile前後fingerprint、control誤差とoracle出典、全ゲート整数metrics、build/control/compileの各wall、worker/aggregate RSSの定義、native exit/reap、call予約・完了・非返却、result SHA、SOURCE/commitを保存する。

実行後は固定task coverage、各軸のcontrol・fingerprint・compiler、終了回収、metricsの整数/欠測、原本の不変性と出典リンクを検査する。cosine/sineで同じRZでも独立した軸記録を保持する。欠測はnullと理由を残し、model値で直接測定の穴を埋めない。費用再会計・順位更新は別の数値解析承認とする。

[将来データ構造](../../artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19/future_data_schema.json)はcandidate/circuit/compiler/observationのkeyを分け、直接観測とmodel/predictionをappend-onlyの別表にする。同じobservation IDに複数のmodel IDの予測を結合できる。旧モデル式・係数・予測値は削除しない。H6/H8の保存S2式 `a+b*q+d*prefix*q` とS4合成仮説 `a+3*b*q+3*d*prefix*q`、保存係数・出典を[モデル登録表](../../artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19/model_registry.csv)へ転記した。既存値をコピーしただけで式を評価していない。

新モデルの学習/holdout分割・候補選択は将来fit前に固定し、既存training anchorや既に見たvalidation値と未使用観測を区別する。誤差の新規計算・fit・統計集約・モデル間優劣の判定は今回は行わない。

## 構造検証・保全・STOP

`python3 scripts/prepare_pf_direct_compile_inventory.py` はstdlibだけで固定Git objectへ再照合し、684候補/1368軸、788新規提案軸、保存値の完全転記、両軸・DF系列分離、H6指定候補のmodel status、未認可状態、既存file変更範囲を検査する。科学runnerやcompilerをimportしない。GitHub固定12ファイルは実取得したbytes/SHAとローカル固定Git objectの一致を確認し、[receipt](../../artifacts/resource_applicability/track_a_pf_direct_compile_plan/2026-10-19/github_read_receipt.json)に保存した。他の出典は固定Git objectと既存公開照合記録に基づく。

構造検証は主担当自身による確認であり、独立科学レビューや外部再現ではない。既存SOURCE/raw結果・grant/ledger/marker/STOP、採用済みプロトコル、論文主表・Figure 2を変更しない。新規科学実行・fit・費用再解析0、詳細科学レビュー・最終evidence freezeは未開始。

次の担当：**ユーザー**が推奨788軸（4系722＋別legacy66）と条件付き再取得48軸の扱いを確認する。**Codex**は指示後に必要な新SOURCE・最終契約を準備し、ユーザーによる実compileの明示承認後に限って実行する。本計画だけで科学実行を開始せず、**mandatory STOP**。
