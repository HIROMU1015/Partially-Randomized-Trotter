# SCVのSOURCE再利用・将来の最小検証

2026-11-07 JST。**固定SOURCEを読んだ技術調査。新実装・科学実行・テスト実行・224-shot再実行計画ではない。**
対象条件は [REPORT](REPORT.md)。本書の「可能」は適合条件を満たした将来の再利用可能性であり、
実H4で接続・完走・効果を検証した意味ではない。

## 1. 実際に確認した固定資料

| 版・出典 | 確認対象 | 役割・版の注意 |
|---|---|---|
| ローカル正式2026-10-25、[同一bytesの公開保存版][policy] | 共通指示、特に§8.1/8.2/8.4/10/10.3 | 読み取り版と公開保存版のbytes一致を確認。新規grant発行を要求しない |
| [560c6092 AGENTS][agents]、元workspaceの適用AGENTS | repository規約、commit/push、保全 | 階層AGENTSは固定tree内ではroot一件。元workspace変更は保持 |
| [560c6092 PROJECT_MAP][project]、[研究概要][synthesis] | 現行の正本・履歴の区別、後続追記 | 概要は全文確認。9月の本文だけを11月のSCV仕様と読まない |
| [560c6092 VALIDATION_STATUS][status]、[manifest][manifest] | SCV準備・理論設計・752-shotの証拠status | SCVはIMPLEMENTED_ARTIFICIAL_ONLY_SCIENCE_NOT_ADOPTED |
| [560c6092 SCV準備REPORT][preparation] | 実装報告、224-shot草案、SOURCEへの案内 | 人工106 PASSは過去のlocal報告。実H4結果なし。本調査で再実行していない |
| [84e021b6 設計directory][design] | STRATIFIED_CV_DESIGN、INTRASTRATUM_CONTROL_CANDIDATES、NEXT_SCIENCE_VALIDATION_DRAFT | 層別分散・Born分解・oracle費用・有限confidence設計の出発点 |
| [8d8a290f A1対B1-OL][a1theory] | A1_VS_B1OL_THEORY | 層内定数controlの限界、B1-OLの公平なoracle比較 |
| [744e1a09 N1][n1] | REPORT | actual Q、全正5層、保存モデル内のtarget |
| [5e9bd080 N2][n2] | REPORT | 保存groundの認証。N2単独では総engine誤差UNKNOWN |
| [fdf00cee engine][engine] | REPORT | 後続の条件付き全support engine誤差認証。新roundへ転用不可 |
| [3c8e55a2 752-shot][pilotreport] | admission側REPORT | 完了したattempt02。旧pilot_science側REPORTはattempt01の0-shot STOPで別物 |
| [a53c1d6e 期待費用設計][expected] | EXPECTED_RZ_COST_GUARANTEE_PLAN | 条件付き/無条件期待費用、価格上界、STOP、engine→ideal移送。設計案 |
| [560c6092 RPE文書][rpedoc]・SOURCE | rpe_resource_accounting、decoder、境界test | strict pi/3、unit-radius specialization、scope/長round制限 |

752-shotは固定報告の結論と設計上の注意を読んだだけで、rawの数値再集計をしていない。
224-shotについては**準備版560c6092だけ**を参照した。
それ以降のactivation・結果・途中journal・manager・cacheは探索・取得していない。
固定版にある未採択表示から、別セッションの現在の採択・進行状況を推測しない。

既知理論の一次資料は[Glynn–Szechtman §2][glynn]、
[Maurer–Pontil v1 Theorem 10][mp]、
[Vershynin First Edition Theorem 2.8.4][bernstein]を確認した。
新しい統計定理の証明や詳細科学レビューは実施していない。

## 2. 照合した実装版

以下のAPIはすべて**560c6092bfb5e71af8eedc4a4be5416922764f98**の版。
再利用旧moduleの7件は84e021b6版とblob一致も確認した。
これは静的な版照合であり、全推移的依存・別環境の再認証ではない。

| module | Git blob |
|---|---|
| scv_scalar.py | 2fa22e7a8488b243915e7e8dce37bedc26e66379 |
| scv224_pilot.py | 17819b753ca165ac974c141505706a45984de384 |
| a1_control_variates.py | 72a42cd6b964a4c408b48bf801d3cacfc67ae10d |
| a1_n3_connection.py | 6645002bf2148d5a297a8a10e94b35bf9e6fd7f0 |
| a1_connections.py | 720863cc3c829cf2624a1d374bff781004e12332 |
| a1_n3_pilot_design.py | 9c5cd09818de25033767bfb768adcb61ff20677e |
| a1_n3_campaign.py | 471d355a4f607f22ebea332b79d709b7e616ab45 |
| a1_n3_managed_runtime.py | a818bda9eaef06f8939cf9c9960749f6b3064027 |
| rpe_resource_accounting.py | 95fd914943d4ee91779e03ba13e83b6f33bdaa87 |

## 3. APIの入力・出力・適用条件・不足

表中の分類は、そのまま可能／最小拡張が必要／当該用途に不適合。
「そのまま」も、新しい科学SOURCE/input/scopeとの照合と将来の実行許可は別途必要。

| 既存実装 | 入力→出力 | 適用条件 | SCVへ | 不足・誤用を避ける点 |
|---|---|---|---|---|
| [n3.ConditionalSampler.from_record/cells/draw][n3src] | actual整数mass、a/b、word源→direct $Q_h$ path | n0/n2、一occurrence、全5層・products→rotation | 固定m0ではそのまま可能 | LAWを再生成しない。一般q/r/Kには不適合。word capのabortを残す |
| [cv.ControlBank.joint_mean/pattern_mass][cvsrc] | indicator条件、保存law→exact joint/mass | 既存の因子化law・有効位置 | 既存SCV内部で再利用済み | 実H4の平均を今回呼び出していない |
| [scalar.metadata_from_bytes/select_fragment_control/FragmentControl][scalar] | LAW/catalog bytes・hash・label対応→群、全$\nu_h$、moment、identity | scalar rotation fragment群、metadataだけ、balance規則固定 | 最初の候補はそのまま可能 | 数値群選択も将来の科学scope。相関・価格を返さない |
| [scalar.ConditionalBank][scalar] | 群、元cell→条件付きmeanとfeature | 元5層とuを別identityで保持 | そのまま可能 | uを層IDにしない |
| [cv.fit_and_freeze/FrozenCV][cvsrc] | train bit/metadata→OLS slope、exact係数、residual/range | trainのみ、indicator、fresh cal/prod | scalar bankで再利用済み | 基底実装のminimum-norm挙動をSCVのrank救済と呼ばない |
| [scalar.fit_cells/CellModels][scalar] | h1 train→rank2/range guard、全層model | weight_bound=1、h1のみfit、他4係数0 | そのまま可能 | CellModel.applyはprod限定。cal用dispatchは未接続 |
| [cv.require_disjoint/evaluation_inputs][cvsrc] | stage/scope/stream→独立性の来歴検査、bounded inputs | 同scope、stream非重複 | 検査原則はそのまま、全層接続は最小拡張 | ID非重複はPRNG独立性の証明ではない |
| [n3.calibrate][n3src] | fixed-n residual、lo/hi、cal delta→count,width,s²,U,delta | k≥2、frozen係数の独立IID cal、真のsupport enclosure | 数値coreはそのまま可能 | 呼出し側が独立性/cell/固定nを保証。sample varianceだけの入力ではない |
| [n3.sufficient_shots][n3src] | Calibration、epsilon/bias、prod delta→十分N、両式、headroom | $e>0$、有効U/D、fresh bounded評価 | 数値coreはそのまま可能 | 全層weight・global bias・返却・familyは外側。最小必要数ではない |
| [intervals.configure/iv/upper/outward_upper/ceil_upper][intervals] | exact rational→interval log/sqrt/上端/ceil | 固定100桁文脈、finite端点 | そのまま可能 | 実variance・価格上界を生成する機能ではない |
| [n3.confidence_cells][n3src] | alpha,H→cal/prod各alpha/(16H) | 旧4方式×2軸 | 最小拡張が必要 | 現2方式なら保守的余剰になるが、round/abortを含むfamilyではない |
| [n3.choose_allocation][n3src] | p,U,price proxy,headroom,delta,cap,units→score/e/n | 全正層、cal後、prod前の有限候補 | coreはそのまま可能 | floor>1や不均等deltaは薄い拡張。proxy最小と真費用最適を区別 |
| [n3.plan_production][n3src] | 旧method/cal/model→ProductionPlan | modelはA1だけ、bankはglobal sampler.bank一致 | 最小拡張が必要 | SCV CellModelsをB1-OLタグで渡すと拒否。層別bounds/meansと明示deltaのwrapperが必要 |
| [n3.production_mean][n3src] | 元cell-ID付きplan、fresh batches→$\sum p_h\overline Z_h$ | 全quota・stage・scope・antecedents固定 | prod合成は既存SCVで再利用済み | plan構築側の保証を代行しない。cal/alphaを検証していない |
| [scalar.paired_evaluate][scalar] | 同じfresh評価rows→raw/residual sample分散・paired covariance | counts≥2、全5層、固定係数 | 診断にはそのまま可能 | precision=NOT_CERTIFIED。variance差のCIや十分Nではない |
| [pilot.allocate_evaluation][pilotsrc] | raw pilot s²/価格→floor付きheuristic quota | 5層、固定総N、raw shrinkage | confidence用途には不適合 | proxyはUではない。SCV residualへcap1を流用しない |
| [n3.SavedPrices.price][n3src] | 保存path/axis/compiler→full compiled metric/cache hit | 保存selected列、固定engine/profile | 固定m0価格は再利用可能 | 真の期待単価は返さない。新round・別compilerは未対応 |
| [a1_connections.selected_full_prices][connectionssrc] | live request、policy定義、compiler→axis full価格 | q1/δ.02/r1/K2/LD3/8qubit/threshold0のguard | 保存payloadにはSavedPricesを優先 | 実build/compileを行う関数。今回呼出し0。一般round価格器ではない |
| [cv.TrialCost/account_costs][cvsrc]、[n3.summarize_costs][n3src] | unique全stage価格→実現量子/古典vector | 同target/scope、UNKNOWN伝播 | そのまま可能 | 期待費用・coverageを証明しない。classical0 placeholderを置換して報告 |
| [scv224.Driver/LegacyCollector/cost_summary][scvdriver] | 注入services/streams/pool→固定224診断・実支出/参照費 | train/prodのみ、一回、固定quota、STOP | 取得・保存原則を再利用 | cal stage拒否、CALL_CAPS=224等。保証用driverとしてそのまま不適合。旧driverを変更しない |
| [Campaign.group/worker_trial/reconcile][campaign] | domain/path→bit先行fsync、価格、全tail分類 | fresh専用output/cache、固定services | 既存SCV collectorが再利用 | acquisition tag A1=SCV train、B1-OL=eval。手法名・purpose会計を分離 |
| [Runtime.request_stop/classify、durable_new][runtime] | 自分の実行scope→耐久STOP・重複拒否・tail分類 | qualified manager/own process identity | 将来の独立campaignで再利用 | 既存224 manager/job/lockに接続・操作しない。可用性保証ではない |
| [rpe.evaluate_rpe_round_candidate/build_rpe_resource_summary][rpesrc] | prep、r/K、PF/alpha/cost→raw N・資源・条件付きstatus | unit radius/effective eigenstate/strict pi3等 | 外側会計・制約の原則を再利用 | SCV校正N・層別価格・pre費・cal alpha・一般半径入力はない |
| [reconstruct_rpe_phase][decodersrc] | round位相・連続2冪q→nearest branch | 共通位相関係・strict eventを外部で保証 | 数学的接続を検討可能 | 各q候補列挙。長round性能・SCV end-to-end未検証 |

## 4. 将来必要な最小差分の設計

以下は**提案API**であり、存在する関数や今回実装した機能ではない。
旧固定SOURCE、scv224.Driver、実行中224-shotを編集する必要はない。

1. **層別cal adapter**：modelsと全5層のcal rowsを受け、元cellのvalidate_batchと
   train/cal間require_disjointを実施。h1はfrozen.apply(cal)とfrozen.output_range、
   inactive層はrawと[-1,1]を選び、既存calibrateへ渡す。
   CellModel.applyのprod限定を解除するより、既存FrozenCVのcal経路を薄く接続する。
2. **plan wrapper**：estimator identity=SCVとacquisition identity=B1-OLを区別し、
   exact masses、cell別Calibration/range、明示delta、global bias、floor/cap、quotaを保存する。
   choose_allocation/sufficient_shots/production_meanを再利用する。
   旧plan_productionのA1-only条件を迂回して「既存保証済み」と表示しない。
3. **独立domain・全費用保存**：新しいpurpose/campaignでtrain/cal/prodを分離する。
   既存Domainのmethod列へ未対応SCV tokenを直接入れる必要はなく、
   準備済みcollector同様、取得規約と推定器の名前を別に保存できる。
   fresh output/cacheと全tail/UNKNOWN、pre費、cal/prod alphaの保存を追加する。
4. **RPE接続は後段**：round座標certificate、半径・bias根拠、層別価格、pre費、失敗familyを
   持つ外側recordを設計する。現RPERoundCandidateのcertified欄へSCV結果を代入するだけにはしない。

新しい回帰法、control、一般統計会計framework、管理システムを作る必要は認められない。
一方、期待費用保証は実現会計関数だけでは得られない。
既存a53c設計を必要範囲で接続する課題は残り、今回は実装・統計方式の正式採択を行わない。

## 5. 既存テストと将来の最小検査

次の固定SOURCEを読んだ。**今回は一件も実行していない。**

| 既存テスト | 既に用意された検査 | 将来の差分に必要な検査 |
|---|---|---|
| [test_scv_scalar.py][testscalar] | toy全pathの条件付きmean/moment、不偏性、h0定数、rank2、range、paired合成 | 同じtoyを使ったcal dispatch、全5層bounds・mean identity |
| [test_scv224_pilot.py][testdriver] | toy224、単一課金、固定quota、SOURCE変化、価格欠測、STOP先行、再実行拒否、CLI run拒否 | 新cal段階とquota freeze順、旧driver不変、cal/prod再利用拒否 |
| [test_a1_control_variates.py][testcv]、[test_a1_n3_connection.py][testn3] | frozen apply、zero sample varianceでも正U、外向き十分N、family union、全正層・cap、価格UNKNOWN | 明示2方式/X/Y/5層family、追加pre費、centering/bias headroom |
| [test_a1_n3_pilot_design.py][testpilot] | heuristic floor/整数quota・scope | heuristicを保証Uとして受け付けないsemantic検査 |
| [test_rpe_certification_boundary.py][testrpe] | pi/3境界、緩和なしphase/alpha和、status/energy scope | SCV座標event・追加cal alpha・一般半径を誤って旧statusで認証しない |

人工検査はAPI恒等式・bound条件・漏洩拒否の確認であり、実H4での分散削減証拠ではない。
新cal/production科学runを行う場合の最小対象は、まず同じm0有限signal、同じh1 scalarだけである。
係数・target・群の固定、$\epsilon/\alpha$、全stage、返却、費用対象とSTOPを別途具体化し、
明示許可を得てから進む。旧224-shotのtrain/evalを繰り返す計画にはしない。

## 6. 224-shot完了後に受け取る項目

以下は将来の受領schemaの意味を定義する表。結果ファイルは今回取得していない。
準備版serializerに存在するものと不足するものを分ける。

| 項目 | 固定準備SOURCEにある出力・処理 | 次の判断での使い方 |
|---|---|---|
| h1 control出現率 | CONTROL_FROZENのexact $\nu_1$、X/Y_FITのtrain_feature_counts、paired cellsのfeature_counts | exact質量とtrain/eval実現countを区別。低出現・全同値なら係数精度を保留 |
| 群・LAWのidentity | CONTROL_FROZEN、metadata/control identity、SOURCE照合 | 探索対象・actual $Q_h$中心化が一致しているか |
| X/Y係数・rank・range | X/Y_FITのactual Fraction係数、rank,width,guard、失敗fitも保存 | guard通過と効果を区別。別軸や別roundへ移植しない |
| raw/residual sample variance | X/Y_PAIRED_RESULTのcells、推定平均分散 | 層内追加効果の探索材料。$U_h$や0母分散へ変換しない |
| paired covariance | 同resultのcell covariance・mean covariance・差のsample-based分散 | paired平均差の不確かさ。variance差のCIとは別 |
| all-equal・低出現count | raw/residual_all_equalとfeature_counts | power不足、非検出、推定不安定を保持。都合のよい層を除外しない |
| 実現RZ・train/eval内訳 | SCV_ACCOUNTINGのphysical/train/evaluation reference、worker reconcileのACCOUNTING/CALL_COUNTS | 実支出一回、baseline評価参照、SCV全費用を区別。全tail完備が必要 |
| classical setup/engine/compile | SCV_CLASSICALのfit/aggregation/driver wall、旧worker timers | metadata setup・outer wall/RSS・manager overheadはactivation側で追加計測が必要。未記録はUNKNOWN |
| 中断・未返却・guard失敗 | STOP_REQUESTED/STOP、RECEIVED、ATTEMPTS、worker journal、finalizer | 完了者だけ選別しない。未価格・未返却を0で埋めない |

Driverの返却envelopeが不完全な場合、SCV_ACCOUNTINGだけで全tail費用が完備したと判断しない。
既存worker reconcileとATTEMPTSを、将来許可された受領時に確認する必要がある。
今回これらのlive fileを読んだり、reconcileを呼んだりしていない。

## 7. 次へ進む意味・縮小・終了を判断する材料

数値の新GO/NO-GO閾値は採択しない。以下は将来の判断に必要な**条件の整理**である。

| 完了報告での状況 | 次の最小判断候補 | 現時点で断定しないこと |
|---|---|---|
| identity・rank/range・tailが整い、h1に出現とfresh residual低下の兆候がある | 既存range式、明示精度、追加train/cal費から保証付き校正に意味があるか検討 | sample低下＝shot削減、期待RZ改善 |
| X/Yで傾向が異なる | 軸別の余地と全X/Y費用を分けて整理。将来scopeの縮小候補とする | 有利な一軸だけでcomplex signal/RPE改善を認定 |
| all-equal・低出現・rank不足・広い不確かさ | 非検出と真の無相関を区別し、範囲を拡げず保留・縮小を検討 | 真分散0、control無効の証明、追加shot自動投入 |
| 学習費に対し改善が乏しく、range/cal/floorで十分Nがほぼ減らない見込み | 同m0の保証校正を縮小・見送り・終了する候補 | 数値break-evenや有望性閾値が採択済みという表示 |
| 層内相関の兆候がなく、追加bitの費用を正当化する根拠がない | 224-shotを探索的負の材料として保存し、追加科学の終了を検討 | 少数pilotから一般的に相関0と結論 |
| technical STOP・価格欠測・不返却 | 科学比較は未判定。既知費・UNKNOWN・STOPを保持 | guardを緩める、refit/retry、欠測再測定 |

決める順は、完了済み固定結果を受領できるか→探索的効果と不足→保証付き最小scopeの価値→
必要なら新scopeの明示採択・許可である。224-shot途中結果を監視して次段階を自動起動しない。
既存係数を凍結して独立calへ進む単発評価と、学習から再実行するpolicy評価を選び分ける。
前者の条件付き費用を後者の全期待費用と呼ばず、過去train費・償却を表示なしに落とさない。

## 8. 今回の保全・実行境界

独立worktree /tmp/o2-scv-finite-confidence-20261107、
専用branch codex/o2-scv-finite-confidence-20261107、base=560c6092。
commit対象は本directoryの5 Markdownだけ。既存main、科学SOURCE、raw、manifest、
研究概要・研究ノート・索引の既存本文を変更しない。
本調査は新研究判断・検証結果・検証経路の登録ではないため、それらのstatus更新は行わない。

新科学計算・保存値再解析・H4 mean/variance/RZ算出・Born・RNG/replay・回路build/compile・
新control・cal/production・詳細GPT/Claude科学レビュー・C1解凍は全て実施しない。
224-shotのjob/process/worker/lock/marker/output/cache/stream/managerは取得・操作しない。
今回の5文書を公開し、remote SHA・本文・blob/treeを照合した時点でmandatory STOPする。

[policy]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/docs/archive/server1_main_collection_20261103/b05c68a693d31e3a15652faccfc4e6f0a05f8c34/research_project_instructions_unified_2026-10-25.md.original
[agents]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/AGENTS.md
[project]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/PROJECT_MAP.md
[synthesis]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/docs/research/研究概要・現状.md
[status]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/VALIDATION_STATUS.md
[manifest]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/artifacts/validation_manifest.json
[preparation]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/docs/research/scv224_preparation_20261107/REPORT.md
[design]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/84e021b6a8decde7fa3ed3fa44d7148cc5428061/docs/research/intrastratum_cv_design_20261107
[a1theory]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8d8a290fa4d4cf24f968aea27953ad5422e418b3/docs/research/a1_b1ol_theory_20261107/A1_VS_B1OL_THEORY.md
[n1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/744e1a09661ebba6841fe5faf456f88c41c47780/docs/research/o2_a1_new_target_n1_20261106/REPORT.md
[n2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5e9bd08068738cdf1070bdc43bee8194b113ae33/docs/research/o2_a1_new_target_n2_20261106/REPORT.md
[engine]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdf00cee3c87d804845497afc89ac6a71f3b3f59/docs/research/o2_a1_engine_qualification_20261106/REPORT.md
[pilotreport]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3c8e55a2728f86a1eda17a2e0105050a637e7c86/docs/research/o2_a1_n3_pilot_admission_20261107/REPORT.md
[expected]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a53c1d6eedea0d85fb484425db9012aa89738129/docs/research/o2_a1_n3_decision_expected_rz_20261106/EXPECTED_RZ_COST_GUARANTEE_PLAN.md
[scalar]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/scv_scalar.py
[scvdriver]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/scv224_pilot.py
[cvsrc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/a1_control_variates.py
[n3src]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/a1_n3_connection.py
[connectionssrc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/a1_connections.py
[pilotsrc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/a1_n3_pilot_design.py
[campaign]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/a1_n3_campaign.py
[runtime]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/a1_n3_managed_runtime.py
[intervals]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trottertracks/algorithm_codesign/df_correction_comparison/intervals.py
[rpesrc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trotterlib/rpe_resource_accounting.py
[rpedoc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/docs/rpe_resource_accounting.md
[decodersrc]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/src/trotterlib/rpe_four_round_phase_validation.py
[testscalar]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/tests/tracks/algorithm_codesign/test_scv_scalar.py
[testdriver]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/tests/tracks/algorithm_codesign/test_scv224_pilot.py
[testcv]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/tests/tracks/algorithm_codesign/test_a1_control_variates.py
[testn3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/tests/tracks/algorithm_codesign/test_a1_n3_connection.py
[testpilot]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/tests/tracks/algorithm_codesign/test_a1_n3_pilot_design.py
[testrpe]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/560c6092bfb5e71af8eedc4a4be5416922764f98/tests/test_rpe_certification_boundary.py
[glynn]: https://web.stanford.edu/~glynn/papers/2002/GSzechtman02.pdf
[mp]: https://arxiv.org/pdf/0907.3740v1
[bernstein]: https://webapps.math.uci.edu/~rvershyn/papers/HDP-book/HDP-1.pdf
