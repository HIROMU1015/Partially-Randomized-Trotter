# A1 N3技術接続（人工・mock検証のみ）

2026-11-06 JST、SERVER1 GPUA100-gleap-01。**技術接続完了／実科学入力の取得は未実行。**
固定対象は保存linear H4/1.00 Å/STO-3G/legacy DF12/8 system qubits/全4電子number sector/L_D3、m0/q1/δ=0.02/r1/K2/coefficient_atol=0、X/Y。N1 actual finite-grid Q、selected `support_run_le_1`/`boundary_optimized`、N2保証groundを保持する。

## 固定資料とSOURCE保全

- N1：`744e1a09661ebba6841fe5faf456f88c41c47780`。REPORT/SOURCE/LAW/POLICY/保存payloadを確認。
- N2：`5e9bd08068738cdf1070bdc43bee8194b113ae33`。REPORT/SOURCE/STATE_RESULTと保存state/witnessのidentityを確認。
- Engine認証：`fdf00cee3c87d804845497afc89ac6a71f3b3f59`。REPORT/SOURCE/ENGINE_ERROR_CERTIFIED/GATE_CERTIFICATESを参照。
- 新target契約：`2b2e6c1bc3b2d201d1a6d1238172187d68aea6ab`。一次targetの採択済み部分と、未採択の旧測定案を区別する。
- 共通指示：実際の `research_project_instructions_unified_2026-10-25.md`、適用AGENTS、PROJECT_MAP・研究概要を確認。旧SOURCE・結果・STOPは変更しない。

認証済み総座標誤差の表示上界 `b_engine≤7.8079008e-7` は引用のみ。新上界・preparation/CDF/state/spectralの再計算はない。
[STATIC_INPUT_CHECK](STATIC_INPUT_CHECK.json)と[SOURCE](../../../artifacts/o2_a1_new_target/2026-11-06/n3_connection_technical/SOURCE.json)に固定identity・bytes確認・追加SOURCEのhashを記録する。

## 実装と再利用

新しい[adapter](../../../src/trottertracks/algorithm_codesign/a1_n3_connection.py)、[人工test](../../../tests/tracks/algorithm_codesign/test_a1_n3_connection.py)、[人工専用入口](../../../scripts/tracks/algorithm_codesign/check_a1_n3_connection.py)を追加した。元の数値kernel・solver・sampler・builder・compiler・テストには差分がない。

| 接続 | APIと再利用先 | 保持する意味 |
|---|---|---|
| 固定input・環境 | `inspect_certified_bundle` →既存 `inspect_fixed_bundle`、固定Git bytesとPROFILE_LOCK。`verify_runtime`→既存profile | N1/N2/proof・旧538 SOURCE。環境/native/dependency差はSTOP。自分専用cacheのpathだけ比較から除外 |
| 保存証明の復元 | `attach_saved_proof`→保存matrixのFraction復号 | 元認証の全matrix/error/keyを復元。新しいinterval生成・qualificationはしない。既存probabilityは毎gateのraw matrix hash・列長を再照合 |
| engineとfacade | `load_certified_services`→既存 `SavedEngine.probability`、`MeasurementFacade.acquire` | N2 stateをprivateに保持。exact gate適用・norm/Born比・[0,1] guard・threshold_countをそのまま使用。X/Y・ordinary diag(I,U)・phase・Sdg→H・bit符号を変更しない |
| actual lawと直接条件付き | `ConditionalSampler`→既存 `OccurrenceLaw` / `proposed_control_banks` | N1の整数count m_i=2^53 Q_iを復号。CDFを再生成せず、nominal probability・float choice・dyadic別proposalを使わない |
| streamとdata | `Domain` / `Streams`、将来の `pcg64_streams` →既存 `Measurement`/`Batch`/`require_disjoint` | method×axis×stage×stratum×path/bitで分離。trial再使用・同handle・別method混入をSTOP。別seedの独立性を証明したとは呼ばない |
| 学習とrange | `merge_training` / `freeze`→既存 `fit_and_freeze` / `FrozenCV` | 3 slopes＋intercept。λはbinary64結果をexact Fractionとして凍結。rank不足・D_CV guard違反でSTOPし、feature削除・clip・再fitなし |
| confidenceと配分 | `calibrate` / `sufficient_shots` / `plan_production`→既存directed interval helper | 独立fixed-n cal、外向きlog/sqrt/ceil。全正層、全方式両軸を同じconfidence routeへ接続。production前にquota/ε_hを凍結 |
| production合成 | `production_mean`、既存CV補正・disjoint検査 | cal/prod混入、層欠落・違うquota・wrapper/weight/metadata/domain不一致はSTOP。exact層質量で合成し、再正規化なし |
| full価格 | `SavedPrices.price`→既存 `SavedEngine.selected_circuit` / `rte_compiled_cost` / cache | 同じselected logical full列に1 bitのmeasurementを追加し、既存compilerで価格化。1軸だけ、状態準備だけ除外。別basis/他axis価格を生成しない |
| 全段階会計 | `collect` / `summarize_costs`→既存 `TrialCost` / `account_costs` | train/pilot（train stage）・cal・prodを全課金。cache hitでもshot RZ課金。古典費は別vector、欠測はUNKNOWN |

既存 `selected_full_prices` はlive `DFPartialS2RepeatedRequest` を前提とする。保存N1 payloadからそのオブジェクトを再preparationで作り直す代わりに、**認証済みselected_circuitを直接呼ぶ最小価格adapter**を用いた。同関数の固定compiler profileを保持し、実compiler/cacheそのものは再利用する。
測定は理想logical列、compileは同じ列の資源指標取得に限る。実compiled回路へのstate作用・native synthesis精度の認証はしない。

## 条件付き取得の正当性

B0/A1のunconditional取得は、保存整数countを逆引きすることでcanonical PRと同じQを与える。元 `iter_sample_rte_events` のSOURCEは変更せず、PRNG列のbitwise一致は要求しない。
Lは適用順products→rotationで保持し、event構成は元SavedEngine内の `_make_event` がrotation-firstへ戻す。
B1-Oはorderを固定し、B1-OLは既存5層のorder・L0=a/非a・L1=b/非bを固定する。その他positionは元label countに従う。
許容集合Aのcount和Sに対する一様整数を、53-bit wordの `floor(2^53/S) S` 未満への整数棄却とmodで得る。これは希少stratumが出るまでpathを棄却する手法ではない。単一labelの条件は直接配置する。
全5正層は必ず残し、真にzero-gridのlabel/層だけ取得対象外にする。nominal event_probabilityや追加importance weightを掛けない。
有限word capは超過時STOPで、分布の切替え・fallbackはない。完走campaignだけを選別してconditional coverageを主張せず、未完了campaignも保存する。

## Oracle境界・独立性・誤差

facadeの戻り値は `Measurement` のbit、path/order/phaseとstream/trial/target bindingのみ。state/vector/E0/gap/Born確率/真signal/真varianceを返さない。価格serviceも回路metricだけを返す。
これはPython interfaceの隔離で、悪意あるintrospectionを防ぐsecurity sandboxではない。将来driverはengine内部参照をCV fit/配分へ渡さない。Born値・path信号の表をcacheしてCVに使う処理はない。
53-bit thresholdは元engineの `ceil(2^53 p)` を使う。state/norm/gate/Born/gridの認証は再利用し、未知誤差を0で穴埋めしない。νと補正がexact Fractionなら追加centering項だけ0。
streamはdistinct SeedSequence children/PCG64 raw>>11を将来の具体実装候補とした。今回PRNGは生成・使用していない。理想の独立uniform-word modelという仮定は残る。

## 人工検証と限界

[TESTS.json](../../../artifacts/o2_a1_new_target/2026-11-06/n3_connection_technical/TESTS.json)：**新規22＋既存A1 connection9＋既存CV15＝46件PASS**。全raw logを同directoryへ保存した。
人工整数lawと全toy pathの層質量、zero support、整数棄却endpoint・cap、bit-only API、X/Yとphase、exact kernelへの2-mode fixture接続、証明復元・環境不一致STOP、rank不足・range guard、cal variance enclosure・外向きshot十分性、全方式/axis union、cal→凍結→prod→全RZ課金、cache hit・price失敗時bit保存を確認した。
初回の人工境界wordを無限に繰り返すfixtureは自分のtest processだけを停止して有限tapeへ修正した。real科学retryは0。既存libraryのdeprecation warningはログに保全した。

人工PASSはCVの改善・測定費用削減・実分子performanceや独立外部再現/CIの証拠ではない。実科学入力でのgate作用/Born/compile/variance/N/費用は0件。
`load_certified_services` / `pcg64_streams` / `collect` / `SavedPrices.price` の**実入力呼出しは別許可まで行わない**。
本入口は人工検証専用で、科学実行CLIは付けていない。採択後に必要な薄いcampaign driver・serializerを同APIで作り、全SOURCE/環境を実入力前に固定する仕様を[契約草案](N3_EXECUTION_CONTRACT_DRAFT.md)に示した。

**MANDATORY STOP：技術接続・草案・成果物の公開照合まで。実N3は未許可・未実行。**
