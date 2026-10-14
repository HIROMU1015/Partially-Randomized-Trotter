# Track A D0：保存scalar再会計の技術準備・実行承認待ち

**技術準備は完成。実H4/H6保存値での旧数値再現、新ε・費用地図・P感度は未実行。** 独立moduleと33件の限定人工検査を固定し、8入力のbytes/SHA・候補集合・出力・一致条件を実行契約へ結合した。新科学計算0、実データの新数値再会計0、H8変更0、新grant0、原稿改訂0。`approved=false`、`runtime_authorization=false`、`allowed_cpus=[]`、`absolute_launch_command=null`、自動retryなし。

資料入口：[bundle README](../../artifacts/resource_applicability/track_a_d0_saved_value_preflight/2026-10-14/README.md)。本書は技術準備記録であり、GPT詳細科学レビュー・科学的採択・実行承認の記録ではない。

## 1. 調査対象・固定SOURCE

設計branch `track-a-resource-evaluation-expansion-design-20261014` のcommit `2ccc79502f043b4885e2337ffa98ff45f1875ae4` と、正式共通ノート `4675730e7e40d25ba3618334a4a0e28b4bd2f145` はorigin SHAと一致した。[追加検証設計](track_a_resource_evaluation_expansion_design.md)と[科学review判断](track_a_scientific_review_decisions.md)を維持する。

適用共通指示は利用者指定の `research_project_instructions_gpt_codex_review_approval_2026-10-12.md`（50,268 bytes、SHA256 `709524f856a3117c85042309c390a3697d9b2c8c8d79af6b6c290e9d20814bcf`）。通常の技術検証を先行し、実データ解析と次stageの実行承認を分離する。

|系列|固定科学／実行SOURCE|原本result commit|
|---|---|---|
|H4 native fixed16|`217c971dd537b1cc7fbc13742bc0f6c263ff7f45`|`0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41`|
|H6 exploration92|science `5a903aea2492c2d211c1e15e7737df0d50694c12`／execution `78a39051d8b6d41c33d592ecfe172f8824adfaeb`|`df99d6ee28942df8d12aac3c4db7cd202df55d77`|
|H6 confirmation union8|`87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`。原確認159group＋追加97groupの来歴は原本に保持|`321cd8c3066194fe98741cbebbcdba1515d1deaa`|
|独立D0解析SOURCE|`fa75c7997e135da7b7a0c4e23949e47252bd7a2e`|今回は準備資料のみ。新数値resultなし|

科学ライブラリ・元runner・元tests・raw/checkpoint/ledger・grant・compiler optionsを編集していない。新しいbranchは `track-a-d0-saved-value-preflight-20261014`。既存69 worktreeのうち68は開始時のHEADとdirty/untracked一覧hashが一致。H8は作業中に別担当の資料commit更新を検出し、公開originとの一致を確認した。H8側を変更・巻き戻しせず、同時更新を別記した。

## 2. 取得済みの科学データ

[入力索引](../../artifacts/resource_applicability/track_a_d0_saved_value_preflight/2026-10-14/d0_input_inventory_v1.json)に108候補の保存cell、reference、signed axis bias、empirical u、log_B／margin、旧shot・適格性、標本数、欠測理由、H/DF/state/SOURCE/result/compiler identityを収録した。科学値は直接projectionであり、新しい算術結果を追加していない。

|入力alias|固定資料|用途|
|---|---|---|
|`h4_signals.json`|H4 resultの `signal_metadata.json`|fixed16の保存scalar・旧precision_rows|
|`h4_wrappers.csv`|同 `wrapper_metrics.csv`|532行、primary528軸＝264pair|
|`h4_old_map.json`|同 `resource_map.json`|旧4εのRZ一致gate|
|`h6_signals.json`|H6探索監査にbyte bindingされた保存 `signal_summary.json`|92候補の正確なlog_B_margin・u・旧precision_rows|
|`h6_exploration_wrappers.csv`|H6探索resultの `wrapper_metrics_v1.csv`|1188行、primary1184軸＝592pair|
|`h6_exploration_old_map.json`|同 `exploration_resource_map_v1.json`|旧4ε×factor1/10の4metric一致gate|
|`h6_confirmation_wrappers.csv`|H6確認resultの `wrapper_metrics_v1.csv`|512軸＝256pair。8候補各32pair|
|`h6_confirmation_old_map.json`|同 `confirmation_resource_map_v1.json`|確認と再利用det baselineの旧4metric一致gate|

8実行入力は合計 **4,984,136 bytes**。公開原本11fileはGit blobとGitHub固定commit URLのbytes/SHAを照合した。H6 scalar JSONは **1,382,011 bytes**、SHA256 **`b9f6cb3dc1a168ddc409833c78a617a7ddd05a3a41e0665e9dec47346d516dfe`** が既存探索監査と一致し、homeの独立受領領域へbyte-onlyで凍結した。元JSONやraw runtimeは新commitへ含めない。全inputの完全SHA・公開リンク・保存roleは索引の`files`が正本。private alias/path mappingはGit管理外に置く。

H4は1.0Å、STO-3G、T=.8、legacy explicit DF12、8mode/α2β2/sector36。H6は1.0Å、STO-3G、T=.8、signed generation-weighted projection・tol1e−8/cutoff0のDF19、12mode/α3β3/sector400。同compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、symmetric directional full wrapper、structured native Gaussian lowering。H4のlegacy stateとH6 saved stateのH/state hashをそれぞれ索引へ固定した。同compiler label/hashから旧host出力bit完全同一性は推論しない。

初回公開確認ではH8は起動commit `38d386c8732295e6fab5d6fcbd47fbe586dc207d` のPENDINGだった。保全監査中に別担当が正式結果を公開し、origin `track-a-h8-resource-map-resume02-20261014` とresult **`681ec23c93fdd1090d325908f4b155a79115afdf`** の一致を確認した。[H8公開受領receipt](../../artifacts/resource_applicability/track_a_d0_saved_value_preflight/2026-10-14/d0_h8_public_result_receipt_v1.json)に正式SOURCE `c31ef424175c02379eafebe8fc6c7548c1d9aabe`、seed `cb8a56c6b2a772827275f9200b02638030f33f72`、入口・完了監査・scalar schema・6metric CSV等6fileの公開bytes/SHAを記録した。状態は **FORMAL_RESULT_PUBLIC_METADATA_RECEIVED_D0_EXCLUDED**。今回の8実行input・候補scopeは拡張せず、H8数値再会計を開始していない。live job・raw・observer・workerには触れていない。

## 3. 既存データだけで再解析できる範囲

[coverage matrix](../../artifacts/resource_applicability/track_a_d0_saved_value_preflight/2026-10-14/d0_coverage_matrix_v1.json)は6分類を明示する。

|項目|分類・扱い|
|---|---|
|保存reference/bias/u/log_B/旧shot、6指標のwrapper整数|SAVED_DIRECT|
|新εのheadroom/適格性/整数shot、未集計weighted metric、paired SE、P envelope|DERIVABLE。今回の実データ値はnull/未計算|
|H4 geometry非RZのpaired SE|RAW_REQUIRED。公開axis meansだけではcovariance不足。今回のnative入力に含めない|
|native B0+I、新signal/trajectory、実state preparation費用、新εの追加確認|NEW_SCIENCE_REQUIRED。自動実行なし|
|H6 B0 p5/q1,2,4,8のcompiled費用|UNAVAILABLE。旧精度集合で不適格につき取得なし。0で補完しない|
|母平均winner、同時保証、hardware critical path/T-depth|UNAVAILABLE。保存平均から保証を作らない|
|H8正式結果・SOURCE・公開wrapper/scalar資料の存在|SAVED_DIRECT metadata受領。H8の新再会計は今回のD0契約外|
|六距離H4 geometryと1Å native、DF政策を統一したH4→H6 size scaling|INCOMPARABLE。別case studyの併記は可能、strict scalingやmixed optimumにしない|

H4 fixed16はB1(full12 S2/S4)8＋B2(prefix3)8であり、native B0/B0+Iは登録されていない。H6探索92はB0 12＋B1 8＋B2 72。確認はB2 union8の32独立pairと探索det baseline20候補の登録（16候補に費用あり）を分離する。探索B2の8pairを確認32pairへpoolしない。H4の固定32pairをH6選抜後確認と同じ統計契約にしない。

## 4. ε=0.0005の解析仕様

後続のdomainは旧 `.05/.01/.005/.001` ＋新 `.0005`、factor1/10。H4 fixed16、H6 exploration92、H6確認union8＋保存det baselineが対象。新εに必要な新scientific callは0だが、数値実行は今回未認可。

`h_a=ε/√2−|bias_a|−factor×u_a`、`log_B_upper=log_B+factor×log_B_margin`。元SOURCEの `axis_headroom`／`u_aware_shots` のfunction textをそのまま独立fileへ隔離した。[凍結規則receipt](../../artifacts/resource_applicability/track_a_d0_saved_value_preflight/2026-10-14/d0_legacy_rule_receipt_v1.json)でH4/H6科学SOURCEとc31ef424…のfunction AST一致を確認する。

`ceil(2 B_upper²/h_a² log80)` の演算順序を保持する。`log_n<log(2**52)` の場合だけ旧binary64整数診断へ入り、それ以上はLOG_DOMAIN_ONLY／整数nullを維持する。NaN/inf/bool、不正な整数metric、float overflowを拒否する。両軸の経験的u/headroom≤.01、整数Nあり、元の三状態適格性を満たす行だけ費用対象。新estimator・certified allowanceは導入しない。

後続出力には各軸bias/u/B_upperの構成・headroom・shot・guard、familyの費用点最小、ε間candidate切替、確認coverageを出す。ここには新headroom/N/Gの数値を載せない。前設計のheadroom censusは保存済み診断として参照できるが、正確なlog_B/ceil/費用再会計のPASSとは扱わない。

`.0005`に上位候補がunion8外で現れた場合、探索8pairの点推定と「確認未取得」を出す。新上位候補の確認を自動追加・旧grant再利用しない。新εは新signal・独立証拠・母平均winnerを増やさない。

## 5. 6指標の費用比較仕様

RZ/CX/RZ_depth/CX_depth/`depth`(total depth)/`size`を独立した単位で会計する。`G_m=Σ_axis N_axis mean(C_m,axis)`。H4旧RZの平均はpaired整数totalのmean、H6は軸mean×Nのsumという元のbinary64演算順序を保持し、正確な有理数の診断値も並記する。候補選択は旧式のG_pointとcell_idの順を維持する。

同一ε/factor/familyごとに、(1)RZ最良候補へ他metricを表示、(2)metricごとに候補を再選択、を別fieldへ出す。one-shot軸mean/max depthとweighted depth-shot sumを分ける。weighted depthはserial会計量であり実機wall time・並列shotのcritical path・T-depthではない。

旧weighted mapはH4がRZ、H6がRZ/CX/depth/size。H4の残る5metricとH6のRZ_depth/CX_depthは保存wrapper整数からDERIVABLEで、今回まだ算出していない。

## 6. P感度の解析仕様

`G_j(P)=G_j(0)+N_total,j P`、P≥0。Pは仮想的な共通1-shot overheadで、実state preparation回路の取得ではない。RZ/CX/各depth/size per shotの単位を混ぜない。

固定pairはP=0で各metricが選んだfamily代表候補の組を固定し、`(G_b−G_a)/(N_a−N_b)` を求める。平行、同一直線、負domain交点を区別する。一方lower envelopeは全適格・費用取得済み候補でPごとに選び直す。exact affine hull、区間、交点のties（区間幅0の候補含む）を保存する。固定pairの交点を全面再選択の切替と表示しない。

Pのline interceptには保存規則のG_pointを使用し、Python整数/binary64値を有理数へ正確に写して交点を診断する。これは新shot estimatorや物理costの換算ではない。P gridを無制限に増やす代わりに非負domainのpiecewise envelopeを出す。

## 7. Missing／uncertaintyの取扱い

native CSVのcos/sinは `(cell_id,phase,replica)` でpair化し、seed/event_digest/compiler_hashの一致を要求する。axis重複・欠測・phase混合・不正整数・標本数不一致はSTOP。ordinary感度4軸はprimaryから外す。

`SE=stdev(N_cos C_cos,i+N_sin C_sin,i)/√n`。nはtrajectory pair数で、2n wrapperを独立標本と数えない。deterministic n1のSEはnull。native保存整数なら全6metricのpaired SEがDERIVABLE。axis meansのみの参照資料ではSEとnをnullにし、covariance未保存という理由を残す。

INELIGIBLE、NUMERICALLY_UNRESOLVED、MISSING(cost未取得)、NOT_SAVED(field未保存)、UNAVAILABLE／RAW_REQUIREDを区別する。sample SEを信頼区間・tail/population boundへ変換しない。選抜と確認のscope、ε追加でも同じ標本である点、未知のrare tailを保持する。

## 8. 数値再会計コード・テストの状況

- [独立module／未認可CLI](../../scripts/d0_saved_value_reanalysis_v1.py)
- [元shot関数の隔離copy](../../scripts/d0_frozen_shot_rule_v1.py)
- [人工unit/CLI fixture](../../tests/test_d0_saved_value_reanalysis_v1.py)
- [人工検査receipt](../../artifacts/resource_applicability/track_a_d0_saved_value_preflight/2026-10-14/d0_synthetic_tests_v1.json)／[log](../../artifacts/resource_applicability/track_a_d0_saved_value_preflight/2026-10-14/d0_synthetic_tests_v1.txt)

**33件PASS、fail/error/skip0**。単一test process・thread1、AS512MiB/CPU60秒内、実測peak RSS27,910,144 bytes。整数ceil・binary64 normalization・log-only・u/headroom・paired covariance・整数積・欠測・confirmation混合拒否・別metric選択・P hull/ties・input hash・one-shot・output STOP・pressure/OOMを人工fixtureで検査した。CLI全経路も人工scalarのみ、資源interfaceはmock。local technical testsであり独立科学review/CIではない。

既存private venvのPython3.12.3で実行し、install/upgrade/設定変更0。[解析profile](../../artifacts/resource_applicability/track_a_d0_saved_value_preflight/2026-10-14/d0_analysis_profile_v1.json)へinterpreter/stdlib hashを記録した。保存資料には歴史的Python3.11.0rc1があるため、function AST一致／人工合格だけで実保存数値のbit同一性は宣言しない。後続のexact旧値gateが不一致なら停止する。

## 9. 後続の実行契約案

[契約](../../artifacts/resource_applicability/track_a_d0_saved_value_preflight/2026-10-14/d0_reanalysis_contract_v1.json)、[検証protocol](../../artifacts/resource_applicability/track_a_d0_saved_value_preflight/2026-10-14/d0_validation_protocol_v1.json)、[予算](../../artifacts/resource_applicability/track_a_d0_saved_value_preflight/2026-10-14/d0_execution_budget_v1.json)が正本。

一度の承認対象は、固定SOURCE＋8scalar/CSV inputsで、旧4εの完全precision_rowsと全公開旧weighted mapを一致検査し、その後に5ε×factor1/10×6metric、P固定pair/envelopeを一度再会計・監査・公開しSTOPすること。候補は16＋92＋28（確認8＋det baseline20）、最大136候補scope／1,360precision行／8,160metric行。探索と確認の同cellは別scopeであり、新標本ではない。実数値検査は最大1,088旧precisionケースを計画した。代表caseにH4 B1 p12/q1とB2 p3/q8/r2/K4、H6 B0 p10/q1・未取得p5/q1・full S4 q8・確認B2 p5/q8/r2/K2を固定したが、今回実行していない。

単一process・thread1・同時CPU1core、affinity変更なし、AS512MiB/RSS256MiB、wall300秒/CPU240秒、output32MiB＋log/terminal reserve2MiB、64inode。fresh available128MiB/128inode、effective memory2GiB以上、host full PSI avg10<1%、可視own cgroup full PSI0、host/ancestor OOM増分0を案とする。資源観測はself/host/own ancestorの読取のみで、追加worker/observer/tasksetなし。現在の小さな読取snapshotはmemory/FS gateに合格したが、将来起動時はfresh検査する。quotaが将来有効ならheadroom receipt未確認のまま開始しない。

outputは候補accounting、旧値一致監査、metric selection、P感度、provenance、terminalの6JSON。streaming保存で一時二重copyを避ける。exclusive outputとdurable one-shot markerを消さず、失敗partialを保持する。未記録exit/timeを補完しない。AS/CPU強制終了等でterminal欠測なら成功と扱わず外部exit statusと欠測を監査する。scientific ledger消費/返却/resetは0。

以下は**未実行command草案**。`approved_d0_authorization_v1.json`は存在せず、新grantは発行していない。利用者の別実行承認後、contractの最終byte digest、解析SOURCE/runtime profile、inputs/outputへ結合した新D0認可を固定する。現在の`absolute_launch_command`はnull。

```bash
env GIT_OPTIONAL_LOCKS=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  TMPDIR=/home/AbeHiromu/projects/h4-handoff-evidence/20261014/d0-saved-value-preflight-v1/planned_outputs \
  /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -P -B \
  /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d0-saved-value-preflight-20261014/scripts/d0_saved_value_reanalysis_v1.py \
  --contract /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d0-saved-value-preflight-20261014/artifacts/resource_applicability/track_a_d0_saved_value_preflight/2026-10-14/d0_reanalysis_contract_v1.json \
  --authorization /home/AbeHiromu/projects/h4-handoff-evidence/20261014/d0-saved-value-preflight-v1/approved_d0_authorization_v1.json \
  --inputs /home/AbeHiromu/projects/h4-handoff-evidence/20261014/d0-saved-value-preflight-v1/input_paths_v1.json \
  --output /home/AbeHiromu/projects/h4-handoff-evidence/20261014/d0-saved-value-preflight-v1/planned_outputs/d0-reanalysis-one-shot01
```

失敗後の自動retry、cap自動増量、source差替、入力補完、次stageはしない。原因診断後、新exclusive outputと必要な新承認へ結合する。

## 10. 残る技術的問題

準備を阻む入力欠落・schema不一致は解消した。H6のlog_B_marginを公開candidate CSVから推測する必要はなく、原scalar receiptを照合済み。

未実行事項は実データでの旧precision/G/SE exact一致、Python/runtime差による数値挙動、実再会計のwall/RSS/output実測。これらは承認後に最初のfail-closed gateとして検査する。人工tests PASSを実データPASSと書かない。原blob/input SHA照合は済んでいるが、旧環境同一性の宣言ではない。

H6 raw scalarはこのhostで取得済みだが新public raw copyはない。他環境で実行する場合には固定SHAのscalar受領が必要。H8は公開正式結果のmetadata受領まで完了し、数値再会計へ追加するには別contract/承認が必要。native B0/I費用・H4 strict tol policy・母平均保証等はD0実装の故障とは別の証拠不足として保持する。push認証障害時は設定を変えず手動commandを報告する。

## 11. GPT判断が必要な科学的論点

R09や既存結論の再reviewは開始していない。後続の数値出力が揃った時点で、(a).0005での点最小と旧union8 coverageをどこまで主張するか、(b)RZ primaryと各metric再選択の表示、(c)仮想P感度の論文上の意味、(d)政策差を明記したnative case studiesの比較範囲、を科学担当へ返す。

newε採用、追加確認、native B0+I、H5/H7、T変更、strict DF統一、主要claim／原稿変更は今回採択しない。これらの科学判断をD0 moduleの合格で代行しない。

## 12. 次の担当・mandatory STOP

次は**利用者によるD0 saved-scalar one-shotの実行承認**。第9節の固定SOURCE・8input・候補/5ε/factor/6metric/P scope・資源上限・旧値一致gate・完了/STOP後停止・自動retryなしをまとめて承認対象とする。

承認後はCodexが新D0認可とfresh resource/profile/input/output bindingを固定し、旧値一致を先に確認して一度実行する。今回は認可済み準備・人工検証・軽量資料公開までで **MANDATORY STOP**。実データ新再会計、D1/D2/D3、H8操作は開始しない。
