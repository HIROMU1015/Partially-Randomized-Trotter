# R依存compiled価格モデル Phase 1実行準備（2026-11-11）

Phase 1専用runner、保存C/Sの接続、paired 3係数fit、Git凍結後の独立検証、別unitのlegal rare診断を実装した。新規14＋既存R adapter回帰9＝**23人工/mock test method PASS**。今回実行したのは技術検査と[PLAN](PLAN.json)のmetadata生成だけ。実分子sampling、回路build/compile、構造S再計数、係数fit、実価格予測/残差評価、signal/N/G・実R最適化は開始していない。

コードSOURCE：`1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7`。基点8457c4435、既存C branch `codex/r-variable-compiled-price-design-20261111`と同じworktreeを継続した。A/Bの作業領域・未公開変更・checkpoint/private raw・process・環境/cacheへアクセス/変更していない。過去のSOURCE/input/raw/result/grant/ledger/marker/KEEP/STOPを保全する。

成果：[runner](../../../scripts/run_r_price_phase1.py)／[校正・保存接続](../../../src/trottertracks/r_price_phase1.py)／[stage制御](../../../src/trottertracks/r_price_phase1_runner.py)／[lazy科学backend](../../../src/trottertracks/r_price_phase1_science.py)／[テスト](../../../tests/check_r_price_phase1.py)／[HANDOFF・実行手順](HANDOFF.md)／[SOURCE索引](SOURCE_INDEX.json)／[技術確認](CHECKS.json)。

## 固定科学scopeと役割

Linear H4/H6・1.00 Å・STO-3G、保存tol-only DF10/DF19・weighted completion、保存H/state/snapshot・fragment/scalar順を保持。H4 S2/δ=.1/PR p1/K2、H6 4次m5_best/δ1.6/PR p7/K2。元Qiskit1.3.0/NumPy1.26.4、rz/sx/x/cx・opt1/seed17、symmetric_directional・測定付きfull Hadamard wrapper、P=0。targetは保存DF-Hの同sector基底エネルギー、保存数値stateの条件付き仮定を厳密ground資格へ昇格しない。h=.1/13round/εE=.00159360010199040/αtotal=.05各方式/βtotal,stat=.4,.2・prior/decoderは固定taskから転記し、計算・変更しない。

| system | qPF | R | T | 役割 | classical価格path数 |
|---|---:|---:|---:|---|---:|
| H4 | 8 | 4 | .8 | 保存calibration | 9探索 |
| H4 | 16 | 4 | 1.6 | 保存calibration（既知の旧独立q点） | 33 |
| H4 | 8 | 8 | .8 | 新calibration・未実行 | 8 |
| H4 | 16 | 8 | 1.6 | 新独立validation・未実行 | 別8 |
| H6 | 1 | 16 | 1.6 | 保存calibration | 9探索 |
| H6 | 2 | 16 | 3.2 | 保存calibration | 8 |
| H6 | 1 | 32 | 1.6 | 新calibration・未実行 | 8 |
| H6 | 2 | 32 | 3.2 | 新独立validation・未実行 | 別8 |

通常4新セル×8 path×X/Y＝最大64 full-wrapper compile。別扱いのrare 2組/system×基準/変更×X/Y＝最大16、計最大80。nは価格の古典標本数で、量子shot Nではない。PF対照H4 B0/p7・H6 Full19およびH6 m0..3の別δ/p5/R32は変更しない。

H6の元順11 signed piece・min-one/largest-remainder allocationを再利用し、Σr_j=R、全event数qPF Rを保持。qRPEとqPFを分離。新seedは通常456 stream＋診断38 streamで、意図したX/Y・診断pair共有を除き494個の相互重複なし。これはseed metadataの照合であり、全履歴coverageやPRNG独立性の証明ではない。

## 実装と再利用

- **保存C/S**：H4 V1 DEVELOPMENTのq8探索、独立q PREDICTIONS＋RESULT＋PROVENANCEのq16、H6 V1 DEVELOPMENTのq1探索、Q2_DIRECTのq2をpath/axis/候補/cohort/compiler/入力bindingでjoinする。H4 q16はevent digestも照合。旧Sは再計数せず、Mq/Affineの補正済み値をSへ代入しない。探索/確認はpoolしない。
- **fit**：SOURCE固定のC7-4 multiresponse fit bodyを再利用。system別の3セルについてpaired C−Sの平均を作り、列[1,q,qR]のrank=3を検査する。等セルweightを明示し、旧equal-q weightingをそのまま継承しない。3セル3係数のlack-of-fit DOFは0。保存Mq/Affine係数を別Rへ移植しない。
- **不確かさ**：校正cohortのpaired residual covariance/nとdesign inverseから名目的6×6係数covarianceを保存する。独立cohort仮定を明記し、完全な係数/rare/law/R-q systematicや総CIを含めない。modelのshared calibration groupとcalibration directのdata groupを対応させ、validation directへfit由来の共分散を割り当てない。
- **validation**：元H4の誤差要約bodyを再利用し、bias/MAE/RMSE/相対誤差・価格cohort分散・paired X/Y residual/mean covarianceを出力する。検証データをfitへ入れず、validation再fitなし。負値をclampせずSTOPする。未観測q/RはPhase 1評価関数の対象外として拒否する。
- **価格取得**：旧sampling、signed compose、native full wrapper、fingerprint、compiler API、Accounting kernelを固定SOURCEから再利用する。新scopeのq/R loopだけを接続し、旧科学runner・authorize/grant処理を呼ばない。inputは保存snapshot/stateのloaderを使い、SCF/DF/ground-state生成を行わない。科学backendは明示execute stageまでloadしない。
- **cache/並列**：旧cacheはfingerprint＋compiler一致の読取だけ。不一致はSTOPし、新compileへfallbackしない。各pathのSをcompiled価格の前に保存する。whole-pathをaffinityとjob数に応じてprocess並列化し、contextをfork共有、X/Yは同一pathを共有する。compiler内の重複poolを避け、共有設定や他processを変更しない。

## 科学実行の順序とSTOP

`prepare`はmetadataのみ。将来の明示承認後に、`calibrate --execute`で既存2セル/systemと新1セル/systemを取得・fitし、FROZEN_MODELSを保存する。そのファイルをGit commitした40桁SHAと同一bytesで確認できる場合だけ、`validate --execute --frozen-commit ...`が独立新価格を取得する。役割・seed・指標は事前PLAN、係数・nominal covarianceは価格を見る前のfrozen commitで固定する。SOURCE変更やplan変更は拒否する。

`rare --execute`は通常validation後に別datasetを取得する。H4はhead/tail先頭と異basisのtail内、H6は正pieceと負pieceを対象とする。専用drawのrotationを保持して全基準orderを0へ置換し、変更pathにだけlegal order2（2 products→rotation）を1個入れる。元component順で事前規則により選び、同じpairはseedを共有する。forced pairを無条件cohort/fit/平均価格へpoolしない。全rare配置や複数order2は網羅せず、全law寄与はnull。

各stageはfresh output/排他的started markerで一回だけ。failureは新STOPへ保存し、同じstageのretry/resumeや別Rへの変更を自動開始しない。完了STOP後は後続科学処理を拒否する。過去runのSTOPは別scopeとして保持。publicとprivate rawを分け、原本上書きなし。

## A/B整合と資格

公開A `5ca2dce3234f3e4a971f4a8c852da676f050e315`のwith_round_Rと、公開B `7bca8f9642d1c42ca3fede4d31b2cdd8dcef46ae`のbuild_menu/request_context/coverageを固定Gitから読取再利用した。新Rのcandidate ID・full identity・allocationとlaw bindingは一致。Bへの価格recordはfull candidate domain、compiler、X/Y unit、cohort、source、共有不確かさを含む。共通N/C APIやN/会計を変更していない。Bの将来最新版・実runtime全体の結合確認ではない。

Phase 1は上表の有限点を供給する準備で、Bの広いR menu全体を埋めない。PR価格は有限cohort、N側ideal lawとの差は未資格。Bの明示finite_cohort_scenarioでdirect点を扱える形式とし、全law期待価格へ昇格しない。modelの採用誤差閾値は未設定なので、校正点/独立検証点を区別してもqualified_for_candidate=falseを維持する。正式な価格採用は指標・必要精度を基に別途判断する。B/attenuationをCへ再乗算せず、未知joint covariance/total CI/Gはnull。

## 検証結果・実施範囲・次の担当

[新14件](TESTS.txt)は人工/mock C/Sによるfit/metrics、3セル分離、Git凍結gate、one-shot/failure/STOP、4保存形式、legal pair、S-before-price、cache hit/mismatch、元構造kernelの人工word、公開A/B metadataを確認した。[旧adapter回帰9件](R_ADAPTER_REGRESSION.txt)もPASS。実価格や実分子descriptorを人工テスト値に使っていない。作者自身の技術確認で、独立科学レビュー・実compiled精度の証拠ではない。

実runtime/private入力は未ロード。実分子backendの動作・価格精度・物理wrapperの新statevector検証は、今回確認したと扱わない。将来の実行時にSOURCE/runtime/input/preparation bindingを検査し、欠落・不一致なら停止する。

**ユーザー**がこのSOURCE/PLANと最大80の具体的なPhase 1科学実行を判断する。承認後は**Codex C**が校正→Git凍結→独立検証→rare診断→公開照合を一括で進められ、細かなstageごとの再承認は不要。**A/B担当**は公開R menu/N条件・有限price unitとの整合を取込み時に確認する。今回はcommit/push・remote SHA/主要成果物照合後mandatory STOP。新しい科学計算へ進まない。
