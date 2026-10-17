# M1-S Stage S one-shot完了・結果保全・mandatory STOP

2026-10-17 JST。`STAGE_S_COMPLETE_AWAITING_USER_JUDGMENT`、failure=null。ユーザーの「その方針でよいので、計算開始まで進めて」を受け、公開候補を正式採択記録Cへ固定し、その同じ発言の計算開始承認を専用Aへ別結合した。固定SOURCEと条件を変更せず一回起動し、初回で正常完了した。SOURCE修正0、retry0。

## 1. 対象と結果

linear H4、隣接距離1.00 Å、STO-3G、8 modes、保存DF rank12、G3/G6/G9、δ={1/8,1/16} Ha^-1、R={2,4}、T={1,4} Ha^-1。保存raw-H理想Task A、元norm²の保存state、非冗長Pauli-tail、同一概念label数のshared ±対ordinary／independent ±。元108方式行inventory（非zero72、zero36）を照合し、zeroを再計算していない。

主対象24対照をcap-edgeで数値評価し、carry24対照と合わせて全48対照を報告した。

|層|新しいcap直下の整数削減の存在証拠|cap直下で同数・全γ判定は未決|旧区分carry・新shot未評価|
|---|---:|---:|---:|
|G3、T=1|0|8|0|
|G3、T=4|8|0|0|
|G6、T=4|0|8|0|
|G6、T=1|0|0|8（旧π3不成立）|
|G9、T=1・4|0|0|16（絶対差未解像）|
|合計|8|16|24|

新主対象の`ARITHMETIC_UNRESOLVED`は0。G3 T4の8対照は4cell×2対照方式であり、8回の独立物理実験や独立holdoutではない。主対象24件はいずれも連続十分設計のR_cont^cap>1を区間で解像したが、整数化すると16件では端点直下のshot数が同じだった。

## 2. 保存された整数設計の表示

下表はG3 T4の保存値の表示。nはXまたはYの一軸shots、N=2nは両軸合計。同じcellのordinary／independent ±はこの整数値では同じとなったが、両方式を統合して評価したわけではない。連続比のexact intervalは全件CSV／rawに残す。

|δ [Ha^-1]|R|n_shared^cap-|n_ordinary^cap-|n_independent^cap-|N_shared / N_ordinary / N_independent|
|---|---:|---:|---:|---:|---|
|1/8|2|26|28|28|52 / 56 / 56|
|1/16|2|25|26|26|50 / 52 / 52|
|1/8|4|25|26|26|50 / 52 / 52|
|1/16|4|24|25|25|48 / 50 / 50|

全48対照は[CSV](../../../artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01/stage_s_comparisons.csv)、正本は[raw result](../../../artifacts/track_b_m1s_stage_s_result/2026-10-17/attempt_01/result.json)。CSVは保存値の表示で、γの新評価・fit・順位付け・shot forecastの追加実行はしていない。

## 3. この結果が示す範囲

採択されたPER_METHOD_ROW、α_row=1/20、fresh-IID/Hoeffding/X/Y等配分、保存半径の条件付きcoverage、p128＋8 guard bits、refinement0の十分設計に限る。各方式行の設計confidenceであり、全72行／全48対照の同時95%保証ではない。物理測定や実機IIDを検証した結果でもない。

capは開端点γ_c=π/3。保存π区間を使い、n_m^cap-=floor(S_m(γ_c))+1とした。`CAP_EDGE_STRICT_INTEGER_WITNESS`は、π/3より厳密に小さいγの非空開領域で、**この十分shot設計**に整数削減が存在するという十分証拠。実用の外生的γを選定・検証したわけではなく、最適shotsの比較でもない。

`CAP_EDGE_NO_WITNESS_GLOBAL_UNDETERMINED`の16件はcap直下で同数であり、より小さなγでの削減を否定しない。G6 T1の旧gate不成立とG9絶対改善未解像もそのまま維持する。相対比でG9を救済しない。

R_cont^capはこの十分設計の連続proxyのsupremumで、開領域内で一般に達成しない。state/prior/全round/実装/SPAM等の未知誤差を0と仮定してRPEへ進んだわけではない。保存総raw-H biasへモデル差abs(T)Bを重ねて加算していない。Stage Cのnative費用、HD合成、state preparation、固定費、RPE総資源は評価していない。科学的GOはnullのまま。

## 4. 固定provenanceと実施承認

- 基点候補公開：`0dd660ece1f2e342d2666d37d4b4536adb5e8ca7`。
- SOURCE S：`7a338c4eba778c4ce1f189cfa8c8f050afb5df08`、実行対象9ファイル不変。
- 正式契約 C：`b4b4c12848d0cc65025e1ce9eb95902aec7d97fe`、SHA256 `9b4b453d9e79ac596f1f5328f260d8f558abc85b13a827759dcdec48779b5843`。
- 専用authority A：`437e6bbc4cc2c550268d8644b8fc2bc0ec56e791`。
- 入力V1：`b34572770e7c2105e16da003af050d789f490662`、SHA256 `bb2d09e6963fdc8b8871170ba28e98c36f4e353f452e50870103a7e97bf2e55d`。
- 入力π3：`81d6103ec89d485cb7336c3a90bd95dad66992aa`、SHA256 `693ce941bf7793a6104af28443746f43e05d4903118c0ebbb6f09729f389fadf`。
- Stage S raw result：67770 bytes、SHA256 `9a11df3fcc60609cfe63ebcd392c6b78585054983487a730dbdf603f8682ab0b`。

今回の発言が採択と開始の双方を承認した文脈を、[採択記録](../../../artifacts/track_b_m1s_stage_s_formal_adoption/2026-10-17/v1/user_adoption_and_execution_context.json)と[専用authorization](../../../artifacts/track_b_m1s_stage_s_execution_authorization/2026-10-17/attempt_01/authorization.json)へ分けて保存した。従来の中間STOPを無断で解除したのではなく、今回の開始まで進める指示を記録した。契約自体はcontract_grants_execution=falseであり、SOURCE-boundの別Aを公開・照合してから起動した。実施承認は今回のStage S一回限りで、retryや条件変更の承認ではない。

## 5. 検査・resource

実行前にS/C/A、入力bytes・SHA256、固定interpreter、clean tracked treeとC/A公開bytesを確認した。同じSOURCE/interpreterで実施済みの人工36テスト（parallel/serial等価を含む）を確認し、科学入力を使う予備runは追加していない。

32利用可能CPUと物理memoryを確認し、12独立cellへ12workerを選択。cell内の3marginを二対照で共有し、36margin、72sin端点call、共通log1callを処理した。bound stepsは1548（上限100000）。人工CPU/RAM/GPU/worker/time capなし。GPU/BLASはexact Fraction backendが利用しないため起動せず、共有設定や他ユーザーのプロセスは変更していない。

science phaseの観測wall約0.093秒、interpreter起動・監視込み約0.69秒。外部timeの最大RSS49152 KiBは報告値であり、12workerの同時aggregate peakではない。各process peakをrawへ保存した。速度向上率や一般的runtime保証は主張しない。result/logは各16MiB以内、適応的再精度化0、retry0。

保存後監査はproduction helperやFractionを使わず、整数対と交差積でfloor+1、n/N、区間順序、RとSの整合、全grid、schema、provenanceを確認した。448件PASS/FAIL0。sin/log/π/H4信号を独立再生成した検証ではなく、元の条件付きcoverageを認証しない。ローカル監査でありCIや外部独立再現ではない。

保全監査14件PASS。旧非index7822 Git blobs、親追跡1284件・HEAD/index/tracked status、既存215 worktreeのHEAD、レビュー原本を保全した。既存worktreeの未追跡内容を全inventoryしたという主張はしない。

## 6. 証拠と次の担当

[preflight](../../../artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01/preflight.json)、[C/A公開照合](../../../artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01/prelaunch_public_identity.json)、[launch receipt](../../../artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01/launch_receipt.json)、[保存値監査](../../../artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01/saved_result_audit.json)、[保全監査](../../../artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01/preservation_audit.json)、[結果summary](../../../artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01/saved_result_summary.json)、[evidence index](../../../artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01/evidence_index.json)、[publication receipt](../../../artifacts/track_b_m1s_stage_s_execution/2026-10-17/attempt_01/publication_receipt.json)。

次は**ユーザー**が結果を確認し、実用位相予算・state/priorの証拠と回路費用の検証をどこまで進めるか判断する。必要な次期技術準備はその指示後に**Codex**が行う。新しいGPT詳細レビューやStage C／Task B、追加科学計算は自動開始しない。

今回の許可された保存信号解析1回を完了。Hamiltonian／Pauli／有限step信号の再生成0、native compile0、RPE simulation0。新専用authorization1、grant0、消費marker1、retry0。旧成果を保持し、結果の公開・同一性確認後**mandatory STOP**。
