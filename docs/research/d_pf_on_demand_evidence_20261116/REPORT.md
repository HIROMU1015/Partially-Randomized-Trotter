# 決定論PFの選択候補を補完し、通常探索へ戻す接続

Full PF/S2・要求幅2hの1候補で、不足信号8 roundとm5 X/Yの2直接価格を取得し、通常探索入口へ戻した。人手のshot seedなしで既存B_PRICE規則から提案が生成され、Q3新規0で停止した。B0Iのp=0〜9を候補化し、p=10はFull PFへ同一IDで対応付けた。全p・元17構成の一括科学計算はしていない。

保存linear H4/1Å/STO-3G/tol-only DF10、h_RPE=.1、M12、P0除外。価格quantityは測定付きfull controlled Hadamard wrapper全体のcompiled RZ数。compilerはQiskit1.3.0、rz/sx/x/cx、opt1、seed17。fragment順、signed decomposition、symmetric_directional control、scalarの総T一回適用、元の作用・compiled-vector gateを維持した。

## 保存証拠と担当境界

元Full PF公開438d6c9a4の17不足構成を[差分CSV](COVERAGE_DELTA.csv)と[固定locator付きmetadata](ORIGINAL17_ASSETS.json)へ整理した。後続B公開133265772とC公開1d44d0739ではFull PFの価格付き保存構成は従来3構成で、17構成の未補完状態を解消する別成果は確認できなかった。17構成とも保存stepと13 referenceを使える。各roundのmean不足と、X/Yの直接価格不足は別欄にした。旧価格候補は新候補へのC受入が未完ならdirectへ昇格しない。

Dは入力補完のみを追加した。候補選択・shot最適化はB2、p7進行分はB、PR-Pauli/DFはD2の既存入口を保持する。元D完了branchは変更せず、B2の2057650a3の実装を固定SOURCE付きで再利用する独立branchを使用した。merge・履歴変更・旧結果の再ラベルはない。

## 取得と受入

対象configurationは `9e96ed75f7f10b5f993ba9e2841684e52ad97b7d4e9df07b7b0070f02c852981`、Full PF/S2/2h。元NEEDSにある1候補を有限受入確認へ選んだもので、最良候補の認定ではない。

|入力|保存再利用|新規|結果|
|---|---:|---:|---|
|complex mean|m0/1/2/3/12の5|m4〜11の8|13/13数値gateとprovider binding通過|
|scalar-free step/reference|保存step・全13 reference|0/0|新しいstep、reference、SCFなし|
|直接価格|m0〜3 X/Yの8軸|m5/q16 X/Yの2|10直接軸、作用gate/C guard通過|
|軸別価格モデル|m2/q2・m3/q4の保存4軸|2軸の係数を凍結|m5を見る前にe093fe760で固定|
|Q3|元保存結果を参照|0|新配分の評価・winner認定なし|

X/Yとも凍結式は307+7783q、m5の予測/観測は124,835 RZで完全一致した。CX=59,344、RZ depth=34,139、CX depth=32,436、total depth=84,137。両軸を個別compileした。[registry](C_RESULT_JOIN.json)から信号、[26軸表](selected/9e96ed75f7f10b5f993ba9e2841684e52ad97b7d4e9df07b7b0070f02c852981/AXIS_PRICES.json)、元receipt、fingerprint、実装証拠へ辿れる。

26軸中10軸だけがdirect、残り16軸は候補固有の条件付きモデルである。m4の2軸も未観測であり、さらにm6〜12の14軸は確認q16を超える（最大q2048）。長q誤差上界・CI・正式E[C]/Gの資格はUNKNOWNで、`qualified_for_candidate=false`を保持した。holdoutをfitに戻さず、新しい2%規則は採択していない。

信号wall3.210s、peak RSS359,128KiB。有効な価格取得wall124.236s、worker trial120.066s、観測worker peak600,690,688bytes。compose5.545s、measured wrapper計6.846s、fingerprint計24.992s、compile/archive計34.243s、作用検査計44.665s。既存fast compose/fingerprint、共有context、保存36次元stepを再利用した。1 evolution cellのX/Yは共有evolutionを保つ元trial経路を使用。独立cellのprocess並列・実資源観測helperを継承し、CPU/RAMの新しい恒常上限は設けていない。同条件baselineとの速度倍率は測定していない。

## 通常入口・既存p7の再利用

[通常入口結果](optimizer_round02/ROUND_RESULT.json)は`ready=true`、不足なし、`GENERATOR_SEED/B_PRICE`、Q3新規0を示す。手入力した整数seedはない。既存PF worker自身のmetadata dry-checkも[DRY_CHECK_PASS](worker_metadata/dry01/DRY_CHECK.json)。別候補やPR routeへ渡していない。

保存p7/Y4/32h/5の完成済みA_SHAPEは、C固定catalogの13信号・26価格・元Q3 templateを使い[再登録](SAVED_P7_REUSE_RESPONSE.json)した。再取得・再fit・Q3は全て0。p7/S2/hの保存10直接軸も新規compile0となるmetadata回帰を追加した。元p7の長qモデルは元の条件付き資格のままである。

B所有p7 runには313 job、312 OUTCOME、1未完了markerがあった。[引継ぎ確認](B_INHERITANCE_CHECK.json)では同じjob keyを認識し、`STALE_JOB/owner_review_required`、Q3新規0、元ファイル不変を確認した。担当branchやjobへcollect・retry・書込みは行っていない。今後のB2 runは`prior_run_dirs`で完成jobを再利用し、生存jobをIN_PROGRESS、不明・終了未完了jobをSTALE_JOBとして保留する。無関係なPF templateの追加で所有権照合が外れないよう修正した。

## pを含む探索の証拠範囲

標準metadata入口はFull PF20構成、B0I90構成（p0〜9×既存indexの9公式/幅）を表す。固定gridにない正有理数幅もcanonical configuration/bodyから渡せる。p10の重複評価はしない。T1全pのshift/scalar/λ_R、T2 p7/8/9の42cellは優先度情報で、Q3棄却条件ではない。p8/9の保存first/m12 meanとstepも自身の物理点で再利用する。

p0〜9すべてを新評価したわけではない。保存Q3・費用評価済みB0I prefixは現状p7のみ。Cの固定保存catalogにおけるp7最安条件付き点は372,035,410RZで、評価済みp集合が{7}の現時点ではp探索評価集合の保存最安も同じ点である。p0〜6/8/9はQ3/価格比較が未評価で、p最適性はUNKNOWN。Full PFの保存最安480,258,064RZも引用値で、今回のFull追加候補にはQ3結果がなく最安判定を更新しない。

## 修正・未対応・検証

通常入口で二つの接続ミスを検出した。最初はcanonical `spec.method=PF`を比較routeとして扱い、科学call0で拒否された。次はFull行にB0I native_rowを適用し、context検査後、cost_trial/build/compile前の既存guardで停止した。旧STOPと空価格は保持した。routeを既存splitから解決し、元Full/B0I emitterを正しく選ぶ回帰を追加した。完了済み8 mean・凍結モデルを再利用し、未実行だった2 wrapperのみ別SOURCE/namespaceで取得した。native contextは失敗前後で計2回、信号再取得と科学的gateの緩和は0。通常再開はこの明示的補完を自動再試行しない。

同実幅の較正2点とlater holdoutがない要求幅は、候補化できても現在のaffine取得方策では補完できない。片軸だけの保存証拠があるcellは新規両軸compileへ進めず保留する。新prefixの実preparation成功、未知長qのcompiler効果、価格CIを人工PASSから認定しない。

著者検査と委譲先の実装検査は区別した。canonical候補7、evidence11、通常接続/所有権/C sync19、provider7、既存optimizer18、人工native runtime2を検査した。独立科学レビューとの呼称は用いない。Cの自動同期には最新1d44d0739を基にした最小consumer差分を提供し、実registryの通常reconcileで26軸を読みQ3_WAITING（費用/N=null）となることを確認した。C branch/scheduler/roadmapは変更していない。

ロードマップ差分は[HANDOFF](HANDOFF.md)。追加科学計算には進まず、本有限作業を公開してSTOPする。
