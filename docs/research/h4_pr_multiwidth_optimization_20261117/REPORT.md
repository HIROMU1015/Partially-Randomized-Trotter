# H4 PR-Pauli／PR-DF 複数幅・並列実探索 R06

今回の目的は、各方式自身のQ3適合条件の下で、13 round×X/Yの条件付き総RZ費用を減らすこと。保存の単一幅最安を固定解として配り直すのではなく、PR-Pauliの0.1/0.2、PR-DFの0.1/0.05、LD1〜9の異なる物理scheduleを既存mapper・Completion・V2で選ぶ。精度・law・target・decoder・loss・exactτ・compiler/loweringは保存契約を維持する。

**実探索をtmux `h4-pr-width-r06`で開始し、切離し後の段階前進と共通queue・実prefix HITを確認した。** 現在の実行状態は [STATUS.json](STATUS.json) と [HANDOFF](HANDOFF.md) が正本。** 本書は起動前の固定SOURCE・開始点・限定修正と、結果回収方法を公開する。探索終了時の幅別最安の最終集計は未実施であり、空欄を不適合・費用0・全域最適として扱わない。

## SOURCEと開始点

公開統合 `5829a7288d28395d2b3e89bfb676c23c690b9ddb` をremote確認後、独立の所有worktreeへ展開した。幅版d695と高速化e839のglueを再実装していない。今回の実行branchは `codex/b2-rpe-pr-multiwidth-optimize-20261117`。Cの既存`codex/b2-rpe-*`同期に載せるためこの名を使い、Cのbranch/config/cronを変更しない。科学実行前の限定実装commit・入力hashは [SOURCE](SOURCE.json)。

[STARTING_EVIDENCE](STARTING_EVIDENCE.json)はC fee61とB2 d8の保存表・検証済み全4,783件を照合したもの。開始点は次の2件であり、古い報告の転記値を探索基準へ固定していない。

|方式|開始時の保存最安適合費用 RZ|保存scope|
|---|---:|---|
|PR-Pauli|410,199,237|LD5/S2、要求幅0.1、C最安と検証済み全完了ledgerの最安が一致|
|PR-DF|155652932221/126|LD7/S2、要求幅0.1、保存Q3適合|

exactτ=`15136964828156312001/5960464477539062500000000`、Z1＋T1、Full DF-H solver proxy、IID deterministic raw、全事象二乗loss。構成・絶対κ・N_X/N_Y・MSE上下界・26軸価格出所と元receiptは同JSONの固定locatorで追える。LD5 κ2の制限集合にある未採用の有効改善点 `3506735579/7` も保存再利用対象。全体最安ではないが、採用点と全完了最安を分ける必要性の実例として保全した。

## 科学scopeと準備済みの実行設定

[ROUND](ROUND.json)と[PROBLEM](PROBLEM.json)を使用する。Pauli0.2はJSON数値のbinary64であり、文字列1/5へ置換しない。h・T_m・exactτはそのまま。元mapperによるq/R/actual T/qを、候補別configurationとSTATEへ保存する。[SCOPE_METADATA](SCOPE_METADATA.json)は実際の設定を通したmapper metadata32 familyと準備待ち4 familyを記録し、新信号・Q3の結果ではない。

既存の有限wide探索規則を使用し、κを中心値に固定しない。CLIのbatch64/cycles200・元maxκ65536等は実際に調べた範囲として報告し、大域最適の保証とはしない。Q3は方法別UNLIMITED設定のもと、選択されたscope内の要求だけを実行する。入力不足・未対応mapper・科学gate不成立・UNKNOWNは元の保留/STOPに従う。配分の不適合を幅全体の不適合へ広げず、未観測を不適合にしない。

ホストGPUA100-gleap02、実際のaffinity64 CPU・観測時約488GiB availableから、16同時job×各3 DP workers（親を含め64 CPU相当）を準備profileとした。独立候補優先のHYBRID、DP prefix-cache、m5/m11 checkpoint、準備と初期READY Q3のoverlapを有効化。32×8を固定上限として引き継がず、OS/共有環境や他processは変更していない。実H4の逐次対照による速度倍率は未測定。

## 開始前に必要だった限定技術修正

実計算前の通常入口・人工確認で次を修正した。量子回路・信号・Q3数値kernel・価格式を変更していない。

1. **方式ごとの探索優先度**：両方式をまとめた価格windowで、安いPauliがDFのrefinementを抑えていた。既存W plannerを方式別に呼び、同じ方法内の規則を保持してproposalを共通queueへ戻す。新しい最適化アルゴリズムや別mapperは作っていない。
2. **完了済み未採用Q3の最安**：executor.finish後、元Executor.prior_records・identity/source・候補自身の単価・exactτ整合を確認し、小さい改善も確認済み最安へ含める。`rule_adopted_best`と`confirmed_all_best`、`CONFIRMED_MINIMA.json`を別保存し、次の計画には確認済みの低い値を使う。
3. **一括準備を避けるwave**：元INIT ladderはper_decisionの例外で222候補を作り、全候補を逐次準備していた。科学処理前のmetadata-only検査を必要範囲で終了し、残るmetadataを保全した。全proposalは保持したまま、既存per_decision（今回8）を方式別・cycle別の準備waveへ適用。familyを巡回し、既知の安いfamilyと未試行familyに機会を配る。残りはCOMPLETIONのまま次cycleへ渡し、WAITING_INPUTSだけならbusy loopせず止める。候補を価格下界やQ3不適合として削除する処理ではない。

[人工回帰](launch_contracts/BOUNDED_WAVE_REGRESSIONS.json)は36件PASS。方式混在の修正前FAIL、最終化後の未採用微小改善、誤価格/誤PASS/identity不一致拒否、222 pending→各方法8件だけdispatch→残り保持・次cycle継続、overlap/通常CLI/並列queue/prefix/元幅検索を確認。原テストのsource-drift検査は人工cycleを進め、元の誤SOURCE拒否assertionを保持した。

## 依存・再利用・非干渉

[PREFLIGHT_DEPENDENCIES](PREFLIGHT_DEPENDENCIES.json)と[RUNTIME_CHECK](RUNTIME_CHECK.json)：実Pauli/DF routeをそれぞれ指定Python3.11.0rc1で初期化し、workers3・元AST guard・prefix context・科学契約不変を確認。kernel/科学callは0。価格環境は既存Python3.12.3/Qiskit1.3.0。faulthandler定期dumpは使わない。

§10.5の既知prior4runは4,229＋185＋85＋284完了で未解決LAUNCHなし。原round05自体は未実行。新worktreeの同名pathを参照して旧結果を取り落とさないよう、round03/04も含め元所有worktreeへの絶対pathにした。D2 READYのPauli LD4/DF LD6・保存packet・価格登録も既存条件で再利用する。候補・round・実幅・law・cohort・compiler/source条件が異なるものを共有しない。

Pauli workspaceは今回の所有checkout。既存shared private rootは候補/step group単位でSOURCE・state・raw存在を検査してREUSE/HELDとなる元経路を維持し、旧private/rawを削除・置換しない。統合worktreeやB/B2/Cのrun/設定を変更していない。初期の公開待ち観測は [INTEGRATION_READINESS](INTEGRATION_READINESS.json) に時点付きで保全し、現在の公開版や修正後の状態と混同しない。

## 結果の回収とロードマップ

起動後はcontroller自身が選択・準備・登録・Q3・取込・次cycle・停止を行う。SSH/chatから候補を逐次投入しない。同じrunの再開には所有flockを使い、既存controller/workerを先に確認する。科学失敗の無断retry、SOURCEの途中変更や旧STOPの解除を行わない。

終了後に各方式について、開始点／基準幅後／新幅／全幅の最安を、全完了した有効実Q3も含めて集計する。LD、要求/実幅、絶対κ、q/R schedule、X/Y配分、MSE上下界、条件付き費用と削減額/率、単価のdirect/model/外挿を元receiptへ結ぶ。実時間と回路RZ費用は別量。UNKNOWN・未評価・入力不足・モデル不適合を区別し、C向けは既存export/join/通常consumerで確認する。チャット終了時に計算中なら同runから後処理を引き継ぐ。ロードマップ本体は編集せず、進行状態と証拠範囲の差分をHANDOFFへ残す。

## 起動後の観測と残る技術的保留

[LIVE_START_SNAPSHOT](LIVE_START_SNAPSHOT.json)の時点で、両方式からQ3を投入し、共通queueの実同時稼働・3-worker/cache profileのrequest転送と実HITを確認した。個別PASS/EXCEEDSは保存し、最終の方式別/幅別最安や削減額はまだ集計していない。DF価格較正は初期8 pathを完了し、m11の4 path×X/Yを処理中だった。CPU生存だけではなく、完了したphase/receiptと追加cacheイベントで前進を確認した。

Pauliの初期8候補は信号入口importで停止し、新信号呼出し0件だった。既に数値callerが要求する固定A/LD3 SOURCEの欠落依存2fileを別worktreeで検証し、所有lock下で**存在しない同一blobだけ**を配置した。[配置来歴](DECLARED_DEPENDENCY_DEPLOYMENT.json)。既存caller・環境・ROUND・数値規則・git HEADを修復処理では変えず、旧8保留を再試行していない。配置差分を隠して初期b025のtreeそのものと一致すると主張しない。未試行LD4・幅0.2の元metadata gateはPASSだが数値成功はまだ未確認。legacy saved-context locatorをPauli descriptorとして転送する別の形式不一致は、将来のclean bridge SOURCEが必要な保留として残る。

またDFの4 wrapper起動は保存template未結合でCASE構築前に終了し、科学APIを呼んでいない。既存有効17件等と混合せず、UNKNOWN/technical-start failureとしてsnapshotへ記録した。新たなSOURCE/条件へ黙って差し替えず、正常なDF較正と他のQ3は継続している。これは両方式の全幅・全LDが通過した報告ではない。controllerは自立して残る候補の選択・準備・探索と自然停止を続ける。終了後は同runから結果回収と未取得領域の報告を行う。
