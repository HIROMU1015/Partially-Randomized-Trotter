# 粗幅Yoshida4固定配分Q3の復旧

対象は旧34783cdのHANDOFF_B/NEXT_STEPSと今回の明示承認に従う。B0I p7／標準Y4／PF要求幅32h/5、configuration dd8bfc24…、X/Y各軸 `[59,55,51,47,43,39,35,31,27,23,19,7,2]` の新規Q3最大1件。新信号/reference/step/回路build・compile・価格・fit・別配分は0。旧q3_run01とUNKNOWNを変更しない。

固定入力は `278fb5214f1091efba2a05b305bc05190cd18ff6:docs/research/d_p7_y4_coarse_signal_q3_20261116/Q3_INPUTS.json`、信号は `f23c3f10ed56d0c218e49efbdbfb31d12ffa8a53` の同ディレクトリBOUND_SIGNAL_INPUT。再生成せず新INPUTS wrapperへ値をそのまま収めた。

実行環境は既存 `/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python`。実体は同project内Python3.11.0rc1、NumPy1.26.4。新install・共有環境変更は0。元kernel710aa69b／accelerator9e294778と既存のAST guardを変更せず、本番PF workerが使うObservedコンストラクタで期待hash3d043e71…へ一致した。

元caller `scripts/run_p7_y4_coarse_signal_q3.py` のq3_runtimeと固定B code snapshotを再利用し、新caller `scripts/run_p7_y4_coarse_q3_recovery.py` はruntime指定・新namespace・固定入力再利用・元supervisor呼出しだけを追加する。元のZ1＋T1/DF10 solver proxy/IID deterministic raw/全事象平方loss/exact τを維持する。

科学call前に、元Bの人工DP3 testを再利用しfull出力・task分割・scope境界のexact一致を確認した。さらに本番と同じM.supervise→F.worker→PF worker→実Accelerator→T.evaluate→API/partials/RESULT/RESOURCE/PERSISTED保存を、inventedな2-round fixtureで検査した。metadata intakeだけをtoyへ差替え、コンストラクタ・guard・DP・数値評価・保存はmockしない。人工fixture名にPauliを含むが保存H4/Pauli科学データは使用しない。5 test PASS、人工評価を実H4 callへ数えない。

科学実行前にINPUTS・caller・DELTA_CHECK・test記録をcommitする。実行コマンドは専用Pythonから `scripts/run_p7_y4_coarse_q3_recovery.py execute`。実行先はこのディレクトリの `run01/`、旧出力と分離。既存8-worker/batch8の並列DP・協調round log・worker自身の保存・親resource/exit監視を使い、CPU/RAMの恒常的上限は追加しない。Python3.11.0rc1のperiodic faulthandlerは使わない。

§10.5：旧STOP以降の関連D/B/A公表namespaceと同ユーザーの同candidate/callerを差分確認。対象の同等結果・進行job・未解決出所は確認範囲で0。B後続S2のm12-one-shot結果やPauli LD3の準備、AのPauli LD3信号は別候補で、今回の同配分Y4 Q3へ流用しない。詳細DELTA_CHECKを参照。科学開始時に関係しないworktree/branchを一括凍結しない。

上界≤元exact τなら適合、下界>τならこの配分でモデル不適合、跨ぐ場合INCONCLUSIVE、input/数値gate不成立はUNKNOWN。後処理では現幅Y4のb4296b110保存値を引用する。粗幅費用はnull。原API/partials/判定・SOURCE・資源を保全し、commit/push・remote照合後mandatory STOP。追加Q3/信号/compile/価格へ進まない。
