# C3 run02：contract参照先修正後の承認済み再実行

2026-11-06 JST。前回結果commit `18d303288d4852b8ec6e5ae378b5e91e7e53965a` のraw構築前のFileNotFoundErrorについて、利用者から修正・再実行が明示承認された。新SOURCEをcommitしbyte照合後、同じC3条件で一度実行する。前回run01・旧SOURCE `f18adc46dffcb69a6b1028d5c38793f9b4c0b948`・失敗・STOPは保持する。許可された再実行であり、前回STOPの抹消や成功扱いはしない。

[元PLAN](PLAN.md)の科学条件、fixture、cut、強い対照、検証・精度・P0・最大8wrapper・STOP・解釈境界をすべて維持する。変更はcontractの参照先を旧origin treeからexecution worktreeへ直すこと、新規出力をrun02へ分けること、対応する読出しtestと来歴のみ。回路アルゴリズムmoduleは変更しない。

- 同H4/1.00Å/STO-3G/Nα=Nβ=2・number-CDF8/元順LD2/S2/s1/r32/K0、各h=δ=.1。元H厳密ground仮定・準備費除外、state-vector/ground solveなし。
- q4=保存sample00→01→02→03、complete step2後cut。q8=sample00→…→07、complete step4後cut。全ordered pathに一つのgadget、fixture/cutの事後変更なし。
- 保存cohort `near_practical_PR_r32_fixed32_v1`、元SOURCE・16入力・追加law・96保存bindingsを同じpreflightで確認する。signed係数、順序、normalization、scalar/identity、真空χ、actual adjointを保持する。
- strong symmetricのq−1 step境界融合、head relative fusion、uncontrolled Gaussian、directional中心、tail同basis共有、scalar/identity集約を維持する。ReferenceはA/B内部q−2境界を同様に融合し、global cutの1境界は保持する。C2にはこの新しい手動cross-step融合が含まれない。
- raw2fixtureを独立processで並行検証し、両PASS時のみ2q×2方式×X/Yの8wrapperをP0で各一度compile。postcheckは既存CAR・全256入力のX/Y observable・複素coherence・位相・leakage検査を使う。
- component abs1e−10、observable Frobenius/signal2e−10、rtol0、phase fitなし。Qiskit1.3.0 opt1 seed17 rz/sx/x/cx、topology-free、位相保持。失敗後の閾値緩和・追加retry/resumeなし。
- 実空きCPU/RAMを再観測し、独立raw2task・最大8compile/postcheckへ配分する。共有環境・他ユーザーには変更・操作しない。

`load_contract`を科学計算を呼ばない一時fixtureでテストする。元の人工3testも実施し、同一SOURCEとして固定する。run02の成果物は `artifacts/track_d_c3_q4_q8/2026-11-06/run02/`、新STOPもここへ記録する。`retry=0`はこの承認済みrun02内での追加retryなしを表し、`authorized_rerun_of`でrun01の結果commitを明示する。

q1/q2を再compileしない。新sampling、別H/endpoint/cut/q、B2、新fit/ground solve、N/成功確率、全13-round費用、q>8、詳細科学レビューは実施しない。結果・raw/compiled回路・資源全指標・q1/q2保存比較・不足証拠を公開し、remote SHAと公開bytesを確認した後mandatory STOP。次の研究判断はユーザー／GPTへ戻す。
