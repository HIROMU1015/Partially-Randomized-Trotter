# C3：中断run02の保存回路による補完

2026-11-06 JST。利用者の修正指示に従い、同じ承認済みC3固定fixtureを完了させる。run02 SOURCE `52b3cab50d5c5edcbd7e3249637168e9978be535`では、q4/q8 rawは両PASS、P0 compileは8本完了し、postcheckは7本PASSした。q8 Reference Yのpostcheck結果と最終result/STOPは未生成で、該当する稼働processを確認できなかった。原因は特定できておらず、科学的correctness失敗が観測されたという意味ではない。

既存runnerはfresh出力専用で、中断後の既存compiled回路を読む入口がない。元runner/adapter/checker/precisionは変更せず、`complete_track_d_c3_run02.py`を追加する。元run02の61ファイルを別の科学input一覧で固定し、旧SOURCEと同じSOURCE/input/environmentであることを既存preflightで確認する。旧run01の失敗・STOP、run02の全ファイルは読み取り専用で保持する。

科学条件は[元PLAN](PLAN.md)と[run02 PLAN](RUN02_PLAN.md)のまま。同H4/1Å/STO-3G/Nα=Nβ=2、number-CDF8/元順LD2/S2/s1/r32/K0/h=δ=.1。q4=保存00..03・complete step2後cut、q8=00..07・complete step4後cut。一つのwhole-path gadget、強い対照の融合・共有、χ/scalar/identity/adjoint、元H厳密ground仮定・準備費除外を維持する。

1. 修正SOURCE・本PLAN・人工test・中断input一覧を専用branchでcommit/pushしremote bytesを照合する。
2. 全保存q4/q8回路・references・7postcheckを新しい`run03_completion`へbyte一致で複製する。元ファイルは上書きしない。
3. `run_track_d_h4_reference.verify_one`を直接再利用し、保存済みq8 Reference Y compiled回路のpostcheckだけを一度実施する。新compile・回路生成は0、他7postcheckの再実行は0。結果に基づくfixture/cut変更はしない。
4. PASS時だけ既存C3のpaired X/Y・vacuum χ検査を両qに実施し、保存q1/q2との資源表を既存関数で作成する。全SOURCE/inputと元run02 bytesを再照合する。
5. 別のresult/STOP/inventoryを保存し、最終成果物を公開・remote SHA/bytes照合後mandatory STOPする。

component abs1e−10、observable Frobenius/signal2e−10、rtol0、global phase fitなし。P0出力は既存Qiskit1.3.0/opt1/seed17/rz/sx/x/cx/topology-freeのbytesをそのまま使う。state-actionのthread数だけは新たな実空きCPU/RAM観測から選ぶ。1残taskに利用可能CPUを配分し、旧runの1threadを恒常上限にしない。個別環境や共有設定には変更を加えない。

長いpostcheckは入出力をローカルlogへ向けた独立sessionで一度起動し、チャット接続の寿命に依存しない形とする。起動PIDはローカルだけに保存する。原因をSIGHUP/OOM等と断定せず、この変更が外部停止を完全に防ぐとも主張しない。実行中processは停止・再起動しない。

SOURCE/input/precision/correctnessの不一致なら新しいSTOPへ記録し、自動retry・再compile・閾値緩和をしない。今回の`resume=1`は利用者指示による中断補完1回を示す。過去のretry/resume禁止やSTOPの抹消ではない。新sampling、B2、別q/endpoint、ground solve/DF refit、N/成功確率、全RPE総費用、詳細科学レビューは実施しない。

最終結果は固定fixtureでの局所binary64技術証拠に限る。q/path/cut/融合条件の差、追加qubit/CSWAP/global cut、RZ/CX/depthの利害を区別する。ユーザー／GPTが次の研究判断を担当し、Codexは公開後mandatory STOPする。
