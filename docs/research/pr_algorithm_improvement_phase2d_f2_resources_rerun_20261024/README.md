# 限定F2：位相保持の技術修正と新規実行

ユーザーが旧停止後の原因修正・再実行と、意図しない技術的中断時の修正・新規実行を明示承認した。
旧結果 `06ce7e5b1002de3cfaec6c17ea901f589c05f8f6` と旧SOURCE `9b1e103102f918bb157dea90c7eaf3acb5d88b6c`、raw／grant／marker／STOPは保全する。
**このSOURCE固定時点は実行前**。修正版、新grant、新outputで全件を新規実行し、旧runをresumeしない。

## 原因と修正

保存済み失敗3回路とPASS対照1回路を用いて、QPY読戻しが独立referenceと一致することを確認した。
失敗3件の保存compiled operatorはreferenceの符号反転と約1e-15以内で一致した。
passごとの診断で、最初の不一致をBasisTranslatorまで局在化した。
人工RY(-.185)でも同じ入れ子のequivalence変換でglobal phaseの欠落を再現し、等価な直接U入力では発生しなかった。
これは科学的な意味論や許容差の変更を必要とする不一致ではなく、固定環境Qiskit1.3.0で観測した変換経路の問題である。

既存の公式UnrollCustomDefinitionsを空の独立EquivalenceLibraryと共に用い、
各gateの公式definitionを固定native basis＋Uまで展開してから、元と同じtranspileへ渡すようgates.compiled_wrapperを修正した。
分解のglobal phaseを明示的に保持し、出力matrixから位相をfitしたり、πを推測して加算したりしない。
共有SessionEquivalenceLibraryやインストール済みQiskit、venv、他jobは変更しない。
公式分解・PassManagerを再利用し、新しい回路合成器を実装していない。

旧失敗3件を修正後のproduction helperで再compileし、operator最大6.28e-15、signal最大5.48e-15で通過した。
この4保存caseではRZ／CX等の整数countは旧compiled回路と同じだった。
全caseへの外挿はせず、新しい全件実行で検査する。
人工operator回帰3件＋既存metadata/cache proxy2件、計5件がPASS。
metadata testは新たな公式lowering段階をmockし、metadata検査の範囲を維持した。
[修正の証拠](fix_evidence.json)、[新しい回帰test](../../../tests/management/test_pr_dictionary_compile_phase.py)を参照。
これらの修正診断は今回の明示承認に基づく限定技術作業で、新しいfixture探索や性能benchmarkではない。

## 維持する比較・実行境界

4 mode・2粒子・occupied(0,1)・L_D=1・残差density-square 2fragmentの同じ合成Hamiltonian。
geometry／AO basis／分子DF rank非該当。T=1/5、q1、delta=1/5の一点、r2、h1/10、K0。
SS/SP/PS/PP、全ordered trajectory6,837、wrapper／logical compile要求各13,676、reference27,356評価を維持する。
独立reference90/128digits、rtol0、phase fitting禁止と全P0閾値は変更しない。
Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、topology-free、approximation_degree1、qubits_initially_zero=Falseを維持し、
修正されたcompile接続は新SOURCEに含める。異なるコンパイラ条件や新しい科学仮説への変更は行わない。

[新しい限定契約](contract.json)と保存B入力7件を新packetへbindingする。
旧統計資格付き契約・gateは変更せず、η／α／γ／interval資格は未成立のまま。
認証N・N加重費用・同精度総資源優位を算出・主張しない。
既存F2 runner、builder、reference、compile cache、exact weighted集計を再利用する。

利用可能CPU affinity32、MemAvailable約53 GiB、低いloadを確認した。
独立trajectory／compileをprocess並列化し、内側threadは各1として過剰競合を避ける。
CPU／RAM／wallの人工capは設けず、旧SOURCE・共有環境・他者jobを変更しない。
新grantとoutputはattemptごとに分離し、一つのgrantを再消費しない。正常完了時はmandatory STOPする。
技術的中断なら今回の最新承認に従い、証拠保全後に必要最小限の修正を固定し、新grantで進める。
科学条件変更や登録外実験が必要なら停止する。

予定outputは `artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/<fresh-run-id>/`。
公開結果は同date directoryの要約・case資源・hash／provenanceと本資料へ記録し、承認原文・raw・内部pathは非公開で保全する。
正式共通指示2026-10-23の§8.4、§10、§10.3を適用する。
完了後の担当はユーザー／GPT：4armのactual per-shot資源とtradeoffを確認し、γ保証や認証N比較へ進む価値を判断する。
