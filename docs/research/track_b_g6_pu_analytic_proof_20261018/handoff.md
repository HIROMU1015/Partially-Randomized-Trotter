# ユーザー・GPTへの結果引継ぎ

2026-10-18 JST。**P_Uは`DISPROVED`。** 全量化域内の反例は

\[
L=2,\quad p_1=p_2=\tfrac12,\quad x=1,\quad m=7,
\quad U_7-B^{\rm form}_{5\oplus o,7}>\tfrac{13}{1024}>0.
\]

[解析記録](analytic_proof_attempt.md)と[反例certificate](proof_or_counterexample_certificate.md)に、有限wordのbinomial計数、次数5/7のempty係数、P5各群、平方根の有理上下評価を記載した。正のpair差だけでなく、root変化とordinary tailの負の差も含めて向きを確定している。

1. **結論**：P_{U,5}の厳密な反例でP_U全体が否定された。部分解決・近似数値による判定ではない。
2. **適用範囲**：指定命題の普遍性を否定する1件。P_{U,3}、他入力での大小、等号条件・成立域を追加調査していない。σや追加のQ間関係に頼らず、固定scalar定義を使った。
3. **Full/Uの区別**：この反例から理想B_Fの方がclosedより大きいとは言えない。既存`B_F≤U≤B_o`の保証も否定していない。形式P5＋tailの実装完成を仮定していない。
4. **追加理論の要否**：P_Uの決着には不要。反例確定後、有利な条件・別次数・別不等式・新上界を探索していない。外部の独立した科学レビューを実施したとも主張しない。
5. **W2b/W2cへの結論**：P_Uの一様上界優位を前提にできない。N・費用比較へ進むにはprecision/bias/range/confidence/ceil、event価格・受理・reject古典費用を別途そろえる。今回新しい資源結果はない。
6. **次の担当**：ユーザーが結果を既存研究状態へ正式に反映するか判断する。GPTは、別途開始承認がある場合に限り結果の科学的範囲・次問題の採否を扱う。Codexに本命題の追加探索を続ける作業はない。新証明・重要レビュー・実装・科学計算への自動移行はしない。

W2a正本は `43d6533970e0eb4c430fe1cff2e798d681b707d4`、今回branchは `track-b-g6-pu-analytic-proof-20261018`。固定出典14件は[source_inventory.json](source_inventory.json)。新しい解析的導出と未検討範囲は[evidence_index.json](evidence_index.json)。管理検査は[verification.json](verification.json)、内容commitの公開照合は[publication_receipt.json](publication_receipt.json)。最終receipt commitはpush後に全公開bytesを独立照合し、チャットでも報告する。

数値プログラム、数式ソフト、science SOURCE/test import・実行・変更、保存値再解析、回路・合成・新grantは0件。管理コードは証明・反例を評価しない。親tracked/直下untrackedのbytesを保全照合し、他worktree（F1/F2、W1を含む）の内容・index・lock・process・出力は参照・変更しないため、それらの独立した変化を監視したとは主張しない。旧STOP/CLOSED/HOLDは保持する。

**成果物公開・保全確認後mandatory STOP。** 本反例を救済するための追加探索、W2b/W2c、別理論、SOURCE実装、新規科学計算、論文執筆、他branch統合へ進まない。
