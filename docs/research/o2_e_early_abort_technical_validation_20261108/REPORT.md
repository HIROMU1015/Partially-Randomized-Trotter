# O2案E：実装・人工correctness・技術的選別

2026-11-08 JST／SERVER1 GPUA100-gleap-01。base/設計081959402f54f06048adb038f97911950091ae1c、branch codex/o2-e-early-abort-20261108、独立worktree /tmp/o2-e-early-abort-20261108。

**instrumentを実装し、新規14＋関連A2回帰2＝16件PASS。技術分類TECHNICALLY_VALID_KNOWN_LCU_APPLICATION。** 標準LCU＋同じ早期終了に対する操作・推定量・論理分岐費用の増分利益0。案E固有の追加構造を確認していない。

## 成果と限界

- [実装・人工検証](E_INSTRUMENT_AND_ARTIFICIAL_TESTS.md)：共通A/Bを一度ずつ使う明示gate列、dynamic flagと三値decoder。独立label別Hadamard/system行列/全branch参照でcomplex平均・M2=p_s・失敗X/Y寄与0を照合。
- [baseline・費用](E_BASELINE_AND_COST_COMPARISON.md)：通常PR、LCUのearly有無、direct/grouping、理想層別化を区別。全attempt prefix・fresh状態・追加制御/flag/feed-forward負担を明示。
- [次の技術判断](E_NEXT_RESEARCH_DECISION.md)：既知法の応用成果として一区切りとする技術提案。人工PASSから実H4科学SOURCEや大きいbenchmarkへ自動移行しない。
- [人工SOURCE](../../../artifacts/o2_e_label_erasure/2026-11-08/artificial_run01/SOURCE.json)／[結果・全分岐](../../../artifacts/o2_e_label_erasure/2026-11-08/artificial_run01/TEST_RESULTS.json)／[log](../../../artifacts/o2_e_label_erasure/2026-11-08/artificial_run01/RUN.txt)／[STOP](../../../artifacts/o2_e_label_erasure/2026-11-08/artificial_run01/STOP.json)。

人工条件は発明した1/2 system qubit・2-label、p=2/5・3/7等、非可換A/B/W、signed/complex phase、X/Y。分子geometry/basis/DF rank/L_D/δ窓は非該当。実H4改善の証拠ではない。

PREP確率ずれ・逆PREP不整合の例でbiasとinactive失敗branch漏れを確認。旧A1 engine誤差上界は継承しない。hardware feed-forward、actual Qの量子PREP、新一様誤差、compiled資源・十分N・期待総費用はUNKNOWN。通常mixtureへのvariance利益高々factor2を、最適層別法への利益と取り違えない。

## 差分・SOURCE・運用

新規はE専用小adapter、test、薄い人工runnerだけ。A2 gate検査・controlled phase・Hadamard bit規約、M人工branch列挙/保存方式、matrix helperを再利用した。A2のβ²/fine−coarse decoderは不使用。旧SOURCEのcode修正は不要で、人工run01は初回PASS、技術修正・再試験なし。旧M/A2全suiteを重複実行せず、関連A2の2件だけ含めた。

SOURCEは実行code/helper/test SHA256、設計commit、依存version、起動時空き資源を保存し、最終公開commitの同じcodeと照合する。Qiskit1.3.0/NumPy1.26.4/Python3.12.3、suite0.536秒、peak RSS542,804KiB。共有環境の既存SyntaxWarningを表示したがfailure/error0。小計算のためCPU/RAM制限・worker管理・共有cache/venv変更は不要だった。

正式共通指示2026-10-25の§8.1/8.2/8.4/10/10.1/10.3、適用AGENTS/PROJECT_MAP、研究概要・目的・方法を確認。今回の人工回路/statevector/Born全分岐はユーザーが明示許可している。追加承認手続き・grant/framework・重要科学レビュー記録は作成しない。

実H4/state/Born、実科学shot/RNG、保存N1 label再分割、旧raw/bit数値再解析、実回路compile/native synthesis、科学variance/N/費用評価は0。M/D保留、C1凍結、元main・dirty/untracked、旧SOURCE/raw/結果/grant/ledger/marker/STOP、他研究worktree/process/cacheを維持。新資料・索引・概要/当日ノート・manifest新項目だけを追記し、既存科学的statusを上書きしない。

専用branchへcommit/公開し、remote SHA・公開全本文・blob/tree、実行SOURCE hashと結果、旧manifest/元main保全を照合する。最終SHAは最終チャットと作業用照合記録に示し、自己SHA追記だけのcommitは増やさない。

**公開照合後mandatory STOP。新科学・実H4 benchmark、M/D追加検証、旧研究再開や重要科学レビューへ自動進行しない。**

公開前の技術照合PASS：文書18参照先、実行SOURCE7ファイルのhash、旧manifest23項目不変、元main HEAD/statusと既存2,136ファイルbyte保全を確認した。これは資料・SOURCE照合で、追加の人工試験や科学計算ではない。
