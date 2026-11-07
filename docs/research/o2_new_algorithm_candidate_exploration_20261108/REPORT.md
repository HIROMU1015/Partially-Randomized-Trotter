# O2新候補探索・技術的選別の報告

2026-11-08 JST／SERVER1 GPUA100-gleap-01。base `d28164b5d51e0ca5a0ade74ded8ff176b443dab8`、独立worktree `/tmp/o2-new-candidates-20261108`、専用branch `codex/o2-new-candidates-20261108`。

**2つの操作案を残し、案M「共通basisのlabel平均を測定へ移す」の限定技術確認を推奨する。** 学術的新規性や科学的GO/NO-GOは確定していない。新アルゴリズム実装、人工量子simulation、科学計算、保存raw/bit数値再解析、旧test再実行、実compile、重要科学レビューはすべて0。

## 成果物と主要結論

- [O2_NEW_ALGORITHM_CANDIDATES.md](O2_NEW_ALGORITHM_CANDIDATES.md)：同じcomplex targetを保存する操作・式・情報アクセス・全費用・既存SOURCEと未実装差分。
- [O2_CANDIDATE_COMPARISON.md](O2_CANDIDATE_COMPARISON.md)：2案、7探索領域の扱い、強いbaseline、保留/除外理由。
- [O2_NEXT_RESEARCH_PROPOSAL.md](O2_NEXT_RESEARCH_PROPOSAL.md)：案Mの限定技術scope案と科学入力へ進む前の境界。

Mは、群内label別のcontrolled gateをsystem readoutと平均observableのancilla測定に置き換える。同じ条件付き平均でM2≤1を記号的に導け、SCVの特徴量変更やA2のherald補正とは異なる。ただしsystem readoutは新しい有料操作であり、A/B/Vが群内で共通という制約、新しい数値誤差認証、full-wrapper/feed-forward費用が未解決。実H4での非自明な適合群の存在・分散削減・量子資源利益はUNKNOWN。分類はKNOWN_METHOD_APPLICATION + ACTIONABLE_CANDIDATE（次の技術段階に限る）。

Dは局所回転差を小係数付きunitaryの加法的別測定へ移す。coarseの平均も有料であり、有限samplingでは実際の確率で補正する必要がある。既知telescoping/角度補間とC-S残差合成との重複、全path係数累積、base/補正の総費用が未解決のためKNOWN_METHOD_APPLICATION + PROMISING_BUT_INCOMPLETEとした。案Mを先に提案し、Dを自動実装しない。

安い価格normalizerを伴わないproposal更新、単なるCV特徴量追加/適応配分、Closed/CTS/D2逆相殺、phase-frame処理の再命名は独立候補から除外した。除外をO2全体の不可能性へ拡張しない。過去の限定T利益、S2/SCV/A2負材料、752-shot INCONCLUSIVE、C1凍結を上書きしていない。

## 確認した資料と範囲

正式共通指示2026-10-25（SHA256 `e1e3e2625e2b728977402bded6583658e7fb762160f5e787fd45572af0095720`）、適用AGENTS、PROJECT_MAP、研究概要・目的・方法と証拠優先順位を確認した。§8.1/8.2/8.4/10/10.1/10.3に従い、SOURCE固定と実行許可を区別し、既存コード再利用・共有環境非干渉・旧STOP維持を守る。

統合主文書/39入口の証拠索引/REPORTを出発点とし、39研究の再調査・raw再集計は行わない。新候補の成立性に必要なRTE順序/符号、selected basis schema、Hadamard X/Y、conditional actual Q/facade、A2明示gate、Return/Parent、価格接口、C-S局所residualを静的に確認した。SOURCEの所在と再利用条件は主文書へ記した。実stateやpayloadの新規数値評価を呼ぶloaderは実行していない。

外部一次資料はdirect measurement置換、TE-PAI、structure-aware variance reductionの3論文に限定した。直接測定置換と角度補間/構造的分散削減が既知であること、channelとcoherent平均の区別に用いた。論文の数値をPRの性能見込みへ転用せず、新規性を網羅的に保証する先行研究surveyともしていない。

## 技術確認・保全・公開・STOP

文書のUTF-8/改行、Markdown参照、固定Git path、append-only既存文書、manifest旧項目不変、変更範囲、元mainのHEAD/status/既存ファイルbyteを確認する。これは科学的correctness試験ではない。新directoryは要求されたMarkdown4点のみで、新しい研究管理framework・証明書・grant等を作らない。

公開前の文書技術確認はPASS。新4文書の15参照と4固定Git参照先、旧manifest18項目、元mainのHEAD/statusと既存2,136ファイルのbyte保全を確認した。既存Markdown5点は追記のみ、manifestは文書設計の1項目だけ追加する。科学module/testをimport・実行しない照合であり、CI/外部再現の証拠とはしない。

元mainのdirty/untracked、旧SOURCE・raw・結果・STOP・grant/ledger/marker・C1凍結を保全する。他worktree/process/cache、共有OS/systemd/cgroup/scheduler/venv、他ユーザーへの変更は0。通常の文書作業だけで、恣意的なCPU/RAM/worker制限は新設していない。

専用branchでcommit/公開し、remote SHAと全変更ファイルの公開本文・blob/treeを照合する。固定commitは最終チャットと作業用照合記録で示し、自己SHA追記だけの追加commitは作らない。

**公開照合後mandatory STOP。案M/Dの実装・人工量子simulation・実科学計算・重要科学レビュー、旧凍結研究へ自動進行しない。** 次の判断は案Mの有限技術確認を行うかであり、実H4 metadata/測定/compile等には別の有限科学契約と明示的な実行許可を要する。
