# O2案M：N1 header binding技術準備の報告

2026-11-08 JST／SERVER1 GPUA100-gleap-01。base `d07a76de8e7a9995b1d55c258556b86fe58b1e3a`、専用branch `codex/o2-m-n0-header-binding-20261108`、独立worktree `/tmp/o2-m-n0-header-binding-20261108`。

**保存schema→literal A/B/V/角/phase/actual law→既存群適格性へのheader binderと、一回metadata-only入口を技術実装した。新人工10件＋関連する既存適格性3件がPASS。実N1の群集計は未実行。** 実H4での非自明群・selected policy binding資格・分散/費用改善はUNKNOWN。

## 成果物・再利用

- [M_N0_HEADER_BINDING.md](M_N0_HEADER_BINDING.md)：保存schema、元SOURCEとの対応、V=保存selected basisのadjoint、application/積順、sign/phase/Q保持。
- [M_N0_ARTIFICIAL_CORRECTNESS.md](M_N0_ARTIFICIAL_CORRECTNESS.md)：人工headerとmock入口の結果・拒否理由・旧suiteとの境界。
- [M_N0_METADATA_EXECUTION_DRAFT.md](M_N0_METADATA_EXECUTION_DRAFT.md)：実行前preflight、固定input blob、一回/retry0、出力・判定・STOPと未採択の実行契約。
- [binder](../../../src/trottertracks/algorithm_codesign/m_n0_header_binding.py)、[実行module](../../../src/trottertracks/algorithm_codesign/m_n0_metadata_execution.py)、[科学CLI](../../../scripts/tracks/algorithm_codesign/run_m_n0_metadata.py)、[新test](../../../tests/tracks/algorithm_codesign/test_m_n0_header_binding.py)、[人工入口](../../../scripts/tracks/algorithm_codesign/run_m_n0_header_artificial.py)、[最終人工証拠](../../../artifacts/o2_m_n0_header/2026-11-08/artificial_run02/)。

既存`CategoricalLaw`、`SlotLabel/qualify_group/partition_supplied_rows`、basis canonical hash/operation dataclass、対角primitive descriptor、N1 canonical/digestを再利用した。元SOURCEのgate/phase/選択規則12箇所はN1科学SOURCEとAST一致。matrixを伴う`restore_operations`やstate/full回路を伴う`SavedEngine`は呼ばない。

既存M adapterがQiskit回路しか受け取れなかったため、immutable `LiteralCircuit`とkey受理を必要最小限追加した。eligibility本体と測定instrument本体は維持し、Literalの量子buildは明示拒否する。旧12件の人工instrument実験を再作成・再実行していない。変更接口に関係する既存3件のみ回帰確認した。

## 今回の取得と未取得

最終人工run02は新10＋関連3、0 failure/error、測定区間0.1162秒、peak RSS 461,208 KiB。run01も新9＋関連3がPASSし、追加分岐/固定範囲を入れた最終版と分けて保全した。人工basisのmatrix hashはschema用の発明した文字列で、matrixを新規評価していない。SOURCE content hashと環境を保存するが、CI/外部再現・実科学資格ではない。

実N1の5入力はGit object名だけを固定し、実catalogからの新しい適格性/群和/qを計算していない。今回の実N1 pass・新量子simulation・state/Born・RNG/shot・回路build/compile・分散/N/期待費用・旧raw再解析・重要科学レビューは0。N1/N2再生成や旧engine認証流用も行わない。

一回入口はfixed SOURCE/環境/input blobを照合し、binder/partitionを各一回呼び、STOPで終わる実装になった。**入力を投入すれば実科学のmetadata集計になるため、現時点では起動しない。** 実保存値の未対応/不一致があればUNKNOWN/STOPとし、実入力での成功や非自明群を未実行のまま保証しない。

次は草案の固定N1、n=0唯一rotation slot、一回metadata-only pass、retry0を明示的に許可するかを判断する。群適合が得られても、Born/shot・新誤差・native価格・科学的採択は別段階である。既知ancilla-free/groupingとの比較を落とさず、M/D・O2全体のGO/NO-GOや独立新規性は採択しない。

## 指示・保全・公開・STOP

正式共通指示2026-10-25（SHA256 `e1e3e2625e2b728977402bded6583658e7fb762160f5e787fd45572af0095720`）、適用AGENTS、PROJECT_MAP、研究概要/目的/方法、前段5資料・実装・STOPと関連生成SOURCEを確認。§8.1/8.2/8.4/10/10.1/10.3に従い、実行許可と技術方針採択を分け、既存コードを優先した。

元main・dirty/untracked、前段d07a76d/154c6e3/d28164b5、N1/N2/旧engine、旧SOURCE・結果・raw・STOP・grant/ledger/marker・C1凍結、O1/O3/C-S、他worktree/process/cacheを保全する。新diffでのM接口拡張と固定旧SOURCEの変更を区別し、旧commit/結果/STOPを上書きしない。共有OS/systemd/cgroup/scheduler/venv/cache等への変更は0。短い人工metadata検査のため新しいresource上限やmanager・grantを新設していない。

公開前に新コード/文書/人工SOURCEのhash、固定Git参照、旧関数AST・旧資料byte・既存索引のappend-only、manifest旧項目、元mainのHEAD/status/2,136ファイルbyteを照合する。専用branchへcommit/公開し、remote SHA・全公開本文・blob/treeも照合する。公開固定SHAは最終チャットで示し、自己SHA追記だけの追加commitを作らない。

公開前の技術照合はPASS。最終人工実行の21 SOURCE hash、新4文書の23参照、固定input5 blob、manifest旧20項目、元mainのHEAD/statusと既存2,136ファイルbyteを確認した。既存Markdown7点は追記のみ。旧Mのeligibility・readout等のASTと数値instrument本体（Literal専用拒否guardを除く）は維持されており、旧test/raw/STOPを変更していない。

**公開照合後mandatory STOP。実N1のmetadata pass、実H4のBorn/shot/compile、案D/O1/O3、旧研究再開、重要科学レビューへ進まない。**
