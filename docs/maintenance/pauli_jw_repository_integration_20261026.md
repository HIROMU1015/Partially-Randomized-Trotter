# Pauli-JW成果統合：文書整備とmain導線案

2026-10-26。文書branch `codex/pauli-jw-research-summary-20261026`。
基点は最新H2結果 `4aea054b3369709bc0d35ff081071e62b5f77cbe`。
この作業は既存結果の引用・文書構造と参照の検査のみで、新しい科学計算、N/G再解析、候補選抜・再ランキング、重要科学レビュー・claim正式採択は行わない。

## 成果物と出典

中心資料は[Pauli-JW研究成果の総合整理](../research/pauli_jw_resource_evaluation_summary.md)。8節で目的・方式・科学条件・最新比較・知見・DFとの関係・変遷・限界・出典をまとめた。
[15行CSV](../research/data/pauli_jw_summary/latest15.csv)にはPF/B0保存対照と最新Hybrid/RTEを対応付け、既存のID/cohort/n/G/比/SE/順位ラベルをそのまま転記した。
14対1は元の保存判定ラベルの件数で、新たにGを比較して決めた順位ではない。5精度は価格標本を共有する。

既存 `integrate_pauli_h2_h4_h6_saved_results.py` の一意行選択・CSV文字列転記を再利用し、新しい2つの最新CSVに合わせた小さな文書adapterを追加した。
同adapterはNumPy、scientific runner、shot/G関数をimportしない。数値演算や候補sort/min、CI計算も行わない。

```bash
python3 scripts/document_pauli_jw_saved_summary.py check
```

転記値、3分子×5精度の一意対応、固定Git blob、既存本文の保全、文書リンクを確認する。科学テストの再実行は不要。
[固定出典一覧](../research/data/pauli_jw_summary/source_references.json)と[文書検査](../research/data/pauli_jw_summary/verification.json)を保存する。

H4 PR探索報告はこの基点treeにファイルが存在しないため、指定d3fdfadのGitHub/Git objectから読んだ。欠落を新しい科学計算で補わず、固定commitへのリンクで依存を明示した。
過去の10/25統合はその時点のsnapshotとして残し、最新版15行とは異なる選抜・確認範囲を中心資料§6で示した。

## 入口と実験履歴を分離する

README、PROJECT_MAP、docs/README、研究概要・現状、研究READMEに短い中心資料リンクを設ける。
PROJECT_MAP等の元のH1より前に積み上がっていた日付付きprefixは、本文を削除せず折り畳みの「過去の実験・追記履歴」に置く。新しい進捗ログの長文は追加しない。
元の分類・研究方針本文は保全し、この限定Pauli系列を他系列の正式研究方針へ置き換えない。

VALIDATION_STATUSは文書案内だけを追加し、既存の科学status、旧auditのoverall_status、certificate未受理を変更しない。

## Manifestの既存不整合

変更前に現行 `scripts/check_validation_manifest.py` を実行したところ、exit1だった。
schema_versionは1だが、validatorはrootを7キーに限定し、現manifestは198キーを持つ。後続の多数の研究entryがunexpected keyとなっている。
これは今回作成したキーによる失敗ではなく、変更前からのschema/validator不整合である。
本作業ではmanifestの新キー追加・schema移行・status再分類を行わず、元bytesのSHAが同じことを確認する。
索引は別の `docs/research/data/pauli_jw_summary/` に保存した。構造検査のFAILと、個別研究の科学的有効性・技術検査PASSを混同しない。
将来manifestを整合させる場合は、対象schemaと既存entryの意味を別の小さな保守作業として確認する。今回その作業や科学レビューへ移行しない。

## Branch構造とmainへの依存

取得したGitHub mainは `0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf`（2026-09-23）。
mainは基点H2 branchの祖先で、比較時点ではmain-only 0／research-only 546 commit、tree差分は20,636ファイルだった。
詳細は[構造記録](../research/data/pauli_jw_summary/repository_structure.json)。ローカル全branchをfreezeしたり変更したりはしていない。

この文書branchをそのままmain向けPRにすると、文書差分だけでなく既存10月研究・実装・証拠全体を取り込む。今回の文書整理の範囲を大きく超えるため、そのPRやmergeは実行しない。
依存は次のように分ける。

| 内容 | 必要な依存 | 今回の扱い |
|---|---|---|
| 最新研究の閲覧 | 中心資料と15行CSV、固定結果commitへのリンク | 文書branchに公開 |
| 数値の出典確認 | H2 4aea054b、H4/H6 10cb3e5c、fairness 288d2206等 | 固定URL／Git blob照合。rawの再複製・再解析なし |
| Raw/QPY復元 | 各元studyのarchive manifest/parts、指定restorer、元SOURCE | 元refの証拠入口へ案内。main移植は今回不要 |
| 実装再実行 | 研究ごとの固定SOURCE/runtime/input/契約、別の実行承認 | STOP維持。今回の文書公開は再実行承認ではない |

## 推奨するmain向けPR案

**mainの現HEADから新しい小さなbranchを作り、READMEに公開済み総合資料の固定commit URLを追加する、文書導線だけのPRを第一案とする。**
main上の既存の旧DF screening/UWCの注意は残し、Pauliの新しい有限・条件付き研究をそのstatusへ混同しない。
この入口だけならSOURCE・raw・結果・既存実行契約・packageをmainへ移植せず参照可能にできる。

PRタイトル案：`Add an entry point to the Pauli-JW H2/H4/H6 research synthesis`

PR本文案：

> Add a README link to the published Pauli-JW resource-evaluation synthesis and its saved 15-condition table. The synthesis separates finite-cohort, hypothetical-rho results from unproved population-cost and implementation-error guarantees, and retains the dated experiment history. This proposed PR changes navigation only; it does not merge the October research branches or adopt new scientific claims.
>
> Validation: fixed result refs and literal table transcriptions checked; linked documentation available. No scientific computation or numerical reanalysis performed.

公開資料はcommit `0e05f061c86ae853e90568abd4842de652c82398` に固定した。
そのURLを使う[README-only patch](pauli_jw_main_readme_entry.patch)と[PR本文案](pauli_jw_main_readme_PR_DRAFT.md)を添付した。
観測したmain SHAのREADMEに対する `git apply --check` と元本文保全を、隔離した一時directoryだけで確認した。[検査記録](../research/data/pauli_jw_summary/main_patch_verification.json)。
mainへのpatch適用・PR作成・main mergeは実行していない。
ユーザーが資料そのものをmainへ同居させたい場合は第二案として、main起点で中心Markdown＋小CSV/出典索引だけを明示パスで移植し、各リンクの依存閉包を再確認する。
巨大な研究branchのcherry-pick/mergeでこの移植を代用せず、rawや科学実装を取り込む判断は別にする。

## 終了と次の担当

専用文書branchをcommit/pushしlocal/remote SHAを照合する。main、旧科学SOURCE/raw/grant/ledger/marker/STOPとmanifestは変更しない。
文書リンク・転記・保全検査が完了したらmandatory STOP。
次の担当はユーザー/GPT：main向けREADME-only PR案の採否と、科学的解釈・中心claim・詳細レビュー・論文反映の要否を判断する。
Codexはその後の明示指示がある場合だけ指定作業を行う。
