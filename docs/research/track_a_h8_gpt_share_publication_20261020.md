# H8：GPTレビュー資料のrepository公開

2026-10-20 JST。利用者から「できるだけリポジトリにあげてほしい」と指示を受け、匿名化済み資料・監査・作成/検査コードを公開対象にした。既存の科学結果・SOURCE・raw・grant・台帳・marker・STOPは変更していない。科学レビューと最終evidence freezeは開始していない。

## 公開資料の入口

- [まとめ読み用ZIP](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/H8_GPT_SHARE_PACKET_20261020.zip)：監査済み18file、202,450 bytes。SHA256 `901e416d0588fe2e011e32705179e628a41d76f9347ea8690def46472ec11bba`。
- [読み方・二条件・検証限界](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/00_README_GPT.md)。[file別の目的・SHA・残余リスク](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/H8_GPT_SHARE_FILES_20261020.md)。
- [結果とSOURCE](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/01_results_and_source.json)、[実行receipt](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/02_execution_receipts.json)、[native worker](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/03_worker_receipts.json)、[grant/台帳/marker/STOP](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/04_grant_ledger_marker_stop.json)。
- [seed/cohort監査](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/05_seed_cohort_audit.json)、[軌道・軸・native対応](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/06_task_axis_links.json)、[input/provider/runtime](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/07_input_runtime_bindings.json)、[原SHAと匿名化範囲](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/08_provenance_and_redactions.json)。
- [不足証拠](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/09_missing_evidence.csv)、[検証できない範囲](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/09_limits.md)、[固定SOURCE抜粋](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/10_source_conditions.md)、[固定commit/URL](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/11_public_sources.json)。
- [公開済み比較表](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/public/comparisons.csv)、[公開192軸](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/public/paired_metrics.csv)、[固定結果報告](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/candidate/public/result_report.md)。
- [作成時の共有前監査](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/audit/PRE_SHARING_AUDIT.json)、[今回の公開前検査](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/audit/PUBLICATION_VALIDATION.json)、[公開file manifest](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/publication_manifest.json)。

## 履歴と今回の承認

candidate内の「未送信」「共有未承認」や監査の`external_send_upload_push_performed=false`は、前回の候補作成時点の記録である。今回のrepository公開承認とは時点が異なる。監査済みcandidateとZIPのbytes・SHAを保持するため、それらの記録は書き換えず、この公開報告とmanifestで後続の公開範囲を示す。これは科学実行の新grantではない。

公開用branchは公開済みcommit `9f33ad0b11b4febe48b802000c38c5a0e3fd267c`から作り、非共有資料を含むローカルcommitを親にしない。非共有原本索引、内部絶対path、seed値とその復元に使える導出入力、private科学raw、未加工grant/承認本文/native identityは公開対象に含めない。原本への相関SHA・byte長・匿名化した実行構造など、残る開示範囲は共有一覧に記載した。

## 検査・再利用コード

```bash
python3 -B scripts/check_h8_gpt_publication.py
```

packetの標準library checkerはZIP内にも同梱する。安全な監査JSON 990件の原SHA、96軌道192軸の対応、grant/marker/台帳、欠測を構造検査できる。10件の改変検知を含む。これは原本の真正性や科学的妥当性の証明ではない。

作成コードも公開した：[抽出](../../scripts/prepare_h8_gpt_share_packet.py)、[packet/ZIP/監査作成](../../scripts/finalize_h8_gpt_share_packet.py)、[offline checker](../../scripts/verify_h8_gpt_share_packet.py)。元のローカル固定pathを、利用者が指定する引数に置換した。元科学SOURCEは変更していない。[変換の記録](../../artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20/code_publication_changes.json)にコードSHAを保存する。

原本を保有する環境で再作成する場合だけ、次の引数を指定する。これらのコードには外部送信機能も科学計算のdispatchもない。今回の公開処理では原科学計算・原結果の再解析は実行していない。

```text
prepare_h8_gpt_share_packet.py --prior-dossier LOCAL_DOSSIER --continuation-run LOCAL_CONTINUATION --origin-run LOCAL_ORIGIN --output-directory NEW_LOCAL_OUTPUT
finalize_h8_gpt_share_packet.py --prior-dossier LOCAL_DOSSIER --registry-file LOCAL_REGISTRY --output-directory NEW_LOCAL_OUTPUT
```

## 科学的な範囲

linear H8、隣接1 Å、STO-3G、tol-only1e-8 DF28、T=.8、P=0、二次partial PF・r4/K4、通常signed bias・allowance倍率1。p8/q2（L_D8、δ=.4）とp7/q16（L_D7、δ=.05）、各n48・96軸の保存結果と実行監査に限定する。標本点費用の再現と、未観測rare eventを含む母平均保証を区別する。探索n8・旧確認n32・条件付きpilotは混合しない。

原compiler profileと保存PR回路は未取得。元回路からのfingerprint再導出・旧compiler出力のbit同一性・数学的な確率独立性は追加認定しない。公開PF監査をPR実行監査に代用しない。新規sampling・compile・統計解析・費用再計算・再ランキング0。

**MANDATORY STOP。** 次担当はユーザー：公開済み資料をGPTレビューへ渡すか判断する。Codex/GPT/Claudeの詳細科学レビューと最終evidence freezeは未開始。
