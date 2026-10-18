# G6/G7 → DF/PR/RTE：End-to-End実装契約の独立技術監査

2026-10-18 JST。Codex Bの静的監査。基点 `5f17a7548b7eef1976f45f5d09bdc979e2e0a81b`、専用branch `track-b-g6-e2e-interface-contract-audit-20261018`。他branchのmergeなし。

**既存SOURCEには、DFの成分・basis取得、G6/G7の有限bit局所生成、DFのcontrolled Z/ZZ回路、ordinary partial-S2／Hadamard wrapperがある。しかし、これらをG6の実確率・重み・事前zero・入力誤差を保持して結ぶEnd-to-End実装は確認できない。とりわけ現行partial-S2 requestはordinaryの角度・確率・共通normalizationを検証し、G6 eventの直接代入を許す契約ではない。回路builderと、backend実行・全trial estimatorも別である。**

本書で定めるのは、既存実装に基づく必要フィールド・不変条件・将来の受入条件である。新adapter、selector、sampler、回路、テスト、実行契約Cは作成しない。数学的に未確立の条件、入力target・precision/confidence・複数の接続方式は採択しない。

## 成果物

1. **[Interface Contract Matrix](interface_contract_matrix.md)**：I01〜I18。入力／出力、SOURCE、保証、型・単位・位相、実装状態、不足、受入条件、認証、次作業。
2. [E1〜E7のSOURCE trace](end_to_end_source_trace.md)：型とcall chain、ordinary専用guard、接続の切れ目。
3. [精度・誤差契約](precision_and_error_contract.md)：入力表現、Taylor、RA、finite-bit、角度、native、統計、PFを分離。
4. [測定・費用契約](measurement_and_cost_contract.md)：全trial分母、weight、事前zero、失敗、量子呼出し、費用vector。
5. [将来の受入検証V1〜V6](verification_acceptance_plan.md)：fixture・期待性質・必要な承認。今回のテスト実行は0。
6. [既存fixture可用性](existing_fixture_availability.md)：保存物と再生成を要するtestを区別。NPZはdecodeしない。
7. [最小の実装gap計画](minimal_implementation_gap_plan.md)：依存順の技術作業と、未採択の選択事項。
8. [引継ぎ](handoff.md)：次の担当と承認、STOP。

[SOURCE inventory](source_inventory.json)、[固定証拠索引](evidence_index.json)、[機械可読interface matrix](interface_contract_matrix.json)、[状態・権限](audit_status.json)、[管理検査](verification.json)、[公開receipt](publication_receipt.json)を相互参照する。

## 固定資料と再利用

- [M1〜M12 Evidence Packet](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d4163acd7b577afaa7d6db8c29db64f76ec03ae7/docs/research/track_b_g6_construction_paper_evidence_20261018/README.md)：既存保証の根拠として再利用し、数学レビューをやり直さない。
- [非Pauli応用調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/128a05765985c1c6aec3a1d0ce35e19ffbb3b3f6/docs/research/track_b_g6_nonpauli_application_survey_20261018/README.md)：別branch固定commitをGit objectから読取。基点に当該ディレクトリがないため、存在を仮定したlocal linkやmergeを作らない。
- [P1〜P12の全生成対応監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f17a7548b7eef1976f45f5d09bdc979e2e0a81b/docs/research/track_b_g6_full_generation_prior_art_audit_20261018/README.md)：access・precision・費用の意味論を再利用。先行法の再探索・独立新規性判定なし。
- G6 `28cfabb1…`、G7 `ab2549f4…`、DF/PR/RTE `321cd8c3…`、G8 `cb60a4a1…`、G9 `0ef2b927…`。完全SHA、実際に読んだ範囲、宣言行、bytes/SHA256をinventoryに記録。作業基点とSOURCE取得元の役割を区別する。

親の共通運用指示2026-10-17を実際に読解し、§8.4を適用した。既読の親PROJECT_MAP・研究概要は前回保存identityと一致を確認して再利用した。私的Claude/GPTレビュー原本は本監査へ取り込まず公開しない。

## 独立性・状態・STOP

Codex Aの `track-b-g6-theory-deepening-triage-20261018` 作業領域、未commit変更、index、lock、process、作業ファイル・結果にはアクセスしない。Aの完了を待たず、理論課題の選別・優先順位を再評価しない。保全監査もAの状態を収集せず、親のtracked状態、以前の自分の公開Packet、既知の消費済みledger、固定Git blob、および今回の書込範囲に限定する。

D＋E、Parent限定成果、G6〜G10、数値探索の区切り、DF/Gaussian等のclosure、他研究と既存STOPを維持する。新科学計算、保存値再解析、科学SOURCE変更/import/実行、formal test作成/実行、回路build/compile/合成、新保証・科学レビュー、実行契約/authorization/grant/marker/claimはすべて0件。公開・保全確認後 **mandatory STOP**。

管理検査44項目はPASS。固定SOURCE/test 32件、固定資料91件のbytes/SHA256・blob、I01〜I18の具体的な関数・行位置とE1〜E7/V1〜V6、リンク、変更範囲を照合した。旧non-index blob 7130件、親の対象tracked状態・直下Markdown identity、以前の自分の公開Packet、既知の消費済みledger 13ファイルを保全確認した。Codex Aの作業領域を監視して保全を主張したものではない。公開receiptはpayloadの公開bytes照合後、metadataのみの子commitで追加し、自己hash循環を避けて最終HEAD・最終全変更fileを別途照合する。
