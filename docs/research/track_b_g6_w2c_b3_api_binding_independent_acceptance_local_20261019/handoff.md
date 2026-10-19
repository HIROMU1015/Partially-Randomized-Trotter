# 引渡し：Q1–Q10

判定：**閉鎖候補の静的・人工技術受入を限定的に完了。実科学と新port入りGlobal Bundleは未検証。Mandatory STOP。**

作業基点 `6bffb5b5819918698b522270ff19b255766d3072`。API候補公開HEAD `82896d53fa97235fa848d6c0833eb193af8b99d6`、人工測定SOURCE `78c571005fe766c61b32dc867e006531099ff608`。科学SOURCE `76ca3c4089b201d55da0fede08e1f77733ac38a5` はbytes/ASTだけを読む。これはローカル技術受入であり、科学的採択・本番実行許可ではない。

1. **Q1 固定bytes**：779件をGit blob/公開bytes/SHA256照合。62 measured source pins、10 input pins一致。raw12 parts/46,334,109 bytes/5,912 member一致。基点と取得元commitを区別。[SOURCE](independent_source_verification.md)
2. **Q2 静的API**：8module/201entryを独立ASTで取得し、4方式A/B/X/Yのcall site対応を確認。正入力canonicalのF3拒否、Full packet/child/requeryを確認。3.12 PEP701のreturn表示2件は3.10再parse未確立。動的schema/lawは未確認。[mapping](scientific_api_mapping_acceptance.md)
3. **Q3 port結果**：正常2件1536/16896完了。故障はS_generator224、P1_angle992完了＋inflight sidecar1。exact event、trace、F01、surrogate angle/hash-retentionを照合。数学的angleではない。[契約](port_contract_acceptance.md)
4. **Q4 seed/identity**：modeはglobal IDに含みseed domainに含めない。S/P1・A/B・phase/passの対応prefix一致、8arm/axis domain分離。bool/int含む改変拒否。科学bitsは生成0。[seed](bitstream_and_seed_acceptance.md)
5. **Q5 authority**：保存snapshotとbindingを照合。新規fake scopeの32競合winner1、spent/STOP/再claim拒否、commit-ACK unknown非返却を実測。保存historyを完全atomic証明と扱わず、実authorityは未実装。[authority](authority_and_preflight_acceptance.md)
6. **Q6 gate**：false/null/typed flag、source/runtime/input/schema bindingと閉鎖3入口を確認。直接9要求拒否、親/spawnでscience load前遮断。実import0。OS sandbox保証ではない。[gate](closed_production_gate_acceptance.md)
7. **Q7 monitoring**：既存/新adapter/実接続後/識別不能を分類。parent＋全worker CPU、outer wall、coarse/fineの非重複、保存量を確認。science費用null。UNKNOWN/Storage V2は固定Bundle回帰の証拠。[費用](monitoring_and_cost_acceptance.md)
8. **Q8 未接続**：port→固定pairと固定Bundle43/35を明確に分離。外部endpoint注入口は未実装、統合E2EはNOT_ESTABLISHED。今回統合変更0。[境界](global_bundle_integration_boundary.md)
9. **Q9 tests/不足**：独立975PASS＝既存845＋新130、Producer73は別。公開port22mutation拒否、Bundle35は32不整合/3証拠不足。今回の確認範囲で新Producer不具合は未確認。codec/real bits/authority/STOP/storage統合は次versionの未承認作業。[回帰](artificial_regression_results.md)、[不一致](producer_consumer_discrepancies.md)
10. **Q10 禁止処理**：実G7/F3 import/sampling、science bit列、新science計算/再解析、science SOURCE採択、実authorization/grant/claim発行/消費、native quantum処理、limited pilot/Full-N、既存claim再使用/STOP解除は全て0。

公開確認は [publication_receipt.json](publication_receipt.json)。receipt自身を含む最終HEADの照合結果は最終chatと所有temp receiptで示す。親worktreeのdirty/untracked、基点SOURCE/raw/contract/grant/marker/STOPを保全。

**次の担当はユーザー/GPT**：この証拠を評価し、別versionのGlobal Bundle統合作業のscope・契約判断を指示する。Codexは追加指示までSTOP。

保全補足：親HEAD/indexは不変。親の進捗資料3件（`docs/progress/README.md`、Track A参照メモ、詳細スライド構成案）は開始時hashとの差を検知した。今回の書込対象外であり、内容を戻さず[保全記録](preservation_verification.json)にbefore/observed hashを保存した。原因・編集者は未確定。新worktree以外の親Git statusは開始時と同じ。固定SOURCE/raw/claim/marker/STOPの改変は0。
