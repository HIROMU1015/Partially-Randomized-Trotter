# Track A Mechanism Analysis integrated readiness

2026-10-19 JST。**READY_FOR_USER_AUTHORIZATION / mandatory STOP**。主要30比較＋H8独立n48主2比較を固定し、runner・人工検査・入力／出力契約の準備を完了した。**実科学データでのG再構成・比率・平均／SE・ランキングは0件。**

## 固定SOURCEと成果物

実装SOURCE：`5c256067db29a326bcaa0a0e244b38f6b87bd138`。branch：track-a-mechanism-runner-readiness-20261019。基準commit：0a8ac1d2d03c5a9aef2acab0295226f8c88b38ce（GitHub commit/treeとローカル一致確認済み）。SOURCEは新規ローカルcommit、公開push未実施。

1. [mechanism_runner_source_readiness.md](mechanism_runner_source_readiness.md)
2. [mechanism_input_binding_final_DRAFT.json](mechanism_input_binding_final_DRAFT.json) — public projection。完全path/SHA・private payloadは同名の内部companionへ分離。
3. [mechanism_input_gap_report.md](mechanism_input_gap_report.md)
4. [mechanism_execution_contract_READY_FOR_APPROVAL.md](mechanism_execution_contract_READY_FOR_APPROVAL.md)
5. [mechanism_test_report.md](mechanism_test_report.md)
6. [h4_future_integration_contract.md](h4_future_integration_contract.md)
7. 本integrated_readiness_report.md、[output_schema.json](output_schema.json)。

## 入力の到達点と未解決

固定32比較のC/N・保存exact G・元SE・PF直接/model・cohortを追跡するprivate入力契約を作成した。42 input roleのbyte snapshot/hashと元参照を保持する。旧nativeの省略bindingはsnapshot/inputのschema差を明示し、完全なA/D sealと登録科学条件を照合した。

64 method record中60で説明fieldを結合。残る4はA18_OLD_PR（H4/.0005）、A58_OLD_PR（H6/.0005）、A82_OLD_PR（H8/.05）、A98_OLD_PR（H8/.0005）のPR側。対象精度の会計未保存など、実際の不足項目はgap reportに列挙し、取得できたB/biasと未取得headroom等を分離した。coreを除外・0補完しない。

これらはmetadata readinessであり、数値CORE_COMPLETE／EXPLANATION_COMPLETEの判定をまだ行っていない。H8 n48のprivate seed/control/native原本や元profile bytesを新たに完全再監査したとの主張もしない。

元結果は64c85f7a…、fc6bd039…、bf8c1905…、e6eea538…に結合し、α343e0c59…／希少層cf8e93ee…は既存の別系列として保持。追加SOURCEの実取得範囲は入力bindingと取得receiptに記録し、読んでいない全closure・全歴史文書を読了扱いにしない。

## 実装・検証・保全

coreは整数/Fractionの費用会計、R_N×R_CeffとPF軸費用重みの恒等式、保存exact値との一致を実装。normalization/headroomは保存値が揃う場合だけ説明し、保存shotを置換しない。SEは転記のみ。

64人工／mock検査PASS。production入口は承認receipt・SOURCE・input/runtime・32 IDs・専用出力をgateし、one-shot／worker再入を拒否。実行前後のgrant/ledger/marker/runtime/inputと結果SHAを受け入れ、native worker exit0/reap後だけ完了する。人工テストと実科学数値一致を混同しない。

過去STOPと失敗準備logを保全した。H4 D/Pの並行worktreeを調査・変更せず、既存科学jobへの操作は0。元A/D SOURCE・結果・grant・ledger・marker・STOPへの書込みは0。最終metadata監査では42入力roleのsnapshotと35原入力roleのbyte不変、Git固定blob6件、32対象、旧STOP不変を確認した。新しい有効grantの発行・消費0、実科学出力rootは未使用、公開push0、新科学レビュー0、AI sub-agent0。

資源は開始時にaffinity64 CPU・空きmemoryを読み取り観測。小さい処理はmetadata準備・scalar workerとして重複を減らし、CPU/RAM/時間の強制capや共有環境変更は行わない。

## 次の担当とGO/STOP

**ユーザー**：固定SOURCE、private入力bindingと契約SHA、32対象、説明4件の欠測許容、統計再計算なしの範囲を確認し、保存値数値解析をone-shotで実行するか判断する。承認文案は実行契約末尾にある。

**Codex**：今回の準備は完了してmandatory STOP。実行承認まではproduction解析、有効grant発行、212補足追加、H4 D/P接続、公開push、新レビューへ進まない。既存科学jobを止める意味ではない。
