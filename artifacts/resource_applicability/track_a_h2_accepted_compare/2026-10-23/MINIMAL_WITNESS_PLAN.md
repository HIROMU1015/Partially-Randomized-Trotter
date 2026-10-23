# PR identity不足に対する最小追加案（実行未承認）

**現時点で実行する追加科学callは0。PR単価再利用はBLOCKED。全67 tasks/134 axesの再sampling・再compileを自動計画しない。**

## まず科学call0で回収するもの

O00002のcanonical locatorまたはsender token mapを得て、期待SHA f011e12b…を照合する。必要ならO00001/f4da8233…へ戻る。元configから対象taskの `events_PRIVATE.json`、piece law、storage manifest/archives、元result/受入/correctness、runtime receiptを追う。

67 taskの対象ID・sample・各axisと、402 archive/134 compiled raw SHAの対応は[判定表](PR_identity_67_tasks_134_axes.json)、[archive照合](archive_raw_SHA_joins.json)に列挙済み。現在は公開SHA metadataの照合で、compressed/raw bodyを回収したものではない。134 compiled raw SHAの完全一致groupは134あり、現metadataからbyte-identical groupの縮約はできなかった。これは134 compileが必要だという意味ではない。

回収後、元eventのdigestとraw/archive SHAを確認する。既存metadataの読取りだけで結合できる項目を先に埋め、同じ入力stream・compiler・runtime・correctnessが十分に証明できるかを判定する。新しいDIAGNOSTICSのprefix1 prep/rotation hash/operation streamは取得済みなので、これを再生成しない。旧runtimeの実load証拠がない場合は欠測を保持する。

元Python3.12と新Python3.11の宣言byte差がある。既存の厳格runtime identity条件を満たせない場合は、そのままACCEPTEDにしない。ユーザーがcompiler関連closureの同等性を別scopeとして受け入れる場合も、根拠資料と新SOURCE/契約の明示判断が必要。今回その例外政策を採択していない。

## 影響task/axisの確定方法

現在「実際に再検証が必要」と確定したtaskはまだない。全67 tasksは判定保留対象であり、再計算対象ではない。各task/axisについて、archive body、event順序、wrapper/lowering、runtime/compilerの不一致が実証された行だけをaffected setへ入れる。設定名やhash欠測だけから全行をaffectedと断定しない。

単価を再取得する必要があれば、affected setのexact task ID・sample・元event digest・axis・原raw SHAを先に確定する。完全byte同一のstreamに代表検査を使う場合も、その同一性証拠を先に固定する。fingerprintやpolicy名だけで他のtaskへ外挿しない。

## 必要なら行う最初の限定probe案

以下は保存資料回収後に新SOURCE/別契約として設計できる**例**であり、今回のCOMPARE契約にも実装にも含めない。科学実行承認・grantはない。全cohortのidentityを証明する最終案ではない。

| prospective対象 | event/piece | axis |
|---|---|---|
| H2E1__H2O32_c7ec590a32a859bc__0 | 保存128 events / 4 signed pieces | cosine, sine |
| H2E1__H2O32_8a2ecdd0294ffb74__0 | 保存64 events / 6 signed pieces | cosine, sine |

sample0は固定候補ごとの最初の保存taskを明示するための選択で、影響が確定したことを意味しない。別のaffected行が判明した場合はその行へ置き換え、前提と予算を再固定する。

上記2軌道/4軸のprobeを採用する場合の例：bootstrap1/setup1/prepare1(prefix1)、archived_request_load2、event_metadata_deserialize192、occupation_trajectory2、native_validation2/compose2/lowering10、wrapper4/compile4/transpile4、simulate12。diagnose/path_signal/reference_rebind/allowance/accounting/ranking/request_factory/samplingは全0。archive replayがRNGを呼ばずSOURCE通りrequestを構成できることを別実装・人工テストで示す必要がある。

これらの新load/deserialization単位は未実装なので、この文書をlive budgetとして使わない。元eventが回収できなければこのprobeもBLOCKEDであり、保存seedからの再samplingへ自動切替しない。代表2軌道の成功を67 task全体のC再利用証明や母平均保証へ格上げしない。

## 次段階

担当ユーザー：O00002またはsender token mapの正確な保存先情報を提供。担当Codex：別作業として原metadata回収・field判定を更新し、affected setを確定した場合だけ最小科学契約を作る。比較/追加witnessの科学実行は、具体的なSOURCE・exact対象/予算・新grant/rootを提示した後の別途明示承認を要する。
