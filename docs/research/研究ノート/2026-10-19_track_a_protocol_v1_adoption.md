# 2026-10-19 Track A共通プロトコルv1の文書正式化

ユーザーがDRAFT 9c54d89f6154525e3c7c814a4da5ecdc37453b8aからの正式文書改訂を明示承認した。AD01/02/04/06/07/12は基本方針、AD05/08/09/10/11は既存区分・閾値・欠測・legacy例外を維持する条件付き採択。AD03の追加PF主比較は保留。旧DRAFTは削除・上書きしない。

H6 Phase 0の6d290fc8とA82回収bd41b376を静的に取得した。前者は準備完了・最終n等未seal、後者は原signal回収・説明数値未検証であり、独立Pや説明完了を増やさない。88行の科学的達成statusはそのまま保ち、政策採択と新証拠metadataを追記した。

新科学計算/数値再解析/公開push0。Codex A/Bや既存科学SOURCE・結果・grant・ledger・marker・STOPの変更0。共通基準を既存契約へ自動適用しない。[統合報告](../track_a_pr_resource_protocol_v1_adoption_20261019.md)。local commit後mandatory STOP。

次担当：ユーザーが公開の要否と各個別stageの追加承認を判断する。Codexは今回それらを起動しない。


## 暫定追補CT01：compile前後のraw状態差1e-7（2026-10-19）

ユーザーの「一旦 －７ にして」により、共通プロトコルのcompile前後raw状態ベクトル2-norm差を、finiteかつ1e-9未満から暫定1e-7未満へ変更した。全Track A系列・PF公式・PR/決定論PF・両軸に共通。既存PASSと事前検査を再利用し、閾値変更だけを理由とする再検査・再compileを要求しない。旧STOPは保持し、新基準での照合は別記録とする。

[プロトコル追補](../track_a_pr_resource_evaluation_protocol_v1.md)と採択判断表AD08を最小更新。他のnative/closure/DF/model閾値、shot allowance、compiler、係数、AD03は維持。88行の充足表・旧adoption_record・証拠bundleは変更しない。新計算・保存値再判定・費用会計・実行コード変更0、local commit後mandatory STOP・公開push0。

次の担当：ユーザーが必要に応じて公開・m5後続作業を判断する。今回の文書追補のための追加計算はない。

検証：既存文書構造checker PASS、人工metadata 18件PASS。88行充足表・旧採択原本・科学SOURCEの変更0を確認。
