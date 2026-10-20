# H8 GPTレビュー用・共有候補パケット

2026-10-20 JST。ローカル作成済み・未送信。実際のGPT添付、外部upload、GitHub pushは承認されていない。科学レビューと最終evidence freezeも未開始。このpacketはH8部分の証拠確認を準備するもので、研究全体の科学的妥当性を判定したものではない。

## 条件と保存結果

linear H8、隣接1 Å、STO-3G、tol-only1e-8 DF28、16 spin modes、sector4900、T=.8、P=0、二次partial PF、r4/K4。通常signed bias・allowance倍率1、ordinary ancilla、full-Fock scalar一回、reference不回転。結果は保存値の転記であり再計算していない。

|scope / PR候補|L_D / q / δ|n / 軸数|保存mean exact / paired SE|固定PF / 保存G|
|---|---|---|---|---|
|H8_CONFIRM_P8_Q2 / B2_p8_q2_2nd_r4_K4|8 / 2 / .4|48 / 96|9840910801/8 / 3175066.353534034|B0I_p11_q1_2nd / 1267321444|
|H8_CONFIRM_P7_Q16 / B2_p7_q16_2nd_r4_K4|7 / 16 / .05|48 / 96|546493203592571/6 / 144211830223.71762|B0I_p15_q4_4th / 94193016652860|

両driverの保存native receiptはexit0/reaped。継続runはp8で47親・94子、p7で48親・101子。p8は同じ新確認cohortで完了した元SOURCEの1軌道2軸を再利用し、探索n8・旧確認n32・条件付きpilotは混合していない。旧STOP/未完了予約も保持する。296 native nodeは旧driver・旧未完了workerを含むため、すべて成功したとの意味ではない。

結果commit：`e6eea5387a6e84d1e39a9adb3b61975acc74e200`。
実行SOURCE：`745a674546f2cb6258526fbfde1d9a00601689f8`。
seed導出とp8再利用1軌道の測定SOURCE：`1549b6f891a15ff48379996c34b344118ba2606c`。
前回回収local commit：`253356942845dddf05c2bb6f4582b026e0bce942`。
公開索引commit：`9f33ad0b11b4febe48b802000c38c5a0e3fd267c`。

## 読む順序と検証できる範囲

1. `01_results_and_source.json`と`public/`：固定主副評価と192軸projection。p8副評価の固定PF不適格/nullを維持する。
2. `02_execution_receipts.json`、`03_worker_receipts.json`：実行時field、SOURCE/technical digest、native exit/reap、匿名化した親子関係。`04_grant_ledger_marker_stop.json`：新旧grantの選択field、承認receiptのscope、非返却台帳、marker、旧STOP、terminal。
3. `05_seed_cohort_audit.json`、`06_task_axis_links.json`：固定96登録・cohort管理の証拠、軸ごとのSOURCE・event digest・fingerprint・整数metricとnative記録。registryの`actual_RTE_event_sampling=0`はregistry作成段階の宣言であり、後の科学実行全体のsampling数ではない。
4. `07_input_runtime_bindings.json`：input65/58件、provider各931件、runtime34,273件のhash結合・前回の原本検証条件。今回は新たなruntime初期化や科学array復元をしていない。
5. `08_provenance_and_redactions.json`：E番号、原本SHA/bytes、原bytesを収録した監査JSONと匿名化projectionの区別。`10_source_conditions.md`と`11_public_sources.json`：固定公開コードの検査条件・URL・SHA。
6. `09_missing_evidence.csv`と`09_limits.md`：原profile・保存PR回路の欠測、匿名化で失われる独立再検証能力、結論への影響。

terminal・STOP・one-shot marker・reservationは安全な許可fieldだけの原JSON bytesを収録しており、その原SHAを再計算できる。他の原SHAは、非共有原本への来歴linkである。匿名化したnative記録・grant等から、原fileそのもののSHAや実PID/starttime/PPID/UID、元承認チャットの真正性を再認証することはできない。

seed値、除外値、raw event列、認証情報、内部絶対path、private科学raw、非共有原本索引は含めない。公開コードと組み合わせるとseedを復元できる導出用digestも除外した。ここに残るseed-plan/registry原fileのSHAは原bytesへの来歴linkであり、その導出用digestとは異なる。seed非衝突・除外の固定導出は原本で前回確認済みだが、受領者はこのpacketだけから再実行できない。seedの非衝突は数学的な確率独立性の証明ではない。

## オフライン構造検査

展開したdirectoryで `python3 -B verify_packet.py --self-test` を実行できる。標準libraryのみを使い、packet SHA、安全な原監査JSONのSHA、192軸projection、親子関係、scope/grant/marker/台帳の対応、欠測表示を検査する。科学計算・統計解析・sampling・compile・費用再計算は呼ばない。これは記録の構造照合であり、科学的妥当性や原本の真正性を証明する検査ではない。

## 解釈上の境界

原compiler profile file bytesと保存PR回路は未取得のままである。保存fingerprintは入力wrapper側の記録であり、元回路から再導出していない。旧compiler出力のbit同一性・final角度列・非Clifford/T数を認定しない。p8の`saved_exact_export_match=false`、p7のtrueを保持する。

標本点費用の再現という保存結論から、未観測rare eventを含む母平均優位性、厳密bias、最終energy/RPE総費用の保証へ拡張しない。CI/p値は保存null。選択済みPF監査をPR実行監査に代用しない。新規科学計算・統計解析・費用再計算・再ランキング0。

**MANDATORY STOP：共有候補作成まで。** 次担当はユーザー。内容・残る開示リスクを確認し、具体的な添付とレビュー開始を別途判断する。
