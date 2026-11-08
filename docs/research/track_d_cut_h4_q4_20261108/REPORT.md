# Track D O5：保存H4 q4一path移送検証の入力preflight

2026-11-08。**入力原本未取得のため、cut選択前にSTOP。実H4の追加depth削減はUNKNOWNで、改善なしとも失敗とも判定していない。** 科学campaignは起動せず、新sampling・cut選択・bundle構築・raw correctness・P0 compile・compiled postcheckは全て0。別pathへの変更、seedからの再生成、中央cutの仮置きは行わなかった。

利用者の実行指示は受領済みであり、承認不足による停止ではない。新たなgrantや承認証明書は不要。停止理由は、指定された保存event内容を取得できず、価格参照前のdecisionを固定できないこと。

## 固定対象と確認結果

対象はLinear H4／隣接1.00 Å／STO-3G／Nα=Nβ=2／元H／number-CDF8／元frame順L_D2／ordered partial S2/s1/r32/K0／h=δ=.1、q4・4 complete step各32 event。元H厳密groundを仮定、準備費除外、全256入力を確認する契約。対象は指定どおり **C6a q4 calibration cohort先頭の `q4_sample00`** 一つであり、独立holdoutではない。

| 項目 | 状態 |
|---|---|
| 人工基点 | `a54d04059bfc61adef43aa2d40fb9c001b62378f` |
| C6a科学SOURCE | `6b6667026ec8b670f610e41141d195883a9eb513` |
| 元science SOURCE | `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7` |
| C7-4特徴 | 固定`64072ba81bb70db1df1b34e94adb08e78926f24b`、変更なし |
| Selector | 既存`track_d_cut_selection.py`、`l_i=1+basis_runs_i+support_work_i`、cut1/2/3、同点時中央優先。変更なし |
| 指定pathのmanifest binding | ID/q/sample/保存seed域・4 seed・event hashを確認 |
| Event内容 | 未取得。公開hashだけで内容確認済みとはしない |
| 対応raw/compiled QPY | 未取得。再利用資格はUNKNOWN |
| SOURCE互換 | C6aの登録6ファイルと関連builder/gadget/phase/compiler4ファイルが当該C6a commitとbyte一致 |
| 個別価格 | decision前のため未参照。比較値を転記していない |
| 追加実装・テスト | 0。既存cut入口の不備は特定されず、入力不足のため重複実装・旧人工test再実行は不要 |

SOURCE互換はデータ再利用の必要条件の一部であり、実event・phase・canonical IR・QPY・PASSとの完全な再利用資格を代替しない。旧結果は無効化していない。入力原本の未取得と、科学的なidentity不一致・correctness FAILを区別する。

## 未取得原本

既存公開索引の期待値：

- event：183,097 bytes、SHA256 `919f2e784920581380cb4090c59fff34741d914e9d340d249bca554eba8cadb1`。
- `paths/q4_sample00/circuit_payload.tar.gz`：6,950,831 bytes、SHA256 `a631cd394073f7770d2723eb0cfa93fc6a8f703534b9cdce88b7c126381a007b`。20 memberのhash索引にはevent、A/B/actual inverse、Strong/Reference raw/compiled QPY、参照/observableが登録されている。

[既存公開方針](../track_d_c6a_paired_price_20261106/RESULT.md)は、archive/event/QPYがローカル保全のみでGitHubに存在しないことを明記している。今回も公開`payload_publication.json`とmember索引を取得・照合した。索引から元payloadを回収できたとはしていない。

SERVER1のtask userが読める既存worktree・home内保存/backup領域・task temporary storageについて、対象event filenameとC6a archiveの所在を限定確認したが、読み取り可能な原本を取得できなかった。これは全サーバー・他ユーザー領域・未接続mountからの消失を意味しない。別の保存先や元実行サーバーの所在地はUNKNOWN。非公開原本の保存サーバー/絶対pathをユーザーへ照会した。

## 資源結果と主評価

| 比較 | RZ count | CX count | RZ depth | CX depth | Total depth | Size | Width |
|---|---|---|---|---|---|---|---|
| 選択cut−中央cut | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| 選択cut−Strong | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |

選択cut、step負荷、score、ΔD、A†/B不均衡、critical path、cutで失う融合位置の実path診断は未計算。欠測価格を0にしない。人工934→646という既知結果を実H4の結果として移送しない。

## 保全・次の担当

[INPUT_AND_REUSE.md](INPUT_AND_REUSE.md) に必要な原本と再利用判定順序を整理した。[SOURCE](../../../artifacts/track_d_cut_h4_q4/2026-11-08/preflight01/SOURCE.json)、[入力状態](../../../artifacts/track_d_cut_h4_q4/2026-11-08/preflight01/INPUT_STATUS.json)、[call数](../../../artifacts/track_d_cut_h4_q4/2026-11-08/preflight01/CALL_COUNTS.json)、[STOP](../../../artifacts/track_d_cut_h4_q4/2026-11-08/preflight01/STOP.json) を独立namespaceに保全する。内部環境情報はprivate localに保持し、実raw/QPY/非公開入力は公開していない。

専用branch `codex/track-d-cut-h4-q4-20261108`。元main・dirty/untracked・旧SOURCE/raw/結果/grant/ledger/marker/STOP・C1凍結を維持。q64のprocess・割当・SOURCE・入力・未完了private出力に触れていない。共有OS/venv/cacheの変更なし。

次の担当はユーザー/GPT。まず既存`q4_sample00`原本を保存しているサーバー・絶対pathの提示/引渡しが必要。原本回収後も本STOPから自動実行せず、同じ一path・不変rule・最大2 bundle/6 P0/6 postcheckという範囲の再開指示を受けて進める。現段階で候補Bの実H4効果を棄却したり、候補Cへ移る根拠は得ていない。実H4再sampling、別path、q≥8、fit/N/全13round、O2未採択契約、候補C、詳細科学レビューへ進まない。公開SHA/tree/bytes照合後mandatory STOP。
