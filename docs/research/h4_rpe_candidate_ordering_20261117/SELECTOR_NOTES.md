# 共通候補列の選択器・時刻推定の実装メモ

2026-11-17 JST。`saved_log_analysis` sub-agentが作成した技術実装メモ。最終REPORTではない。候補生成・実行・予算・科学的な採否はrootの既存接続が担当し、このmoduleは純粋な候補順だけを返す。

## API

`src/trottertracks/rpe_candidate_ordering.py` の `Selector(policy='ROUND05'|'IMPROVEMENT_PER_TIME_V1', explore_every=4).choose(actions, method_best=..., tau=..., now=..., observations=...)` は、入力action自身を`chosen`、全actionの使用特徴・理由を`trace`として返す。archive、OUTCOME、mapper、価格登録表へのアクセスは持たない。

- ROUND05は`a_rank`の昇順、同rankは入力順を厳密に保つ。
- Bの改善量は既知point価格についてだけ `max(method_best - cost, 0)`。`price_floor`はboundとして表示するが、適合点の予測価格には使わない。
- 完了したpublic PASS/FAILの比率へBeta(1,1)の縮約を加える。同configurationの観測があれば優先し、なければ同method、観測0なら1/2の中立値と明示する。これは経験的な順序特徴であり、精度保証ではない。
- 正のscoreは「即時価格削減×経験的PASS率÷推定総秒数」。既知parent MSEのheadroomはtie-breakの特徴に限る。shotに対するMSE単調性を仮定せず、scoreから不適合や除外を作らない。
- 時刻根拠や正のscoreがない場合は、既知pointの即時削減・headroom・a_rankへ戻る。未測定を0秒・1秒にしない。
- 4選択ごとの周期探索を既定とし、unknownのNEW_SCHEDULEを優先する。新しい未評価SHOTのIDが増えても未知scheduleを押し出し続けない。候補IDごとの選択回数とa_rankで決定論的に選ぶ。
- `snapshot()` / `Selector.restore(snapshot)` はpolicy・周期・選択回数・ID別回数だけを保存/復元する。幅はopaque値のままtraceへ残し、binary64 `.2` と厳密`"1/5"`を正規化しない。

## 選択前に使う時刻

`estimate_seconds()` が使う観測は `finished_at <= now` の完了分だけ。同じ選択候補ID自身のheld-out recordは除外する。actionの`q3_seconds`は明示した仮定priorとして表示し、観測値とは呼ばない。

観測の`kind='SHOT_Q3', seconds=...`はQ3 legの所要時間と定義する。NEW_SCHEDULEの`seconds`は準備込みの可能性があるため、そのQ3時間は明示`q3_seconds`がある場合だけ使う。両kindの総時間を混ぜて準備時間を二重加算しない。時刻群は同method・同profileを要求し、同configurationの完了群を優先してmedianを使う。profileを変えた速度を過去実測から認定しない。

準備時間はsignal/priceが再利用済みかつ較正不要なら、存在しないstageとして0秒にできる。それ以外は明示`preparation.seconds`を使い、未測定ならnull。較正が必要なのに時間不明なら、その理由もtraceへ残す。総時刻は準備/Q3のいずれか不明ならnullである。

hidden replayの実サービス時間、全保存jobから計算した未来込み平均、未完了jobの最終wall/CPU/MSEは選択器へ渡さない。callerは完了イベントの時点で観測を追加する。保存答えがない候補は未評価であり、時刻未測定も別に保持する。

## 幅探索との境界

既知の並行branch `codex/a-pr-step-width-search-20261117` は読取時点でHEADがB2 `d8e806265`、新しいWidthDriver/StepPriceRegistry等は未commitで、該当remote refもなかった。このmoduleはそこで作業中のmapper・監査・registryを重複実装しない。

draft HANDOFF `docs/research/pr_requested_step_search_20261117/HANDOFF.md` の `requested_widths`、`width_family_key`、requested/evaluation IDの区別、signal/price別の再利用と較正要求をcallerから受ける設計に合わせた。未公開draftの存在を固定公開SOURCEや独立検証済みの証拠とは扱わない。

## 人工検査

`PYTHONDONTWRITEBYTECODE=1 python3 -B tests/test_rpe_candidate_ordering.py`：11/11 PASS（作成者自身、約0.002秒）。cheap schedule優先と高価な較正時の逆転、未来/自身のheld-out情報排除、完了後だけの時刻可視化、unknown scheduleの周期探索、profile分離、pointと価格boundの区別、二種の幅の保持、snapshot再開を確認した。実H4の信号・較正・compile・Q3や科学的性能比較は行っていない。
