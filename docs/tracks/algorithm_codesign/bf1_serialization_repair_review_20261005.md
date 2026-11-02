# BF-1 serialization repair: technical review handoff

2026-10-05 JST。`SYNTHETIC_SERIALIZATION_CHECKS_PASSED_SCIENCE_STOPPED`。
この文書は実装修正の記録であり、BF-1の再実行・再採点・研究GOを認可しない。

## 作業分担とidentity

利用者の指示に従い、GPT側は研究全体の方針、RQ、新規性、論文着地点、追加検証の必要性・範囲を判断する。
Codex側は承認済み仕様に沿った検証、保存値の再集計、実装・テスト、provenance監査、レビュー資料の整理を担う。
[AGENTS.md](../../../AGENTS.md)にこの分担と、GPTへ渡す前に必要資料をcommit・pushする継続ルールを記録する。
今回のCodex作業はJSON保存の最小修正と限定synthetic検査だけである。
中断したBF-1の解釈、再実行例外の可否、BのRQ・新規性・着地点はGPT側のreviewへ戻す。

| 項目 | Identity |
|---|---|
| 元のscience source S | `e59344a564e70d64dc3ea39d640581c72676df31` |
| 元のauthorization-only A | `cc971e4a2bff9b0c5708003fde7b9519eed27241` |
| 中断結果の公開commit | `09c9e89555032213b52a6a60b38f55563d07a34d` |
| 修正branch | `track-b-bf1-serialization-repair` |
| 修正worktree | `/home/abe/Project/prt-worktrees/track-b-bf1-serialization-repair` |
| 修正のbase | 上記result commit |
| 証拠scope | local synthetic regression checks; CI・外部再現ではない |

このpacketは同branchへcommit・pushし、引き継ぎ時に完全SHAと固定commit URLを提示する。
以下のcode/test、repair reports/auditは同じcommitに含め、元の結果・source・authorizationは公開済み履歴を参照する。

[元の結果照合](bf1_one_shot_result_validation_20261005.md)と原artifactは保持する。
このbranchの修正はSのsealed inventoryと一致しない。旧authorizationを修正sourceへ継承しない。
Track A worktree、共有`src/trotterlib`、Aのstatus/manifest/APIは変更しない。

## 再現された不具合と最小修正

`parameter_score()`のfeasible objective値は`numpy.float64`となる。
その値の比較・加算でcross-scoreの`union_rank`が`numpy.int64`となり、
runnerの標準JSON encoderへ渡すと`TypeError: Object of type int64 is not JSON serializable`になる。
実際のscore関数とencoderを用いたsynthetic検査で、順位順・同点・不適格点を含む3 caseすべてに再現した。

修正はB専用[cross_objectives.py](../../../src/trottertracks/algorithm_codesign/cross_objectives.py)の
順位式を`int(1 + sum(...))`とするだけ。順位値・同点規則・目的関数・探索・primary判定・数値guardは変更しない。
共通JSON encoderやrunner、保存checkpoint policy、authorization、実行IDは変更しない。

元runはtraceback/phaseを保存していないため、これは保存された例外と一致する失敗経路の再現である。
元science runの正確なthrow locationを確定したとは主張しない。

## 限定検証

[追加regression tests](../../../tests/tracks/algorithm_codesign/test_bf1_cross_objectives.py)は、
synthetic costsから得た実際のscore型で、cross-scoreのcallback行と全resultをJSON round-tripする。
Python float版との同一性、順位・同点・不適格点、cache件数、探索点の不変性も確認する。
分子入力、science evaluator、科学的候補探索は使用しない。

- 修正前：追加3 caseが同じ`int64`例外で失敗、既存6 caseはdeselected。
- 修正後：`tests/tracks/algorithm_codesign`の49 testsが通過（既存46＋追加3）。
- 既存検査はsynthetic small matricesとauthorization用temporary fixtureを含む。
- repository full suite、science runner、NPZ操作、trajectory、circuit/compile、GPU query/useは行わない。

正確なcommand、環境、stdout、source/test SHAは
[修正前report](../../../artifacts/track_b_bf1_serialization_repair/2026-10-05/before_fix_report.json)、
[修正後report](../../../artifacts/track_b_bf1_serialization_repair/2026-10-05/after_fix_report.json)、
[repair audit](../../../artifacts/track_b_bf1_serialization_repair/2026-10-05/repair_audit.json)に記録する。
旧preparationを再生成せず、これらは新しいrepair namespaceへ保存する。

## 保存範囲とGPT側へ戻す判断

元resultは`INCONCLUSIVE`、`BF1_INCOMPLETE_MANDATORY_STOP_NO_RETRY`のまま。
保存済み1,269 cellは各係数のidentityを持つが、arm別search record、係数定義、
primary decision、cross-objective、bridgeはresultへ保存されていない。
これは直接保存された情報の棚卸しであり、復元が一般に不可能だと証明したものではない。
adaptive探索点の再構成、既存cellの再採点、BF-A/B/Cの再分類は今回行わない。

GPT側の判断対象は、この不完全な証拠を受けた研究方針、保存記録の救済を別途検討する価値、
no-retry契約への例外を検討するかどうかである。Codexから再実行を提案・認可しない。
一回実行済みmarkerはconsumed、retry falseのまま保持する。BF-2も未認可。
研究上の判断が必要な段階では、利用者/GPT側の具体的な判断と認可されたscopeを受けて作業を再開する。
