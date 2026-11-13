# D748 expected_scalarとC価格bindingの接続修正

2026-11-13。Cの価格request生成・再構成へDのoptional `expected_scalar`を転送し、schedule anchorとresolved path検査へ追加した。**人工・回帰25件PASS、新しい実H4科学計算0。** T2の実preparation scalar不一致の原因や正しい値は判定していない。

[HANDOFF](HANDOFF.md)／[テスト記録](TESTS.txt)。既存C branchを継続し、他担当のworktreeは変更していない。ベースライン有限科学計画 `c5466946a` は未承認・未実行のまま保全する。

## 再利用したSOURCEと差分

- C元価格接続 `5cc7c07e112ce97205948465c94441cba0042416`。今回の基点は同branch上の計画追加 `c5466946ab0ef9b50511210ca1e58fb7147bb0f0`。
- D request `748be6875b5007707d864e16aa7a4fc4250f3714` の元 `candidate_price_adapter.from_b/from_a_job/check_request/pending_price_records`を再利用する。Cの固定読込SOURCEを旧 `0c6e16342` から更新した。4つの読込moduleのうちdiffはcandidate adapterのoptional scalar・runtime scalar gateだけで、compiled_price_fast/efficient_controlled_zz/p1_efficient_zzは両SOURCEで同一。
- D長時間HANDOFF `39678f5a063490641621e836e53737cb39eb6979:docs/research/d_long_time_price_plan_20261113/HANDOFF.md` の接続差分表を確認した。
- D T2 `752b9747a363eb4f1e510092230ef8dc243187ca:docs/research/d_ld2_m12_t2_20261113/REPORT.md` の停止理由を確認した。旧run/raw/STOPは変更せず、再実行していない。

## 実装

1. `I.price_requests`と`P.pending_requests`へ`expected_scalar=None`を追加し、元D constructorへ転送する。scalarのみを指定してSOURCEを省略する呼出しは、一般pendingへ落とさず元SOURCE必須検査で拒否する。
2. `_requests`で元requestのscalarをconstructorへ引き継ぐ。request/context IDはD自身が生成・検査し、CでhashやB identity/lawを再定義しない。
3. schedule anchorへoptional scalarを加える。scalar値の違い、指定と未指定の混在を拒否する。0も指定として扱う。
4. `_paths`ではscalar指定時にD runtimeと同じ`float.hex()`完全一致を要求する。許容幅を追加せず、1 ULP差や正負zeroの差を受け入れない。
5. 未指定時は元D0c6 constructorとrequest内容/IDが一致する。元の一般pending、片軸/欠測/有限cohort/外部mean・B会計経路も回帰確認した。

`expected_scalar`は固定input/tail SOURCEに結び付いた既存metadataを呼出し側から渡す。今回の修正はscalar値やその科学的定義を選択する機能ではない。Cは実H/context/preparationを復元せず、渡されたSOURCEの物理的正当性を新しく認定しない。Dの実scalar gateは元SOURCEのまま保つ。

## 確認結果と限界

新5件＋既存C価格接続8件＋統合10件＋pinned B metadata2件＝25 PASS。新テストは人工scalar/metadata/path/外部価格のみ。既存統合testのsignal/N/G算術も人工fixtureであり、実H4結果ではない。既存1件はA公開configurationのmetadata読取だけを行う。詳細な独立科学レビューは実施していない。

本修正ではpreparation、sampling、実回路build/compile、保存科学値再解析、実E[C]/G、T2 retryを行っていない。旧T2の実scalar値・差・原因はUNKNOWNのまま。API整合が通ることは実preparation gateのPASS、平均価格資格、科学実行許可を意味しない。

成果をcommit/pushし、remote SHA/tree・主要成果物照合後 **mandatory STOP**。後続Dのpreparation-only診断は別scope。本C技術修正からT2又はベースライン科学実行を自動開始しない。
