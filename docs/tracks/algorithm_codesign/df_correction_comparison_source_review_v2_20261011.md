# DF補正A/B比較 v2：source / contract 整合確認への引き継ぎ

2026-10-11 JST。**SOURCE_PREPARATION_COMPLETE / EXECUTION_NOT_AUTHORIZED / mandatory STOP**。
基点 `73b2102a305f018d22348418adc7939cae63f0b2`、独立branch
`track-b-df-correction-comparison-v2-source-preparation-20261011`。
source Sの完全SHAはpublication receiptに固定する（自己参照commitを作らない）。
publication-only childのHEADをscience sourceと混同せず、将来のauthorization AはSの直接子とする。

## 受領と今回の範囲

[GPT review原本](../../research/track_b_df_correction_comparison_preregistration_review_20261011.md)を
34,280 bytes、SHA256 `8c3bd4d712c37f60038bad1f6256385cabcdb97cc43d2ee993c35dee36a1984f`
としてそのまま保存。入力commit `7dbbc0b098231446aebd4c97302dd62a5ab87a70` をpushし、
GitHub再取得byte/hash一致後に実装へ進んだ。
旧ZIP/レビュー/監査・source・結果・marker・STOPは変更していない。

[契約v2](df_correction_comparison_preregistration_v2_20261011.md)、
[実行source計画](df_correction_comparison_execution_plan_v2_20261011.md)、
[machine contract](../../../artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/contract_v2.json)
へM1–M7を同期し、専用namespace/runner/guard/off-domain testsを準備した。
shared APIは変更せず、原本やv1を履歴として保持。

## source確認で重く見る点

1. **Executed law**：exact outer coin、48-bit tuple/conditional Pauli law、親確率付きcomplete-law L1。
   cost、実装weight、二次moment、range、weightとgate誤差の相関を同じlawで会計。
2. **理想input / proxy**：元float recipeを保ち、理想frameをexact ordered Gram-Schmidtで定義。
   区間input距離とconditional IEEE/BLASのBall誤差を伝播。
   exact rational computational Pauli proxyと理想Tの距離を課金する結果前amendmentを明記。
   computational zeroを理想zeroと呼ばない。threshold drop/rank dropなし。
3. **Signal / resource**：CDは条件付き精度・十分shot予算・記号費用。Eは別fixtureのlocal中心。
   E単価×CD頻度、実測PR総費用の勝敗、oracle shot削減を行わない。
4. **Strict phase / T**：raw-rad/pi/atanを明示分岐。backendもdps100を明示し自動dps変更を抑止。
   Wと4pi/RZ相対位相を保持。<=64 inventoryを合成前freeze、B完了→phase→A。
   partial Tはmultiplicity付き固定加法completion下界だけ。未知全T、coverage、局所分離は別field。
5. **Containment**：外側parentが50msでlive/reaped CPU・aggregate RSS・wall/outputを監視。
   cap検出時process groupsをkillしqueue取消、kernel per-child CPU制限、exclusive marker/job-start。
   RSSの瞬間的overshootゼロを保証しない。完了後・metadata境界もtotal cap確認。

## 検証証拠

[最終focused tests](../../../artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/focused_tests_final_v2.json)：
**62 passed / 0 failure**。最終suite wall約1.05s、CPU約0.85s、peak RSS約124.27 MiB。
有限bit費用/moment、B weighted mean、独立性/符号・controlled phase、frame/rank、Pauli proxy/zero、
raw-angleとClifford strict guard、T multiplicity/coverage、CPU/RSS/wall停止、
direct-child/hash/path/marker保護、元float helperとのbit identityを別入力で確認。

最終suiteのoff-domain compileは1件、準備期間全体では5件（小さい別入力のみ）。
これは登録E最大22件とは別である。初回interval conjugate bugの修正履歴、途中60件のreportも保持。
point fallbackなし。比較性能結果・immutable CI・外部再現として扱わない。
既存77/15/18/63検査へ加算して科学的evidenceを増額しない。

登録CD6task、登録E compile、Clifford+T synthesis、量子shot/trajectory、分子/NPZ、GPU、LP、
旧G10/RA-D0再開はいずれも**0**。登録入力frameの生成・開封もしていない。
現在のauthorization=false、runs0、source_commit=null、新execution markerなし。
metadata verifier / source manifestは[準備artifact](../../../artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/)に保存。

## 残る判断と限界

今後必要なのは、公開source Sとv2契約の整合確認、そして別の明示的one-shot実行認可。
本番入力でのnumeric margin、4 A fingerprints、22 compile内の実行可能性、literal64-key inventory、
T coverage/certificate/資源上限内の完走は未取得であり、focused PASSだけで保証しない。
registered margin/capが不成立ならtechnical STOP。source/precision/fixture/denominator変更による救済なし。

数値guardは記載した計算モデル依存、CD native費用/実装精度は未知。
全A T未取得は非優位の証拠ではない。局所固定実装の分離をPR総資源や一般whole法へ拡張しない。
source準備からalgorithm採択、G10再開、分子production、追加baseline/gridへ自動移行しない。
今回完了後はmandatory STOPし、必要なsource整合確認・次scopeの判断をGPT/利用者へ戻す。
