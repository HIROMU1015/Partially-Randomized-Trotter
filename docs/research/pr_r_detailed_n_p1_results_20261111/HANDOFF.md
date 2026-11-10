# B/C HANDOFF：R可変PR P1の12 N record

2026-11-11 JST。科学SOURCE `a1a79213a9b5331145e20e8778a3be9d9fc4e497`、固定計画 `d33380a8c2fe68d0be1f2c5faaf7fc3b3a956b58` のP1のみ。全12件がCONDITIONAL_ROUND_PASS、actual N/全RPE/C/Gは未資格。[REPORT](REPORT.md)・[N_HANDOFF](run01/N_HANDOFF.json)・[診断CSV](run01/SIGNAL_DIAGNOSTICS.csv)・[CHECKS](CHECKS.json)。

## 受入方法とscope

JSONの `shots` は既存identity/plan/actual/diagnostics/source/originと、B `law_binding`、`N_comparison_policy=R_AWARE_EMPIRICAL_ACTION_COUNT_V1` を持つ。`menus.H4/H6` は今回P1の比較menu、他round/PFも元taskのmetadataとして残るが、新しいNを提供したのは下記12 IDだけである。

| system/m | R順 | canonical candidate ID（同順） | N_X=N_Y（同順） |
|---|---|---|---|
| H4/0 | 2,4,8 | 53e00d8b07d1547d19a7 / 03a51c8d13a91a403742 / 656998c3354fb8b614e9 | 764,734,719 |
| H4/6 | 4,128,256 | 65c51c5cb3e39a2266e3 / 2a8c6718220eec59d419 / ae0187ce74cd82207748 | 9638,766,735 |
| H6/4 | 11,16,32 | 5e2581e9c4f25bdf8529 / 2d2af6ca830dff14a695 / 4269eafc8c0ef8922c23 | 729,721,712 |
| H6/8 | 16,128,256 | 53797d9e5a2eeec31a13 / 55a8eb0ae50ec3f7c713 / 00ab807ee1f600b30440 | 1015,735,720 |

Bのrequest_contextと完全identity/lawが固定計画と一致することを確認済み。A0 normalizationの元law宣言は旧結果のまま保全し、今回N recordには実行済みfinite mean SOURCEに対応するBのN-law bindingを付けた。actual sampler距離の資格を追加したものではない。

H4 m0/R4・m6/R4、H6 m8/R16は保存signalを同policyでscalar再評価した別record。H6 m4/R16は不足していた独立reference/normを新計算した。旧baselineのNが数値的に同じでも、その旧recordを再ラベルしない。今回のrecordとSOURCEを比較laneへ使い、既存旧SOURCEの在庫は別に保つ。H4巨大N正本N_HANDOFFとm12、H6他roundの保存Nは不変。

NはPython intと十進文字列で保存。軸epsilon/alphaに対する既存Decimal140/180 ceil検査を通した十分shot数で、minimum Nではない。actual Nはnull、全RPE成功はfalse。未知implementation/law/ground同定誤差を0補完しない。

## 科学/回路binding

Linear1Å/STO-3G、保存tol-only1e−8 DF10/DF19、元weighted completion・H/state/snapshot/sectorとfragment/scalar順。H4 S2/δ.1/p1/K2、H6 m5_best/δ1.6/p7/K2。qはqPF、qRPE=2^mは別field。H6 allocationは元11piece・負時間と整数tieの順序を維持し、event数qPF R_outer。scalar/identityは総Tへ一回会計する。

h=.1/13round、εE=.00159360010199040、各方式αtotal=.05・軸.05/26、β=.4/.2、保存prior/decoder、無shift・P0・元compiler/control/full measurement wrapperをidentityへ保持。PFやH6早期別domainを変更していない。corrected/rawは独立伝播、exact-tail対照を保持しており、normalizationをNに一回含める。CにB/shot係数を追加乗算しない。

## C/不確かさと未評価範囲

今回C recordは空、Gはnull。固定計画でCとexact候補一致したのはH6 m4/R16（保存finite cohort）とR32（Phase1予定）の2件だけ。Cのその後の実行状況はこのN成果に取り込んでいない。小q CモデルをH4 q64やH6 q16・新R128/256へ適格移送しない。

Cはfull candidate/axis/binding/compiler/law/cohort/SOURCEを一致させ、direct/model/外挿/欠測、有限cohortと全law期待値、共有校正・paired XY/unknown covariance・systematicを保持する。Bは不足Cを0補完せず、今回のN改善だけでRを選ばない。新G/順位/全RPE成功保証には別の科学scopeと資格が必要。

A0全24件のうち今回Nを評価した12件以外はそのまま残る。P2/P3の最終時間6候補と保留6候補は未評価。元86候補や上側未探索範囲を切り捨てない。今回の完了STOPはP2/P3への許可ではない。

次：**Codex B**が公開Nを取り込み、**Codex C**が自身の許可範囲で価格domainを検証、**ユーザー**が次の有限科学scopeを判断する。公開・remote SHA/tree/主要成果物照合後mandatory STOP。追加N/C/G・R最適化・詳細科学レビューへ自動進行しない。
