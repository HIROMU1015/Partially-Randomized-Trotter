# N3旧条件族診断 v1：固定S1二例・一回結果

2026-10-12 JST。固定planの一回診断を完了。**両入力でCERTIFIED_NORM_AND_OLD_FAMILY_SEPARATION、独立有理数監査PASS、Mandatory STOP。**
指定した保存OCC幅で、NORM不成立と旧族全κ/μの不可行性を厳密なdualにより認証した。FIVEはこの旧族に含まれる。
新規性・中心テーマ・研究継続の科学判断は行わず、新結果の資料としてGPTへ渡す。

## 固定条件・provenance

source13 `1aa12ab0b777b28f1428d9bb8417af93c4376aae`、fingerprint `46baa537269db923dd8b8c95e4788555471c60bad1572d3a01ec1e70245c6488`。
execution authorization commit `f40b1bc3fe1e3cffcfdc67b9d8e3c2dcbc9450ee`、plan SHA `ef3c66d2e01e47027d9d382afc324aab9b7263aab3d550c3576c31e0f5bd6ff5`、input fingerprint `81e90cb760d6e6e9774aad808ed40953bae0b461c18532bff78d7ead6e4e584d`。
[認可](n3_old_family_v1_diagnostic_authorization_20261012.md)、[数学・比較契約](n3_old_family_v1_mathematical_contract_20261012.md)、[固定plan](n3_old_family_v1_execution_plan_20261012.json)。

S1_edge_pair/S1_distributed_threeの同じstar型係数、実対称one-body＋非負unordered density、一外部mode8、full10→active9、ν4、scale1.25。
化学geometry/basis/DF rank/policy/L_D/PF・time-evolution窓は適用なし。元δgoal=.005は不変。
今回のδは各保存OCCの正の有理high=positive_proof_deltaそのもの。同じ保存U=−1951/1064、c_star=−233352716459809/183068686024704、v、β²=1/576を使用した。
新入力や独立移送ではなく、既存S1のposthoc development診断。元primary NO_ADDITIONAL_ACCEPTANCE_OBSERVED/追加保証0/2-base未達は保持する。
保証は基底エネルギー射影差0≤a−E0≤δ。真の誤差・追加mode削減・量子総費用を評価していない。

## 結果と数学的証拠

表は表示小数。全判定は保存有理数のstrict inequalityで行う。
D=c−U+δ、Mδ(κ)=κ(h_A−U I/ν)−vvᵀ+(δD/ν)I、κ∈[0,D]。

| input | 保存OCC δ（表示） | δD−β² | tr(WMδ(0)) | tr(WMδ(D)) | 判定 |
|---|---:|---:|---:|---:|---|
| S1_edge_pair | 0.0024579588333 | -0.000356135481461 | -0.000280006092587 | -0.0670430164266 | 分離 |
| S1_distributed_three | 0.00307821749914 | -5.99340822845e-06 | -3.12782815474e-05 | -0.00511496743798 | 分離 |

Wは両入力でexact PSD、trW=1。両端traceが負で、traceのaffine性から全κ区間で負となるためMδ(κ)はPSDになれない。
任意の許容μ≥μ0をμ0へ下げるとPSDが保たれるので、この診断はμ0だけでなく全許容μを覆う。
同じδでNORMのδD−β²も厳密負。FIVEの有効候補は同じold_candidate条件を満たすため包含される。
旧族の有限保存候補数・optimizer終了・探索失敗を不可行性の根拠に使用していない。

edge_pairの証拠はW=e0e0ᵀ（active順序のcoordinate0、full mode0）。
distributed_threeはW=t e1e1ᵀ+(1−t)vvᵀ/β²（coordinate1、full mode1）、0<t<1の保存有理t。
保存Wとのこの表現の完全一致も確認した。追加のW探索やδ最適化ではなく、取得証拠の表現確認である。
[小さい証拠summary索引](../../artifacts/n3_old_family_v1/2026-10-12/execution_audit/case_summaries_index.json)、[一次result](../../artifacts/n3_old_family_v1/2026-10-12/diagnostic1/result.json)。

## 独立監査と費用

runner内の独立auditと、さらに保存結果CLI auditの両方で全2件PASS。
WのPSD、元係数からの両端再構成・負trace、NORM、input/source/caps/event数/STOPを独立確認した。
新dual checker conditions合計236、PSD2件。保存OCC/旧U,cの原checker replayは2,873,507 conditionsで別にPASS。
新境界やOCC証拠を生成せず、元ledger・勝者の有効性を再確認した。same local environmentで、immutable CI/外部科学再現とは区別する。
[独立saved監査](../../artifacts/n3_old_family_v1/2026-10-12/execution_audit/independent_saved_verifier.json)、[run audit](../../artifacts/n3_old_family_v1/2026-10-12/diagnostic1/run_audit.json)。

numeric eigh0、primal PSD probes0、analytic rays5、混合scalar probes2で両証拠を取得。
固定grid/数値golden経路へ進む必要がなかった。新dual2件、Wの独立PSD検査2件、原proof replayの費用を別に記録した。
worker case wallはedge .005124s、distributed .007863s（並列なので合計は実elapsedではない）。
全一回wall 3.853623s、parent CPU 3.965134s。保存proof監査を含み、cold standaloneや生成だけの費用ではない。
CPU32/affinity32、Python3.12.3/NumPy1.26.4、2 spawn workers、各numeric thread1、GPU0。開始前available55,426,924,544 bytes。
analytic優先と両端trace cacheを適用。比較benchmarkをしていないためspeedup倍率を主張しない。
最大producer rational729bits、演算12912、cap4096bits/2,000,000以内。Fraction内部temporary整数全体の監視ではない。
parent peak RSS79,949,824 bytes、max recorded worker peak RSS79,949,824 bytes。
sampled aggregate peak112,648,192 bytes、4GiB違反なし。sample21、目標50ms、最大gap0.315971s、間の瞬間peakは未測定。
output11,582 bytes、最大shard6,118 bytes、log53 bytes。wall/CPU capなし、各output/log/call/bit/memory制限を保持。
23 synthetic PASSはsource identity付きarchiveを再利用し、source変更・再実行なし。global manifestの既存description欠落FAILを保持し、新entry/prefix保全を別に確認する。

## 主張の範囲とSTOP

認証したのは、同じU/c/指定δにおけるNORM＋FIVE＋全旧LMI κ/μへの分離。この2方向は同じS1 baseであり、2-base移送成功ではない。
旧族の最小幅を求めていないので、既存20.424%/0.343%を最適旧族との差率へ変更しない。
全既知手法への優位、新入力汎化、一般二体、多外部、実分子、新規性、量子資源削減を確定しない。
新input/U,c/OCC/E_forced/真ground-gap/compile/shots/元batch再実行/N1-N2再開は0。
attempt1/retry0。元source/input/結果/STOPとroot dirty・BentoSlide・別Trackを保全して公開・GitHub再取得確認へ進む。
[STOP](../../artifacts/n3_old_family_v1/2026-10-12/diagnostic1/MANDATORY_STOP.json)、[command記録](../../artifacts/n3_old_family_v1/2026-10-12/execution_audit/commands_record.json)、[aggregate summary](../../artifacts/n3_old_family_v1/2026-10-12/execution_audit/aggregate_summary.json)。
**次はGPTによる新しい分離結果の科学判断。** 科学レビュー開始と追加実験の認可は別。中心テーマnull、次段false、Mandatory STOP。
