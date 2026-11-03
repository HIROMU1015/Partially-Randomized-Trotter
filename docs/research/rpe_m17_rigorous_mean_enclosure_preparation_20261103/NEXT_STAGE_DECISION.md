# m17包含評価の準備完了と次段階

2026-11-03 JST。**218成分/basis export、CDF/grid、ideal測定平均の包含adapterと固定実行runnerを準備した。**
実H4のSOURCE replay・CDF・operator/norm/mean・統計予算の新評価は未実行。

1. 成分export経路は既存の保存H snapshotとprepare_df_partial_s2を再利用し、人工preparationで動作確認した。
   実H4でhashと218成分を再現できるかは将来の許可された1回のSOURCE replayで確認する。
2. component/orderのbinary64→NumPy CDF→exact 53-bit massを接続した。
   IID uniform53の仮定と固定PCG64列を区別し、RNG replayや標本頻度は使わない。
3. python-flint0.9.0/FLINT3.6.0を専用環境へ導入し、Acb＋global radius方式を人工検証した。
   exact-parameter ideal gateのsector不変性を使い、過去の数値leakageを厳密0としない。
4. 第一科学実行SOURCEは本folderを含む公開commitで固定する（full SHAは最終チャット）。
   [計画](SCIENTIFIC_EXECUTION_PLAN.md)は同じm17条件のcomponent/basis replay1、CDF各1、最大3包含mean。
   96 bitから、区間判定不能の場合だけ160/224 bitへ進む。
5. 保証対象は、保存stateのnormalized ray、指定IID grid law、exact-parameter ideal uncompiled gateによるν_cat。
   RTE builderのπ64加算誤差は含める。ν_exec全体や、過去の未保存payloadの完全同一性は保証しない。
6. PRNG→IID、float仮想測定/clip/Bernoulli、compiled/control内部合成・noise、state preparation、
   historical payload identityのbridgeはnull。これらを0にせず、categorical判定と全体statusを別に出す。
7. 大qのradius増幅を抑える方法とstrict予算判定は人工例で確認できたが、実H4の区間幅・source identity・予算内判定は未確認。
   categorical条件が通っても、全18roundのenergy成功や総費用の保証へ拡張しない。

**次の担当：ユーザー**が、この公開SOURCEと[固定計画](SCIENTIFIC_EXECUTION_PLAN.md)の
新SOURCE replay/CDF/ideal categorical包含評価を明示許可する。
**許可後のCodex**が固定1セルを計画内で1回実行し、包含・条件付き判定・未知bridge・原本を報告してSTOPする。
承認の証明書・ID・grant等は不要。今回の技術準備指示を科学実行許可とは扱わない。

旧INCONSISTENT、旧u_num_bound=null、RPE_tolerance_certified=false、科学UNDETERMINEDを保持。
準備完了後はmandatory STOPし、B-1b/B-2、別round、sampling/shot/compile、数値再解析、独立詳細レビューへ進まない。

成果物：[実装報告](IMPLEMENTATION_REPORT.md)・[保証scope](ERROR_BOUND_SCOPE.md)・
[科学実行計画](SCIENTIFIC_EXECUTION_PLAN.md)・[人工検証](VALIDATION.txt)。
branch：`codex/rpe-m17-rigorous-mean-enclosure-preparation-20261103`。
