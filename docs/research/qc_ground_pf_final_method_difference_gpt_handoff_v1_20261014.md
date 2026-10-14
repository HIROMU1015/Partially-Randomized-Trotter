# 厳密ground PF再分割：最終方法差監査のGPT引継ぎ

2026-10-14 JST。Codexの限定技術監査完了。GPT科学レビューは開始していない。第一候補の正式採択・継続・終了・第二候補への移行を決定しない。[方法差・計算量報告](qc_ground_pf_final_method_difference_audit_v1_20261014.md)

## 1. 結果

**`REDUCES_TO_KNOWN`：指定された同情報・同恒等式・同自由度・同cacheの比較では、1-RDM応答生成はcompact/grouped BCHの特殊化として同等に実現できる。独立した生成費用・情報量・保存量の改善は残らない。**

最強対照B*を、未変更tailの差をexpectation取得前に除き、既存ground交換子恒等式とone-body CARでsimplifyし、同じmatrix trace/edge演算とcacheでg,Qを作る手順として具体化した。元提案のblack-box generatorを呼ぶだけの対照ではない。

両者ともresponseに1-RDMの同じlinear functionals、初回の同じd0取得を用い、full response表はdense arithmetic O(n³)/追加storage O(n²)、K dense候補はO(Kn²)、fixed directionはℓ/q取得後scalar O(1)。RDM・d0取得とinput/frame費用を戻しても、それらは共通費用であり現法だけの差にはならない。未知実単価を0としない。

前回の数式と保存27検査を否定・変更していない。前回未確定だった「強いcompact対照も同じ表・費用へ達するか」を、今回の公平な比較条件内で解消した。閉式の歴史的初出性や理論記録価値の判断とは別である。

## 2. 六つの問いへの回答

| 問い | 技術監査の答え | GPTに残る範囲 |
| --- | --- | --- |
| ① 1-RDM応答生成は既知compact BCHの特殊化か | 指定された固定先頭pair/equal-accessの範囲ではYES、REDUCES_TO_KNOWN。1-RDMとcacheはB*も使える | その特殊化や閉式の理論記録価値、先行性をどう扱うか |
| ② 本方法だけの具体的効率改善はあるか | この生成classでは特定できず、同等の具体演算列が存在する。愚直full-BCH対照への利得は独立差ではない | generic実装のengineering改善と研究寄与を区別する科学判断 |
| ③ d0・1-RDM費用を戻しても意味のある差か | 両法が同じ取得を共有・再利用できるため、取得費用から固有差は作れない。delta表からabsolute d0まで無料にはならない | 未知実費の一般的重要性。追加数値取得を自動で指示しない |
| ④ 強い補正PF・既知再配分との比較に科学的価値があるか | 生成独立性を根拠とする比較理由は残らない。finite-h物理利益はUNDETERMINED、今回はPhase B未実施 | scalar-corrected S2/S4/外挿等と別の目的で比較する価値があるか |
| ⑤ 数値pilotへ進む具体的技術根拠は残るか | 「既知法より安い独自response generatorを確認する」という現理由は支持されない。現計画の準備・freeze・実行へ進まない | 既知手順の応用評価へscopeを変える科学的合理性があるか。変更・計算は別認可 |
| ⑥ 第一候補継続か第二候補か | Codexは決定しない。今回の前提での独立生成claimを踏まえてGPTへ戻す | 第一候補終了/技術記録化/別目的で継続/別候補検討の選択 |

第④⑤で「物理性能が悪い」「第一候補を正式終了した」「第二候補を開始する」と判断していない。未判定物理量の存在を新規性・継続根拠と扱わない。

## 3. 限定性と旧pilot案

同値としたのは、外側one-body Aと最初のdensity B、Xがそのframeでdiagonal、fixed S2、exact eigenstateと同じground恒等式・precision・legal cacheを使用できる場合のleading response生成。別ordering/非隣接/複数非可換更新/frame変更/有限h全log/overlap/compile/全energy taskには拡張しない。

既知FFFのvariance目的やpartitioningのreflection/L1 heuristicと、現energy目的は同一ではない。しかし既知自由度とstate-specific BCHを接続した強い対照B*に、同じ応答目的と取得情報を許すと生成差は消える。現在のgeneric libraryが閉式を自動認識すること、実backend runtime/bit一致は確認しておらず、理論上のmatching constructionと区別する。

[旧pilot案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0bb4350e599e557f7897e64222854d0fe45d620/artifacts/qc_exact_ground_pf_repartition_preaudit_v1/2026-10-14/minimal_pilot_design.json)は固定byteのまま。未固定/未認可/非execution-readyという状態を維持し、入力・source・direction・h・capsを追加で固定していない。Phase A STOPによりPhase Bのphysical mechanism/pilot情報価値解析を新たに進めていない。

## 4. 判定語と次の担当

- `SUPPORTED`：前回固定responseの能力・数学的supportは保持する（再検査なし）。
- `REDUCES_TO_KNOWN`：最強の許可B*が同じ情報/表/演算量/storage/cacheで生成できる、という今回の限定技術所見。
- `CONDITIONAL`：同じexact-state/gamma/d0/precision accessが与えられた場合の静的費用上限。bit長/実測費用の最適性ではない。
- `UNDETERMINED`：歴史的先行性、記録価値、有限h量子利益、研究継続/終了/移行。GPTが別明示開始承認後に判断する。

次は、ユーザーによる**GPT重要科学判断の別明示開始承認**。同じ第一候補の追加技術監査を反復しない。新計算・pilot準備・第二候補を開始する包括認可はない。

## 5. 証拠・公開・停止

Base `c0bb4350e599e557f7897e64222854d0fe45d620`、独立branch `qc-ground-pf-final-method-difference-audit-20261014`。[固定9参照のcommit/SHA](../../artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/reference_identity.json)、[一次本文](../../artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/literature_registry.json)、[scope](../../artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/scope.json)、[method status](../../artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/method_status.json)、[404旧files保全](../../artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/preservation_index.json)、[公開receipt](../../artifacts/qc_ground_pf_final_method_difference_audit_v1/2026-10-14/publication/receipt.json)。

旧148manifest records/non-result fields/global description欠落FAIL、元worktree135pathsを保全。新科学数値calls/保存数値集計/新形式program checks/旧27検査再実行/production/pilot準備/Track A–Bや旧系列再開/GPTレビューはすべて0。N1STOP/N2HOLD/N3CLOSED/THRIFT現template終了、旧source/結果/認可/STOPは不変。commit・push・GitHub byte/SHA照合後Mandatory STOP。
