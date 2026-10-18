# 不足証拠と最小追加検証案

[入口](README.md) / [条件対応](comparison_conditions.md) / [既存結果](existing_strong_baselines.md)。**計画の整理であり、新しい実験契約・実装指示・実行許可ではない。** 不足があることだけを理由に全項目の取得を推奨しない。

## 1. 比較可能性の分類

| 分類 | 対象 | 現在の扱い |
|---|---|---|
| **既存SOURCE・結果だけで比較可能** | G9 v2内の同じP5/provider/direct・22-axis policyによる6方式の期待T/K・CX表示、helper5 rows、model内workspace | 旧matched表・affine判定を引用すれば閉じる。新compile/N評価不要 |
| **契約・評価規則の統一が必要** | G7または新B3/C1とG9のN・native総量の横断比較 | β3対β8、α/axes/resource failure、log、helper/direct、primitive/全event費用scopeを合わせる |
| **技術実装だけが不足する部分** | 物理Qとpolicyを既存G9に固定した後の監査/price/resultのbinding、必要なproduction/保存adapter | 固定SOURCEを上書きせず別版が必要。今回は実装しない。完成度はCodex Aの公開報告から後日確認 |
| **新規数値評価が必要な場合** | 新しい共通policyのN/ceil、既存native費用の新倍率/役割別集計、共通条件の全trial古典費用 | 保存値からの再会計も科学数値再解析なので別途承認。今回は0件 |
| **新規native取得が必要な場合** | provider/angle/precision/gate set/compiler/最適化/費用座標が旧証拠を超える | 一致する旧primitive/rowsは再利用し、不一致部分だけ対象化。未取得depthのためだけに全再compileしない |
| **科学的判断が必要** | generic query-only対I1、fixed workload対期待μ、主要費用座標、prep/readout込み範囲、sampling/precision最適化を含めるか | ユーザー/GPTが採否を決める。Codexは採択せず、重要レビューも開始しない |
| **現時点では比較不能・未解像** | 未指定providerのC1対G9、異なるbias/confidenceのnative総量、depth・実state/backend総費用、Closed P5/CTSの監査込みμ | 価格や未実装費用を0にしない。未知はUNKNOWN、重複区間をtieにしない |

G9のnative費用はliteral exact modelの資源会計であり、hardware実行費用ではない。B3のCPU秒・F2 bit上界・native T/CX/depthを無換算に加算しない。[G9R][G9R] [ABE][ABE]

## 2. P0：既存G9比較の引用で閉じる（最優先）

| 項目 | 内容 |
|---|---|
| 検証目的 | 登録P5で強い対照を含む既存資源主張の範囲を正確に固定する |
| 比較方式 | General Full、Closed P5、matched CTSとG9登録direct6方式 |
| 再利用証拠 | G9R/G9AFF/G9CSV/G9AUD/G9MAN、固定SOURCE/契約。今回の引用表と対応表 |
| 不足情報 | この**既存scopeの引用**には追加数値情報なし。一般CPU/native・科学的採択は別 |
| SOURCE・契約変更 | 不要。G7/B3/C1契約へ値を混ぜず、G9の独立証拠として記す |
| 新規計算・compile | 不要。旧数式確認・native取得・保存値verifierも再実行しない |
| 変わる研究判断 | 「Fullが強い既知対照を含め全座標で最良」とは主張できないこと、m=5 fast pathと一般構成の役割分担を判断材料にできる。新規性/GOは認定しない |
| STOP・承認 | 今回の監査で完了。次の研究方針や原稿claimの正式採択はユーザー。追加研究を開始せずSTOP |

この範囲で十分ならP1以降は不要である。B3/C1の完成を待たずに保存できる結論である。

## 3. P1：B3/C1公開結果とのidentity・scope照合だけ

| 項目 | 内容 |
|---|---|
| 検証目的 | 新費用結果がG9の既存native証拠をそのまま再利用できるか、差分だけを特定する |
| 比較方式 | 新結果の4方式と、同じimplementation/費用scopeを持つG9 row。強い対照にもdirectを許す |
| 再利用証拠 | G9のS/契約・19 keys/strict phase・native/compiler・各policy。Codex Aが将来公開する固定result/manifest/handoffだけ |
| 不足情報 | 新結果の固定commit、cost estimand、N/bias/confidence、provider/control/adjoint、layer/angle/cache/保存scope。現作業ではCodex Aのprivate成果を取得していない |
| SOURCE・契約変更 | 原則不要。まずpublic metadataの照合。diffがあれば計画だけ返し、Codex BがAのSOURCE/schema/候補を変更しない |
| 新規計算・compile | identity照合のみなら不要。新価格への再縮約・新N評価はP2の独立scope |
| 変わる研究判断 | G9既存T/CX・helper診断の引用を添えられるか、別条件として分離するかが分かる |
| STOP・承認 | ユーザーの後日の照合指示で実施可能。private worktreeへ立ち入らず、未公開/identity不一致なら保留・STOP |

G9にはすでに明示3-qubit providerとdirect/helper native rowsがあるため、**同じcontextを選ぶ場合のC1を一から再取得する必然性はない**。これはCodex Aへの新provider採択・実装変更指示ではない。

## 4. P2：どうしても単一policyの新費用表が必要な場合だけ

| 項目 | 内容 |
|---|---|
| 検証目的 | 異なるG7/G9 budgetの値を混ぜず、採択した一つの規約でnative費用を比較する |
| 比較方式 | 少なくともGeneral Full/Closed P5/matched CTS、必要なら登録direct6方式全体。同じfinite P5/provider/合法direct機会 |
| 再利用証拠 | G9の保存per-trial/event費用・exact係数/π、group/CTS構成、19取得列。G7の表を勝手に換算しない |
| 不足情報 | 共通bias/confidence/resource policyと対象axes、同費用座標の定義。G7の3ρがCTSまで妥当という既存保証は示されていない |
| SOURCE・契約変更 | 同じG9 policy/値の引用で足りるなら変更不要。別policyが必要な場合だけ、新しい共通budget/集計仕様を別版で固定。新非自明保証が必要ならSTOPして科学判断へ戻す |
| 新規計算・compile | 別policyのN/ceil・native期待量再縮約は新数値再解析で別途承認。provider/keys/compilerが完全一致なら新compile/synthesisは不要であり、既存値を再利用する。新座標が不足すれば差分だけ判断 |
| 変わる研究判断 | 新しい単一policyでの各費用座標が判定可能になる。ただし強い対照最適sampling・一般Q・全mの優位ではない |
| STOP・承認 | ユーザーによる共通条件の採択と数値再解析の明示承認。G7の低いbiasをCTSへ無根拠に拡張、全event無料取得、未承認N/新価格/新keyが必要ならSTOP |

単一policy化そのものを研究上の目的にしない。元G9は6方式をすでに同じpolicyで比較しており、既存結論を確認するためだけのP2は重複投資になる。

## 5. P3：強い対照まで含む古典費用が判断を変える場合だけ

| 項目 | 内容 |
|---|---|
| 検証目的 | General Fullの追加local/guard/requery費用が、Closed P5/CTSとの全trial比較で許容されるか |
| 比較方式 | 最小はGeneral Full/Closed P5/matched CTS。同じ採択policy・監査/保存・情報access・cost estimand |
| 再利用証拠 | Codex AのB3結果は**対応が一致する範囲**だけ。G9のP5 sample API/CTS有限event law、setup/Pauli/referenceの既存費用記録、F2/F3の費用層分離 |
| 不足情報 | Closed P5/CTSの同等preflight/guard・production接続・全N output/cache費用。CTSの構成/法則は実装済みだがG9は全N sampler benchmarkではない。固定trace平均を母期待μにしない |
| SOURCE・契約変更 | まず必要なadapter差分を特定し、必要時のみ別承認で実装。F3 Full固有検査の流用や比較側監査0は禁止。現在のCodex Bでは実装/契約候補を作らない |
| 新規計算・compile | 古典profile/全trial workload取得は科学実行として別途承認。旧N7での全N実測をN9へ比例外挿しない。native compileは古典比較には必須でない |
| 変わる研究判断 | CL条件を強い対照へ広げ、古典/量子のトレードオフを判断できる。CPUとTの未採択scalar合成やC2総資源は別 |
| STOP・承認 | ユーザーがこの不足が研究判断に必要と判断した場合だけ実装・実行承認。費用estimand、監査、情報取得、保存、precision/identityが揃わなければ保留・STOP |

## 6. 追加取得を今推奨しないもの

depth、hardware noise、実state preparation/readout/backend、任意cost-aware IS、最良compiler、mやproviderの新探索は不足/未検証である。しかし**登録P5のT-primary/CXトレードオフを正確に記録するだけなら不要**。End-to-Endや一般性を別の研究目的として正式採択しない限り、取得を自動推奨しない。[G9R][G9R] [APLAN][APLAN]

未取得=0、区間重複=tie、同じp/x/m=同じ全費用契約、少ないN=少ない総費用という置換を禁止する。C_cl=S+Nμ+Ω、C_q,d=NΣπcのW2c-A条件はそのまま使うが、未知変数を埋めず、upper同士で実費順位を作らない。[ABE][ABE]

次は**ユーザー**がP0の既存限定証拠で判断できるかを検討する。必要なら**Codex**へP1の公開identity照合を依頼し、P2/P3や新native取得は対象・固定SOURCE/入力・精度・call・保存・STOPを明記した別途採択/実行承認後だけ行う。科学的に重要な新判断は論点として返すだけで、レビューを自動開始しない。**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[ABE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/864097c1285f1f06399040cf0f7c0f42334d9252/docs/research/track_b_g6_w2c_total_cost_conditions_20261018/break_even_conditions.md
[APLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/864097c1285f1f06399040cf0f7c0f42334d9252/docs/research/track_b_g6_w2c_total_cost_conditions_20261018/missing_cost_evidence_and_execution_plan.md
[G9R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
