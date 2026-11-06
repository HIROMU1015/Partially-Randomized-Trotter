# A1対B1-OL：理論整理の報告

2026-11-07 JST。静的資料調査・既知の数理枠組みからの導出・文書作成。詳細科学レビューではない。研究方針・科学的GO/NO-GOは採択していない。

## 結論

1. **現在の3-control A1に、理想的な最適B1-OLを超える固有の分散優位はない。** controlは5層内で定数であり、A1は層間成分の一部を除く。B1-OLは既知質量による合成で層間出現数の揺らぎを除き、同shot数／同期待RZ評価予算の連続最適配分でA1を下回らない。
2. **有限B1-OL実装に対する優位は条件付きで可能。** 整数quota、最低shot、probe誤差、価格・calの保守性、全段階追加費用の差がoracle gapを上回る必要がある。現在のH4でその条件が成立するかは未評価。752-shot草案はA1の期待学習費がB1-OLのprobe費より大きく、「学習費が安い」を理由にできない。
3. **現行法の継続には有限実装比較としての意義がある。** 独自の追加分散削減を目指す研究候補としては、層内で変化するcontrolや層別＋CVに理論的動機がある。継続・拡張・終了の判断はユーザーに残す。

## 成果物

| 文書 | 内容 |
|---|---|
| [A1_VS_B1OL_THEORY.md](A1_VS_B1OL_THEORY.md) | 全分散・層別分散、patternのrankと残余、有限学習、同shot/期待RZのoracle関係、hybrid式 |
| [A1_ADVANTAGE_CONDITIONS.md](A1_ADVANTAGE_CONDITIONS.md) | 有限quota・全費用での必要十分条件、range/cal/STOP、可能性との区別 |
| [A1_FUTURE_RESEARCH_DIRECTIONS.md](A1_FUTURE_RESEARCH_DIRECTIONS.md) | 現行法の意義、H1–H7、752-shotの診断範囲、次の判断と担当 |
| 本書 | 結論、SOURCE対応、確認範囲、未解決事項、公開後STOP |

## 数式から確定すること

一軸でW=Σp_hσ_h²、R(λ)=Var_h[μ_h−λᵀ(g_h−ν)]、A=W+Rとすると、

$$
V_A=A/n_A,\qquad V_{\rm OL}=\sum_h p_h^2\sigma_h^2/n_h.
$$

同Nの連続最適層別は(Σp_hσ_h)²/N≤W/N≤A/N。同期待評価RZ予算B_evalの連続最適層別は(Σp_hσ_h√c_h)²/B_eval≤c_QW/B_eval≤c_QA/B_eval。

現在の3-controlは4patternに飽和し、5層のうちSOURCEのh=1とh=2が同じ(0,0,0)になる。任意のμ_hを表現できず、oracleでも

$$
R_*=\frac{p_1p_2}{p_1+p_2}(\mu_1-\mu_2)^2
$$

が残る。独立な有限学習はこれに非負の係数損失を加える。均等conditional trainのunweighted OLSは、同patternの2層の合成にQ質量比を使わない点も明示した。これらは新しい実H4数値を計算した結果ではない。

## 固定資料・SOURCE・既存資産との対応

- Repository：HIROMU1015/Partially-Randomized-Trotter、remote origin。
- 独立worktree：/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/a1-b1ol-theory-20261107。
- 専用branch：codex/a1-b1ol-theory-20261107。
- 基点：[5c27684d904aed9f4f9f2787752c917bb46d684c][base]。同基点が752-shot設計を含み、N1 744e1a0を祖先に持つ。実行中jobのworktreeを編集しない。
- [N1 744e1a09661ebba6841fe5faf456f88c41c47780][n1]：LAW/CONTROLSとREPORTの定義・表を静的に読む。保存数値から新しい分散・価格・順位を計算しない。
- [N3 a53c1d6eedea0d85fb484425db9012aa89738129][n3]：N3_RESULT_DECISION_FRAMEWORK.md §6の分散分解と相殺、EXPECTED_RZ_COST_GUARANTEE_PLAN.md §2,6–8の期待費用の意味を再利用。a53cは基点の祖先ではないため、原commitのblobを直接参照する。基点内のo2_a1_n3_interruption_20261106/referencesにも原文保存コピーがある。
- [正式共通指示2026-10-25][policy]：既知のworktrees共通文書を読み、基点内のarchive保存版とbytes一致を確認した。旧版§8/§10も併読し、最新版を優先。これは科学的・運用的条件の出典であり、許可証明のための保存ではない。
- 固定5c27684のa1_connections.proposed_control_banks、a1_control_variates.fit_and_freeze/FrozenCV、a1_n3_connection.ConditionalSampler/freeze/production_mean、a1_n3_pilot_design.allocate_evaluation/pilot_meanを静的に確認した。実分子関数、load_certified_services、fit、sampler、価格providerは実行しない。
- 既存人工testsのCV・層別・range・pilot集計を確認した。既存codeを変更・再実装する必要はなく、数式の適用条件とSOURCEとの関係の記述に再利用した。
- 一次文献：[Glynn–Szechtman (2002), Some New Perspectives on the Method of Control Variates][glynn]の§2、Theorem 1、§7/Theorem 6を参照した。L²射影・CVと層別の既知関係との対応確認に限定し、網羅的文献レビューや新規性認定を行っていない。Owen章PDFも取得を試みたが本文取得できず、根拠には使用しない。

## 確認範囲

主要な根拠は文書内の数式導出である。付随する人工・静的確認は以下のとおり。科学pipelineは実行していない。

- stdlib Fractionだけによる決定論的人工検算12項目PASS。全分散、共有pattern残余、oracle周りの二次損失、層別合成での相殺、均等trainの重み差、比例配分、独立学習の非負損失、費用最適配分・Cauchy、配分penalty条件、有限floor例、startupでの順位反転を確認した。
- 人工例はp=(1/10,1/2,1/5,3/20,1/20)、μ=(-1/2,1/4,-3/4,1/2,-1/4)、各層の二値応答確率(1±μ_h)/2。価格配分の検算は全μ_h=0とc=(1,4,9,16,25)。有限floor例はp=(24/25,1/100,1/100,1/100,1/100)、全μ_h=0、全価格1、評価quota=(112,4,4,4,4)を使用した。いずれも任意の人工値で、N1や測定結果から抽出していない。例のPASSは実H4の性能の証拠ではない。
- project moduleのimport、保存結果の数値load、RNGは0。既存人工testsの目的を確認したうえで、今回固有の数式だけをone-offの人工例で検算し、新しい計算pipelineや恒久test frameworkは作らなかった。
- 新規4文書のローカルリンク8件と固定Git出典リンク15件の参照先、数式区切りを確認した。N3 a53c原文2点と基点内の保存コピーのbytesも一致した。

公開時には新規4文書だけのcommitであること、専用branchのremote SHA、公開4文書の内容を照合する。具体的な完了結果は最終チャットで報告する。commit自身のSHAを本文へ自己埋め込みするための追加commitは作らない。

## 未解決事項

- 実際のμ_h、σ_h、c_h、R_*、係数損失、配分penaltyの大きさと不確かさ。
- 学習・probe・cal・評価の全期待費用、guard/STOPを含む返却確率と同精度サービスの比較。
- 752-shotの当該結果、seed間の安定性、保証付き本検証への接続。今回は閲覧・判定していない。
- 有用な層内controlの存在、既知平均の取得方法、有限学習・追加費用を含む利得。
- energy、長round、他条件、実機noise、最終RPE総費用への移送。今回の導出だけでは確定しない。

固定5c27684の752-shot文書が当時草案だったという履歴を保全する。ユーザーが今回言及する実行中検証について、新しいstatusを取得したり、実行契約へ本書を自動反映したりしない。

## 保全・次の判断・mandatory STOP

変更は本directoryの新規Markdown4点に限定する。既存SOURCE・入力・結果・STOP・科学契約・manifest・実行中jobを変更しない。新しい実分子計算、科学RNG、Born評価、回路build/compile、保存結果再解析、processへのsignal送信、共有設定変更は行わない。sub-agentは使用しない。

次の作業（いずれも未着手）：

1. **ユーザー**：現行3-controlの有限実装比較を継続するか、層内control／層別＋CVの検討を追加するか判断する。
2. **Codex**：別途指示後、許可された結果報告または選ばれた仮説の限定設計を担当する。新しい科学取得・保存結果再解析は、そのSOURCE・条件・scopeへの明示実行許可後に限る。

**本作業はcommit/pushと公開内容・remote SHA照合後にmandatory STOP。** これは本理論文書作業の停止であり、実行中科学jobの停止指示ではない。科学的GO/NO-GO、新control採択、追加pilot・本検証・レビューは自動開始しない。

[base]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/5c27684d904aed9f4f9f2787752c917bb46d684c
[n1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/744e1a09661ebba6841fe5faf456f88c41c47780/artifacts/o2_a1_new_target/2026-11-06/n1
[n3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/a53c1d6eedea0d85fb484425db9012aa89738129/docs/research/o2_a1_n3_decision_expected_rz_20261106
[policy]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5c27684d904aed9f4f9f2787752c917bb46d684c/docs/archive/server1_main_collection_20261103/b05c68a693d31e3a15652faccfc4e6f0a05f8c34/research_project_instructions_unified_2026-10-25.md.original
[glynn]: https://web.stanford.edu/~glynn/papers/2002/GSzechtman02.pdf
