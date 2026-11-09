# 元Track A（PRアルゴリズム改善）の研究整理と次段階判断の準備（2026-11-09）

> 記録：2026-11-09にClaude Codeが静的調査として作成し、repository外の作業報告ファイル `.agent/docs/review.md` で報告した内容です。2026-11-10にユーザーの指示でrepositoryへ保存しました。本文は報告時のままで、「変更ファイル」などの節は報告時点の記述です。技術的な意見は採択ではありません。新しい科学計算は行っておらず、mandatory STOPを維持しています。

読み手：GPT（研究方針の科学的判断）とユーザー。
作業範囲：既存資料・SOURCE・保存結果の静的調査のみ。新しい科学計算、数値再解析、build/compile、sampling、保存値の集計・ランキング、repository変更、サブエージェント利用はいずれも0件です。報告後は **mandatory STOP** です。

### 要点

1. **「元Track A」のアルゴリズム改善系列は、2026-11-06〜08に行われた3本です。**
   - **O1-M1**：同じ物理tailの辞書を替える（Pauli collect と DF回転Z/ZZ）。
   - **O1-M2**：圧縮core＋厳密残差。
   - **B2-P1**：既存B2方式（DF prefix L_D3＋tail RTE）を、全18round RPEへ接続する。
   - M1から分岐した **C-S V0**（SERVER1）も含めます。
   - 10月までの旧Track A（旧PR-2 M1/M2、PM、AX、H4/H6/H8、PF588、長時間B2/B3）は**リソース評価**です。10/14の正式ノートも「新PR algorithmを達成条件にしない」と明記しています。
2. **元Track A内で、強Full PFに対するPR＋RPE総費用の優位を示した結果はありません。**
   - M1 C0の条件付き総CXは、PRが強DF-PFの約2.9e18〜5.8e18倍です。
   - B2-P1は価格が未取得です（取得済みはm0の1 trajectoryだけ）。
   - 近い条件の **Track C（C-H4）** は、PR内部を最適化した後もPR/強Full = RZ 2.3086倍でした。H4ではTruncated PFがさらに安い結果です。
3. **B2-P1のshot数の差は、長roundのnormalization減衰だけで生じています。**
   - PRは7,805,784 shots、強PF 8構成はそれぞれ18,540 shotsです（引用2値の比で約420倍）。
   - m17だけで98.49%を占めます（A=.0116）。初期roundはPRも強PFも515 shots/axisで同じです。
   - 原因は全round共通の固定 r4/K2 です。round別のr（resolution）選択は、9月のopt2 modelとTrack Cで既に扱われています。
   - Track Cでは、その結果trial数がFull並みになり（16,791対14,225）、費用の主因がRTE event単価へ移りました。
4. **静的に読める中心的な制約（GPTに確認してほしい点）**
   - SOURCE式を主要次数で読むと log A_m ≈ −λ_R² δ T_m / r_m です。
   - 減衰を抑えるには r_m ∝ T_m が必要になり、1shotあたりのRTE event数は λ_R² T_m² に比例します。
   - 固定T（≤2）の資源評価でPRが有利だった領域を、T_max=2621のRPEへそのまま移せない理由はここにあります。
   - 今後の改善機構は、長roundの λ_R、event単価、T_max のどれかを動かす必要があります。
5. **PR内部の改善候補の多くは、他Trackで実施済みか負結果があります。**
   - r/h/αの共同選択、Closed P3、joint isometry、D内部BCH はTrack C。
   - IS/CV/LCU はO2。
   - C-S はTrack C pilotとSERVER1 V0。
   - Reference干渉は新AとTrack D。
   - 元Track Aの守備範囲（表現・分割）で未着手なのは主に次の3つです：D/R/drop三分割（K3）、元H groundでの表現接続（K4）、RPE長roundでの round別 D/R 分割。
6. **記述の食い違いを1件見つけました。**
   - 11/09の横断調査 `2e79d80d` は「強PF 5 STOP armは未再資格」と書いています。
   - しかし 11/07 の `3e316e07`（HP 80/120桁で5 arm全90点PASS）と `e162ef70`（全8 armのbranch/shot PASS）で、5 armは再資格済みです。
   - 本報告では後者を正とします。repositoryは修正していません。

---

## 0. 作業の前提・方法・用語

### 0.1 指示と実行範囲
- 指示は、チャットに直接貼り付けられた「Codex指示：元Track Aのアルゴリズム改善研究の整理と次段階判断の準備」です。`comment.md` には10/18付けの別指示（H8 RARE_SENSITIVITY）が残っていましたが、古いものと判断し、実行していません。
- 調べた対象は、ローカルclone `partially-randomized-trotter` のGit object（固定commitに対する `git show`）だけです。
  - remote-tracking refを、GitHubの `git ls-remote` 228 heads（2026-11-09取得）と照合し、全件一致しました。fetchは不要でした。
  - 研究moduleのimport、test/runnerの起動、配列のdecodeはしていません。
  - 実行中の他プロセス（`prefetch_h46_full_h_t20.py` 群）には触れていません。
  - projects全体の横断検索もしていません。
- 報告で参照した72個の固定commit/pathは、すべて `git cat-file` で存在を確認しました。

### 0.2 元Track Aの範囲（本報告での定義と根拠）

| 区分 | 内容 | 根拠 |
|---|---|---|
| 旧Track A＝リソース評価 | 既知PRをDF Hamiltonianへ実装し、有限時間 z(T) の同一精度で、測定込み回路資源と適用域を地図化する | [10/14正式ノート](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4675730e7e40d25ba3618334a4a0e28b4bd2f145/docs/research/track_a_shared_research_status.md)、R09は「PR全勝/new algorithmを採用しない」（[判断索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4675730e7e40d25ba3618334a4a0e28b4bd2f145/docs/research/track_a_scientific_review_decisions.md)） |
| 元Track A＝アルゴリズム改善 | 文献調査のO1（表現・分割）を `track-a-o1-*` で実施（M1/M2）し、`track-a-b2-*` でB2をRPE全体へ接続 | [O1–O6定義](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/RESEARCH_OPPORTUNITIES.md)、[O1 preflight](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/docs/research/o1_hamiltonian_partition_preflight_20261106/REPORT.md)、[11/08横断](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_rpe_cost_refocus_20261108/REPORT.md)の「Track A／表現担当」 |
| 他系列（統合しない） | O2＝sampling/推定器、O3＝Track C、O5＝Track D と新A（Reference干渉）、研究C＝PR-aware DF、研究B、Track B（G6–G10） | [11/09横断 A](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e79d80dc9cce2885580e50fe6f1705da8fa41af/docs/research/pr_rpe_cross_track_20261109/EXISTING_RESULTS_AND_OVERLAP.md) |

**名前の衝突に注意してください。**
- 旧PR-2の「M1/M2」（H4開発210候補と1.30 Å移送）は、O1-M1/M2 とは別の研究です。
- H8の「Phase B2」は決定論S4価格の段階名で、PR方式B2ではありません。
- exact engineの「O1/O2」とH8 G04の「O1」は、研究機会O1とは無関係です。
- C-Sは2つあります。Track C初期pilotのC-S（SERVER2）と、M1から分岐したC-S V0（SERVER1、`pr-o1-m1-delta-cs`→`cs-v0-*`）です。どちらも機構はO3系です。

---

## ① Track Aの研究到達点

### 1.1 リソース評価（旧Track A）で得られたもの

**この節は既知PRの適用評価であり、アルゴリズム改善の成果ではありません。** いずれも有限時間 z(T)、T≤2、ε_complex を対象にしています。RPEの全round費用ではありません。

| 研究 | 主な保存結果（引用） | 改善研究への意味 |
|---|---|---|
| 旧PR-2 M1/M2・PM-0/1/2（H4 DF12, T=.8, S2） | B2 L_D3/q1/r4/K2 が6指標でPareto（G=130,774,896.66 RZ）。1.30 Åへの移送を支持。PM-2では精度で最良q/Kが変わる（[M1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8e0814e70c14ecf526444fac8a2142799610dc96/docs/pr2_matched_accuracy_m1_b1_result_validation.md)、[M2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/054bb7b2b2ca788b47ebdc5c14c115f24c8b6379/docs/pr2_matched_accuracy_m2_transfer_result_validation.md)、[PM-2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5a1adffad780f0ec4272f5e8bb94713f9ff0f2bc/docs/pr2_pm2_precision_resource_result_validation.md)） | B2-P1の代表候補「B2-rank3-q1-r4-K2」の出所。分割（L_D）と r/K の選択が精度に依存することを示す |
| H4 geometry・B0+I | 元4方式では1,812点すべてでB2が最小。既知scalarを戻す対照B0+Iを加えるとB0+I 397点／B2 1,415点（[geometry](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/docs/research/track_a_h4_geometry_results_20261013.md)、[B0+I](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/docs/research/track_a_h4_b0i_map_results_20261014.md)） | 対照を強めるとPR優位は縮む |
| H4/H6/H8固定6確認・PF588・長時間 | H4/H6は粗い精度でPF、細かい精度でPR。H8は両点でPR。PF588ではH2は全点PF。長時間（T≤2）ではB2が33/40で低費用。B3（L_D0）は0/40（[PF588](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/docs/research/track_a_h246_pf_common_final_20261022.md)、[B3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/REPORT.md)、[H468統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/track_a_h468_paper_evidence_integration_20261019.md)） | 短時間かつ高精度ではPRが有利。全H純RTE（11/09）も固定点で高価（[Pure-H](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9b83b96b83f62142c9d87858dc24a94db7ac5d61/artifacts/resource_applicability/df_full_h_pure_rte_light/2026-11-09/REPORT.md)）。分割の極端（LD0）は不利 |
| 9月の全RPEモデル（M06-F coherent opt2、H4 DF12 L_D3対L_D12、δ=.02、18round） | モデルGはPR 1.263e12対Full 1.328e12 RZ（点推定でPRが4.858%低い）。shotsは13,588対11,162。区間は重なる。q>32はproxy、opt2 compile（[M06-F](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research_direction_full_opt2.md)、[landscape](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/pr_repository_scientific_landscape_20261101.md)） | round別のr最適化でshot爆発は回避できる。ただし、後にTrack Cがopt2の微小角省略問題を指摘し、compiler A・強対照の下ではPR/Full=2.31倍。**両結果の関係は未整合**（§6） |
| AX、H6希少次数・s2 pilot、H8希少層 | 契約・強対照（spin-aware relative fusion等）の整備と、希少eventを含む期待価格推定の方法 | 評価方法の整備であり、PR方式の変更ではない |

### 1.2 アルゴリズム改善（元Track A）の系譜

| 日付 | 段階 | 固定commit |
|---|---|---|
| 11/05–06 | 3環境の改善証拠を統合 → PR＋RPE文献調査（O1–O6）→ O1 preflight（候補をM1/M2の2件に絞る） | [3571477](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/REPORT.md)、[e8b19ff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/RESEARCH_OPPORTUNITIES.md)、[aeb9d18](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/docs/research/o1_hamiltonian_partition_preflight_20261106/O1_IMPROVEMENT_MECHANISMS.md) |
| 11/06 | O1-M1：実装 → H4 operator → 回路・信号 → C0総費用 → 改善調査 → r4/r8減衰 → 必要単価 → CX下界 → r8価格pilot | 下表 1.3 |
| 11/06 | O1-M2：実装 → H4 operator（mapper cutoff FAIL）→ 微小係数修正 → 残差検証（NOT_QUALIFIED）→ Hermitian-B（operator PASS）→ core-only上界 → 改善調査 | 下表 1.4 |
| 11/06–07 | B2-P1：総RPE接続調査 → 実装 → 元H/DF12対応 → H_phys ground → 減衰screen → P1条件 → m0/m17 → 残り16round → branch → shot → 強PF 8 arm → 数値診断 → HP再資格 → branch/shot → m0価格 → 長round価格方式 → streaming → fingerprint・memory | 下表 1.5 |
| 11/08 | M1分岐：重複を排除してC-S同位置接口へ（ユーザー指示の順序：O1-M1→O3-C-S→C-N→C-P→Multilevel）→ C-S V0（1回STOP、修復して再実行し完了） | [c077de7](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c077de79713f793939f7e33c18ca81f4c819588c/docs/research/pr_o1_m1_delta_cs_preparation_20261108/REPORT.md)、[7b57654](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7b5765453cf29532d8c8280d5aa581a87eaaf464/docs/research/cs_v0_repair_results_20261108/REPORT.md) |
| 11/08以降 | Track A名のbranchへの更新なし（remote最終は 11/07 23:41 のB2 memory pipeline と、11/08 02:31 のC-S V0）。11/08・11/09の横断調査がTrack A成果を引用 | [654b850](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_rpe_cost_refocus_20261108/REPORT.md)、[2e79d80](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e79d80dc9cce2885580e50fe6f1705da8fa41af/docs/research/pr_rpe_cross_track_20261109/REPORT.md) |

### 1.3 O1-M1（同じ物理tailの辞書変更）とC-S分岐

| 項目 | 内容 |
|---|---|
| 解決しようとした問題 | 同じ H_D/H_R のまま実装辞書（global collected Pauli と DF回転Z/ZZ）を替え、λ_R・normalization・event価格の交換で N×full価格を下げられるか |
| 従来PRからの変更 | split（H4-A：保存rank10、共通 T_eff/V_square、**LD0**＝二体全部がR）を固定し、tail辞書だけを替える |
| 期待した効果 | 同じ補正meanのまま、λ_R が小さく単価の安い辞書で総費用が下がること |
| 確認された成果 | ① C0（δ=.05、m0..6、r2/K2、ε_E=.1 Ha）の全7round信号が条件付きPASS。21/21の価格pointも取得。辞書間の条件付き総CXはDF/P=0.50384（[C0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/docs/research/track_a_o1_m1_c0_parallel_science_20261106/REPORT.md)）。② q64でrを増やすと減衰が軽くなる：A は r2→r4→r8 で P 1.74e-11→1.38e-6→1.00e-3、DF 3.18e-11→1.97e-6→1.21e-3（[減衰](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185104f64a4594355a3813d04d9e50bd4abdb6b5/docs/research/track_a_o1_m1_attenuation_science_20261106/REPORT.md)、LIMITED_SIGNAL_PROBE_COMPLETE） |
| 負結果 | ① C0の条件付き総CXは P 5.69e28／DF 2.87e28 に対し強DF-S2は 9.83e9 で、baseline比は約5.8e18／2.9e18。q64のshot/axisは P 3.7e24、DF 1.1e24、強PFは1,134。② q64だけにbaseline全7round予算を許す楽観的な必要単価でも、r4は約1e-5 CX/shot未満、r8は4.34／6.29 CX/shot未満が必要（[必要単価](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c7e9587fb7398ebdab156d83abc869d82e5f42c9/docs/research/track_a_o1_m1_cost_feasibility_science_20261106/REPORT.md)）。③ r8の価格pilot（各8 trajectory）は平均 P 14,280.4／DF 24,313.3 CXで、必要単価を大きく超えた。固定r8の2候補は研究上の終了が提案された（[r8](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0daf01d51c022977a1c0a7d29dde34a1c4c82d7c/docs/research/track_a_o1_m1_r8_price_pilot_20261106/REPORT.md)）。④ 11/08、M1には既存の collect/selector/matched Stage1–4 へ還元されない操作上の新しい差分は見つからなかった（[重複判定](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c077de79713f793939f7e33c18ca81f4c819588c/docs/research/pr_o1_m1_delta_cs_preparation_20261108/EXISTING_EVIDENCE_AND_DUPLICATION_CHECK.md)） |
| 失敗の原因（静的な読み） | 辞書間の差は約2倍なのに対し、強PFとの差は約1e18倍。この負結果は辞書の違いより、**LD0による大きいλ_Rと長roundの減衰**が支配していると読める。なお固定T=.8（P3、11/02）ではRZでPauli/DF=0.45273とPauliが安く、C0（CX、RPE）とは辞書の優劣が逆になる（[P3 Stage4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a1d3f32aba6416fd1e200cd66a97d1a825b5dd73/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage4/run01/REPORT.md)）。指標・taskが違うため、比較そのものは条件付き |
| 未検証・未資格 | CX下界は4候補ともUNDETERMINED（非負の0しか確立していない、[下界](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2263000e203f8b9b075f7296a9e33f34e8ea24db/docs/research/track_a_o1_m1_cx_lower_bound_20261106/REPORT.md)）。価格CIなし、not_certified。LD>0の分割での辞書効果、round別r/Kの全RPE最適化は未実施 |
| C-S分岐（O3系） | 回転角を coarse＋有限residual に分け、元のslotに挿入する接口を実装（人工9件PASS）。V0：D/R各1局所specでfinite≤1e-12、局所≤3e-6を通過。局所literal T は通常S0の70に対し、S1 ≈142/140、S2（Dのみ）≈142/70で、**T削減は見られない**（[V0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7b5765453cf29532d8c8280d5aa581a87eaaf464/docs/research/cs_v0_repair_results_20261108/REPORT.md)）。Track C初期のC-S pilotでも終端log(range)が.557から約386.8へ増え、trial負担が極端に増えた |
| STOP・承認 | 全段階mandatory STOP。r8終了提案を正式に採択したかはUNKNOWN（11/08にユーザーがC-Sへの移行を指示）。C-S V1、full-wrapper、C-N/C-P/Multilevel（SERVER1系列）は未承認・未着手。有効な実行承認はない |

### 1.4 O1-M2（圧縮core＋厳密残差）

| 項目 | 内容 |
|---|---|
| 解決しようとした問題 | 実装しやすい圧縮coreに残差 H−H_core を補償的に付けて元Hを保ち、backboneを軽くする |
| 入力 | 研究Cの update005（number-CDF rank8、全8 frame、own-DF fit由来）と元h/g。H4・1 Å・STO-3G、Nα=Nβ=2/sector36 |
| 確認された成果 | ① 残差に微小係数が全部消える問題を解消（360項保持）。② 規約B（明示Hermitian化）で C_H+R_H=H_H、R_H mapperとCARの差 8.3e-25 などで **operator-only PASS**（[B](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/68ef4feee3835ed7135d254ac879e8d88c69c895/docs/research/track_a_o1_m2_hermitian_b_science_20261106/REPORT.md)）。③ core-onlyの省略誤差は保守的なCAR係数上界で **≤6.93e-8 Ha**（[core-only](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c75d3cfd2115051814c80e14b0af6e55bc51b779/docs/research/track_a_o1_m2_core_only_science_20261106/REPORT.md)） |
| 負結果・問題点 | 固定SOURCEでの残差検証は、保存binary64のanti-Hermitian成分（8.39e-17）によりNOT_QUALIFIED（[残差](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa7dc379cd16975dea317220148c473a0976a9f8/docs/research/track_a_o1_m2_retained_residual_validation_20261106/REPORT.md)）。研究Cのnative-cost fitはTECHNICAL_NO_GO。同じJ/Uの強Dに対し、旧LD2 PRは主RZで1.97倍（own-DF state・別条件、[M2調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f891f6e3265e4f0124c598fddfb7775bc1eddd1b/docs/research/track_a_o1_m2_improvement_survey_20261106/REPORT.md)） |
| 静的な読み | 残差の上界（6.93e-8 Ha）はP1のε_E（1.59e-4 Ha）より3桁以上小さい。この入力ではcore-only（決定論）で足りる可能性が高く、**残差をランダム化する必然性自体が成立していない**（調査報告も「core-onlyが十分安ければM2の費用仮説は支持されない」と記載）。M2はPR改善というより、決定論の圧縮表現（研究C・UWC系）の問題に近い |
| 未検証 | H_physの定義と保存bitの扱い、model差の許容（physical_model=UNASSESSED、budget=DEFERRED）。PF/RTE信号、native価格、N、全RPE費用はすべて未取得 |
| STOP・承認 | mandatory STOP・保留。次段階には、model許容とH_physの判断（ユーザー/GPT）と別途の実行承認が必要 |

### 1.5 B2-P1（既存B2方式の全RPE接続）

**固定科学条件（P1、[7bc9527](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7bc952772e451f27c1654dab6803192f2b204c9f/docs/research/track_a_b2_signal_conditions_20261106/REPORT.md)）**
- 対象：linear H4、1 Å、STO-3G、cutoff前のH_phys、Nα=Nβ=2/sector36、元H保存基底state（厳密基底状態と仮定、準備費は除外）。
- 表現：保存DF12。H_physとの差は数値診断で約9.27e-15 Ha（[対応](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/275813d0a12a4bca62a8c972312c80386829de22/docs/research/b2_original_df12_correspondence_20261106/REPORT.md)、[ground](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/89165cf70958e7e4676d0580e3acb7faf588772c/docs/research/b2_hphys_ground_state_20261106/REPORT.md)）。
- 方式：generation-prefix L_D3（決定論はone-bodyとfragment 0..2、Rは3..11、λ_R=.5833、非identity成分324）、partial-S2、**全round r4/K2**、δ=.02、m0..17（Q_max=131072、T_max=2621.44）。
- RPE条件：ε_E=1.5936e-4 Ha、95%、β=.4、prior [−128,128]。位相予算は model .02／PF .06／RTE .04／stat .24／impl .02／reserve .02、α_axis=.05/36。

| 項目 | 内容 |
|---|---|
| 解決しようとした問題 | 固定T資源評価で優位だったB2が、元Hの同一energy精度・成功確率の総RPE費用でも有利かを検証できる形に接続する |
| 従来PRからの変更 | **PR方式自体は変えていない**。変えたのはtarget（DF-state→元H state）とtask（z(T)→18round RPE）と会計の接続。改善機構の提案ではなく、改善課題を見つけるための基盤 |
| 確認された成果 | ① 全18roundで条件付き信号PASS（m17：PF位相.01399 rad/cap .06、RTE disk 7.5e-8、A=.01157、B=86.41）（[m0/m17](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ea2c0915ba2f7e11a6dac58dc01cf1af3c0d0ff5/docs/research/track_a_b2_p1_two_round_20261107/REPORT.md)、[残り16](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d7a55ac181de71cd2b2c49644810373cafd0a450/docs/research/track_a_b2_p1_remaining_rounds_20261107/REPORT.md)）。② 既存decoderで全18roundのbranchを一意に復元。E_hat=−2.16638211 Ha、誤差5.34e-6 Ha（[branch](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d424bb8bec1aaf1224a33513a06e64801749a69f/docs/research/track_a_b2_p1_branch_20261107/REPORT.md)）。③ 条件付き十分shot：計 **7,805,784**。m17が7,688,270（98.49%）、m16が88,976、m0–m10は515–552/axis（[shot](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d5ca357ccfd8fcc850bef800c5bcadfde2a153e/docs/research/track_a_b2_p1_shots_20261107/REPORT.md)） |
| 強Full PF 8構成 | Full S2 s1/s2/s4/s8 と Yoshida4 s1/s2/s4/s8。旧SOURCEでは3 PASS／5 STOP。5 STOPの原因は独立経路の数値一致>1e-9（基本stepの非unitarityが長い累乗で増幅）で、PF位相cap超過ではない（[8 arm](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2583ce4957ed44e914ce6571e3e676830439dfa7/docs/research/track_a_b2_p1_strong_pf_20261107/REPORT.md)、[診断](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7dc2113e1325ef87b594b04f2d638fef641de9b7/docs/research/track_a_b2_p1_strong_pf_h4_diagnosis_20261107/REPORT.md)）。HP 80/120桁の新SOURCEで5 arm全90点PASS（[HP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e316e071fe671aa61b31bc89f83bf4380d0e92f/docs/research/track_a_b2_strong_pf_hp_requalification_20261107/REPORT.md)）。**全8 armで条件付きbranch/energy/shot PASS、各18,540 shots（515/axis/round）**。energy誤差はS2s1の5.3e-6からY4s8の3.0e-13 Ha（[旧3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/55fa5f5eb121ee01ae963da7ab6e5bbe9a507173/docs/research/track_a_b2_p1_strong_pf_branch_shots_20261107/REPORT.md)、[8 arm比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e162ef708d7f721cd9c95ae07a833f7301d2b449/docs/research/track_a_b2_p1_hp_branch_shots_20261107/REPORT.md)）。各行は独立候補で、8 arm同時の95%保証ではない |
| 価格pipeline | ① m0/Q1の固定1 trajectoryについて、native→X/Y full wrapper→Qiskit1.3.0 opt1 seed7 の接続がPASS。CX 3,262、RZ 6,025、depth 5,504。native 812命令（[m0価格](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2d21c23879aaae063a796fa95c1433fb3d0b5738/docs/research/track_a_b2_p1_compiled_cost_20261107/REPORT.md)）。② 同一gate列のstreamingは人工18/18、実H4 m0・Q2で全6指標がEXACT_MATCH。boundary fusion（14/18）と分割compile（18/18）は価格が変わるため不採用（[方式](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1bed45be1869f447cfd90cef443877fcaed7af6c/docs/research/track_a_b2_p1_long_cost_method_20261107/REPORT.md)、[m0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f6eea2f28be1f15b6ff5e2954b9c8fb48f3c5a0f/docs/research/track_a_b2_p1_m0_streaming_h4_20261107/REPORT.md)、[Q2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ce2e6736ca1178a4adafc4af344852d958951ea6/docs/research/track_a_b2_p1_q2_streaming_h4_20261107/REPORT.md)）。③ fingerprintとmemoryの省力化を人工回路で確認（26万命令のhashでpeak RSSが3.30→0.42 GiB、[memory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ef1108ebf57ec53e7aaa0a2d2a4f0e40ee68947c/docs/research/b2_memory_pipeline_20261107/REPORT.md)）。**いずれも古典側の技術高速化であり、量子費用の削減ではない** |
| PRのshotを増やす要因（保存値からの読取り） | ① 全roundで固定のr4により、A_m は T_m とともに指数的に小さくなる（下記の静的読解）。N∝ρ_raw^−2 なので、m17のA=.0116が総数を支配する。② λ_R（L_D3で.5833）。③ Hoeffding十分式と均等α・stat .24。ただしこれはPRと強PFに共通で、A=1の強PFでも515/axisになる。④ PF半径、RTE disk、model誤差は主因ではない（m17でもρ_PF,lb≈1、disk 7.5e-8） |
| 静的読解（新しい数値評価ではない） | SOURCE式 A=B_K(τ)^{−qr}、τ=λ_Rδ/r、B_2(τ)≈1+τ² を主要次数で読むと **log A_m ≈ −λ_R² δ T_m / r_m**。報告値（λ_R=.5833、δ=.02、T=2621.44、r=4 → A=.01157）と主要次数で整合する。A_m を一定以上に保つには r_m ∝ λ_R² δ T_m が要り、1shotあたりのRTE event数 q_m r_m は λ_R² T_m² に比例する。一方、決定論tailの仕事量は T_m/δ に比例する。したがって長roundでPRが勝つには、おおよそ λ_R² T_max δ が「決定論tail費/1 event費」より小さい必要がある。**この不等式の定量評価はしていない。GPTの確認対象** |
| 既存PRの費用を下げるために残っている問題 | ① round別のr_m（resolution）、h、αの選択（既知の対処で、Sept opt2とTrack Cで実施済み）。② その後に残るRTE event単価（basis変換が支配的。Track C旧段階でRTE RZの91.44%）。③ 長roundのλ_R自体（分割・drop・表現）。④ PRと強PFを同じcompilerで価格付けすること（B2-P1はQiskit opt1/seed7、Track Cと新Aはcompiler A） |
| 価格取得の技術的成立範囲 | 成立：Q1・Q2の固定trajectoryのwhole compile、streamingの同値性、cacheとmemoryの改善（人工）。未成立：m≥2の実H4価格、m17（1 trajectoryあたり524,288 event・1,048,576 deterministic block append）のwhole compileの実用性、価格MCの統計規則（標本数・seed・paired SE/CI・rare event）、同じcompilerでの強PF価格。全18round総費用は未算出（null） |
| STOP・承認 | 全段階mandatory STOP。価格評価の対象・compiler・MC規則はユーザー判断待ちで、有効な実行承認はない |

### 1.6 3つの達成段階での到達度

| 系列 | (a) 改善機構の成立 | (b) 既存PRに対する改善 | (c) 強Full PFに対するPR＋RPE総費用の優位 |
|---|---|---|---|
| O1-M1 辞書変更（H4-A rank10、LD0） | 同じ物理splitで補正meanが一致することを限定確認。rを増やすと減衰は軽くなる（q64） | 条件付き総CXではDF辞書がPauli比0.504倍（C0）。r増による減衰改善は価格込みでは未改善（r8で必要単価の数千倍） | **不成立**（C0：約1e18倍劣る）。固定r8候補は終了が提案済み（数学的不可能性ではない） |
| C-S V0（M1分岐、O3系） | 局所strict資格は成立 | **負**（局所T 70→約142） | 未評価 |
| O1-M2 core＋残差 | operator-only PASS（規約B）。物理modelはUNASSESSED | 未評価。残差が微小なため、PR化する必要性自体が未成立 | 未評価 |
| B2-P1 | 全18roundの条件付き信号・branch・energy・shot PASS（接続基盤として成立） | 改善機構の検証ではない。費用支配の要因（m17の減衰）を特定 | **未評価**（PR価格はm0の1 trajectoryのみ、強PF価格なし）。shotはPR 7.8M対強PF各18,540 |
| 参考：Track C（C-H4、近い条件・別SOURCE） | Closed P3、D内部BCH、joint isometry、対角融合 | PR内部改善あり（例：旧固定Closed比RZ −6.03%、D-BCH RZ −26.2%） | **H4で不成立**：PR 1.3039e12 対 Full 5.648e11 RZ（2.31倍、CX 1.80倍）。Truncated PFは 3.34e11（A）／3.86e11（保守B）でさらに安い。H2ではPRが最安（[C統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3c81c2677988c7b9491eeee31fbcc0f9b29b078/docs/research/track_c_o3_integrated_20261108/TRACK_C_O3_INTEGRATED_REPORT.md)、[C共同最適化](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md)） |
| 参考：9月 opt2モデル | — | — | 点推定でPRが−4.858%（区間は重なる、opt2・proxy）。Track Cのcompiler A結果と未整合 |

---

## ② 改善候補一覧

区分の記号：【済】既に実施済み、【負】既存の負結果で制限される、【他】他Trackと重複・他Trackの担当、【未】Track Aの範囲で未着手。番号は識別用で、優先順位ではありません。

| 改善候補 | 変更する機構 | 既存の成果 | 負結果・問題点 | 未解決事項 | 関連Track・SOURCE |
|---|---|---|---|---|---|
| **A1 RPE下でのD/R分割（prefix L_D）の再選択** 【未（H4）／他（H2・H6）】 | λ_R（長roundの減衰とevent数）と決定論sweep費の交換 | 資源評価（T≤2）でL_D・q・r・Kの最良が精度で変わる。B2-P1はLD3でλ_R=.583、M1はLD0で減衰が極端。Track CはH2でLD0/1/2、H6でLD3/9/14/17をRPEで選択 | LD0（B3、純RTE）は資源評価で不利。L_Dを増やすとD費とPF誤差が増える。11/09横断調査は「既存familyのparameter再探索」と分類 | H4の全18roundで、L_Dとround別rを同時に選ぶ評価は未実施。λ_R²T_maxδ と単価比の関係（§1.5） | 旧PR-2/PM/PF588、[B2-P1 shot](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d5ca357ccfd8fcc850bef800c5bcadfde2a153e/docs/research/track_a_b2_p1_shots_20261107/REPORT.md)、Track C DF比較 |
| **A2 D／R／drop三分割（K3）** 【未・他（資産はTrack C）】 | 小さい成分をdropし、中間をR、大きい成分をDに置く。random poolのλ・range・basis出現を減らし、drop誤差を払う | H4のTruncated PF（coeff_L1_2で204係数をdrop）がFullとPRより安い。PRは全tailを保持 | H4ではR=0の端点が両方式を支配。drop誤差は長時間で累積。M2の微小残差はoperator-onlyまで | 中間点の総費用（未評価）。初手は保存m17 retentionの1組（横断調査C-K3案） | [K3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e79d80dc9cce2885580e50fe6f1705da8fa41af/docs/research/pr_rpe_cross_track_20261109/IMPROVEMENT_CANDIDATES.md)、Track C truncation `da8b2c3f`・`92495a04` |
| **A3 round別のD/R分割（長roundほどλ_Rを小さく、またはR→drop/D）** 【未】 | 減衰とevent数が支配的なm15–17だけ分割を変え、λ_R²T_m を抑える | Truncatedの採択policyはroundごとにretention・公式・hを切り替える（H4：m0..14はprefix7_I0/S2、m15..17はcoeff_L1_2/Y4）。費用はm16–17が支配（B2-P1 shotの98.5%、Track Cでm17/m16がRZの62.7%/21.9%） | 短roundだけの最適化は効果が小さい（UWC L03の最終round支配）。異なる構成の信号を同じenergyへ接続する必要がある（O4） | PRのround別分割は全Trackで未検証。branch・aliasの整合と、表現を切り替える費用 | O4（[O1–O6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/RESEARCH_OPPORTUNITIES.md)）、Track C DF比較 |
| **A4 同じ物理tailの辞書変更（O1-M1）** 【済・負】 | Pauli collect と DF Z/ZZ の交換でλ_Rと単価を変える | 補正meanの一致、C0のDF/P=0.504（CX）、P3のPauli/DF=0.453（RZ、T=.8） | C0で約1e18倍劣る。r8の価格。11/08に新しい操作差分なし | 価格下界はUNDETERMINED。LD>0の分割での効果 | [C0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50ee55bc30f30838e1ac0704bc94f720914640de/docs/research/track_a_o1_m1_c0_parallel_science_20261106/REPORT.md)、[r8](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0daf01d51c022977a1c0a7d29dde34a1c4c82d7c/docs/research/track_a_o1_m1_r8_price_pilot_20261106/REPORT.md) |
| **A5 元H groundでの表現接続：core＋厳密残差（O1-M2）、Z再配置、ENTRY/PR-aware DF（K4）** 【済（operator）・保留・他（研究C）】 | 線形Z項を安いone-body Dへ移す、または安いcoreに小さい残差を付けて、range・basis価格を変える | M2 operator-only PASS、省略上界6.93e-8 Ha。研究Cの13round比較（own-DF ground）：CDF0/r64 2.32e11、P/r16 8.75e10、E2/r16 7.56e10 RZ | 研究Cはown-DF groundで、元Hの比較ではない。native-cost fitはNO-GO。M2の残差は微小で、core-onlyで足りる可能性 | 元H stateでの再接続、model許容、PF/RTE/N/価格 | [M2調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f891f6e3265e4f0124c598fddfb7775bc1eddd1b/docs/research/track_a_o1_m2_improvement_survey_20261106/REPORT.md)、[研究C三者](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/709813805d41e470676e311cd63a740af23bf33c/docs/research/research_c_three_method_z_relocated_comparison_20261108/REPORT.md) |
| **A6 round別のr_m（resolution）・h・αの共同選択** 【他（Track Cで済）】 | 減衰・event数・単価・shotの交換 | M1でr増により減衰が改善。9月opt2モデル（shots 13,588）。Track CでRZ −6.03%、trial 16,791（Full 14,225） | M1 r8は単価で不利。Track Cの最終でもPRはFullの2.31倍。normalizationを最小にしても総費用が最小になるとは限らない | B2-P1条件（opt1/seed7、sector36）では未実施。有限menuで大域最適ではない | [C共同最適化](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md)、[M06-F](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research_direction_full_opt2.md) |
| **A7 Taylor cutoff K・finite law（Closed P3/P5、return集約）** 【他（Track B/C）】 | 同じfinite meanを少ないnormalization・range・回転で実現 | Closed P3はcanonical比RZ −1.97%（固定policy）／−2.11%（共同最適化後）。G9 Closed P5（synthetic） | P5/K4ではtargetが変わる。S2では単価の低下がNの増加で相殺された | H4でのP5のnative資格。Track AはK2固定 | [Closed P3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e38af3b12200c48c4f1b793638d223baff24e76/docs/research/track_c_closed_p3_20261108/REPORT.md)、[Track B調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a8299889de93c201bf21e727415d629f5e4b9f48/docs/research/track_b_existing_results_survey_20261108/REPORT.md) |
| **A8 sampling lawの変更（IS、Parent、whole-path、CV、LCU）** 【負・他（O2、HOLD）】 | 安いpathへlawを寄せ、likelihood weightでmeanを保つ。推定器を替える | 技術資産。syntheticでの限定的なT利益 | SAMPLING-FREEDOMはNO_DISTINCT_FREEDOM。S2は相殺。SCV224/448で学習費の回収なし。A2 LCUで測定noiseが減らない。非同値性選別で推薦0 | 安価なconditional summaryの具体化 | [O2統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d28164b5d51e0ca5a0ade74ded8ff176b443dab8/docs/research/o2_research_results_integration_20261108/O2_RESEARCH_SUMMARY.md)、[非同値性選別](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1bf88b251fd09caa0a227972ee0f1b56c5888e6a/docs/research/o2_distinct_mechanism_screening_20261108/REPORT.md) |
| **A9 角度のcoarse＋residual分割（C-S）** 【負】 | 高精度回転を安いcoarseとランダムresidualに分ける | 同位置splice接口、V0の局所資格 | Track C pilotで range爆増（log 386）。V0で局所T 70→約142 | 別のcoarse・精度・slot、V1 full-wrapper | [V0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7b5765453cf29532d8c8280d5aa581a87eaaf464/docs/research/cs_v0_repair_results_20261108/REPORT.md) |
| **A10 D側のPF構成（次数、D内部BCH、外側5段族、次補正の制御K6）** 【他（Track C）・一部負】 | PF位相の余裕を増やしてhを大きくし、D呼出しを減らす | 強PF 8 armのP1資格（Y4s1のPF位相3.2e-6）。D内部BCH相殺でRZ −26.2%、CX −30.7%（登録r固定） | D-only Y4、混合2MN、全体partial Y4（負のR時間でnormalization増）、C2-A STOP。5段族はa=1/4で半刻みS2二回と同値 | K6の成立性は未確立。現状はRTEが支配的なので、PF精度改善だけでは総費用に効かない可能性 | [C統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3c81c2677988c7b9491eeee31fbcc0f9b29b078/docs/research/track_c_o3_integrated_20261108/TRACK_C_O3_INTEGRATED_REPORT.md)、[K6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e79d80dc9cce2885580e50fe6f1705da8fa41af/docs/research/pr_rpe_cross_track_20261109/IMPROVEMENT_CANDIDATES.md) |
| **A11 RTE eventの単価（basis共有、joint completion、control・融合、K1）** 【他（Track C/D）】 | 往復するbasis変換・controlの費用を減らす | joint isometryでRZ −3.355%。融合でRZ −18.2%（大半は既知control）。PR固有の融合は約0.135% | always-partialで悪化、depth増 | 異basis間のrelative Gaussian橋（K1） | Track C、[Track D C1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/141fa7624d74aa5deacbd752d09df516cad9206f/docs/research/track_d_o5_c1_head_tail_h4_q4_20261108/REPORT.md) |
| **A12 Reference干渉によるcontrolの削減（K2）** 【他（新A／Track D）。統合しない】 | evolutionのcontrol負担を、追加registerと干渉の費用に交換 | 新AはB′比CX≥4.68%・depth≥41.0%減、RZ≥0.15%増。V6-P（11/09）はB1仮定下で条件付きの総RZ優位。Track Dは13round modelでCX −11.2%、depth −54.7% | RZ増、追加8 qubit、未資格項目多数 | 最新のClosed＋融合への接続 | [V6-P](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f672ce6884bd49231366875a9e24a539fcaec7a/docs/research/pr_reference_v6p_results_20261109/REPORT.md)、[D q64](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fe361b206a6572142775ac01972545d312e5fb14/docs/research/track_d_c8_1_q64_20261108/COMPLETION_RESULT.md) |
| **A13 有限confidence会計（Hoeffding以外の界、K5）** 【未】 | 同じlaw・回路のまま十分Nを減らす | Track Bのmoment・range会計。旧短roundの二項検証 | O2 S2でM2・rangeの増加により相殺。±1 outcomeではHoeffdingの余地は限られる可能性 | 最新Closed weightへの適用。PRとFullの両方に効く可能性 | [K5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e79d80dc9cce2885580e50fe6f1705da8fa41af/docs/research/pr_rpe_cross_track_20261109/IMPROVEMENT_CANDIDATES.md) |

補足：
- 候補の分類（§1.5の静的読解に基づく）
  - λ_R を下げる候補：A1、A2、A3、A5。
  - event単価を下げる候補：A7、A9、A11、A12。
  - T_max や N の側を変える候補：A13、および推定器familyの変更（O6。PR＋RPEの範囲外としてユーザー判断事項）。
- shot数と1shot費用のトレードオフの評価手段（候補ではない）：M1の必要単価screen（`o1_m1_cost_feasibility.account_saved_c0_signal` / `optimistic_unit_price_ceiling`）とTrack Cの共同選択が既にあります。B2-P1条件へ適用すると保存値の新しい科学再解析になり、承認が必要です。

---

## ③ 研究の重複と再利用可能な資産

### 3.1 重複を避けられる研究（既に他所で実施済み）

| Track Aで考え得る作業 | 実施済みの場所と状態 |
|---|---|
| B2のround別r・h・αの最適化と、shot爆発の回避 | Track C（C-H4、Closed P3、共同選択。trial 16,791）、9月のopt2モデル |
| H4・同じRPE条件でのPR対強Full対Truncatedの総費用比較 | Track C DF比較（H2/H4/H6）。H4はTruncated＜Full＜PR |
| D側の高次PF・BCH・外側構成 | Track C（BCH相殺は成功、Y4/2MN/processingは負） |
| C-S（coarse＋residual） | Track C pilot（range爆増）、SERVER1 V0（T増） |
| sampling・推定器の変更 | O2（HOLD）、Track B（G6–G10、Parent） |
| Reference干渉 | 新A（CT、B′、V6-P）、Track D（C1、q64） |
| M1のadapter・selector・matched接続 | P3 Stage1–4（11/02）と O1-M1（11/06）。11/08に「新しい差分なし」と判定 |
| 強PF 8 armのP1資格・branch・shot | B2-P1で完了（HP再資格を含む）。価格だけが残る |

### 3.2 再利用可能な実装（科学条件と制約）

| 資産（固定commit） | 機能 | 元の科学条件・制約 |
|---|---|---|
| `src/trotterlib/b2_rpe_connection.py`（`prepare_b2`、`B2RPERound`、`strong_baseline_schedule`、[66a41ad](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/docs/research/track_a_b2_rpe_implementation_20261106/REPORT.md)） | B2のprefix・event・native request、強PFのschedule | DF12・L_D3・P1。219件の人工PASSは実H4の証拠ではない |
| `b2_p1_signal_validation.py`、`b2_p1_branch_validation.py`、`b2_p1_shot_accounting.py`（`ea2c091`/`d424bb8`/`0d5ca35`） | 元H stateの5信号、既存decoderへの接続、実半径からのshot会計 | P1の予算・allowance・prior。他の条件では再資格が必要 |
| `b2_p1_strong_pf_validation.py`、`b2_strong_pf_precision.py`、`b2_strong_pf_hp_requalification.py`、`b2_p1_hp_branch_shots.py`（`2583ce4`/`75cb38e`/`3e316e0`/`e162ef7`） | 強PF（S2/Y4×substeps）の信号、HP 80/120桁経路、branch・shot | HP経路の結果は、実native emitterとcompiled gate列の一致の証明ではない |
| `b2_p1_compiled_cost.py`、`b2_streamed_native_cost.py`、`streamed_circuit_fingerprint.py`、memory-efficient `MatchedNoPrepCostProvider`（`2d21c23`/`1bed45b`/`ef1108e`） | full no-prep価格、同一gate列のstreaming、省メモリfingerprint | Qiskit1.3.0 opt1 seed7、all-to-all。m17の実用性は未確立。fusion・分割compileはwholeと非同値 |
| `src/trotterlib/o1_m1_resource_bridge.py`（`account_radius_round`、`radius_phase_diagnostics`、`aggregate_registered_rounds`、`build_strong_df_baseline`）、`src/trottertracks/o1_m1_parallel_price.py`、`src/trotterlib/o1_m1_cost_feasibility.py`（`50ee55b`/`c64007e`） | 実半径の会計、並列価格MC、必要単価screen | M1のpreparation（T_eff/V_square、LD0）は、B2の準備としてそのまま使えない |
| `src/trottertracks/o1_m2_hermitian_b.py`、`o1_m2_core_only_bound.py`、`o1_m2_core_only_qualification.py`（`68ef4fe`/`c75d3cf`/`10902e8`） | 完全残差のmapper、CAR照合、省略上界、予算helper | update005（own-DF由来）。物理modelはUNASSESSED |
| `src/trottertracks/algorithm_codesign/same_position_residual.py`（`c077de7`） | coarse/residualの同位置splice | 局所sliceのみ。full IRとの対応はない |
| 既存RPE部品 `rpe_four_round_phase_validation.reconstruct_rpe_phase`、`rpe_resource_accounting`、`rte.finite_rte_attenuation`、`finite_rte_signal_validation._hoeffding_shots` | decoder、horizon、減衰、shot | 連続した2の冪のround列を要求する。Hoeffdingは十分条件 |
| Track C：`track_c_closed_p3*`、`track_c_joint_isometry`、`track_c_diagonal_fusion`、`track_c_closed_p3_joint_opt*`、`track_c_bottleneck.rigorous_shots`、`track_c_df_truncation*`、compiler A | Closed law、回路の共有・融合、共同選択、truncation、modular価格 | C-H4条件（DF12生成順・LD3/218成分・sector70・compiler A）。B2-P1と同一入力かは未照合 |
| Track D：`c1_h4_strong_baseline.StrongBuilder`、`track_d_head_tail_h4`、`track_d_native_resources` | relative Gaussian、head–tail接口、native stream計数 | number-CDF8/LD2/K0/P0、別条件 |

**承認の流用について**：上記資産の再利用は、各研究の実行承認・grant・STOP状態をTrack Aへ移すことを意味しません。Track Cの価格やNをB2-P1の結果として扱うこともしません。

---

## ④ 次の研究判断に必要な事項

### 4.1 既存の証拠だけで判断できること
- 元Track A内には、(c)「強Full PFに対するPR＋RPE総費用の優位」を示す結果はありません。近い条件のTrack C（H4）でも不成立で、PRはRZで2.31倍、Truncated PFはさらに安い結果です。H2ではPRが最安です（Track C）。
- B2-P1の過剰なshotは、全round固定r4のもとで長roundが減衰することだけで生じています。初期roundのshotは強PFと同じです。これはround別のr選択（既知手法）で緩和できます。その後の課題がevent単価であることはTrack Cが示しています。
- M1 C0の負結果は、辞書の違いよりLD0の大きいλ_Rが支配していると読めます（辞書間の差は約2倍、対強PFは約1e18倍）。辞書を替える方向だけを追う根拠は弱い状態です。
- M2の入力（update005）では残差が極めて小さく（≤6.93e-8 Ha）、「PRで補償する」構図はこのH4では成り立ちにくいと読めます。
- 固定T（≤2）の資源評価で得たPR優位を、RPE（T_max=2621）へ移すことはできません。
- C-S、IS/CV/LCU、辞書変更の追加実装を正当化する新しい情報はありません。

### 4.2 追加の技術調査で判断できること（新しい科学計算なし）
1. B2-P1条件とTrack C C-H4条件の差を静的に照合する。対象は入力hash、sector36と70、成分数（324と218）、予算・α規約、compiler（opt1/seed7とA）。Track Cの強Full価格とClosed PRの結果を、Track Aの判断材料としてどこまで参照できるかが決まります。
2. 9月opt2モデル（PRが−4.858%）とTrack C compiler A（PRが+131%）の不一致を、資料照合で切り分ける。候補はopt2の微小角省略、強対照の差、proxy外挿。
3. 交差調査 `2e79d80d` の「5 STOP arm未再資格」という記述と、HP再資格（`3e316e07`・`e162ef70`）の関係を確認し、注記する（文書修正は別途の指示が必要）。
4. compiler Aのmodular価格を、B2のnative経路（m17）に適用できるかをコードで静的に読む。whole compileの代替となるかを確かめる作業です。
5. A2（K3）とA5（K4）の入力適合を静的に照合する。前者はLD3のDとTruncated retentionが衝突するか、後者はupdate005とH_physのbindです。

### 4.3 新しい科学計算が必要なこと（それぞれ明示の実行承認が必要）
- B2-P1条件での強PF 8 armとB2の全round／X/Y価格、および総RPE費用（または、Track C pipelineで代替するという判断）。
- 必要単価screenを保存shotへ適用すること（保存値の再解析）。
- round別r_mを入れたB2の信号の再資格とshotの再会計。
- A1（L_Dの再選択）、A2（三分割）、A3（round別の分割）の信号・誤差・価格・N。
- M2（core-onlyとPR）のPF/RTE/N/価格。
- 減衰とevent数の関係（λ_R²T_maxδ と単価比）を、実際の価格で検証すること。

### 4.4 GPTによる科学的判断・レビューが必要なこと
1. **Track A（表現・分割）を独立に続けるか**、それとも近い条件で費用改善が進んでいるTrack C pipeline上の「分割変数」として扱うか（研究系列・SOURCEは統合しない前提で）。
2. 主資源指標（RZ・CX・T）、compiler profileの統一、強対照の範囲（FullとTruncatedの両方）。
3. §1.5の静的読解（log A ≈ −λ_R²δT/r、event数 ∝ λ_R²T²）の妥当性と、PRが長roundで有利になり得る条件の理論的な見通し。
4. A1・A2・A3・A5のうち、UWC、Truncated/hybrid PF、RC-DF等の既知法に対して追加の性質を持ち得るものはどれか（新規性の判断）。
5. M2の物理model許容とH_physの扱い。保証の水準（engineering allowanceか厳密か）。
6. 推定器familyの変更（O6）を研究対象に含めるか。

### 4.5 既存の負結果・STOPにより、再開を推奨しにくいこと
- M1の固定r4/r8候補での価格MCの拡大、C0格子の拡張、同じ辞書比較の繰返し。
- 同じ登録政策でのC-S（Track C pilot、V0）。
- O2のIS/Parent/CV/LCUの同じ機構の再実装（HOLD）。PS48/singleton、P-A/P-C/P-D/R3、PR-3外挿、BF/BM。
- D-only Y4、混合2MN、固定processor（C-P）、always-partial。
- 研究Cのnative-cost fit（TECHNICAL_NO_GO）と、汎用のrank/fit grid。
- 旧opt2価格の流用。LD0（純RTE・B3）方向（正式なSTOPではないが、資源評価で不利）。
- 強PF 5 armの旧STOP（旧SOURCEの記録として保持。HP結果で書き換えない）。

### 4.6 静的読解で分かる機構と、数値実験が必要な事項の分離

| 静的読解で確認できること | 数値実験が必要なこと |
|---|---|
| 減衰の式（A=B_K(τ)^{−qr}）、主要次数のスケーリング、round別rで減衰を抑えられること | その結果の総費用（event単価との交換） |
| 強PFはA=1なので、shotはHoeffding下限（515/axis）で決まること | 強PFの全round価格 |
| L_Dを増やすとλ_Rは下がり、D費は増えるという向き | 最適なL_D、およびround別分割の利益 |
| Truncated（R=0）とPR（drop=0）が三分割の両端であること | 中間点がどちらかに支配されるかどうか |

---

## 5. 主要な固定commit・REPORT

- 旧Track A横断資料：[PROJECT_MAP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4675730e7e40d25ba3618334a4a0e28b4bd2f145/PROJECT_MAP.md)、[共通ノート](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4675730e7e40d25ba3618334a4a0e28b4bd2f145/docs/research/track_a_shared_research_status.md)、[判断索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4675730e7e40d25ba3618334a4a0e28b4bd2f145/docs/research/track_a_scientific_review_decisions.md)（`4675730e`、10/14時点）、[11/07 Track A現在地と研究Cの関係](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2e3df45043e874a5a182d856c21c9d96b4073d3/docs/research/track_a_research_c_integration_assessment_20261107.md)
- O1：[M1改善調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8820c41bc5beb32208447fe78fee6854088cbcd5/docs/research/track_a_o1_m1_improvement_survey_20261106/REPORT.md)、[M2 core-only準備](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10902e8835885dd99e3def965015961d0df2db77/docs/research/track_a_o1_m2_core_only_qualification_preparation_20261106/REPORT.md)、ほかは§1.3・§1.4の各リンク
- B2-P1：[総RPE接続調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e17f4d39ec3733c5e1f58eda6871d69b06a651c/docs/research/track_a_b2_total_rpe_connection_20261106/REPORT.md)、[減衰screen](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e69a51de70651ced673c22d2b215683a9f30059b/docs/research/track_a_b2_rpe_attenuation_20261106/REPORT.md)、[数値診断（人工）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e217d145bd99c2b34302017390ecb57987c06b90/docs/research/track_a_b2_p1_strong_pf_numerical_diagnosis_20261107/REPORT.md)、[精度adapter](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/75cb38e686ad8b82d00d787531d767e816e1331d/docs/research/track_a_b2_strong_pf_precision_20261107/REPORT.md)、[大規模人工memory](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c6f3f07914feca4f1233fd54951212aeffee001d/docs/research/large_artificial_memory_20261107/REPORT.md)、ほかは§1.5
- 横断：[11/09 横断REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e79d80dc9cce2885580e50fe6f1705da8fa41af/docs/research/pr_rpe_cross_track_20261109/REPORT.md)、[最小検証計画K1–K6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e79d80dc9cce2885580e50fe6f1705da8fa41af/docs/research/pr_rpe_cross_track_20261109/MINIMUM_VALIDATION_PLANS.md)

## 6. 確認できなかった資料・UNKNOWN

- 他SERVERに未公開のTrack A作業、別チャットで進行中の作業：UNKNOWN（公開remoteでは、11/08 02:31以降にTrack A関連の更新なし）。
- private state.npz、HDF5、配列、raw：今回は読んでいない（静的調査では不要）。保存metadataの記載を引用しただけです。
- B2-P1とTrack C C-H4が同一入力かどうか：UNKNOWN（成分数の数え方とsector規約が異なる。横断調査も「source・state規約・予算・価格が異なる」と記載）。
- 9月opt2モデルの結果が、opt2の微小角問題の影響を受けたか：UNKNOWN（再評価なし）。
- M1 r8終了提案をユーザーが正式に採択したか：UNKNOWN。
- 11/09横断調査の「5 STOP arm未再資格」：HP結果との食い違いを確認（§要点6）。原因（参照範囲の違いか）は未確認。
- C-N/C-P/Multilevel（SERVER1系列の後続順位）の着手状況：公開branchでは見当たらない。Track C初期のC-N/C-Pとの関係も未整理。
- 文献（UWC、RC-DF等）の原著は今回再読していません。既存の文献調査の記載を引用しただけです。

## 7. 実施内容・変更ファイル・検証

- 実施：固定commitの資料とSOURCEを静的に読み（`git show`、`git cat-file`、`git ls-remote`）、整理し、本報告を作成しました。科学計算、保存値の再集計、build/compile、test実行、module import、branch操作はいずれも0件です。
- 変更ファイル：
  - `.agent/docs/review.md`（repository外の作業報告ファイル）（本報告で上書き。前回の10/17報告を置換）。
  - Claude Codeのauto-memoryにreference 1件（PRTのTrack名・O番号の対応と横断調査の入口）を追加。
  - PRT repository、他のworktree、共有環境は無変更です。
- 検証：本報告の72リンク（固定commit＋path）がすべてGit objectとして存在することを確認しました。数値はすべて原REPORTからの転記です。新しく算出したものではありません。例外は、明示した引用2値の比（約420倍）と、§1.5の主要次数での式の読み合わせだけです。

## 8. 次の作業（mandatory STOP）

- **ユーザー**：本報告をGPTへ渡し、元Track Aを継続する範囲を判断する。
- **GPT**：§4.4の科学判断を行う（Track A継続とTrack C利用の関係、主資源とcompiler、λ_R²T問題の見通し、A1/A2/A3/A5の選択と既知法との差、M2の扱い）。あわせて、詳細レビューが必要かも判断する。
- **ユーザー**：候補を選んだ場合、§4.3のうち実施する新しい科学計算の対象・条件・範囲を明示的に承認する。§4.2の技術調査だけを先に行う場合は、その旨を指示する。
- **Claude Code／Codex**：上記の指示と承認があるまで、実装・科学計算・文書修正には自動で進まない。今回のmandatory STOPを維持する。
