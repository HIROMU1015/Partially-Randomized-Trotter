# D1/D2 保存値の最終統合分析とGPT引き継ぎ

2026-10-13 JST。**D2_FINAL_SAVED_INTEGRATION_COMPLETE**。
利用者の指示に従い、全方式の感度・quantum Pareto、r=1/2選択、相殺深さの構造を一束で整理した。
新science、219,303 pair再評価、event IRの解析・生成、回路生成、compile、合成、行列、sampling、trajectory、LP、DF/NPZ/GPUは0。原eventのbytes/hash照合は来歴保護として行い、意味内容をparseしてpairを作る処理は行わない。
旧D1/D2分類・source・contract・authorization・marker・STOPは不変。GPT科学レビューは未開始、mandatory STOP。

## 固定証拠と比較条件

- D1 science source `874ca12d347f8f35e5d8d64f65f24abc4a2df523`、D1 result `fc96571881f301435a655f057709df4d6c1678e0`。
- D1保存分析 `2dcd56737be35526f6436a52835f59d817adf30c`、key機構・D2 source `28fc3ca17ec51bb7656a498e3eea9b72966eb3fa`。
- D2 execution A `9e753a0d807df683c86f7742ed0c14ddcd4e58ba`、D2 result `d1c59b28f4370cc7be65cff60d1ec877fd1491f5`（今回のbranch基点）。
- [D1 GPT科学レビュー原本](../../research/track_b_d1_results_and_resource_mechanisms_scientific_review_20261013.md)、[key機構分析](d2_key_mechanisms_and_boundary_preparation_20261013.md)、[D2実行/監査](d2_boundary_results_and_gpt_handoff_20261013.md)。

対象は3-system-qubit explicit synthetic involution、p=(1/5,3/10,1/2)、総時間τ=5/7、σ=+、state=000、P5、r=1,2。Q0=Z0、Q1=V1†Z1V1、Q2=V2†Z2V2、V1=R_XX01(π/4)、V2=R_XX12(π/4)R_ZZ01(π/4)、R_P(θ)=exp(−iθP/2)。分子geometry/basis/DF rank/split L_Dは適用外。同じ既知I1 providerで、独立validationではない。

各axis ε=1/200、α=49/24000、各row resource failure=1/12000、probability160/root256 bit、primitive strict error10⁻⁶。旧policy・実sequence価格・N/biasを保持し、保存値の再最適化・新しい結果分類はしない。

比較に使用した費用は
\[
G_{i,k}(h_k)=A_{i,k}+B_i h_k,\qquad B_i=2N_iZ_i^{r_i}.
\]
T/CX/1Qそれぞれに独立な共通準備・読出し単価h_k≥0を考える。Tは1Qに含まれ、合算scoreを作らない。実際の価格vectorにはh_1Q≥h_T等の物理的制約があり得る。以下は形式的な非負領域で、未取得の実価格を設定したものではない。classical online費用は未知でquantum Paretoから除外した。

## 全六方式の最小費用領域

以下はD2 r=2相殺後。数値は表示用の丸めで、exact rationalの交点とtieは[感度JSON](../../../artifacts/track_b_d2_final_saved_integration/2026-10-13/v1/sensitivity_D2_r2_CX_v1.json)等が正本。

| 指標 | hの領域 | 最小方式 |
|---|---|---|
| T | 全h_T≥0 | closed P5 |
| CX | 0≤h_CX<6.332292668 | CTS |
| CX | 6.332292668<h_CX<6.637184831 | partial-return |
| CX | h_CX>6.637184831 | closed P5 |
| 1Q | 0≤h_1Q<6320.486372 | closed P3 |
| 1Q | h_1Q>6320.486372 | closed P5 |

境界では隣接する方式が同率。**CXにはpartial-returnが最小となる領域が生じた。** 旧D1 r=2加法モデルではCTS→P5の切替が約10.056256で、partialは全方式の最小包絡線に現れなかった。

P5対partialの約6.637185は二方式の交点。CTS対P5の直接交点は約6.335384だが、その点ではpartialのCX費用が小さい。したがってこの直接交点を全方式の切替として扱わない。

旧1Qのclosed P3→P5交点は約6317.245619、相殺後は約6320.486372。Tの六方式内最小はP5のまま。[更新前後の全最小領域](../../../artifacts/track_b_d2_final_saved_integration/2026-10-13/v1/minimum_envelope_updates_v1.json)を保存した。

各r=2について全15 pair×3 metricの交点・半直線・全順位/tieも保存した。JSONの`domain_breakpoints_including_origin`には0が必ず含まれる。実際のpair交点は`actual_pair_nonnegative_crossings`で区別する。

## 全hのquantum Pareto

既知coordinate (G_T(h_T),G_CX(h_CX),G_1Q(h_1Q),workspace)を使用し、三費用がすべて非増加、少なくとも一coordinateがstrict改善する支配を採用した。workspaceは全方式1。未知classical費用を含む全資源Paretoではない。

### D2 r=2の六方式

| 方式 | nondominatedとなる領域 |
|---|---|
| closed P5 | 全非負領域 |
| partial-return | h_CX<6.637184831 |
| closed P3 | h_1Q<6320.486372 |
| CTS | h_CX<6.332292668 |
| ordinary / local full | 全領域で支配される |

これらの境界では、対象metricでtieでも別metricでstrict改善する方式があるため、支配される側はParetoから外れる。最小metricのtieと多資源Paretoを混同しない。

### D1 r=1とD2 r=2を合わせた12候補

| 候補 | nondominatedとなる領域 |
|---|---|
| r=2 closed P5 | 全非負領域 |
| r=2 partial-return | h_CX<6.637184831 |
| r=2 closed P3 | h_1Q<6320.486372 |
| r=2 CTS | h_CX<6.332292668 |
| r=1 closed P5 | h_T<8.328647321 **かつ** h_1Q<11.620966212 |
| r=1 CTS | h_CX<0.942514350 **かつ** (h_T<10.928976256 **または** h_1Q<32.209779832) |
| その他6候補 | 全領域で支配される |

h=0のfrontはこの六候補で、旧joint D1のfront集合と同じ。ただし候補ごとの費用領域は更新された。

完全な[六方式の15支配領域](../../../artifacts/track_b_d2_final_saved_integration/2026-10-13/v1/pareto_D2_r2_v1.json)と[12候補の67支配領域](../../../artifacts/track_b_d2_final_saved_integration/2026-10-13/v1/pareto_D1r1_D2r2_joint12_v1.json)、さらに[簡約した同値な領域](../../../artifacts/track_b_d2_final_saved_integration/2026-10-13/v1/simplified_pareto_domains_v1.json)を保存した。元の各領域が保持領域へ包含され、unionが変わらないことを82件のexact interval包含で確認した。full equalityの例外点も保持している。

## 共通回路規則によるr=1/2の比較

r=1にはstep境界がない。D1 r=1のstep内exact pass済み保存費用と、D2 r=2の共通境界pass後の費用を比べることで、どちらも同じfull-trial隣接inverse規則を使用する。

一方、有限平均はM_r=[P5(−iτR/r)]^rでrごとに異なる。r=1→2ではTaylor/native bias・統計残余幅・N・角度価格も異なる。以下を純粋なrepresentation改善や、時間分割だけの因果効果とはしない。

### 全12候補のmetric最小

三指標とも上のD2 r=2の最小領域と同じで、**全領域で最小候補のrは2**。

Tについて、旧joint D1はh_T<約0.586820でr=1 P5、より大きいh_Tでr=2 P5だった。境界相殺後はr=2 P5がh=0から最小。r=2 P5のT切片はr=1 P5より約0.200632%低い。

ただしr=1 P5とCTSは多資源Paretoに残る。単一coordinateの最小候補と、他coordinateとの交換を含む候補を区別する。

### 同じ方式内でのr選択

以下は各metricを個別に最小化するときのr=1/2二候補の境界。有限集合内の比較で、r≥3やprecision最適化は行っていない。

| 方式 | Tを選ぶr | 1Qを選ぶr |
|---|---|---|
| ordinary | r=2、全h_T≥0 | r=2、全h_1Q≥0 |
| partial | h_T<12.749596でr=1、それ以上でr=2 | h_1Q<3.494282でr=1、それ以上でr=2 |
| closed P3 | r=2、全h_T≥0 | r=2、全h_1Q≥0 |
| local full | r=2、全h_T≥0 | h_1Q<7.648871でr=1、それ以上でr=2 |
| closed P5 | r=2、全h_T≥0 | h_1Q<11.620966でr=1、それ以上でr=2 |
| CTS | h_T<10.928976でr=1、それ以上でr=2 | h_1Q<32.209780でr=1、それ以上でr=2 |

表の交点では同率。CXでは全六方式についてr=2が全非負領域で低費用。全[18 exact同方式比較](../../../artifacts/track_b_d2_final_saved_integration/2026-10-13/v1/same_arm_r1_vs_r2_all18_relations_v1.json)と、旧D1 r=2を用いた境界との対照を保存した。旧90 factor分解・誤差係数の帰属を再実行していない。

## 相殺深さから説明できることと限界

深さkはk個のinverse gate pairが除去され、2k個のliteral IR tokenが消えることを表す。Tは1Qに含まれるため、tokenとgate-resourceの計数も重複させない。zero-fillを除いた条件付き深さ分布と、全trialに対する無条件分布を分けた。

| 方式 | nonzero pair確率 | nonzero条件付き相殺確率 | 条件付き平均除去IR token数 | 保存されたpositive深さ |
|---|---:|---:|---:|---|
| ordinary |1|32.0639%|5.902849|1,7,10|
| partial |1|32.7467%|6.028543|1,7,10|
| closed P3 |1|32.6866%|6.015534|1,7,10|
| local full |0.913836507|32.6961%|6.017783|1,7,10|
| closed P5 |1|32.6961%|6.017783|1,7,10|
| CTS |1|7.1957%|0.658341|1,2,4,6|

local fullの無条件相殺確率は約29.8789%であり、上の条件付き32.6961%と混同しない。

静的sourceではV1は7 tokens（T1/CX2/1Q5）、V2は10 tokens（T2/CX4/1Q6）のprovider基底列で、境界V†Vはexact隣接inverseに適合する。保存深さの無条件mass P_kに対して、次の**構造モデルの整合確認**を行った。

\[
S_T=2P_7+4P_{10},\quad S_{CX}=4P_7+8P_{10},\quad
S_{1Q}=2P_1+10P_7+12P_{10}.
\]

CTS以外の五方式すべてで保存されたaggregate削減量とexact一致し、S_CX+S_1Q=2ΣkP_kも一致した。これはV1/V2のbasis列相殺と整合する説明である。

**depthごとの価格を実測した証拠や、各pairが必ずこのgate列を除去した証拠とはしない。** 保存物にはdepth×削減gate/costのjoint表、pair identity、除去gate traceがない。異なる深さに相関した別のcost配分でも同じaggregateを作れる可能性を、保存値だけから排除していない。per-key節約量、depthごとの完全なT/CX/1Q帰属は未取得として残す。

CTSの保存T削減はexact0で、使用し得る保存R key価格はすべて正。非負費用とpositive lawのもとでは、T-priced Rの除去がなかったことが従う。Clifford部分の簡約と整合するが、depth1/2/4/6のgate種類別・価格別帰属は未確定。新たなpair/IR生成で補っていない。

方式別のexact mass、期待除去token数、aggregate削減量、整合確認は[structure索引](../../../artifacts/track_b_d2_final_saved_integration/2026-10-13/v1/structure_index_v1.json)、不足情報は[帰属の限界](../../../artifacts/track_b_d2_final_saved_integration/2026-10-13/v1/attribution_limits_and_remaining_unknowns_v1.json)を参照する。

## 実装・検証・保護

- [保存値analyzer](../../../scripts/tracks/algorithm_codesign/analyze_d2_final_saved_integration.py)はstdlib Fraction処理のみ。既存のtested affine/支配box関数をimportするが、旧analyzerのmain/科学runnerを呼ばない。
- 新しい[人工scalar tests](../../../tests/tracks/algorithm_codesign/test_d2_final_saved_integration.py)16 PASS。旧D1/D2 tests、1218監査、1890 event/36 sequenceの旧検査は繰り返していない。
- 本analyzerのwall約0.222秒、CPU約0.177秒、peak RSS18,688 KiB。18 scalar rowを共有しsingle processで処理。GPU query0、登録kernel性能や速度倍率を測ったとはしない。
- 入力JSONは固定D1/D2 commitのbytesと一致。現在のcost/budget/分類を変更せず、結果前契約を追記修正していない。入力一覧とSHAは[identity](../../../artifacts/track_b_d2_final_saved_integration/2026-10-13/v1/input_identity_v1.json)、公開資料一覧は[最終manifest](../../../artifacts/track_b_d2_final_saved_integration/2026-10-13/v1/final_evidence_manifest_v1.json)を参照。
- 本分析のPareto・h感度は保存されたモデル内の技術的な関係で、科学的GO/STOP再分類やalgorithm採択ではない。
- D1/D2のsource/result/contract/authorization/marker/STOPを含む対象195ファイルのgit原bytes一致、および親のHEAD/status/index・27 dirty tracked path hash一致を[保護記録](../../../artifacts/track_b_d2_final_saved_integration/2026-10-13/v1/protection_and_scope_receipt_v1.json)へ保存した。親worktreeや他研究のファイルへ書込みを行っていない。

## 次に渡すものとSTOP

これで指示された最後の保存値分析一束を完了する。追加の技術分析を自動で反復しない。

次の担当はGPT。利用者の開始承認後、D1原結果、key価格分析、D2原結果、本資料を一束で科学レビューする。特に、

1. CX最小方式にpartial領域が現れ、T最小のr選択が変わったことを、どの限定claimとして残すか。
2. basis相殺と整合する説明と、未取得のdepth/gate帰属を分けたうえで、一般化・追加検証に情報価値があるか。
3. 別synthesis条件、whole compiler、独立provider、古典online費用のうち、次に必要な最小範囲と論文化の着地点は何か。

別precision/seed、whole compiler、独立provider、online sampling、新r/分子/DF/SP、G10/RA-D0再開は未認可。
**mandatory STOP。GPTレビュー未開始、新規科学計算へ進まない。**
