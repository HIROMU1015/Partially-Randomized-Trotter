# Track A 研究全体の独立科学レビュー（Level 1）

| 項目 | 内容 |
|---|---|
| 作成日 | 2026-10-16（JST） |
| レビュー担当 | Claude（Anthropic）。過去のGPTレビューの結論に拘束されない独立レビュー |
| 依頼 | 利用者提供「Claudeへの研究レビュー依頼：Track Aの研究全体の科学的再評価」（重要度Level 1） |
| 対象 | `HIROMU1015/Partially-Randomized-Trotter` Track A（量子化学シミュレーションにおける部分ランダム化Trotter法のリソース効率と適用条件） |
| レビュー状態 | **完成**（本書が完成版） |
| 新しい科学計算 | **0**。SCF/DF/state生成、signal/reference、trajectory sampling、native build/compile、既存結果の再実行、保存値の再集計・再ランキング、数値最適化のいずれも実施していない |
| GitHub・正式registry | **無変更**。本書は研究方針の提案・推奨であり、正式な採択は利用者が行う |
| 実行承認 | **含まない**。第10・11節の検証はすべて別途承認を要する |

---

## 0. 結論の要約

### 0.1 総合判断

Track Aの現成果は、「部分ランダム化（PR）が量子化学で資源効率的である」ことの実証としては、PR原論文と2026年の関連研究に対して新規性が弱い。

一方で、次の点を示す条件付きbenchmarkとしては、独立した科学的価値がある。

- 強化した決定論的打切り対照（B0/B0+I）の下で、PRの実compile費用上の利益が**どこから生じるか**。
- その利益が**どの精度・時間の範囲に限られるか**。

現データは、この中心主張の大部分を支持する。ただし、利益が残る範囲の両端（緩い精度と厳しい精度）は、比較候補が欠けているため未確定である。

### 0.2 主要な判断

| 論点 | 判断 |
|---|---|
| 費用優位の機構 | ε≧.005では $G_{PR}/G_D\simeq B^2\,\bar C_{PR}/C_D$ が、公開値で相対0.01%以内に成立する（H6 T panel、紙上算術、付録B1）。PRの利益は「中間重みDF断片の不偏な圧縮」から生じる。1外側stepあたり約10〜14断片step分の決定論回路を、約1.3〜1.4断片step相当のrandom eventで置き換えている。PR回路費用の約8割は決定論prefixである |
| 登録domain内の信頼性 | **高い**。点費用の標本SEは0.30〜0.54%で、確認集合からの選択効果は1%未満の規模である。未観測の希少Taylor次数が平均費用へ与える寄与は、構造的に約0.2%以下と見積もられる。allowance factorは適格性を変えない。主なリスクは統計ではなく、**比較domainの構造**にある |
| 中間精度（ε=.01〜.001） | 観測された利益は約50〜56%。決定論prefix格子を最も不利に見積もり、打切りS4を加えても、**概ね32〜40%以上が残る見込み**（条件付き下限、4.4節・付録B7）。中心主張の根拠として最も頑健 |
| 緩い精度（ε=.05） | 格子に依存し、利益は0〜29%の範囲で未確定。H4では打切り対照が勝つ点がある |
| 厳しい精度（ε=.0005） | 未確定。H8では、未登録の打切りS4（p21/q4）がPRとほぼ同費用（差1〜2%）と見積もられる。23.83%の利益は打切りだけで説明できる可能性が高い。H6では打切りS4に対しても約37%の利益が残る見積もり |
| 時間依存性 | 固定δでは、Bを一定に保つためのrandom event数が $T^2$ で増え、決定論部分は $T$ で増える。打切り対照に対するPRの利益は、H6/H8規模では概ね $T\approx17$〜24 a.u.で消える見積もり（紙上概算、付録B9）。energy/QPEの長時間領域へは外挿できない |
| 理論ゲート数モデルA/B | 単独の研究価値は限定的。用途を次の三つに絞れば、本研究の道具として有用：既存compile値への接続検証、希少次数の解析的閉包、非Clifford回転の区別。それ以上の開発は別課題 |
| 今後の方針 | **H5/H7開殻拡張は推奨しない**。最小検証V1（prefix分解能と打切りS4）を最優先とする。限定的な費用モデル接続（V2）で範囲を整理し、既存成果で区切る |
| 着地点 | 中心主張は「公平な打切り対照の下でPRの利益が残る精度・時間の範囲と、その機構（prefix圧縮×normalization×bias headroom）」。PR効率の一般的実証や、理論とcompileの相違は中心にしない |

### 0.3 最小の次の一手

最も少ない追加作業で研究判断を変え得る検証は、**V1**である（第10節）。

- **内容**：H6で、B0+Iと打切りS4の最小適格prefixを整数格子で確定し、決定論側の最良費用を更新する。
- **必要な計算**：新規signal計算と、4軸以下の検証compile。実行承認を要する。
- **得られるもの**：中心主張の数値幅と、範囲の両端の確定。

---

## 1. レビュー対象・範囲・状態

### 1.1 対象と中心課題

依頼書§1.3の三点を主対象とした。すなわち、独立研究としての価値と新規性、今後の方向、中心に据えるべき科学的主張である。あわせて、§3.1〜3.7と§4の判断項目すべてに回答する。

個々の数値結果の正しさの再検証ではなく、研究の科学的価値と将来の方向を中心に置いた。従来方針の維持は前提とせず、継続・縮小・方向転換・区切りの各案を比較した（8.1節）。

### 1.2 実施したこと

- 指定commitのGitHub資料の取得と読解（取得状況は第2節）。
- 理論ゲート数モデル実装の静的読解（一部）。
- PR原論文（v2本文の主要節と付録の該当箇所）と関連文献の調査。
- 公開値（文書・CSVに記載済みの数値）を用いた紙上の数学的推論と、少数の説明用算術。

算術はすべて、出典と前提とともに付録Bに列挙した。これは保存科学データの再集計・再ランキングではなく、新しい資源比較のための数値最適化でもない。正式な結論として採用する場合は、承認下で再計算すること。

### 1.3 実施していないこと

次のことは実施していない。

- 新SCF/DF/stateの生成、signal/referenceの計算、trajectory sampling、native build/compile。
- 既存結果の再実行、保存値の再集計・再ランキング。
- リポジトリのコード実行、GitHubへの書き込み。

### 1.4 出力規則に関する経緯（記録）

本レビューの前回応答（同一会話）は、ツール使用の上限により、ファイル保存の前に打ち切られた。その際、システム側の指示に従い、暫定所見の要約をチャットへ出力した。これは依頼書第0節の出力規則（ファイル未提出時は「レビューを中断しました。」の一文のみ）からの逸脱であり、ここに記録する。

本書はその暫定所見を置き換える。両者に実質的な矛盾はない。本書で追加した内容は次のとおりである。

- 機構の定量検証（4.2節）
- 断片換算の閾値（4.3節）
- 条件付き下限（4.4節）
- 打切りS4の見積もり（4.5節）
- 時間範囲（4.6節）

なお、依頼書が参照する登録済み「研究レビュー依頼テンプレート.md」は、本レビュー環境から参照できなかった。出力規則は、依頼書第0節と第6節の記載に従った。

### 1.5 記号と略記

| 記号 | 意味 |
|---|---|
| $L$ | DF断片数（H6で19、H8で28） |
| $p$ | 決定論prefixの断片数 |
| $q$ | 外側step数（$\delta=T/q$） |
| $r$ | 1外側stepあたりのRTE event数（$R=qr$） |
| $K$ | Taylor打切り次数 |
| $B$ | finite-RTEのnormalization |
| $\lambda_R$ | random tailの1-norm |
| $h_a=\varepsilon/\sqrt2-b_a$ | 軸$a$のheadroom（$b_a$は軸bias） |
| $c_f$ | 1断片・1外側stepあたりの決定論回路費用（RZ） |
| $c_e$ | 1 RTE eventあたりの回路費用（RZ） |
| $p^\ast(\varepsilon)$ | 精度εで適格となる打切り対照の最小prefix |

方式の略記は次のとおり。

| 略記 | 方式 |
|---|---|
| B0 | 打切り |
| B0+I | 打切り＋identity位相の保持 |
| B1 | full PF（S2/S4） |
| B2 | PR（外側S2＋finite-RTE tail） |

### 1.6 依頼書の各要求と本書の対応

| 依頼書 | 本書 |
|---|---|
| §3.1 価値と新規性 | 第0節、第6節 |
| §3.2 主張の妥当性 | 4.7〜4.11節、第5節 |
| §3.3 資源削減の機構 | 4.1〜4.3節 |
| §3.4 理論リソース評価 | 4.12節、7.1節、V2 |
| §3.5 方針と優先順位 | 第7〜9節 |
| §3.6 着地点 | 8.3節 |
| §3.7 反対意見と代替解釈 | 7.2節 |
| §4.1 成果の評価 | 第0節、第5節、6.3節 |
| §4.2 今後の方針 | 第8〜9節、10.2節、11.4節 |
| §4.3 最小限の次の検証 | 10.1節 |
| §5 独立性と権限 | 付録A、1.3節、第11節 |
| §6 成果物の12項目 | 第1〜12節 |

---

## 2. 取得した原資料・固定commit・取得状況

取得状況は「全文」「一部」「未取得」「取得不要」で区別する。リンクやcommit文字列を確認しただけのものは、取得済みとしていない。

### 2.1 リポジトリ資料（取得済み）

| 資料 | commit | 状況 | 用途 |
|---|---|---|---|
| H8 native B0+I全登録候補結果 `docs/research/track_a_h8_native_b0i_full_family_results_20261016.md` | `5f79257f` | 全文 | H8主比較、適格性、P境界、回路費用 |
| H4/H6/H8強化対照の整理 `docs/research/track_a_h4_h6_h8_strengthened_baseline_status_20261016.md` | `5f79257f` | 全文 | 系列間の条件差 |
| 科学レビュー判断索引 `docs/research/track_a_scientific_review_decisions.md` | `5f79257f` | 全文 | R01〜R10の採否 |
| H8 D0保存値再解析結果 `docs/research/track_a_h8_d0_saved_value_result_20261015.md` | `5f79257f` | 全文 | full PF比、PR選択候補、ε=.0005 |
| H8比較CSV `…/track_a_h8_native_b0i_full_family_result_v1/2026-10-16/comparison_P0_v1.csv` | `5f79257f` | 一部（列定義とfactor1・RZ行） | 選択候補、paired SE |
| 同artifactのfile一覧 | `5f79257f` | 全文（file名のみ） | 取得範囲の確認 |
| 正式共通研究ノート `docs/research/track_a_shared_research_status.md` | `5f79257f` | 一部（冒頭〜未解決事項） | 目的とscope |
| 原稿v0.4 `docs/manuscripts/track_a_resource_study_geometry_native_v0_4.md` | `5f79257f` | 一部（Abstract〜§6.2） | 現行の主張候補 |
| `docs/research/先行研究と未解決点.md` | `5f79257f` | 一部（冒頭〜§3.5） | PR論文の理解と既存整理 |
| 先行研究gate `docs/research/pr2_matched_accuracy_prior_art_gate_v1.md` | `5f79257f` | 全文 | 既存の新規性判断 |
| `docs/research/研究目的・研究課題.md` | `5f79257f` | 一部（冒頭〜§1.1） | 目的の変遷 |
| 独立技術監査 `docs/research/track_a_independent_technical_audit_20261015.md` | `5f79257f` | 一部（判定〜人工検証） | 実装の信頼性 |
| H6 native B0+I結果 `docs/research/track_a_h6_native_b0i_full_family_results_20261015.md` | `34efc914` | 全文 | H6主比較、回路費用 |
| H6比較CSV `…/track_a_h6_native_b0i_full_family_result_v1/2026-10-15/comparison_P0_v1.csv` | `34efc914` | 一部（factor1・RZ行） | 選択候補 |
| H6固定δ T依存性結果 `docs/research/track_a_h6_fixed_delta_t_dependence_results_20261015.md` | `043f58ac` | 全文 | bias分解、B、回路費用、希少次数 |
| H4 B0+I地図結果 `docs/research/track_a_h4_b0i_map_results_20261014.md` | `98cdb35c` | 全文 | geometry依存と精度依存 |
| 理論ゲート数モデル資料 `docs/research/track_a_native_df_theoretical_gate_model_20261016.md` | `1ea42fbf` | 全文 | 4.12節 |
| 実装 `src/trotterlib/native_df_gate_model.py` | `1ea42fbf` | 一部（全体構造、次数確率、期待値DP） | 静的確認 |
| R10統合科学レビュー `docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md` | `3d910ee1` | 全文 | 既存判断との比較 |

### 2.2 未取得・取得不要

| 資料 | 状況 | 影響 |
|---|---|---|
| 研究レビュー依頼テンプレート.md（登録済み） | 未取得（参照不能） | 出力規則は依頼書本文に従った |
| H8のP感度JSON、P区間CSV、終了監査、軸別compile指標CSV | 未取得 | 文書記載の値を使用。結論への影響は小さい |
| H6 T依存性のartifact（主表CSV、回路平均表、会計JSON） | 未取得 | 付録B1の前提「同じp/qの決定論回路のRZ数はTに依らない」は未確認 |
| 理論モデルの人工検証receipt、tests、runner | 未取得 | 27件PASSは文書記載に依拠 |
| R09（10/14のH4全面結果後レビュー）原本 | 未取得 | R09の採否は判断索引で把握 |
| H6希少次数解析（`eee1343f`）、s2 pilot結果 | 未取得 | 局所補正0.00734%はR10の記載に依拠 |
| 研究概要・現状.md（212KB）、研究方法・解析手順.md、追加検証設計 | 未取得 | H5/H7の設計はR10の記載で把握 |
| 実行log、handoff、server運用資料 | 取得不要 | 科学判断に不要 |

重要な原資料が欠けたまま、科学的判断を推測で確定した箇所はない。付録B1の前提だけは未取得のH6 T依存性artifactで確認できるので、その確認をV2に含める。

### 2.3 外部文献

| 文献 | 状況 |
|---|---|
| Günther, Witteveen, Schmidhuber, Miller, Christandl, Harrow, *Phase estimation with partially randomized time evolution*, PRX Quantum 7, 020332（2026）; arXiv:2503.05647 | v2 PDFの本文主要節（I〜VIII）と付録の該当箇所を読了。HTML v1も参照 |
| Miller et al., *phase2: Full-state vector simulation of quantum time evolution at scale*, Commun. AI Comput. 1, 5（2026）; arXiv:2504.17881v2 | 本文と補足を読了 |
| Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741（2026） | abstractのみ |
| Hagan, Wiebe, *Composite quantum simulations*, Quantum 7, 1181（2023） | abstract |
| Pocrnic et al., *Composite Qdrift-product formulas for quantum and classical simulations in real and imaginary time*, PRR 6, 013224（2024） | abstract |
| Martyn, Rall, *Halving the cost of quantum algorithms with randomization*, npj QI 11, 47（2025） | abstract |
| Kiss et al., *Early fault-tolerant quantum algorithms in practice*, Quantum 9, 1682（2025） | abstract |
| Microsoft QDK/Chemistryの部分ランダム化builder文書 | 概要 |
| Wan–Berta–Campbell（PRL 129, 030503, 2022）、Campbell（PRL 123, 070503, 2019）、Childs et al.（PRX 11, 011020, 2021）、Jin–Li（2021）、Motta et al.（npj QI 7, 83, 2021）、Oumarou et al.（Quantum 8, 1371, 2024）、Kronenberger–Erakovic–Reiher（Mol. Phys., 2026） | 本レビューでは未読（書誌のみ）。主張の根拠にはしていない |

---

## 3. 現在の研究目的と到達点

### 3.1 研究目的とestimand

研究目的は、数回の再定義を経ている。当初の目標は、full RPE interrogationの総費用モデルだった。その後、PR-2の先行研究gate（2026-09-29）とPM-1/PM-2後のreviewを経て、現在の正本は次のとおりである。

対象は、保存されたDF Hamiltonianと指定stateに対する有限時間complex signal

\[
z(T)=\langle\psi|e^{-iH_{\rm DF}T}|\psi\rangle
\]

である。各方式を同じsignal精度εへ揃え、測定込みのcompiled資源を比較する。これは限定的なresource/applicability studyであり、RQ-Rが主、RQ-P1が補助、RQ-P2（truth-free選抜）は未達である。

主費用は次式で定義される。条件はP=0、主指標はRZ、各軸α=0.025のHoeffding型十分shot数である。

\[
G=\sum_{a\in\{c,s\}}N_a\,\bar C_a,\qquad
N_a=\left\lceil\frac{2B^2\ln 80}{h_a^2}\right\rceil,\qquad
h_a=\frac{\varepsilon}{\sqrt2}-b_a .
\]

### 3.2 到達点

| 系列 | 条件 | 主な結果 |
|---|---|---|
| H4 geometry＋B0+I | legacy DF12、6距離、B0+I prefix 3/4/5/6/9、B2 prefix 3、ordinary control、ε=.005〜.1の302点 | 点最小はB0+Iが397点、B2が1,415点。ε=.05ではB0+Iが0.70/0.80/1.40 Åで勝ち、ε=.005では全距離でB2が勝つ。B0とB0+Iの差は全240軸でRZ等すべて0 |
| H6 native＋B0+I | DF19、prefix 5/10/15、T=.8、確認union8×独立32 | P=0・RZで、全5精度でPRが低い（29.35/49.69/53.27/56.14/50.37%） |
| H8 native＋B0+I | DF28、prefix 7/14/21、確認union6×独立32 | 全5精度でPRが低い（28.77/52.72/52.73/56.12/23.83%）。B0+Iの追加でε=.05の利益は39.48%から28.77%へ縮小 |
| H6固定δのT依存性 | δ=.2、T=.4/.8/1.6、固定5構成 | ε≧.005でPRが約50〜55%低い。厳しいεではS4が有利になるか、PRが不適格になる。主な制限要因はouter-PF bias |
| 理論ゲート数モデルA/B | 明示分解profile、finite-RTE期待値、人工27件PASS | 実分子とQiskitへの接続は未検証 |

H6/H8では、全60条件（5精度×allowance 2種×6指標）で、PR/決定論の点費用比が1未満だった。指標の選び方に対して、結論は安定している。

### 3.3 既存GPTレビューの判断の要点

- **R09（10/14）**：対照を強化した有限時間signal資源の適用領域（claim B）を主とした。件数（A）は補助、別条件のnative H6/H8（C）は構成要素として採択した。
- **R10（10/15）**：研究の継続を決めた。あわせて、H8 D0、native B0+Iの完成、H5→H7の中性ROHF拡張、H6固定δのT比較、ε=.0005の採用を採択した。

本レビューとの一致点・相違点は、付録Aにまとめる。

---

## 4. 重要な数理的・科学的検討

### 4.1 費用比の分解

同じεでPRと決定論方式Dを比べる。整数ceilと軸間の差を無視すると、次が成り立つ（決定論方式は$B=1$）。

\[
\frac{G_{PR}}{G_D}
=\underbrace{\frac{N_{PR}}{N_D}}_{\text{shot比}}\cdot
\underbrace{\frac{\bar C_{PR}}{C_D}}_{\text{回路費用比}},
\qquad
\frac{N_{PR}}{N_D}\simeq B^2\left(\frac{h_D}{h_{PR}}\right)^2 .
\]

この式自体はほぼ恒等式である。科学的な内容は、次の二つの経験事実にある。

1. **biasがほぼ等しい。** T依存性の表では、T=.4でPR p5が3.3328e-4、B0 p15が3.3376e-4、full S2が3.3382e-4である。外側S2の誤差は上位断片の交換子で決まり、tailを打ち切るか、確率的に扱うか、決定論的に扱うかではほとんど変わらない。PR原論文も、部分ランダム化のTrotter誤差を完全決定論の誤差推定で代用している。そして、$\lambda_R$が$\lambda$の大きな割合でない限り、$L_D$を減らしても誤差はあまり変わらないとしている。本研究は、この経験則を有限時間signalの意味で、小分子について確認したことになる。
2. **headroomが十分なら $h_D/h_{PR}\approx1$。** このとき次が成り立つ。

\[
\frac{G_{PR}}{G_D}\simeq B^2\,\frac{\bar C_{PR}}{C_D}.
\]

したがって中間精度までは、PRの利益は「回路圧縮」と「normalizationの罰則 $B^2$」の積だけで決まる。

厳しい精度では、$h\to0$ によって外側S2系の全方式（PR、B0、B0+I、full S2）のshot数が発散する。このとき比較は「PR対決定論」ではなく、「外側S2対S4」という公式の次数の問題に変わる。

### 4.2 T panelでの機構の定量検証（紙上算術）

H6固定δ panelについて、次の公開値を使った（付録B1）。

- PRの1軸平均RZ $\bar C_{PR}$
- 決定論対照の1軸RZ $C_D$。ただしT=.4/1.6のB0 p15は、同じp/qのT=.8におけるB0+I実測値で代用した。
- $B$

| T（q） | $\bar C_{PR}/C_D$ | $B^2$ | 積 | 公開された費用比（ε=.05） |
|---:|---:|---:|---:|---:|
| 0.4（2） | 25,889.78 / 59,013 = 0.43871 | 1.034256 | 0.45374 | 0.453740 |
| 0.8（4） | 51,234.16 / 117,229 = 0.43704 | 1.069686 | 0.46750 | 0.467465 |
| 1.6（8） | 101,586.31 / 233,661 = 0.43476 | 1.144228 | 0.49746 | 0.497423 |

三つのTすべてで、相対0.01%以内で一致した。T=.8・ε=.005でも、headroom比の補正（約0.9995）を入れると0.46728となり、公開値0.467275と一致する。

つまりε≧.005では、compilerの偶然、標本誤差、bias差といった隠れた要因は費用比に寄与していない。Tとともに利益が約55%から約50%へ縮むのは、$B^2$の増加だけで説明できる。

H8のε=.001でも同じ構造が確認できる。

- 回路比は $\bar C_{PR}/C_D=253{,}005.8/618{,}608\approx0.409$ である（B0 p21/q8の費用はB0+Iの値で代用）。
- 公開された総費用比0.4388から逆算したshot比は1.073である。
- この値は、R10の希少次数表の層確率から推定した $\lambda_R\approx1.33$ Haによる $B^2\approx1.073$ と一致する（付録B5・B6）。

### 4.3 回路費用の線形構造と「断片換算」の閾値

H6のB0+I回路費用は、prefix長とqに対して極めて線形である（付録B2）。

\[
C_D(p,q)\approx 797+q\,(269+1{,}922.6\,p)\qquad[\text{RZ/軸}].
\]

4つのqすべてで、傾きは1断片stepあたり1,922.6 RZに厳密に一致する。旧pilotのp5/q1実測値10,670に対し、この式は10,679を与える。

H8も同様に線形で、1断片stepあたり約3,610 RZである。S4を3つのS2 pieceとして扱うと、同じ式がfull S4 p28/q4の実測値1,232,704を0.2%以内で再現する（付録B4）。

この式でPRのprefix部分の費用を求め、T panelのPR平均費用から差し引くと、1 eventあたりの費用は次のとおりとなる（付録B3・B4）。

| 系 | $c_e$ | $c_e/c_f$ | PR回路に占める決定論prefix |
|---|---|---|---|
| H6 | 約670〜680 RZ（3つのTで一貫） | 約0.35 | 79% |
| H8 | 約1,210 RZ（p7への外挿を含む） | 約0.34 | 85% |

同じq・同程度のbiasのもとで、PRが打切り対照に勝つ条件は次のとおりである。ここで$a$は1 stepあたりの固定費用、$\bar C_{PR}^{\rm step}$は1 stepあたりのPR費用を表す。

\[
B^2\,(a+p_{PR}c_f+r c_e)<a+p^\ast(\varepsilon)\,c_f
\;\Longleftrightarrow\;
p^\ast(\varepsilon)-p_{PR}\;\gtrsim\;\frac{r\,c_e+(B^2-1)\,\bar C_{PR}^{\rm step}}{c_f}
\]

H6・T=.8（r=4、$B^2=1.07$）では、右辺は約1.9断片となる。つまり、**打切り対照がPRより約2断片以上多くの断片を必要とする精度では、PRが有利になる**。

この「断片換算の閾値」は、次の三点を一つの式で説明する。

1. H6/H8の中間精度で利益が大きいこと（$p^\ast-p_{PR}\approx10$〜14）。
2. H4の緩い精度で打切り対照が勝つ点があること（ε=.05でのB0+Iの最良prefixは3〜5で、B2のp3との差が閾値の近傍にある）。
3. 費用比が実装の細部（$c_e$）に鈍感であること（event部分はPR回路の15〜21%なので、$c_e$が30%変わっても比は高々約6%しか変わらない）。

これは**検証可能な予測**であり、V1・V2で確かめるべきである。

### 4.4 決定論prefix格子の粗さと分母の感度

登録prefixはH6で5/10/15、H8で7/14/21である。$L/4$刻みであり、H4（12中3/4/5/6/9）より粗い。

H6のB0+I p10は、ε=.05でのみ適格である。したがってε≦.01での真の$p^\ast$は11〜15のどこかにあるが、実測されているのはp15だけである。H8でも、真の$p^\ast$は15〜21のどこかにあるが、実測はp21だけである。より小さい適格prefixがあれば、決定論の分母は縮む。

線形費用式を使い、最も不利な仮定で下限を見積もった（付録B7）。仮定は次のとおりである。

- H6は$p^\ast=11$、H8は$p^\ast=15$とする。
- shot数は、p15/p21と同じかそれ以上とする。
- H8のε=.001では、打切りS4 p15/q4も分母の候補に含める。

| 系・ε | 公開された低減率 | 最も不利な格子での下限見積もり |
|---|---:|---:|
| H6 ε=.01 | 49.7% | 約32% |
| H6 ε=.005 | 53.3% | 約37% |
| H6 ε=.001 | 56.1% | 約40% |
| H8 ε=.01 / .005 | 52.7% / 52.7% | 約34% / 34% |
| H8 ε=.001 | 56.1% | 約37%（打切りS4 p15を含む） |

**中間精度での利益は、格子を最悪に見積もっても、概ね32〜40%以上残る見込みである。**

一方ε=.05では、H6の$p^\ast$は6〜10の範囲にある。$p^\ast=7$なら利益はほぼ0、$p^\ast=8$なら約13%となる（付録B7）。緩い精度の端は未確定である。

注意点が二つある。

- 細かい格子では、打切りbiasと外側S2 biasの符号付き相殺を偶然に選ぶ選択効果が生じうる。その場合、shot数が減って上の仮定が崩れる。V1の事前登録で、扱いを決める必要がある。
- PRも、常に最小の登録prefix（$\lceil L/4\rceil$）を選んでいる。これは登録domainの端である。より小さいprefixで改善する可能性はあるが、$\lambda_R$の増加による$B^2$の増加と相殺する。

### 4.5 厳しい精度：S2系の限界と打切りS4の欠落

ε=.0005では、PRはq8（登録の上端）の外側S2にある。決定論側にはfull S4しかなく、打切りS4は登録されていない。

H8に線形費用式を当てはめると、打切りS4 p21/q4の1回路費用は約927k RZであり、full S4の約0.752倍となる（付録B8）。p21の打切りbiasは小さく、S2 q8でもε=.0005で適格になる程度である。したがって、shot数はfull S4とほぼ同じとみてよい。この場合、総費用は約1.403e14となる。PRの1.4205e14とほぼ同等で、打切りS4のほうが1〜2%安い。**H8のε=.0005における23.83%の利益は、決定論的打切りだけで回収できる可能性が高い。**

H6では同じ見積もりで、打切りS4 p15/q4に対してもPRが約37%安い。厳しい精度での優劣は系に依存し、現データでは確定できない。

PRに外側S4やq16を許した場合の挙動は未知である。4.1節の機構からは、PR-S4は打切りS4に対してprefix圧縮の利益を保つと予想される。ただしYoshida S4は負の時間pieceを含み、tail時間の総和は1 stepあたり約4.4倍になる。その結果、同じr/pieceでは$\ln B$が約6.5倍になるため、この予想は不確かである。

結論として、厳しい精度での「PR不利」も「PR有利」も、現時点では次の二つの性質によるものである。

- **外側S2に限定したPR候補集合**
- **打切りS4を欠く決定論集合**

いずれも、PRに固有の性質とはいえない。

### 4.6 時間依存性と、PRの利益が残る時間範囲

finite-RTEでは$\ln b(\tau)\simeq\tau^2$、$\tau=\lambda_R\delta/r$である。したがって固定δでは、次が成り立つ。

\[
\ln B=R\,\tau^2=\frac{T\,\lambda_R^2\,\delta}{r}.
\]

公開されたBから$\lambda_R$を推定すると、H6（p5より後のtail）で約0.92 Ha、H8（p7より後のtail）で約1.33 Haとなる（付録B5）。この推定値から計算した1 eventの次数2確率は、公開値0.0010518と一致する。

Bを一定に保つには$r\propto T$が必要で、random eventの総数は$R=qr\propto T^2$で増える。一方、決定論部分は$T$でしか増えない。

4.3節の費用式を使い、各Tで$r$を最適化した場合を概算した（付録B9）。打切り対照（H6はp15、H8はp21）に対してPRが安い範囲は次のとおりである。

| 系 | δ | $r$を最適化した場合 | $r=4$に固定した場合 |
|---|---|---|---|
| H6 | 0.2 | $T\lesssim17$ a.u. | $T\approx10$ a.u.まで |
| H8 | 0.1 | $T\lesssim24$ a.u. | $T\approx10$ a.u.まで |

この概算は次の前提に依存しており、検証を要する。
- PRと打切り対照のbiasが同等であること。
- 打切りが長時間でも適格であること（H6 p15の打切り成分はT=.4で6.5e-8で、Tに比例する）。
- 線形費用式が成り立つこと。

この結果が意味することは二つある。

1. 本研究の結果は、短時間・多shotのsignal推定についての結論である。初期FTの浅い回路や、高overlap stateでmax depthを下げる運用（PR原論文の$\xi$）がこれにあたる。
2. エネルギー精度1 mHa級のQPE/RPEでは、数百〜数千a.u.の長時間が必要になる。この領域では、H6/H8規模の小さい系に対して、打切り対照のほうが有利になると予想される。

PR原論文での利益は、決定論tailが巨大な大規模系（$L_D\ll M$）で生じるものであり、この予想と矛盾しない。

### 4.7 推定対象（有限時間signal）とenergy/QPE/RPEの関係

stateが$H_{\rm DF}$の固有状態であれば、$z(T)=e^{-iET}$である（H4の目標値は保存された$e^{-iET}$であり、H6/H8も同様と理解している）。このとき$|\hat z-z|\le\varepsilon$は、概ね$\varepsilon/T$のエネルギー誤差に相当する。T=.8では、ε=.05/.01/.005/.001/.0005が、それぞれ約62/12.5/6.3/1.25/0.63 mHaにあたる。

これは、単一時刻の統計的位相推定（shot雑音律速、$\varepsilon^{-2}$）という運用点である。PR原論文のRPEとは性質がまったく異なる。

| 項目 | 本研究 | PR原論文のRPE |
|---|---|---|
| 1 roundのshot数 | 多い | 少ない（$11+4(M-m)$に$B^2$を掛けた数） |
| 時間 | 短い（T=.8） | 長い |
| 精度の定義 | 有限時間signalの絶対誤差 | energyのRMSEと、各roundの位相margin |

したがって、両者のεは同じ量ではない。比較できるのは、4.1節の式の構造（shot数が$B^2$倍になること、biasが外側公式で決まること）が両regimeで共通であるという点に限られる。

固定Tでの比較にはもう一つ注意が要る。エネルギー精度$\varepsilon_E$を固定するなら、方式ごとに最適なTが異なる。S4はbiasが小さいので長いTを使える。そのため、同一ε・同一Tの表をエネルギー性能の比較として読んではならない。既存のT panelを$\varepsilon_E=\varepsilon/T$の軸で読み直す作業（V4）は、論文での解釈に役立つ。

状態準備費用Pの境界（H8）は、ε=.001で約1.21e6 RZ/shot、ε=.0005で約1.08e5 RZ/shotである。これはPR 1回路費用（約2.53e5 RZ）のそれぞれ約4.8倍と約0.43倍にあたる（付録B12）。

- HF程度の安価なguiding stateなら、P=0の結論はほぼ保たれる。
- 厳密基底状態の準備費用は1回路費用と同程度になりうるので、厳しい精度では結論が反転しうる。

### 4.8 費用指標（RZ）の意味

主指標は、rz/sx/x/cxへtranspile（optimization level 1）した後のRZ命令数である。FT費用（T/Toffoli数）の代理としては、次の三つの不正確さを含む。

1. H、CZ、XX+YYの分解や1-qubit再合成に由来する**Clifford角のRZ**を含みうる。リポジトリ内に、非Clifford回転を区別する処理は見当たらなかった。
2. 汎用1-qubit unitaryが最大3つのRZとして数えられるなど、表現に依存する。
3. FTでの回転合成の精度や、同角回転の一括処理（PR原論文が用いるHamming weight phasingなど）を反映しない。PRのrandom eventは同角の回転が多いので、FTではさらに有利になりうる。

ただし4.3節のとおり、費用比はprefix比と$B^2$が支配しており、event費用の比重は小さい。指標を変えても、中間精度の結論の符号が変わる可能性は低い。6指標すべてでPR<決定論だったことも、これを支持する。

非Clifford回転数の併記は、論文の頑健性を示す低コストの補強として推奨する（V2）。

### 4.9 希少Taylor次数の構造的上界

k次のeventは、k個のproductと1個のrotationからなる。各applicationが最大1組のbasis変換を伴うとする（モデルAと同じ構造）。このとき、次が成り立つ。

\[
\mathbb E[C_{\rm event}]=\bar c_R+\mathbb E[k]\,\bar c_P,
\qquad
\mathbb E[k]=\sum_k k\,p_k\approx 2p_2+4p_4 .
\]

0次だけから推定した平均$\bar c_R$に対する相対的な過大分は、$\mathbb E[k]\,\bar c_P/\bar c_R$である。$\mathbb E[k]$の値は、H6のr4/K2で約0.0021、H8のq8/r4/K4で約0.0011、H8のq4/r4/K4で約0.0044である。

event部分はPR回路の15〜21%なので、trajectory全体への影響は約0.02〜0.07%となる。$\bar c_P/\bar c_R$を悲観的に3としても、約0.2%以下である（付録B10）。この評価の前提は、compile後の費用がblockに対してほぼ加法的であることで、4.3節の線形性がこれを支持している。

この見積もりは、既存のH6 s2 pilotの局所補正0.00734%とも整合する（いずれも十分に小さい）。R10が示した逆転の目安（非ゼロ層の平均がs0平均の33〜74倍）には、構造上到達できない。1つの非ゼロ次数eventが追加するのは、高々数個のbasis sandwichだからである。

**希少次数は、平均費用の比較の結論を左右しない。** この論点はモデルAの期待値で形式的に閉じれば十分であり、追加samplingは不要である。

### 4.10 統計・選択・allowance・compiler同一性

- **標本SE**：H8でのpaired SEは、点費用の0.30〜0.54%である（付録B11）。
- **選択効果**：union6から点最小を選ぶことによる効果は、最大でも約1.3 SE（0.4〜0.7%）の規模である。24〜56%の差に比べて無視できる。
- **allowance**：factor 1と10で適格性は変わらない。$u/h\le.01$ならshot比への影響は2%以下である。
- **compiler同一性**：旧host出力とのbit同一性は未検証である。ただし、compiler hashとprofileは固定されている。H4でB0とB0+Iの差は0であり、4.2節では0.01%の一致が得られた。これらから、系統差は小さいと判断する。

**結論**：登録domain内の点比較について、統計的・数値的な信頼性は高い。これ以上の監査、sampling、allowanceの精緻化は情報価値が低い。

### 4.11 DFモデル・政策差・系の代表性

- H6/H8はtol 1e-8のDFであり、元のSTO-3G Hamiltonianとの差は、本研究の精度に比べて無視できる規模と考える。ただし対象は、あくまで保存$H_{\rm DF}$と定義しておく。
- H4（legacy DF12、ordinary control）とnative H6/H8は、政策も実装も異なる。統制されたsize scalingへpoolしないという既存判断に同意する。
- H6/H8は1.0 Åの単一geometryで、相関は弱い。DF断片のスペクトル（したがって$p^\ast(\varepsilon)$と$\lambda_R$）は、geometryと相関の強さで変わる。4.3節の閾値式を使えば、系を増やさなくても「何が変われば結論が変わるか」を示せる。

### 4.12 理論ゲート数モデルA/Bの位置付け（依頼書§3.4）

| 問い | 判断 |
|---|---|
| 独立した科学的価値があるか | 単独では低い。固定した分解の命令数を数えることは標準的な作業で、Qiskitとの差は実装上の注記に留まる。本研究の道具としての価値はある |
| PRと決定論PFの費用差を説明できるか | 説明の骨格（prefix、event、wrapper、basis変換の寄与）は数えられる。ただし比の大部分は、既存compile値の線形性だけで説明できる（4.3節）。モデルが新たに加えるのは、eventの内部構造（次数、component、basis再利用）である |
| 理論モデルによる候補選択は有用か | screeningには有用。最終比較には、compile値か、affineに較正した値を使う |
| 希少Taylor次数の費用問題をどこまで扱えるか | モデルAの線形期待値は全次数を正確に含むので、この論点を形式的に閉じられる。集約位相RZの幅は1 occurrenceあたり最大1で、回路費用（1e4〜1e6 RZ）に比べて無視できる |
| 期待理論ゲート数と実compile費用の違いをどう解釈するか | opt1による1-qubit併合やCX相殺などの「compiler残差」として扱う。残差がPRとPFで共通なら、比の解釈は変わらない。異なる場合は、その差自体が機構の一部となる |
| モデルA/Bに含めるべき最適化の範囲 | A（最適化なし）とB（同一basisの相殺）で十分。隣接する異なるbasis変換の融合（Givens網の合成）だけは、感度解析として追加する価値がある |
| 必須の検証か、別課題か | 既存compile値への接続（V2）は推奨するが、必須ではない。一般identity位相の厳密期待値、history依存のbasis plan、routing、多数eventでのDP性能は別課題 |

技術的な留意点は次の三つである。

- **Clifford角**：モデルのRZ数も、Hに含まれる2つのRZ(π/2)などのClifford角を含む。角度の分類を加えるのは低コストなので、推奨する。
- **絶対値の意味**：明示XX+YY分解（12 RZ）が効率最適でないことは、資料自身が認めている。したがって絶対値に意味はなく、意味を持つのはcompile値とのaffineな関係だけである。
- **完成度の認定**：人工検証27件のPASSは、明示profileの内部一貫性を支持する。しかし、実分子への接続、eventの独立同分布の仮定、τとnormalizationの対応はいずれも未確認である。現段階では「技術基盤として整っている」に留める。科学的比較指標として採用するかどうかは、V2の後に判断する。

---

## 5. 結果を支持する証拠と不確かさ

| 主張候補 | 支持する証拠 | 主な不確かさ | 信頼度 |
|---|---|---|---|
| 登録domain内・P=0で、PRの点費用は登録決定論最良より低い（H6/H8の全5精度） | 独立32 trajectoryの確認、SE 0.3〜0.5%、全6指標で同じ結論、allowanceで不変 | 母平均winnerではない（ただし差はSEの数十倍） | 高 |
| 費用比は$B^2\times$回路圧縮比で説明される（ε≧.005） | T panelで0.01%以内の一致、H8 ε=.001でも整合 | T=.4/1.6の決定論回路費用は代用値 | 高（前提を確認すれば確定） |
| 中間精度（.01〜.001）で、打切り対照に対し約32〜40%以上の利益が残る | 線形費用式による下限、PRとS2対照のbias同等性 | $p^\ast$は未測定、符号付き相殺の可能性、打切りS4は未測定 | 中〜高 |
| 緩い精度（.05）でPRが有利 | H6・H8とも29% | 格子に依存し0〜29%。H4では打切りが勝つ点がある | 低 |
| 厳しい精度（.0005）でPRが有利 | H6で50%、H8で24% | 打切りS4が未登録（H8ではほぼ同等の見積もり）。PRはq8端かつS2限定 | 低 |
| 厳しい精度での利益は外側S2 biasで制限される | T panelのbias分解（outer成分がRTE成分より4桁大きい） | 候補集合がS2限定 | 高（S2系に限れば） |
| 時間を延ばすと利益は縮み、短時間に限られる | $B^2$の増加（実測）、$\ln B\propto T$の式 | $T^\ast$は未検証の概算 | 定性は中、数値は低 |
| 打切り対照の強化で、full PF比の利益の一部が回収される | H8：full PF比54〜66%が強化後29〜56%へ（付録B13）。H4：B0+Iが397点で最小 | なし（定義上の事実） | 高 |
| 状態準備費用を加えても結論は保たれる | P境界は1e5〜5e6 RZ/shot | 実際の準備費用は未測定 | 中 |
| energy/QPE/RPE、FT総費用、一般分子への主張 | なし | 推定対象が異なる | 主張不可 |

**PRの優位性はどこまで科学的に信頼できるか（依頼書§3.2）。** 四つの段階に分けて答える。

1. **登録domain内**：差は統計的にも数値的にも頑健である。
2. **中間精度で、強化した打切り対照（S2/S4）に対する優位**：符号は頑健で、大きさは約32〜56%の範囲にあると考える。
3. **緩い精度と厳しい精度での優位**：現データでは主張できない。
4. **energyやFTへの外挿**：本研究の範囲外である。

---

## 6. 先行研究との比較と新規性評価

### 6.1 PR原論文との比較（観点を揃えた対照）

| 観点 | Günther et al.（PRX Quantum 7, 020332） | Track A |
|---|---|---|
| 対象 | 基底状態energy（単一ancillaのRPEなど）。H鎖（大規模まで）、FeMoco、小分子群 | 有限時間signal $z(T{=}0.8)$。H4（6距離）、H6/H8（1.0 Å、STO-3G） |
| 誤差定義 | energyのRMSE。$\varepsilon^2=\varepsilon_{qpe}^2+\varepsilon_{trot}^2$と配分し、経験的$C_{gs}$を用いる | 複素signalの絶対誤差。軸ごとの予算$\varepsilon/\sqrt2$、Hoeffding型（α=.025/軸）、厳密signalによるbias、経験的allowance |
| 計算モデル | Pauli回転をToffoliへ換算（Hamming weight phasingなど）、2-qubit数も評価。解析的な期待費用。FT論理層 | Qiskit 1.3.0でcompileした測定込みHadamard wrapper（rz/sx/x/cx、opt1、routingなし）。random回路を標本してcompile |
| 比較baseline | 決定論PF（full、経験的$C_{gs}$）、qDRIFT、qubitization（文献値） | full S2/S4、打切りB0、B0+I（登録domain内） |
| 最適化 | $L_D$、δ、κ（r）、ξを解析的に最適化。軌道とλも最適化 | 有限の登録格子（p、q、r、K）。探索と確認を分離。軌道最適化なし |
| 保証 | randomized部分についてRPEの厳密解析。Trotter定数は経験的 | RTEの不偏性（有限Kのbiasは計算済み）、Hoeffding型shot数。母平均費用の保証なし |
| 費用指標 | Toffoli数（総数・回路あたり）、論理qubit数 | compiled RZ×shot（補助にCX、depthなど）。T/Toffoli数ではない |

両者の量は直接比較できない。それでも、定性的には次の三点で一致する。

- 部分ランダム化の利益は定数倍程度である。原論文自身も、部分ランダム化は実際には控えめな定数倍の改善になる傾向があると述べている。本研究の約2倍も、これと整合する。
- 部分ランダム化のTrotter誤差は、完全決定論とほぼ同じである。
- 分割位置が費用を決める。

原論文は、DF表現での部分ランダム化（大きいDF項を決定論とし、残りを基底回転した2-qubit Pauliとしてランダム化する）を既に扱っている。したがって、**DF-nativeなPRそのものは本研究の新規性ではない**。

一方、原論文は、決定論的なDF打切りでは良い閾値の選び方が曖昧であると指摘している。打切り対照を体系的に強化したことは、本研究が原論文に対して加える具体的な差分になる。

### 6.2 その他の関連研究

- **決定論とランダムの合成**：Hagan–Wiebe（Composite、Quantum 2023）、Pocrnic et al.（PRR 2024）、Jin–Li（2021）。分割の原理はこれらで既知である。
- **RTE**：起源はWan–Berta–Campbell（PRL 2022）である。
- **分子Hamiltonianでの部分ランダム化の数値実証**：phase2（Miller, Günther, Witteveen et al., Commun. AI Comput. 2026）が、Ru錯体の活性空間で最大40 qubitまで部分ランダム化の時間発展を状態ベクトル模擬した。経験的$C_{gs}$が解析上界より1〜2桁以上小さいことも示している。したがって、「分子でのPRの初めての数値適用」は主張できない。
- **最適化product formula**：SPRINT/GRADE（Casares et al., 2026）は、near-integrability、randomization、symmetry protection、QROMなどを統合した。実課題で、Toffoli数を先行法の1/4.5に削減したと報告している。本研究の決定論対照（full S2/S4と打切り）は、最先端のPF設計から遠い。したがって、**「最良のPFに対するPRの優位」は主張できない**。
- **ツール**：Microsoft QDK/Chemistryに部分ランダム化builderが実装されている。PRが標準的な手段になりつつあるので、独立した実compile benchmarkの実用的な意義はむしろ増している。
- **ランダム化による誤差抑制**：random順序PFやMartyn–Rall（npj QI 2025）など。厳しい精度でのS2 bias（4.5節）に対する別の対処として関係する。
- **初期FTでの実践**：Kiss et al.（Quantum 2025）は、shot数と精度の関係を実装に即して扱っており、本研究の位置付けに近い。

### 6.3 新規性評価

| 区分 | 内容 |
|---|---|
| 既知（主張しない） | PRアルゴリズム、DFでの部分ランダム化、RTE、Hadamard testの枠組み、C±/directional制御、決定論とランダムの合成、PRと決定論のbias同等性（経験則）、分子Hamiltonianでの数値実証 |
| 本研究の差分（主張できる） | (1) 打切り対照（B0/B0+I）による帰属：full PF比の利益のうち、決定論的打切りで回収できる部分を、同一signal精度・実compile・選択を考慮した会計で定量化した。(2) 機構の定量化：費用比を$B^2\times$回路圧縮比に分解し、ε≧.005で0.01%以内に一致することを示した。これは断片換算の閾値として整理できる。(3) 利益が残る範囲：緩い精度では打切りが競合し（H4）、中間精度ではPRが約30〜56%有利で、厳しい精度・長時間ではS2系の限界と$B^2$の増加が効く |
| 限定付きで主張できる | 状態準備P感度の補助結果。選択を考慮した確認の方法論（主張ではなく方法として） |
| 主張できない | 一般的なPR優位、最良PFへの勝利、energy/QPE/FT総費用、size scaling、新アルゴリズム |

**新規性の水準**：アルゴリズムとしての新規性はない。評価研究としての差分は、「公平な打切り対照の下での、実compileでの帰属と範囲」である。これは、PR原論文の大規模・FT寄りの解析を、小規模・実compile・短時間の側から補完するものである。

ただし、「既知の構成と評価法の組合せにすぎない」という批判に十分反論できるのは、(2)の機構と(3)の範囲を主張の中心に置いた場合に限られる。

---

## 7. 研究方針の選択肢と比較

### 7.1 選択肢の比較

| 方針 | 研究上の情報価値 | 新規性 | 必要な追加証拠 | 計算負担 | 既存成果への影響 | 判定 |
|---|---|---|---|---|---|---|
| A. H5/H7開殻への拡張 | 低い。H4〜H8では傾向を同定できず、開殻・ROHF・spinという新しい交絡が加わる | 低 | SCF/DF/state、sector、native正しさの確認、40軸を超えるcompile | 大 | なし（別系列） | **採用しない（当面）** |
| B. 理論資源モデルの適用・発展 | 接続検証に限れば中〜高（希少次数の閉包、未登録候補の予測、非Clifford指標）。発展まで含めると低 | 単独では低 | 既存H6候補のgate listとcompile値の対応 | 接続は小、発展は大 | 補強 | **限定採用（V2）** |
| C. H4/H6/H8結果の統合 | 高い（着地に必須） | 主張の立て方による | 範囲の両端の確定 | 小 | 中心 | **採用（主軸）** |
| D. 追加の精度・時間・候補探索 | 狙いを絞れば高（prefix分解能、打切りS4）。それ以外は低 | 低 | 絞った新規signalと少数compile | 中（絞れば小） | 数値幅の修正 | **V1・V3のみ採用** |
| E1. 較正費用モデルによる精度×時間の範囲の整理 | 高い（4.3・4.6節の式の検証） | 中 | V2の接続 | 小 | 主張の中心を形成 | **推奨（V4と併せる）** |
| E2. FT指標（非Clifford数、Clifford+T換算） | 中 | 低 | 既存回路の再解析または再compile | 小〜中 | 頑健性の補強 | 任意（V2に含める） |
| E3. 外側S4やnear-integrableなPRの設計（SPRINTとの接続） | 中（新しい方法研究） | 中 | 新しい理論と実装 | 大 | 別研究 | 別課題 |
| E4. 大規模系への資源見積もり | 低い（PR論文と重複） | 低 | — | 大 | なし | 採用しない |

### 7.2 反対意見・代替解釈への回答（依頼書§3.7）

| 反対意見 | 評価 |
|---|---|
| 既存PR研究との差別化が不十分ではないか | 「PRの効率」を主張するなら不十分である。打切り対照による帰属・機構・範囲を中心に置けば、差分は成立する（6.3節） |
| 比較baselineがまだ不足していないか | 不足している。prefix格子（4.4節）と打切りS4（4.5節）が欠けており、範囲の両端を左右する。より強い対照としては、HF期待値による打切り位相補正や、断片間のbasis変換融合もありうる。ただし中心主張に必要なのは前の二つだけである |
| RZ総費用だけでは主張が弱くないか | FT費用の代理としては弱い（4.8節）。ただし費用比はprefix比と$B^2$が支配しており、6指標で結論は変わらない。非Clifford数を併記すれば十分である |
| 現在の回路実装に特有の結果ではないか | 中間精度の結論は概ね構造的である。比は$p_{PR}/p^\ast$と$B^2$でほぼ決まり、実装に依存する$c_e$への感度は小さい（4.3節）。basis変換の融合などでevent費用が下がれば、PRはさらに有利になる。一方、絶対値はQiskit opt1に依存する |
| 数値allowanceや有限標本の限界が結論を左右しないか | 左右しない（4.9・4.10節） |
| H5/H7の追加は本当に価値を高めるか | ほとんど高めない。範囲を決める量（$p^\ast$、$\lambda_R$、$c_e/c_f$）は、DF断片のスペクトルから安価に推定できる。compileを伴う全面比較は不要である |
| 理論モデルの開発は本筋から外れないか | 接続検証を超える開発は本筋から外れる。接続に絞れば本筋を支える（第8・10節） |
| これ以上検証を増やさず、現成果で一区切りとした方がよくないか | ほぼ同意する。ただし、中間精度の数値幅と範囲の両端は、小規模なV1だけで大きく確定できる。V1（必要ならV2も）の後に区切るのが、費用対効果として最良である。V1を実施しない場合も、範囲の両端を「未確定」と明記して区切ることは科学的に成立する |

---

## 8. 推奨する方向性と根拠

### 8.1 継続・縮小・方向転換・区切りの比較

| 案 | 評価 |
|---|---|
| 継続（R10の拡張計画どおりH5/H7、T比較を進める） | 推奨しない。情報価値に比べて費用が大きい |
| **縮小して区切る（V1とV2の後に統合して着地）** | **推奨** |
| 方向転換（E3：PR-S4やnear-integrable設計、またはFT資源見積もり） | 別研究として価値はある。本研究を完結させた後に検討する |
| 現成果で直ちに区切る | 成立する（範囲の両端を未確定と明記する）。V1の費用が小さいので、次善の案とする |

### 8.2 推奨する進め方と根拠

推奨は、**C（統合）を主軸に、D（V1）を最小限加え、B（V2）を接続検証に限定し、E1で範囲を整理して区切る**ことである。

進める順序は次のとおり。

1. **V1**：H6でprefix分解能と打切りS4を確定する。H8は、H6の結果が範囲の両端を変えた場合にだけ行う。
2. **V2**：費用モデルを既存compile値へ接続する。V1と並行してよい。
3. **V4/E1**：較正した式で、精度×時間の利益の範囲を整理する。
4. 統合し、着地する。V3（PR-S4）は任意とし、厳しい精度を主張に含めたい場合にだけ行う。

根拠は四つある。

1. 中心主張の弱点は、統計ではなく比較domainの構造にある。V1はそこを、直接かつ低費用で確定する。
2. 機構は既に定量的に同定できている（4.2節）。系や時間を増やすより、式を検証して範囲を地図化するほうが、情報量あたりの費用が小さい。
3. A（H5/H7）は中心主張の真偽を変えない。結果がどうであっても、主張の形は「1.0 Åの小規模H鎖での条件付き結果」のままである。
4. 研究の結論はほぼ出揃っており、拡張を続けると完成が遅れる。これまでの作業は監査・来歴管理の比重が大きかった。それは数値の信頼性を高く保つのに有効だったが、残る主要リスクに対する情報価値は低い。

### 8.3 研究の着地点（中心的な科学的主張）

推奨する中心主張は次のとおりである。

> 量子化学DF Hamiltonianの有限時間signal推定において、部分ランダム化（PR）の実compile資源上の利益は、強化した決定論的打切り対照の下でも、中間精度で約30〜56%残る。その大きさは、「中間重み断片の不偏圧縮（回路費用比）」と「normalizationの罰則$B^2$」の積で、ほぼ完全に説明される。利益が残る範囲は、緩い精度側では、打切り対照の最小適格prefixとPRのprefixとの差（約2断片）で区切られる。厳しい精度・長時間側では、外側公式の次数と$B^2$の増加で区切られる。

依頼書§3.6の候補との関係は次のとおりである。

| 候補 | 位置付け |
|---|---|
| PRの優位性が成立する適用条件の特定 | **主**（上の主張そのもの） |
| 強化した決定論的対照による既存評価の再検討 | **主の構成要素**（帰属の定量化） |
| Bias・shot数・回路費用のトレードオフ | **主を支える機構**（4.1〜4.3節） |
| PRの量子化学資源効率の実証 | 主にしない（原論文と重複し、系が小さい） |
| 理論リソースと実コンパイルリソースの相違 | 主にしない（V2の結果次第で補足に留める） |

**主張の範囲**は次のとおりとする。

- 対象：保存DFモデル、1.0 Åの小さいH鎖。
- 時間：T=0.8（T panelではT≦1.6）。
- 状態準備：P=0を主とする。
- 指標：RZを主とし、6指標と非Clifford数を補助とする。
- 比較domain：V1で拡張した後の登録domain。

energy/QPE/FT総費用、size scaling、最良PFへの勝利は主張しない。

---

## 9. 採用しない・後回しにする方向性

| 方向 | 理由 |
|---|---|
| H5/H7開殻拡張（R10の採択の見直しを提案） | 中心主張への情報価値が低い。ROHF、spin、sectorという新しい交絡と、大きな工数を伴う。範囲を決める量はDFスペクトルから推定できる。既に正式採択された方針の変更なので、利用者の判断を要する |
| 一律の追加sampling、希少次数pilot、全supportの上界 | 平均費用への寄与は構造的に約0.2%以下（4.9節）。モデルAで形式化すれば足りる |
| 追加のT点、別のδ、q16の一律追加 | 4.6節の式とV4で代替できる。q16は、V3で厳しい精度を扱う場合に限り検討する |
| 監査・runner・quota機能の追加campaign | 実行制御の信頼性は既に高く、残るリスクとは無関係である |
| 理論モデルの全面的な発展（一般identity位相、history依存のbasis plan、routing、多数eventのDP） | 本研究の主張には不要である。別課題とする |
| 大規模系の資源見積もり（E4） | PR原論文やSPRINTと重複し、本研究の強み（実compileと公平な対照）を活かせない |
| 外側S4やnear-integrableなPRの方法開発（E3） | 価値はあるが、別の研究である。本研究ではV3での候補追加に留める |
| H4を新しいDF政策で再生成すること | H4は別系列として保持すれば足りる（R10に同意） |

---

## 10. 必須・任意の追加検証

### 10.1 最小限の次の検証

#### V1（必須・最優先）：打切り対照の最小適格prefixを確定する

| 項目 | 内容 |
|---|---|
| 確認すること | H6で、B0+Iと打切りS4の最小適格prefix $p^\ast(\varepsilon)$を整数格子（S2はp=6〜14、S4はp=11〜15）で確定し、5精度での最良決定論費用を更新する。PR側は、p=3〜4の外側S2 biasを厳密signalで計算し、Bを解析式で求める |
| 影響する判断 | 中心主張の数値幅、範囲の両端、区切るかどうか |
| 成功・失敗で何が変わるか | 中間精度の利益が25%以上残れば、主張が確定し、両端の符号も確定する。全精度で15〜20%未満なら、「対照に敏感な結果」の報告へ切り替える |
| 担当 | 計算はCodex。比較規則の事前登録と解釈はClaude |
| 新しい科学計算 | 必要。新規signalは、B0+I/打切りS4の追加prefixと、PRのp3〜4のbias。費用は線形式で内挿し、検証compileは4軸以下とする。新SCF/DF/state、samplingは不要 |
| 再レビュー | 結果が出たら、対象を絞った再レビューを行う（Level 1は不要）。11.3節の条件に当たる場合だけ範囲を広げる |

#### V2（推奨）：費用モデルを既存compile値へ接続する

| 項目 | 内容 |
|---|---|
| 確認すること | 理論モデルA/B（または線形の経験費用式）を、既存H6候補のcompile値へ接続する。affineな一致、残差のPR/PF差、event費用$c_e$、非Clifford回転数を確認する。希少次数を含む期待費用をモデルAで計算する。あわせて、付録B1の前提（T panelの決定論回路費用）を確認する |
| 影響する判断 | 4.3節の閾値式の妥当性、未登録候補の予測可能性、希少次数の論点を閉じられるか、FT指標での頑健性 |
| 成功・失敗で何が変わるか | 残差が小さくPRとPFで共通なら、V1・V4でcompileを省ける。系統的な差があれば、compilerの効果を機構の一部として報告する |
| 担当 | 計算はCodex（保存event metadataからgate listを作る。新規samplingなし）。解釈はClaude |
| 新しい科学計算 | 必要（軽量）。native buildのみでcompileはしない。保存値との照合を含む |
| 再レビュー | V1と同時に確認する |

#### V3（任意）：厳しい精度でのS4系同士の比較

| 項目 | 内容 |
|---|---|
| 確認すること | H6のε=.001/.0005で、外側S4のPR（p=5）またはPR q16を、打切りS4と比べる |
| 影響する判断 | 厳しい精度での劣位がPRに固有なのか、候補集合の性質なのか |
| 成功・失敗で何が変わるか | PR-S4が打切りS4に約40%有利なら、「利益は公式の次数に依らずprefix圧縮として残る」という、より強い主張ができる。そうでなければ、厳しい精度を範囲の外とする |
| 担当 | 計算はCodex、設計はClaude |
| 新しい科学計算 | 必要（新規signal、PR trajectory、compile。中規模） |
| 再レビュー | 実施した場合のみ |

#### V4（任意・低費用）：$\varepsilon_E$軸での読み直しと時間範囲式の検証

| 項目 | 内容 |
|---|---|
| 確認すること | 既存のT panelを$\varepsilon_E=\varepsilon/T$の軸で読み直す。4.6節の時間範囲式を、V2の費用式で検証する |
| 影響する判断 | 論文での推定対象の解釈、QPEへ外挿しないことの根拠 |
| 成功・失敗で何が変わるか | 式が既存の3点を再現すれば、範囲図を主要図にできる |
| 担当 | 紙上の解析は主にClaude。保存値の再集計はCodex |
| 新しい科学計算 | 新規simulationは不要。ただし保存値の再集計にあたるため、承認は要る |
| 再レビュー | 不要 |

**V1の事前登録で決めておく規則**

- 決定論の分母は、{B0, B0+I}×{登録済みprefixと追加prefix}×{q=1,2,4,8}×{S2, S4}のうち、適格な候補の最小とする。
- 細かい格子では、符号付きbiasの相殺を偶然に選ぶおそれがある。そのため、通常の最小値と、「|bias|の上界で適格性を判定した場合の最小値」を併記する。
- PR側に候補を追加する場合は、決定論側と同じ精度で行う。
- 線形内挿の許容誤差と、検証compileとの不一致が出た場合の停止条件を、あらかじめ決めておく。

### 10.2 必須と任意の区別

| 区分 | 検証 |
|---|---|
| 必須（中間精度の数値幅と範囲の両端を論文で述べる場合） | V1 |
| 推奨 | V2（V1の費用を下げ、希少次数とFT指標の論点を閉じる） |
| 任意 | V3（厳しい精度を主張に含める場合）、V4（解釈と図のため） |
| 不要 | H5/H7、追加sampling、追加T、全support上界、監査campaign |

---

## 11. 次の担当・承認事項・STOP条件

### 11.1 担当

- **Claude**
  - V1の比較規則の事前登録。
  - V4の紙上解析。
  - V1/V2の結果に対する、対象を絞った科学レビュー。
  - 着地点（主張と範囲）を文章化する方針の策定。
- **Codex**
  - V1/V2/V3の計算と抽出。
  - 保存値との照合と公開。
  - 技術的な細部や不具合の調査。

### 11.2 利用者の承認が必要な事項

| 検証 | 承認が必要な内容 |
|---|---|
| V1 | 新規signal計算（追加prefixのB0+Iと打切りS4、PRのp3〜4のbias）、検証compile（4軸以下） |
| V2 | 保存event metadataからのnative build（compileなし）、保存compile値との照合 |
| V3（実施する場合） | 新規signal、PR trajectory、compile |
| V4 | 保存値の再集計（新規simulationなし） |
| H5/H7方針の見直し | R10で採択済みの方針を変更すること |

### 11.3 STOP条件と再レビュー条件

- 各検証は、完了した時点でmandatory STOPとする。
- 自動retryや、次stageへの自動移行は行わない。
- 既存の結果、grant、ledger、registryは変更しない。

次のいずれかに当たる場合は、対象を絞った再レビューに戻す。

- V1で、中間精度の利益が10ポイント以上変わる。
- ε=.05またはε=.0005で、利益の符号が変わる。
- V2で、compile残差がPRとPFの間で5%以上異なる。
- 線形内挿が検証compileと2%以上ずれる。

中心主張が変わる場合（たとえば全精度で利益が15〜20%未満になる場合）は、Level 1の再レビューに戻す。

### 11.4 研究を区切る条件

1. **主張成立による区切り**：V1（必要ならV2も）の後、中間精度での利益が強化対照に対して約25%以上で頑健なら、データ収集を終了し、統合して着地する。
2. **否定的結果による区切り**：全精度で利益が15〜20%未満なら、「対照に敏感な結果」のnoteとして区切る。この閾値は、実装に依存する要因（分解法、basis変換の融合、指標の選択）が比を約10〜20%動かしうることに基づく。
3. **時間による区切り**：V1/V2を小規模に実施できない場合は、範囲の両端を「未確定」と明記したうえで、現成果で区切る。この場合でも、中間精度の結論は4.4節の下限によって維持できる。
4. いずれの場合も、H5/H7、新しいT、追加samplingには進まない。

---

## 12. 参照資料

### 12.1 リポジトリ資料（固定commit）

完全なcommit hashは次のとおり。

| 略記 | 完全なhash |
|---|---|
| `5f79257f` | `5f79257f432bc7e0317f122090f62fec379e2a1b` |
| `34efc914` | `34efc914b2f485ec694c7d6016025d5c76829be6` |
| `043f58ac` | `043f58ac9336d662a3181caff508526568f1e57a` |
| `98cdb35c` | `98cdb35cd08469714f20d6cc3b982d9c80908f46` |
| `1ea42fbf` | `1ea42fbf1bd1c8341ffd410b8067467b4ba4e170` |
| `3d910ee1` | `3d910ee1de4597b8895e8506500553dbad5a5085` |

- [R1] H8 B0+I結果：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/track_a_h8_native_b0i_full_family_results_20261016.md
- [R2] H4/H6/H8強化対照の整理：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/track_a_h4_h6_h8_strengthened_baseline_status_20261016.md
- [R3] 科学レビュー判断索引：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/track_a_scientific_review_decisions.md
- [R4] H8 D0結果：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/track_a_h8_d0_saved_value_result_20261015.md
- [R5] 正式共通研究ノート：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/track_a_shared_research_status.md
- [R6] 原稿v0.4：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/manuscripts/track_a_resource_study_geometry_native_v0_4.md
- [R7] 先行研究と未解決点：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/先行研究と未解決点.md
- [R8] 先行研究gate：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/pr2_matched_accuracy_prior_art_gate_v1.md
- [R9] 研究目的・研究課題：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/研究目的・研究課題.md
- [R10] 独立技術監査：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/docs/research/track_a_independent_technical_audit_20261015.md
- [R11] H8比較CSV：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f79257f432bc7e0317f122090f62fec379e2a1b/artifacts/resource_applicability/track_a_h8_native_b0i_full_family_result_v1/2026-10-16/comparison_P0_v1.csv
- [R12] H6 B0+I結果：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34efc914b2f485ec694c7d6016025d5c76829be6/docs/research/track_a_h6_native_b0i_full_family_results_20261015.md
- [R13] H6比較CSV：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34efc914b2f485ec694c7d6016025d5c76829be6/artifacts/resource_applicability/track_a_h6_native_b0i_full_family_result_v1/2026-10-15/comparison_P0_v1.csv
- [R14] H6のT依存性結果：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/043f58ac9336d662a3181caff508526568f1e57a/docs/research/track_a_h6_fixed_delta_t_dependence_results_20261015.md
- [R15] H4 B0+I地図結果：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/docs/research/track_a_h4_b0i_map_results_20261014.md
- [R16] 理論ゲート数モデル資料：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1ea42fbf1bd1c8341ffd410b8067467b4ba4e170/docs/research/track_a_native_df_theoretical_gate_model_20261016.md
- [R17] 理論ゲート数モデル実装：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1ea42fbf1bd1c8341ffd410b8067467b4ba4e170/src/trotterlib/native_df_gate_model.py
- [R18] R10統合科学レビュー：https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md

### 12.2 外部文献

- [L1] J. Günther, F. Witteveen, A. Schmidhuber, M. Miller, M. Christandl, A. W. Harrow, "Phase estimation with partially randomized time evolution," PRX Quantum 7, 020332 (2026). doi:10.1103/ynxb-p2xq. arXiv:2503.05647 — https://arxiv.org/abs/2503.05647
- [L2] M. Miller, J. Günther, F. Witteveen, M. S. Teynor, M. Erakovic, M. Reiher, G. C. Solomon, M. Christandl, "phase2: Full-state vector simulation of quantum time evolution at scale," Commun. AI Comput. 1, 5 (2026). doi:10.1038/s44488-026-00002-2. arXiv:2504.17881 — https://arxiv.org/abs/2504.17881
- [L3] P. A. M. Casares et al., "Theory and practice of Trotter product formulas for quantum chemistry," arXiv:2606.30741 (2026) — https://arxiv.org/abs/2606.30741
- [L4] M. Hagan, N. Wiebe, "Composite quantum simulations," Quantum 7, 1181 (2023). doi:10.22331/q-2023-11-14-1181
- [L5] M. Pocrnic, M. Hagan, J. Carrasquilla, D. Segal, N. Wiebe, "Composite Qdrift-product formulas for quantum and classical simulations in real and imaginary time," Phys. Rev. Research 6, 013224 (2024). doi:10.1103/PhysRevResearch.6.013224
- [L6] J. M. Martyn, P. Rall, "Halving the cost of quantum algorithms with randomization," npj Quantum Inf. 11, 47 (2025). doi:10.1038/s41534-025-01003-2
- [L7] O. Kiss et al., "Early fault-tolerant quantum algorithms in practice: Application to ground-state energy estimation," Quantum 9, 1682 (2025). doi:10.22331/q-2025-04-01-1682
- [L8] Microsoft QDK/Chemistry, partially randomized product formula builder — https://microsoft.github.io/qdk-chemistry/api/api_autogen/qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.partially_randomized.html
- [L9]〜[L15] 未読（書誌のみ）：Wan–Berta–Campbell, PRL 129, 030503 (2022)／Campbell, PRL 123, 070503 (2019)／Childs et al., PRX 11, 011020 (2021)／Jin–Li, partially random Trotter (2021)／Motta et al., npj QI 7, 83 (2021)／Oumarou et al., Quantum 8, 1371 (2024)／Kronenberger–Erakovic–Reiher, Mol. Phys. (2026)

### 12.3 未取得資料

第2.2節を参照。

---

## 付録A　既存GPTレビュー（R09/R10）との一致・相違（依頼書§2.6・§5.1）

### A.1 一致する判断と、本レビュー独自の根拠

| R09/R10の判断 | 本レビュー独自の根拠 |
|---|---|
| 一般的なPR優位は認定しない | 利益は$p^\ast(\varepsilon)-p_{PR}$の関数であり、精度によって符号が変わりうる（4.3節） |
| 「高精度ほどPRが有利」とはしない | 厳しい精度では外側S2系の全方式がheadroomを失う。比較は公式の次数の問題になる（4.1・4.5節） |
| B0+Iを採用し、B0も保持する | B0+Iは小さいprefixでの打切りを成立させ、緩い精度側の端を決める（4.4節） |
| 希少次数のための一律sampling増量は不要 | 平均費用への寄与は、構造的に約0.2%以下である（4.9節） |
| H4は別系列とし、size scalingへpoolしない | 政策も実装も異なり、範囲を決める量も別である（4.11節） |
| P感度は補助結果として扱う | 境界は回路費用の0.4〜5倍の規模であり、運用上の意味が明確である（4.7節） |
| R09のclaim B（対照を強化した適用領域）を主とする | 同意する。本書はこれに、機構と範囲の式を加えて具体化する |

### A.2 異なる判断

| 論点 | R10 | 本レビュー | 理由 |
|---|---|---|---|
| H5/H7 | 中性ROHFで採択（H5→H7の順） | 当面は採用しない | 中心主張の真偽を変えない。新しい交絡と大きな工数を伴う |
| 希少次数 | 未解決の不確かさとし、逆転の目安を提示 | 平均費用については実質的に閉じている | 構造的上界があり、既存pilotとも整合する |
| 最大の未確定点 | 対照のcoverage（24軸）、H8 D0、開殻、T | 打切りprefix格子の粗さと、打切りS4の欠落 | 4.4・4.5節の見積もりで、範囲両端の符号が変わりうる |
| 時間依存性 | 固定δでのbiasとnormalizationの蓄積を確認する | これに加え、$T^2$対$T$による時間範囲を明示し、QPEへ外挿しない根拠とする | 4.6節 |
| 指標 | RZはT/Toffoliと同義ではないと注記 | 加えて、Clifford角と1-qubit再合成の問題を指摘し、非Clifford数の併記を推奨 | 4.8節 |
| 原稿化 | 保留し、研究ノートで継続 | V1（とV2）の後、統合して区切る方向へ移る | 機構は同定済みで、追加拡張の情報価値は低い |

過去のGPTレビューは参考資料として扱い、本書の結論を拘束するものとはしていない。

---

## 付録B　レビュー内の算術一覧（公開値と前提）

以下はすべて、文書やCSVに記載済みの値を用いた手計算である。保存科学データの再集計・再ランキングではない。正式な結論にする場合は、承認を得たうえで再計算すること。

**B1. T panelの機構検証（4.2節）**
- 出典：[R14]（PRの1軸平均RZ、B）、[R12]（B0+Iの回路費用）、[R14]の主表（費用比）。
- 算術：$0.4387132\times1.034256=0.453742$（公開値0.453740）。$0.4370434\times1.069686=0.467499$（公開値0.467465）。$0.4347594\times1.144228=0.497464$（公開値0.497423）。
- 前提：T=.4/1.6の比較対照であるB0 p15/q2・q8の回路のRZ数は、T=.8のB0+I p15/q2・q8（59,013と233,661）に等しい。B0とB0+Iは同じp/qで同じ回路構造をとり、H4ではRZの差が0だった。未取得の[R14]回路平均表で確認できる。

**B2. H6の線形費用式（4.3節）**
- 出典：[R12]の回路費用表。
- p10からp15への傾き：q1は9,613/5=1,922.6、q2は3,845.2、q4は7,690.4、q8は15,380.8。いずれも$1{,}922.6\,q$に一致する。
- 切片：q=1,2,4,8でそれぞれ1,066、1,335、1,873、2,949。$797+269q$に一致する。
- p5/q1の予測は10,679で、旧pilotの実測10,670と一致する。

**B3. H6のevent費用（4.3節）**
- PRのprefix費用は$797+9{,}882q$とした。
- 差し引きでtailに残る費用：q2で5,328.8（R=8、1 eventあたり666）、q4で10,909.2（R=16、682）、q8で21,733.3（R=32、679）。
- prefixの占める割合は約79%。
- 前提：PRのprefixはB0+I p5と同じ費用構造を持つ。occurrence単位の固定費用は$c_e$に含まれる。

**B4. H8の線形費用式とevent費用（4.3節）**
- 出典：[R1]の回路費用表。
- 傾き：1断片stepあたり3,610.1。切片：$1{,}592+1{,}314q$。
- full S4 p28/q4の予測：$1{,}592+4(3\cdot1{,}314+28\cdot3\cdot3{,}610.1)=1{,}230{,}364$。実測1,232,704（[R18]§3.2）との差は−0.19%。
- p7/q8のprefixは214,273と予測される。これを[R18]のPR平均253,005.8から差し引くと、tailは38,733となり、32 eventで割ると1 eventあたり約1,210。prefixの割合は約85%。
- 前提：p14/p21の値からp7へ外挿している。

**B5. $\lambda_R$の推定（4.2・4.6節）**
- H6：$\ln B(T{=}.4,R{=}8)=0.016841$。1 eventあたり0.0021051。$b=\sqrt{1+\tau^2}+\tfrac{\tau^2}{2}\sqrt{1+\tau^2/9}$を解くと$\tau^2=0.0021078$、$\tau=0.04591$、$\lambda_R=\tau r/\delta=0.918$ Ha。このとき次数2の確率は0.0010517となり、公開値0.0010518と一致する。
- H8：[R18]のπ2=0.0346709（q4/r4/K4、R=16）とπ4から、$p_2=0.0022030$、$\tau^2\approx0.0044255$、$\lambda_R=1.3305$ Ha。q8/r4（π2=0.0175284、R=32）からは$\lambda_R=1.3303$ Haとなり、一致する。

**B6. H8 ε=.001でのshot比（4.2節）**
- 回路比：$253{,}005.8/618{,}608=0.40899$。総費用比：0.438795。逆算したshot比：1.07288。
- B5から得る$B^2$：$\exp(2\cdot32\cdot0.0011061)\approx1.0733$。
- 前提：B0 p21/q8の費用をB0+Iの値で代用した。

**B7. prefix格子の下限（4.4節）**
- H6（$p^\ast=11$）：q2で$43{,}632/59{,}013=0.7394$、低減率は1−0.50314/0.7394=32.0%。q4で0.7376、36.6%。q8で0.7367、40.5%。
- H8（$p^\ast=15$）：q4で$223{,}454/310{,}100=0.7206$、低減率は34.4%。q8で0.7199、39.0%。打切りS4 p15/q4（667,184 RZ、shot数はfull S4の36,390,063）の場合、$G\approx2.428\times10^{13}$で、低減率は36.7%。
- H6 ε=.05（p10/q1の分母20,292に対して）：$p^\ast=7$なら14,524で低減率1.3%、$p^\ast=8$なら16,447で12.8%、$p^\ast=9$なら18,369で22%。
- 前提：shot数は減らないとする（符号付き相殺を除く）。

**B8. 打切りS4（4.5節）**
- H8 p21/q4の回路費用：$1{,}592+4(3{,}942+21\cdot10{,}830.4)=927{,}112$。full S4の実測値に対する比は0.7521。
- 総費用：$0.7521\times1.86505\times10^{14}=1.4027\times10^{14}$。PRの1.42055e14より1.25%低い。
- H6 p15/q4：モデル上のfull S4（442,378）に対する比は0.7914。総費用は$5.2025\times10^{13}$で、PRの3.2628e13はこれより37.3%低い。
- 前提：shot数はfull S4と同じとする。H6のfull S4費用はモデル値である。

**B9. 時間範囲（4.6節）**
- H6の費用比：$\rho(T,r)=e^{0.33724T/r}\,(9{,}882+677r)/29{,}108$。$r$を最適化すると、$T=17$で$r\approx12.45$、$\rho\approx0.997$。$T=18$で$\rho\approx1.024$。r=4に固定すると、$\rho=1$となるのは$T\approx9.9$。
- H8の費用比：$\rho=e^{0.354T/r}\,(26{,}585+1{,}210r)/77{,}126$。$T=23$で$\rho\approx0.986$、$T=24$で$\rho\approx1.005$。r=4に固定すると$T\approx10.1$。
- 前提：bias同等、δ固定、打切りprefixは適格のまま、B3・B4・B5の値を用いる。

**B10. 希少次数（4.9節）**
- $\mathbb E[k]$：H6 r4/K2で0.0021036。H8 q8/r4/K4で0.0011051。H8 q4/r4/K4で0.0044093。
- 回路全体に占めるevent部分の割合：H6で約21%、H8で約15%。
- trajectoryへの影響は0.02〜0.07%。$\bar c_P/\bar c_R=3$としても約0.2%以下。

**B11. 標本SE（4.10節）**
- H8 factor1・RZ：0.543%、0.428%、0.428%、0.296%、0.296%（ε=.05〜.0005の順）。
- 6候補から最小を選ぶことによる期待効果は約1.27 SE。

**B12. P境界と回路費用の比（4.7節）**
- H8：ε=.001で$1{,}212{,}916.6/253{,}005.8=4.79$。ε=.0005で$108{,}369.8/253{,}005.8=0.43$（同じ候補p7/q8/r4/K4）。

**B13. 打切りによる回収分（第5節）**
- H8で$G_{\rm det}/G_{\rm fullPF}=\rho_{\rm fullPF}/\rho_{\rm det}$を求めた。ε=.05：0.45708/0.71232=0.642。ε=.01：0.754。ε=.005：0.754。ε=.001：0.780。
- full PFに対するPRの見かけの低減（54〜66%）のうち、約3分の1〜3分の2は決定論的打切りで回収される。ε=.001のfull PFはS4/q4であり、公式の次数の差も含む。

---

*本書はレビューと提案であり、計算の実行承認、GitHubの正式な科学判断registryの変更、原稿の執筆・改訂を含まない。*
