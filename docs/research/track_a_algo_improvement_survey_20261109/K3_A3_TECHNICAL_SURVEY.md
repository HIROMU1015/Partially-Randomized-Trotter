# 元Track A：K3（D/R/drop三分割）とA3（round別D/R分割）の技術調査と最小検証計画（2026-11-09）

> 記録：2026-11-09にClaude Codeが静的調査として作成し、repository外の作業報告ファイル `.agent/docs/review.md` で報告した内容です。2026-11-10にユーザーの指示でrepositoryへ保存しました。本文は報告時のままで、「変更ファイル」などの節は報告時点の記述です。技術的な意見は採択ではありません。新しい科学計算は行っておらず、mandatory STOPを維持しています。

読み手：GPT（研究方針の科学判断）とユーザー。
作業範囲：既存資料・SOURCE・保存結果を静的に読んだだけです。新しい科学計算、数値再解析、信号・shot・価格の計算、build/compile、テスト実行、repository変更、サブエージェント利用はいずれも0件です。報告後は **mandatory STOP** です。

### 要点

1. **A3（round別D/R分割）は、多くがTrack Cで実装・実行済みでした。**
   - Track CのDF比較（`da8b2c3f`）は、PRのLD（D/R境界）をroundごとに選んでいます。保存policyでは、H2がLD0→LD1→LD2、H6がLD3（m0–4）→LD9（m5–16）→LD14（m17）です。長roundほどDを増やしています。
   - 共通のdecoderで全18roundの資格も確認済みです。
   - **H4だけは例外**で、PRはc5a3由来の固定LD3（218成分にhard-code）を使っています。H4のLD4以上を全RPEで評価した結果は、確認した範囲ではありません。
2. **H4のLD3は、大きいfragmentをRに入れています（保存drop normからの静的な読み）。**
   - Track C CANDIDATESの保存値：prefix3→prefix4でdrop norm .6045→.0517（I1では .512→.036）、prefix6 .026、prefix7 2.6e-4、prefix9 2.1e-7。prefix10以降は5.7e-14（実質0）。
   - 4番目のDF fragmentがtail normの大部分を占めると読めます。L1≥operator normなので、λ_R(LD4)≥.036 は確実です（LD3は.583）。
   - Track Cの式（log range ≈ λ_R² h T/r）を主要次数で読むと、m17でのRTE event数はおおよそ λ_R² に比例します。**H4でLDを4〜7へ移すと、event数がLD3より桁で減る可能性があります。** λ_R(k)の正確な値は未算出で、これは仮説です。
   - H4のPRがFullの2.31倍になった主因は、m17のR費用（同roundのRZの80%）です。
3. **K3はH4ではほぼA3に帰着します。**
   - H4のtailのうち、drop可能な部分（fragment 8–12）はλが小さい（L1は3.13e-4 Ha）。fragment 11–12は実質0です。
   - 横断調査（`2e79d80d`）のK3初手案（D=LD3、Truncated coeff_L1_2の除去集合をdrop）は、λ_Rを約.05%しか変えません。**既存PR LD3とほぼ同じになり、判別力が低い**と判断します。
   - K3に固有の機構が効き得るのは、保守的評価B（相殺に依存しない評価）でTruncatedが追加のfragmentを保持させられる場合です。例えばH6 m17では、Aでprefix13、Bでprefix15になります。この場合に「Dはprefix13のまま、落とした部分をRで補完」が意味を持ちます。
   - **評価Aでは逆に不利になり得ます。** Truncatedの優位には、切捨て位相とPF位相の符号付き相殺が含まれます（H4 m17：−.0219/+.0213 rad、H6 m17：+.280/−.314 rad）。Rで切捨てを補うと、この相殺がなくなります。
4. **旧opt2とTrack Cの逆転は、主にFull baseline側の変化と整合します。**
   - 両者は同じH4入力（DF sha256 `13e4b10d…`）、同じε・18roundです。
   - PR総RZは 1.263e12→1.304e12 とほぼ同程度でした。Fullは 1.328e12→5.648e11 です。
   - 旧opt2のFullは全roundでS2・δ=.02に固定されていました。Track CのFullはroundごとにh・公式・順序を選び、m17ではY4・h=.16・16,384 stepです。
   - これにcompiler（opt2→compiler A）、q>32のproxy外挿、opt2の微小角問題が重なります。**寄与の分解はUNKNOWN**で、新しい価格計算が必要です。
5. **B2-P1とTrack C C-H4は、同じDF12数値に近い別条件です。**
   - λ_Rは.5832704246339983と.5832704246339503で、差は5e-14。
   - 成分数324と218の差は、coefficient_atol（0と1e-12）の違いで説明できる可能性が高い（未検証）。
   - 異なる点：sector（36と70）、state、stat予算（固定.24と残余）、α（均等と最適化）、compiler（opt1 whole と A modular）、保証の水準、外側h（δ固定と可変）。
   - **B2-P1 pipelineは h=δ が型で固定され、h>δ を扱えません。** そのため、今後のK3/A3にはTrack Cのpipeline（`track_c_df_truncation(_pr)`）を、Track A自身の契約で再利用するのが技術的に合理的です。
6. **技術的な意見（採択ではありません）：** 最初の判別にはA3-1（H4 m17、LD4–7）が最も安く、情報量も多いと考えます。K3は、評価Bを比較基準にする場合に限り、K3-1（H6 m17）を別に行う価値があります。

---

## 0. 前提・取得状況

- 指示：チャットに貼り付けられた「Codex指示：元Track AのK3・A3アルゴリズム改善調査と次段階の技術準備」を受けて作業しました。前回の調査報告（[TRACK_A_RESEARCH_SURVEY.md](TRACK_A_RESEARCH_SURVEY.md)）を基点にしています。
- 資料の取得：ローカルclone `partially-randomized-trotter` のGit objectから、固定commitを `git show` で読みました。
  - `git ls-remote`（作業中に再照会、229 heads）とローカルrefは一致しています。
  - 前回以降の新しいheadは `codex/algo-improve-new-a-total-cost-20261109`（新A）と、Track D long-qの更新だけでした。どちらもK3/A3とは別系列のため、読んでいません。
  - 研究moduleはimportしていません。読んだのはSOURCEのテキストとAST、保存済みのCSV/JSONの該当行だけです。
- 確認した固定成果：
  - 横断調査 `2e79d80d`
  - Track C：`b3c81c26`（統合）、`c5a3c102`（Closed P3共同最適化）、`da8b2c3f`（Truncated比較）、`92495a04`（相殺P1/P2）、`a78054dc`（H4 pilot入力）、`100377da`（旧統合）
  - B2-P1：`7bc95277`、`0d5ca357`、`55fa5f5e`、`3e316e07`、`e162ef70`、`66a41ad9`
  - 旧opt2：`4591b8d7` の `docs/research_direction_full_opt2.md` と保存coherent analysis JSON
- 強PFの記録：旧5 STOP（`2583ce49`）は旧SOURCEの記録として残っています。HP SOURCEによる再資格（`3e316e07`、`e162ef70`）で、**全8 armが条件付きbranch/shot PASS（各18,540 shots）**です。
- 以下の数値はすべて原報告・保存CSV/JSONからの転記です。「目安」と書いたものは、転記値と既存の式を用いた静的な算術で、科学的な結果ではありません。

---

## A. K3（D／R／drop三分割）の技術調査

### A.1 既存実装との対応

| 構成 | 既存で実現しているもの | SOURCE |
|---|---|---|
| Truncated PF（R=∅） | DF block列から `kept(c, candidate)` でHkeepを作る。候補は2系統：①DF prefix k（I0は末尾fragmentをidentityごと除去、I1はidentityを保持）、②全blockの非identity係数を小さい順に累積L1 budget（1.5936e-4×{.01…64} Ha）まで除去。roundごとにretention・公式・h・順序を選び、元H groundに対するcombined信号を直接囲む（評価A）。P2では個別位相の上界を加算する評価Bも実施 | `src/trottertracks/track_c_df_truncation.py`（`candidates`、`kept`、`det_qualification`、`circuit_piece`、`price_piece`、`choose`）、`track_c_df_cancellation_conservative.py`（`92495a04`） |
| PR（drop=∅） | `partition(n, ld)` は D=blocks[:ld+1]、R=残り。R成分は署名付きZ/ZZ、λ=Σ\|c\|、確率は\|c\|/λ。Closed P3 law（`cp.CachedCanonical`、256-bit proposal、weighted estimator、range Lで1回正規化）。`qualify_point` は元H groundでPF×exact tailの信号を元E0と比較（A型）し、finite tail誤差・root角・SOURCE/native/joint isometry disk・range を加える | `src/trottertracks/track_c_df_truncation_pr.py`（`partition`、`generator`、`screen`、`qualify_point`、`build_isometries`、`event_circuit`、`price_one`、`independent_one`）、`track_c_closed_p3*.py`（`c5a3`） |
| H4のPR | c5a3由来の固定LD3（`track_c_closed_p3.py` が `assert len(ps)==218` で固定）。D内部BCH調整（tuned_0321）、625 joint isometry、h・resolution・αの共同選択 | `c5a3c102`、`b3c81c26` |
| 中間（D＋一部R＋drop） | **未実装**。ただし `kept()` はdrop集合を、`partition()` はLD境界を既に持つため、両者の合成は小さい差分で済む | — |

両端との機構上の関係：
- Truncated = K3でR=∅。PR = K3でdrop=∅。
- K3の自由度は、tail成分ごとに D／R／drop のどれに置くかです。DF表現ではRに置いたeventごとにそのfragmentのbasis変換が必要になるため、**実用的な単位はfragment単位、またはfragment内の係数集合**になります。

### A.2 保存値から読める、H4とH6のtail構造

H4（Track C CANDIDATES、drop operator norm [Ha]）：

| 保持prefix k（D=one-body＋k fragment） | I0 drop norm | I1 drop norm | Truncatedとして十分screen内か |
|---:|---:|---:|---|
| 3 | .6045 | .5119 | 不可 |
| 4 | .0517 | .0361 | 不可 |
| 5 | .0398 | .0308 | 不可 |
| 6 | .0259 | .0218 | 不可 |
| 7 | 2.633e-4 | 2.180e-4 | 可（m0–14はI0、m15–17はI1相当のcoeff_L1_2） |
| 8 | 9.53e-5 | 8.31e-5 | 可（最終選択されず） |
| 9 | 2.08e-7 | 1.56e-7 | 可（同上） |
| 10、11 | 5.67e-14 | 5.67e-14 | 可（Fullと同じ。実質ゼロfragment） |

- coeff_L1_2（m15–17で採択）は、除去L1が3.13e-4 Ha、|Hdrop| と dE がprefix7_I1と一致します。**実質的に「fragment 8–12の非identity部分を落とす」操作**です。
- H6のm17：評価Aではprefix13_I0（末尾5 fragmentを除去、|Hdrop| 4.63e-4、dE −1.07e-4 Ha）。評価Bではprefix15_I0を選んでいます。

### A.3 「Truncatedで省略可能とされた項の一部を、あえてRTEで補完する」ことの追加利益

| 想定される利益の経路 | 根拠と限界 |
|---|---|
| **(i) 評価Bで、切捨てbiasをRで置き換える** | Truncated-BはB_T＋B_P＜.4を満たすため、H6 m17でprefix13→prefix15（2 fragment多い）とY4・h=.16へ切り替えた（`92495a04`）。Rで補完すれば prefix13のD＋小さいR でB_T≈0にできる可能性がある。Rのλが小さいので減衰は小さく、eventはr=1程度/stepで足りる見込み（静的な読み）。H4でのB評価m17はB_T=.0219、B_P=.0131、β=.3651。N∝1/sin²β なので、B_Tを0にしてもm17のNの減少は約1割が上限（目安：sin².3651/sin².3870≈.895）。H4ではDも同じなので、利益はこの範囲に限られる |
| **(ii) whole fragmentを積極的に外し、Rで補う** | 実体は「LDを大きくしたPR」で、A3と同じになる（B節） |
| **(iii) fragment内の小係数を、同じbasisの中でランダム化する** | 既存builderにはない（R eventはD sweepの中央に置かれ、独立したbasis遷移を持つ）。新しい配置機構が必要で、今回は候補として記録するだけ |
| 反対方向（不利） | **評価Aでは、Truncatedの優位に切捨てとPFの符号付き相殺が含まれる**（H4 m17：trunc −.021852／PF +.021345／combined −.000507 rad。H6 m17：+.2798／−.3138／−.0340）。Rで切捨てを消すと、PF誤差が相殺されずに残る。H6のS2・h=.08ではPF ≈.31 radが残り、βが急減する |

### A.4 誤差・normalization・shot・価格の接続（既存の式）
- 信号：z_m = 元H groundに対する D(PF)×R(finite Closed P3 mean) の信号。drop有りの場合、評価Aは元E0との位相を直接囲むので、qualify_pointの構造（PF×exact tailを元E0と比較）をそのまま使える。評価Bには、Hkeep spectral証拠と積diskが必要（`track_c_df_cancellation_conservative`）。
- 誤差の項目：drop誤差（A：combined、B：B_T）、PF（D/R分割とD内部）、finite tail（`M e^x x⁴/24`、x=λh/r）、root角、SOURCE/native、joint isometry、weight丸め、scalar・時刻。round位相予算は.4 rad。
- normalization：log L ≈ M·log W_max ≈ λ_R² h T_m / r（主要次数）。ρ = radius/L − disk。
- shot：N = ⌈4 log(2/α)/(ρ² sin²β)⌉、Σα≤.05（36軸）。
- 価格：Dはmoduleの first＋(steps−1)middle＋last、Rは完全occurrenceの価格MC（root/no-root層別）、それにwrapper。指標はRZを主とし、CX・size・serial depth上界を同じmoduleから取る。

### A.5 比較要件（同じ元H・state・精度・成功条件）
C-H4の条件に合わせます。
- 元H＝保存行列のHermitian部分。元H exact ground（N=4 sector70）。準備費は除外。
- 18round、δ0=.02、ε_E=1.5936e-4 Ha、failure≤.05、位相予算.4/round。
- alias条件は `.02·‖H‖_F+.4<π`（qualify_pointの実装）。
- compiler A、modular価格。PR価格は独立holdout。
- Full・Truncated・PRに同じh・公式・順序のmenuを与える。

### A.6 中間構成が両端に支配される可能性と、最小の判定法
- 最小の判定法：**m17（総費用の支配round）の単一roundで、固定α（screenと同じ1/720）でのG_17を、同じh・公式で端点と比べる。** 中間点のG_17がTruncated（A/B）とPR（同じLD）の両方以上なら、そのfamilyは支配されていると判断してSTOPします。
- H4では、drop可能な集合のλが極めて小さく、drop次元はほぼ縮退しています。横断調査K3案（D=LD3＋coeff_L1_2の除去集合）は既存PR LD3とほぼ同じで、判別力が低いと判断します。

### A.7 K3の分類
- 既存コードで実現できる部分：
  - 候補とdrop集合（`candidates`、`kept`）、Closed P3のR、A型の資格（`qualify_point`）、B型の位相disk（`track_c_df_cancellation_conservative`）。
  - D/R moduleの価格、R occurrenceの価格MC、独立check、α配分（`choose`）、decoder。
- 最小限の追加実装：
  - drop引数付きの `partition(n, ld, drop)` と、残す係数で作るR用 `block_data`。
  - drop後のR supportに合わせたjoint isometry catalogue。
  - 「Truncatedと同じD公式（S2/Y4、kept D block）＋中央R(h)」という新しいschedule種別。既存の `track_c_d_internal.schedule` はordered/D4/Dtunedなので、これに1種類追加する。
  - PR＋dropのB型adapter。
  - molecule loopの拡張（下記B.4）。
- 既存結果で否定・制限されている条件：
  - H4では、R=∅の端点（Truncated）がA・Bの両方でFullとPRより安い。
  - Aでは相殺を失うと不利。
  - 負のR時間を使うpartial Y4は負結果。
  - H2では実fragmentを落とせず、非Fullは極小係数の除去だけ。K3の対象外。
- 未解決の科学的論点：
  - AとBのどちらを比較基準にするか（GPT判断）。
  - R 1 eventのbasis費が、落とせなかったfragmentのD費より安いか。
  - 極小τでのfinite-bit law、range、root確率の扱い。
  - 相殺を利用する「R量の調整」は、wrapped終点位相への過適合になり得るため推奨しない。

---

## B. A3（RPE round別D/R分割）の技術調査

### B.1 B2-P1で長roundのshotが増えた機構とSOURCEの依存関係
1. `b2_rpe_connection.B2RPERound`：P1は全roundで r=4、K=2。`c.evolution_time==δ`（外側h=δ固定）、generation-prefix、`coefficient_atol==0`、2次の公式を `__post_init__` で強制している。
2. `rte.finite_rte_distribution`／`finite_rte_attenuation`：event 1回の b_2(τ)、τ=λ_Rδ/r。A=B^{−qr}。
3. `o1_m1_resource_bridge.radius_phase_diagnostics`／`account_radius_round`：ρ_raw = A(ρ_PF − ε)。
4. `finite_rte_signal_validation._hoeffding_shots`：β_stat=.24固定、α=1/720。

結果として N ∝ A^{−2} ≈ exp(2λ_R²δT_m/r) です。m17のA=.0116で、総shotの98.5%を占めます。

### B.2 Track Cで実施済みのround別最適化との区別

| 自由度 | 動かす量 | 実施状況 |
|---|---|---|
| r_m（resolution） | λ_Rは固定のまま、eventを細かくして減衰を抑える。1shotあたりのevent数 ∝ λ_R²T_m² | Track C H4（m17はh=.32、768 microstep/D step、log range .36）、旧opt2（r≤32） |
| h_m・公式・順序・α_m | PF誤差と単価、shot配分 | Track C（Full・Truncated・PRの全方式） |
| **D/R分割L_D^(m)そのもの** | **λ_R^(m)を下げ（event数は二乗で減る）、D費を上げる** | **Track C DF比較でH2・H6は実施済み**（保存policy：H2はLD0[m0–4]→LD1[m5–6]→LD2[m7–17]、H6はLD3[m0–4]→LD9[m5–16]→LD14[m17]）。**H4は未実施**（LD3固定） |
| Truncatedのround別retention | Hamiltonian近似をroundごとに替える | Track C（H4：m0–14 prefix7_I0、m15–17 coeff_L1_2。H6：prefix13、coeff_L1_8、prefix14/15） |

r_mだけの調整と分割の変更は、別の手段です（前者はλ_R固定でevent数を増やす、後者はλ_R自体を下げる）。**ただし、後者もH2・H6ではTrack Cの有限menuとして既に使われています。** 「round別に分割を変える」こと自体を新しいアルゴリズムとして扱うべきではありません。

### B.3 新しい自由度は存在するか
- 一般的にはありません（Track Cに存在します）。
- 残っているのは次の4点です。
  1. **H4のLD menu**（LD4–9）を、Closed P3・joint isometry・BCH-Dの枠内で使うこと。既存familyのparameter拡張ですが、H4の費用の支配要因（m17のR）に直接作用します。
  2. H6のより細かいLD（LD10–16。既存menuは3/9/14/17のみ）。
  3. 非prefixのfragment選択。旧RQ-02で具体的な差が確立されておらず、優先度は低い。
  4. λ_R²h_mT_m/r と単価比に基づく選択則。古典探索を減らす規則で、量子機構ではない。

### B.4 既存実装でD/R分割のround別変更を扱えるか

| pipeline | round別L_D | 制約 |
|---|---|---|
| `track_c_df_truncation_pr` | 可（`screen` がLD×kind×h×rを各round mについて評価し、`choose` で選ぶ） | `bch()`、`probes()`、`build_isometries()`、`shortlist()` などで `for n in (2,6)` が固定されている。H4用のLD menuとisometryは未作成 |
| `track_c_closed_p3*`（H4） | 不可 | 218成分（LD3）に固定 |
| `b2_rpe_connection`（B2-P1） | データ型上は可（`B2RPERound` がroundごとにpreparationとL_Dを持ち、`aggregate_b2_rounds` はcontextの同一性だけを要求） | h=δ固定、opt1 whole compile（m17は実用性未確立）、atol=0、sector36。強PF baselineもh≤δ（substeps）に限られ、Track CのFull（m17でh=.16のY4）より弱い |

### B.5 全roundで同じ元H energyを推定するための条件
- 既存decoder（`reconstruct_rpe_phase`）に渡すのはphase列とq列だけです。各roundが元E0に対して位相誤差≤β_m（.4−誤差）とalias条件を満たせば、**round間で構成が同じである必要はありません。** Track Cはroundごとに異なるretention（Truncated）やLD（PR）で全18roundのdecoderを検証済みです。
- roundごとに必要なもの：
  - 元ground stateと、そのroundの構成の支配固有branchのoverlapが.5を超えること（Track Cのnonresonant screen）。
  - PF・RTE・drop・SOURCE/native diskの合計が予算内であること。
  - ρ>0、β>0、36軸のΣα≤.05。
- B2-P1のpriorは[−128,128]（係数和S）、Track Cは `.02‖H‖_F+.4<π` です。どちらも状態に依存しない上界ですが、条件としては別物です。

### B.6 分割切替の費用の扱い
- roundは別々の回路で実行するので、round内での切替gateは不要です。
- control構造・wrapper・準備（除外）は同じです。
- 古典側の費用は別欄に記録します（menu探索・BCH root・isometry catalogue・資格の計算。Track CではH2/H6で902のdet cellと209のPR probeの規模）。量子費用へは算入しません。

### B.7 公平な比較の与え方
Track Cの枠組みでは、Fullはroundごとにh（.02–2.56）・S2/Y4・先頭順序、Truncatedはretentionも選んでいます。A3を評価する場合も、FullとTruncatedは既存policyをそのまま使えます（同じ条件で資格済み）。PRだけに新しいmenuを与える形になるので、改善が出た場合は「PR内部の改善」と「対照との差」を分けて示す必要があります。

### B.8 H4でA3が効き得る理由（静的な読み。数値未検証）
- H4 m17の保存値：
  - PR：h=.32、8,192 step×768 microstep、N=303/軸、1shotのRZ 1.381e9、round G 8.36e11（うちR 6.68e11）。
  - Full：Y4、h=.16、16,384 step、N=139、1shot 1.099e9、G 3.04e11。
  - Truncated（A）：Y4、h=.16、N=130、1shot 6.93e8、G 1.79e11。
- 規模の目安：Truncated m17では1stepあたりのD費が約4e4 RZ（6.93e8/16,384）。R 1 eventは数百RZ規模（Track C旧統合：m17代表occurrenceで1,024 eventが総RZ約174,800）。
- LD3（λ_R=.583）ではr=768が必要で、Rが支配的になります。LD k≥4ではλ_R(k)が大きく下がり得ます（下界はI1 drop normで、LD4≥.036、LD6≥.022、LD7≥2.2e-4）。必要なrはλ_R²に比例して減り、Dは5〜8 blockになります。**FullやTruncatedより安くなる窓が存在し得る**というのが仮説です。λ_R(k)の上界、PF誤差、BCH-Dの再調整は未検証です。

---

## C. 科学条件の対応表

| 項目 | B2-P1（Track A） | Track C C-H4（PR／強Full／Truncated） | 旧opt2（M06-F、9月） |
|---|---|---|---|
| Hamiltonian | energy targetは cutoff前のH_phys（HDF5 `a3c88f6a`）。DF12 snapshotは `bf5d3b24…/pr2_s0_s1_validation/h4_1p00_rank12_development_v1.npz`（file `3bc92e92`、H hash `de7a5492`）。H_physとの差は9.27e-15 Ha（数値診断） | legacy DF12生成順（`4591b8d7…/rte_connected_cluster…hamiltonian.npz`、df sha256 `13e4b10d`）。元H＝保存行列のHermitian部分。DF再構成差は5.67e-14 Ha | Track Cと**同じsnapshot `13e4b10d`** |
| identity・scalar | 核scalar 2.29310124732を1回。tail identity .0926を抽出して相対位相に | 同様にscalarを分離（数値は今回未照合） | — |
| D/R分割 | generation-prefix LD3（one-body＋fragment 0–2）。R=fragment 3–11の非identity **324成分**（atol=0）、λ_R=.5832704246339983 | Partial LD3（D4 block）。R=**218成分**、7 basis。atol=1e-12（threshold誤差4.83e-14）、λ_R=.5832704246339503 | LD3/LD12 |
| state | Nα=Nβ=2、**sector36**、H_phys ground（S0 `1a2df11b`） | **N=4 sector70**、元H ground（ref `6a350e7d`） | — |
| finite RTE | canonical paired K2（degree 3）、**全round r=4**、補正B | **Closed P3**（256-bit、weighted、range L）、roundごとのresolution（m17で768/step）、D内部BCH | canonical K2、r∈{1…32}（m17でr=32、A=.573） |
| 外側step | **h=δ=.02固定**（q=2^m step） | roundごとのh（.02–2.56） | δ=.02固定 |
| RPE | m0..17、ε=1.5936e-4、β=.4、prior[−128,128]、stat .24固定、α=1/720均等 | m0..17、同じε、failure .05、位相.4/round、βは残余、αはcost-sensitiveな整数配分、alias `.02‖H‖_F+.4<π` | 同じε・18round、β・α・scheduleを再最適化（stat .386） |
| 数値保証 | engineering allowance u=1e-8、独立照合1e-9。強PFはHP 80/120桁 | 192-bit区間・directed disk・spectral residual・ground区間（保証A）。PF/drop個別のB | 経験的PF位相proxy（pf_coefficient）。local dirty-worktree evidence |
| compiler | Qiskit1.3.0 opt1 seed7、whole compile（m0、Q2のみ取得） | Qiskit1.3.0＋**compiler A**（定義に基づくlowering、固定inverse cancellation、角度cutoffなし）、modular | Qiskit1.3.0 **opt2** seed17、q≤32を直接compile、それ以上はaffine proxy |
| 回路費用 | full Hadamard X/Y、no-prep、6指標 | 同じscope。RZ/CX/sizeは加法、depthはserial上界 | 同じscope（モデル） |
| baseline | 既存PR（B2自身）、強PF 8 arm（S2/Y4×substeps、h≤δ）。価格なし | 強Full（h・公式・順序をround別に最適化）、Truncated A/B、Closed PR | Full LD12、S2、δ=.02、r=0 |

**再利用の分類**

| 区分 | 内容 |
|---|---|
| そのまま再利用できる | decoder `reconstruct_rpe_phase`。十分shot式（同じ形：4 log(2/α)/(ρ² sin²β)）。各pipelineの自条件内の処理（B2-P1の信号/branch/shot、HP強PF。Track Cの全stage） |
| adapterで再利用できる | `track_c_df_truncation_pr` のH4 LD menuとdrop引数。B2のnative IRをcompiler A modularへ渡す接続（B2側で価格を取る場合） |
| 再資格が必要 | sector36↔70、atol 0↔1e-12、engineering allowance↔directed保証、opt1/opt2/Aの間の価格、β・αの規約変更（B2-P1のshotをTrack C式に直すと別の結果になる） |
| 数値を直接比較できない | B2-P1のshot（7.8M・18,540）とTrack Cのtrial（16,791・14,225）。B2 m0価格（opt1）とTrack C価格（A）。旧opt2の総額とTrack Cの総額（compiler・proxy・baselineが異なる） |
| UNKNOWN | H_physとTrack Cの `original_H_in_legacy_basis` がbit一致するか。DF factorがbit一致するか（λ_Rの一致は強い示唆だが未照合）。324→218の内訳（atol以外の要因）。Track Cのtail identity値 |

---

## D. 旧opt2とTrack Cの結果の相違

**資料から直接確認できた差分**

| 観点 | 旧opt2（`4591b8d7`） | Track C（`da8b2c3f`／`c5a3`） |
|---|---|---|
| 入力 | snapshot `13e4b10d`、ε=1.5936e-4、18round | 同じsnapshot・同じε・18round |
| 総RZ | PR LD3 1.263e12、Full 1.328e12（PRが点推定で4.858%低い、区間は重なる） | PR 1.304e12、Full 5.648e11（PRが2.31倍） |
| m17 | PR：q=131072、r=32、A=.573、G 7.60e11。Full：S2・δ=.02・131072 step、G 5.85e11 | PR：h=.32、8192×768、G 8.36e11。Full：Y4・h=.16・16,384 step、G 3.04e11 |
| Full baseline | LD12、S2、δ=.02固定（r=0）、h・順序・公式の選択なし | roundごとのh・S2/Y4・先頭順序、PRと同じ効率的なcontrol・対角合成 |
| compiler | opt2。Track Cの後の調査で、CRZ(1e-5)が空回路になる例、Full Y4 q1でX期待値差1.4077e-5を観測（`a78054dc`、`b3c81c26`）し、opt2での価格付けをSTOP | compiler A（微小角cutoffなし、44人工ケース・512次元operatorで照合） |
| 長roundの価格 | q=1,2の校正によるaffine proxy。holdoutはq=8（fresh）とq=16,32（旧）。**q>32の直接検証なし**（m17はq=131072） | modular直接価格（Dはfirst/middle/last、Rは完全occurrence MC） |
| PR機構 | canonical K2、r≤32、support_run_le_1 | Closed P3、BCH-D、joint isometry、融合、h・resolution・α |
| 誤差・shot | PF位相は経験的proxy、β・αを再最適化。shot：PR 13,588／Full 11,162 | directed保証。trial：16,791／14,225 |
| 価格不確かさ | local-5%、per-r、25% transferの各区間が重なる（「点の選好であり区間での優越ではない」と自ら明記） | PRはnominal MC SE 4.83e9、Fullは決定論 |

**整理**
- 保存総額の比較では、PR側は同程度（1.263e12→1.304e12）で、変化はFull側（1.328e12→5.648e11）にあります。**逆転は主にbaselineの強化と整合します。**
- 旧結果は、自らの範囲内（点の選好と明記）では無効とまでは言えません。ただし「PRがFullより安い」という証拠としては使えません。具体的な根拠は次の3点です。
  1. Full baselineが、Track Cで示されたround別のh・公式選択の機会を持っていない。
  2. q>32が4096倍の外挿である。
  3. opt2の微小角問題が後に記録されている。
- 新しい計算が必要な事項（UNKNOWN）：
  - compiler差・baselineのh/公式差・control合成差・proxy外挿誤差のそれぞれの寄与。例えば「旧型のFull（S2・δ=.02）をcompiler Aで価格付け」「Track CのFullをopt2で価格付け」するようなablationが必要です。
  - 微小角問題が旧PR価格に与えた影響。

---

## E. 再利用可能な実装（固定commit・適用条件）

| module・関数 | commit | 適用条件・注意 |
|---|---|---|
| `track_c_df_truncation`：`replay_model`/`model`（H2/H4/H6の公開replay）、`candidates`、`kept`、`det_qualification`、`circuit_piece`、`price_piece`、`wrapper`、`choose`、`qualification_supplement`、`independent_one` | `da8b2c3f` | C-H4（H4 sector70）とH2/H6の保存入力。新しいSCF・fit・ground solveなし |
| `track_c_df_truncation_pr`：`partition`、`generator`（`cp.CachedCanonical`）、`bch_one`、`probe_one`、`screen`、`qualify_point`、`build_isometries`、`label_price`、`conditioned_records`、`event_circuit`、`price_one`、`independent_one` | `da8b2c3f` | **round別LD選択の既存実装。** H4では `for n in (2,6)` の拡張、LD menu、H4用のisometry・BCHが必要 |
| `track_c_d_internal.schedule`（ordered/D2/D4/Dtuned/partial_Y4/full_S2/full_Y4） | `da8b2c3f`の依存 | K3には「Truncated型のD公式＋中央R」の種別を追加 |
| `track_c_df_cancellation_conservative`／`_validation`（評価B：Hkeep spectral、積disk、時刻差） | `92495a04` | 決定論PF用。PR＋dropへの拡張が必要 |
| `track_c_closed_p3*`、`track_c_closed_p3_joint_opt*`、`track_c_bottleneck.rigorous_shots`、compiler A | `c5a3c102` | H4 LD3専用（218成分に固定）。既存PR policyの再利用・対照として使う |
| `b2_rpe_connection`（`B2RPERound`、`prepare_b2`、`strong_baseline_schedule`）、`b2_p1_*`、`b2_strong_pf_*`、`b2_streamed_native_cost` | `66a41ad9`ほか | P1条件（sector36、h=δ、atol0、opt1）でのみ。K3/A3の主pipelineには不向き |
| `rpe_four_round_phase_validation.reconstruct_rpe_phase`、`rpe_resource_accounting` | 共通 | 連続した2の冪round列。round間で構成が違っても可 |

---

## F. 次の研究判断のための比較

| 項目 | K3（D/R/drop三分割） | A3（round別D/R分割） |
|---|---|---|
| 技術的成立性 | 既存部品の合成で実装可能（drop付きpartition、新schedule種別、B型adapter、isometry） | H2/H6では既に動いている。H4はmolecule loopの拡張とH4 LD menuで実装可能 |
| 既存研究との差分 | 両端（PR・Truncated）は実施済み。中間点は未評価。fragment内の同一basisランダム化は新機構（未実装） | H2/H6は実施済み。**新規なのはH4のLD4–9**と、より細かいmenu・選択則だけ |
| 負結果・既存STOP | H4ではR=∅がA・Bの両方で優位。Aでは相殺を失って不利。partial Y4（負R時間）は負。H2は対象外 | B2-P1の固定r4ではshotが爆発（既知）。H6のPRはround別LDでもTruncated-Aに負け（1.62e12対1.22e12）、Truncated-Bには勝つ（1.87e12）。C2-AはSTOP |
| 総費用へ接続する機構 | 評価Bで切捨てbiasを小さいRで置き換え、Dのfragment数を減らす | 長roundでλ_Rを下げ、event数をλ_R²に比例して減らす。代わりにD blockが増える |
| 不足する科学的証拠 | Rで補完したときのG_17（A/B）。λ_R(R集合) | λ_R(k)（k=4–9）、各LDのm17信号・資格・価格・N |
| 次の最小検証 | K3-1：H6 m17（評価Bが主） | A3-1：H4 m17（LD4–7） |
| 別途必要な承認 | 科学計算・build/compile・価格MCの実行承認と、A/Bどちらを基準にするかの判断 | 同左。加えて、Track AがC-H4条件で検証すること（Track Cの承認は流用しない）の判断 |

### K3とA3の関係
- **H4**：drop可能なtailは極小（L1 3.13e-4、fragment 11–12は0）なので、K3はA3（LD k）とほぼ同じになります。H4ではA3-1を一本行えば足ります。K3のH4専用cellは推奨しません。
- **H6**：A3はTrack Cで既に実施済み（LD3/9/14）。K3はdrop引数を加えて、評価Bでの中間点を調べる追加検証です。
- 共通基盤：どちらも `track_c_df_truncation(_pr)` を使います。依存関係はK3 ⊃（A3と同じpartition拡張）＋drop＋B型adapterで、**共通基盤を一度整えれば両方に使えます**。
- 組合せ（roundごとのLDとdrop）は個別検証の後に行います。削減率は加算しません。
- 既存の手法で十分な可能性：H4でPR LD k≥4がTruncated-A（3.34e11）を下回れなければ、H4はTruncated PFで十分です。H6でもK3がTruncated-Aを下回る見込みは低く、K3の価値は評価Bを基準にする場合に限られます。

---

## 9. 最小検証計画（未実行・未承認）

### A3-1：H4長roundのLD再選択

1. 問い：H4のm17（次にm15・m16）で、LDを3から4–7へ移すと、λ_Rが下がってevent数が減り、D blockの増加を差し引いてもG_mが下がるか。比較は3段に分ける：(a) 既存PR LD3に対する改善、(b) 強Fullとの比較、(c) Truncated A/Bとの比較。
2. 変更する機構：round別のgeneration-prefix L_D^(m)。Closed P3 law・r・hの選択枠、D kind（ordered/D4/Dtuned）は既存のまま。
3. baseline：Track C H4の既存policy。PR（LD3）m17 G 8.36e11・総1.304e12、Full m17 3.04e11・総5.648e11、Truncated A 1.79e11・総3.338e11、Truncated B 総3.865e11。
4. 再利用：E節の `track_c_df_truncation(_pr)`、`track_c_d_internal.schedule`、compiler A、`choose`／`rigorous_shots`、decoder。
5. 最小追加：`bch()`・`probes()`・`build_isometries()`・`shortlist()` などのmolecule loopにn=4を追加。H4のLD k用BCH root、R support用のjoint isometry catalogue。新しいSCF・DF・stateは作らない。
6. 科学計算の範囲：
   - ① λ_R(k)（k=3..9）を保存係数から厳密計算（安価）。
   - ② m17について LD∈{4,5,6,7}×kind×既存hのmenu×r でprobe（70次元の元H ground信号）→ `screen`。
   - ③ LDごとの上位2点をdirected資格（A型。必要ならB型）。
   - ④ D moduleのcompile（compiler A）、R occurrenceの価格MC（Track C同様、training 2×32、holdout 2×64 trajectory/cell程度）、wrapper。
   - ⑤ 固定α=1/720での単一round G_17。
7. 評価量：λ_R、r、log range、ρ、β（PF/RTE/SOURCE/native/joint disk）、N_X/N_Y、1shot RZ（D/R/wrapper別）、G_RZ。副指標としてCX・size・serial depth上界（同じmodule）。T countは対象外。
8. 成立条件：
   - 局所GO：min_k G_17(LD k) が既存PR m17を明確に下回り、かつFull m17に近いか下回る。
   - その後の別承認で全18round policyを組み、独立holdoutで総G＜Full（5.648e11）なら「強Full比改善」とする。
9. 負・未資格：
   - LD≥4の全点が資格不成立（PF誤差、overlap、BCH rootなし）。
   - G_17がPR LD3以上、またはFull以上。
   - λ_R(k)が大きく、必要なrが減らない。
   - いずれの場合もこのfamilyをSTOPし、menuを事後に拡張しない。
10. STOPと承認：上記①–⑤の実行承認が必要（λ算出も保存値の新しい解析に当たる）。単一round段階の完了でmandatory STOP。全18round化は別途承認。

### K3-1：H6 m17、評価Bでの中間構成

1. 問い：相殺に依存しない評価Bで、Truncated-Bが落とせなかったtailの一部をRTEで補完し、prefix13のDを使うと、Truncated-B・PR・Fullより低いG_17を得られるか。同時に、評価Aで相殺を失う不利も測る。
2. 変更する機構：D=one-body＋13 fragment（Truncated-Aと同じretention）。R=fragment 14–18の非identity成分（全部、または|c|上位の一部で、残りはL1 budgetでdrop）。Closed P3。
3. baseline：Track C H6の既存点。Truncated-A（prefix13 S2 h=.08、G_17 5.86e11・総1.223e12）、Truncated-B（prefix15_I0 Y4 h=.16、総1.868e12）、PR（LD14 D4 h=.32 r=1、G_17 8.56e11・総1.618e12）、Full（Y4 h=.16、G_17 1.07e12・総2.082e12）。
4. 再利用：A3-1と同じものに加え、`kept`のdrop集合と `track_c_df_cancellation_conservative`。
5. 最小追加：`partition(n, ld, drop)`、drop付きの `block_data`、「Truncated型D（S2/Y4）＋中央R(h)」のschedule種別、PR＋dropのB型adapter、fragment 14–18のisometry。
6. 科学計算の範囲：
   - R集合のλ_R。
   - m17だけで、D公式2種×h∈{.08,.16}×r∈{1,2}×R変種≤3（≤24 cell）のprobe・screen。
   - A・B両方の資格、価格（Dが端点と同じなら保存価格を再利用、R occurrenceはMC）。
   - 固定αでの単一round G_17。
7. 評価量：A3-1と同じものに加え、B_TとB_Pを別々に記録。
8. 成立条件：評価Bで G_17^{K3} ＜ min(G_17^{Trunc-B}, G_17^{PR})。評価AでのG_17も併記する。
9. 負：A・Bの両方で端点以上。Bが閉じない。Rに多くのeventが要る。いずれもSTOP。
10. STOPと承認：A3-1と同様。加えて、評価AとBのどちらを科学的比較基準にするかをGPTが先に判断する。

**技術的な優先順位の意見（採択ではありません）**
- A3-1を先に行うことを勧めます。H4は元Track Aの系列で、pipeline拡張が小さく、計算も軽く、H4のPR費用の支配要因に直接作用するためです。
- K3-1は、評価Bを基準にする判断が出た場合に限り、実施する価値があります。
- 横断調査のK3初手案（D=LD3＋coeff_L1_2の除去集合）は、上記の理由で推奨しません。

---

## 主要資料のリンク（固定commit）
- [Track C Truncated比較REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/docs/research/track_c_df_truncation_20261108/REPORT.md)／[ROUNDS.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/ordering/ROUNDS.csv)／[CANDIDATES.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/artifacts/track_c_df_truncation/2026-11-08/ordering/CANDIDATES.csv)／[PR module](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/src/trottertracks/track_c_df_truncation_pr.py)／[Truncated module](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/da8b2c3f934aa7410f2e7eb90bb1b5575b6dfc13/src/trottertracks/track_c_df_truncation.py)
- [Track C 相殺P1/P2 REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/92495a04d52c65142ed16b3abe977345065f9745/docs/research/track_c_df_cancellation_validation_20261108/REPORT.md)
- [Track C 最終統合](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3c81c2677988c7b9491eeee31fbcc0f9b29b078/docs/research/track_c_o3_integrated_20261108/TRACK_C_O3_INTEGRATED_REPORT.md)／[Closed P3共同最適化](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md)／[H4 pilot INPUTS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a78054dc48737b06b9396e649f53ecb9efd4bd3b/artifacts/track_c_h4_pilots/2026-11-06/run01/INPUTS.json)
- [横断調査 K3案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2e79d80dc9cce2885580e50fe6f1705da8fa41af/docs/research/pr_rpe_cross_track_20261109/MINIMUM_VALIDATION_PLANS.md)
- [B2-P1 shot](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d5ca357ccfd8fcc850bef800c5bcadfde2a153e/docs/research/track_a_b2_p1_shots_20261107/REPORT.md)／[強PF 8 arm比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e162ef708d7f721cd9c95ae07a833f7301d2b449/docs/research/track_a_b2_p1_hp_branch_shots_20261107/REPORT.md)／[B2 RPE接続 module](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66a41ad9219f12377e7d325e272c1a90d66ab7e5/src/trotterlib/b2_rpe_connection.py)
- [旧opt2 M06-F](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research_direction_full_opt2.md)

## 10. 確認できなかった事項・UNKNOWN
- λ_R(k)（H4 k=4..9、H6のtail部分集合）の正確な値：未算出。下界（drop norm）だけを引用しました。
- Track Cのraw tar（MENU/INPUTのexact support、R occurrenceのraw）：未展開。CSV/JSONの該当行だけを読みました。
- B2-P1とC-H4の、H・DF factor・stateのbit一致：未照合。
- 旧opt2の逆転要因の寄与分解、微小角問題の影響：要計算。
- H4 LD k≥4でのBCH root（Dtuned）の有無、PF誤差：未評価。
- 新Aの新branch（`algo-improve-new-a-total-cost-20261109`）とTrack D long-qの更新：別系列のため読んでいません。

## 11. 実施内容・変更ファイル・検証
- 実施：固定commitの資料・SOURCE（テキストとAST）・保存CSV/JSONの該当行を静的に読み、K3/A3の対応関係・差分・再利用方針・最小検証計画を作成しました。新しい科学計算、数値再解析、build/compile、テスト実行、module import、branch操作、repository変更はいずれも0件です。
- 変更ファイル：
  - `.agent/docs/review.md`（repository外の作業報告ファイル）（本報告で上書き）。
  - Claude Codeのauto-memoryに、本調査の要点のproject memoryを1件追加。
- 数値はすべて原報告・保存値からの転記です。§A.3、§B.8の「目安」は、転記値と既存の式による静的な算術で、科学的な結果ではありません。

## 12. 次の作業（mandatory STOP）
- **ユーザー**：本報告をGPTへ渡す。
- **GPT**：次の点を科学的に判断する。
  - (1) A3-1とK3-1のどちらを行うか（または両方か）。
  - (2) 評価A（combined）とB（個別上界）のどちらを比較基準にするか。
  - (3) Track AがC-H4条件・Track C pipelineで、自身の契約として検証してよいか。
  - (4) §B.8の静的読解（λ_R²T と単価比）の妥当性。
  - (5) 詳細レビューが必要か。
- **ユーザー**：選んだ計画の、§9の6項に挙げた科学計算・build/compile・価格MCの範囲を明示的に承認する。
- **Claude Code／Codex**：承認があるまで、実装・科学計算・文書修正には自動で進まない。今回のmandatory STOPを維持する。
