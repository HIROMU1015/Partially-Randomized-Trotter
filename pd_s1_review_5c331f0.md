# P-D S1レビュー：B1aの位置付け、Case A相当の解釈、次の研究方向

レビュー日：2026-09-26
対象：`HIROMU1015/Partially-Randomized-Trotter`
branch：`all-r-coherent-opt2-reoptimization`
commit：`5c331f0af5c3ab97b8b3dda9c2d9fc590528119f`

## 0. 結論

1. **B1aは主たる資源比較baselineから外し、内部workを省いた診断用ablationとして残す。** 結果が悪かったから除外するのではなく、最適化変数m_Dを増やす費用を目的関数が評価しないためである。
2. **事前登録上のCase B＋undetermined_boundaryは保存する。** 別の事後解釈として「B1b/B2/B4の選択は、今回の有限候補集合とB4参照のもとでCase A相当」と記録してよい。元の判定を書き換えない。
3. **P-Dの独立した研究上の新規性は、現在のS1結果では実証されていない。** 有限補正を追加する選択上の利益は得られず、B1aの失敗も新しいRTE効果ではない。P-Dを主題としてS2へ自動進行させない。
4. **次の方向の第一候補はR3の再設計**。ただし、既知のPR優位性図を描き直すのではなく、固定DF表現に対する実装構造・PF誤差・tail負担から、分割と決定論/部分ランダム化の選択を予測する問いへ絞る。R6は特定の表現・抽出法を変える必要が見つかった場合、R8は深さ制約という別タスクを研究対象として選ぶ場合の候補。

これは正式方針をリポジトリへ書き込んだものではなく、指定snapshotに対するレビューである。新しい量子シミュレーション、compile、テスト全suiteの実行はしていない。保存資料とコードの照合および記載数値の算術を行った。

## 1. 照合した証拠

指定の5資料を入口とし、選択・判定コードを追加確認した。

- [S1] `docs/research/研究概要・現状.md`
- [S2] `docs/research/pd_primary_research_contract.md`
- [S3] `docs/research/pd_s1_fair_comparison_preregistration.md`
- [S4] `docs/research_direction_pd_fair_comparison.md`
- [S5] `artifacts/research_direction_pd_fair_comparison/2026-09-26/pd_s1_fair_comparison_v2.json`
- [S6] `src/trotterlib/research_direction_pd_fair_comparison.py`
- [S7] `partial_randomized_trotter_research_redesign_20260925.md`

[S5] result fingerprint：`0ba7764da7b7d8b7e195a5c315d3dc0a65c2c79ce01c51cf021a2685977487d2`。
Contents経路の空contentを空ファイルとは扱わず、blob SHA `f983a2d50809e456defeba977bd525262029d11d` で内容を取得して選択結果を確認した。

## 2. 実際のS1結果

固定条件はH4直鎖、1.0 Å、STO-3G、8 qubit、4電子、DF rank 12、L_D=3、共通時間T=0.8 a.u.、総位相誤差予算8.0e-7 rad。比較はcomponent-action proxyとB²によるモデル目的であり、compiled gate数でも実測shot数でもない。[S2–S5]

| 範囲 | B1b/B2/B4の共通選択 | B4目的値 | B2目的値 | B2のB4 regret |
|---|---|---:|---:|---:|
| nested | 新4次、delta=0.2、m_D=16、R=16、K=2 | 2854.982343 | 2854.321195 | 0 |
| native | 新4次、delta=0.2、R=16、K=2 | 215.801160 | 215.733579 | 0 |
| combined | nativeの上記設定 | 215.801160 | 215.733579 | 0 |

B1bにも選択点のB4 false acceptanceはない。B2の目的値の相対差はnested約-0.0232%、native約-0.0313%。限定K=4感度でもPF・delta・m_D・Rの選択は変わらず、目的値変化は約4.5e-7 relativeである。[S4–S5]

## 3. B1aをどう扱うか

### 3.1 資源baselineとしての問題

B1aはouter exponential-stage数だけを目的とする。したがって固定したPFとdeltaについて、内部m_Dを増やしても主目的値は変わらない。一方、コードは同点時にdeterministic phase boundを優先する。今回の候補列では、内部誤差が小さい大きなm_Dへ選択が動く。[S3,S6]

これは「精度向上と計算負担の最適な釣合いが探索上端にある」という観測ではない。主目的がm_Dを識別せず、二次的な精度優先規則が上端を選んだものである。m_Dに対する誤差単調性を一般定理とはしないが、今回の上限依存を説明できる。

H_Dの指数演算を単位費用のoracleとする別モデルならB1aには意味がある。しかし、内部fragment workを実際に比較する今回の研究では、そのoracle modelとrealized-work modelを同じ資源baselineにしてはいけない。

**扱い：B1aは履歴とablationとして保存し、B1bを最小の主資源baselineにする。** B1aを削除したり、過去の失敗をなかったことにしたりしない。

### 3.2 m_Dをさらに増やさなくてよい理由

B1aの選択点は `nested_4th_new_2_d0.4_m128_r16_k2`。

- deterministic phase bound：7.294033104325027e-7 rad
- finite-RTE phase bound：1.8178592578904054e-6 rad
- 合計：2.5472625683229083e-6 rad
- 共通予算：8.0e-7 rad
- finite signal-error bound：1.8178592577905343e-6

finite-RTE成分だけで予算の約2.272倍である。固定したformula、delta、R、Kではこのfinite signal boundはm_Dに依存しない。unitaryなdeterministic参照の半径は高々1なので、半径を最大の1としてもB4の位相上界は予算を超える。[S5,S6]

従って、**同じtail設定でm_Dだけを256等へ増やしても、この候補のB4不適格は解消しない**。これは実際の物理誤差が必ず予算を超えるという主張ではなく、採用しているB4上界による受理が不可能という意味である。

## 4. Case A相当と解釈してよい範囲

### 4.1 保存すべき二つの判断

- 事前登録の一次結果：`primary_case=B`、`undetermined_boundary=true`。
- 事後の研究解釈：**内部workを評価するB1b/B2/B4に限れば、このS1の有限候補集合で同じ最良設定を選んでおり、選択上はCase A相当。**

B1aを結果後に取り除いて「事前登録Case Aを達成した」とは書かない。post-hoc interpretationとして別ファイルへ追記する。

### 4.2 言えないこと

- B2が全candidateのfeasibilityを正しく判定する。
- 全候補でcost/shot誤差が小さい。
- R=16より下、未評価K=4全grid、他のT・精度・分割でも最適設定が同じ。
- finite-RTE誤差を今後無条件に無視できる。
- B4が実際の回路・測定の正解である。

regret=0は「そのモデルが選んだ設計を、今回のB4有限参照集合で再評価すると最良だった」という意味である。B4自体も未検証要素を持つ解析モデルである。[S2–S6]

### 4.3 Case Bのreasonは因果的な証拠ではない

コードはB1aまたはB1bが失敗すればBを立て、reasonに `absolute_tail_time_model_is_needed_but_finite_correction_is_not` を入れる。しかし今回B1bもB2/B4と同じ設計を選んでいる。従って「絶対tail時間を入れなければ選べなかった」とは示されていない。[S6]

支持されるのは、内部workを無視するstage-only比較が不適切だったことと、B1b以上のモデル間で現在の選択差がなかったこと。

## 5. Case Dについて残る分類上の注意

事前登録は内部精度・nested/native構成の影響が大きい場合をDとしていたが、実装の `construction_difference` はnested/nativeで選択した**formula_labelの不一致だけ**を検査する。[S3,S6]

今回、同じ新4次でもB4目的値はnested/nativeで約13.2297倍異なる。従って「PF名が同じだから構成の影響は小さい」とは言えない。

ただし、この比をそのまま新しい回路優位性とはしない。現在の費用はcomponent-action proxyであり、nested内部substep、融合の数え方、nestedがouter/inner位相差の和を使うこと、nativeが直接位相差を使うこと、m_D gridなどの影響を区別する必要がある。[S3,S6]

ここは新しい大型実験より、既存rowsのwork内訳・feasibility余裕を照合する追加解釈が先である。**資源モデルの選択能力はCase A相当でも、構成の影響が消えたわけではない。**

## 6. P-Dの新規性の判断

元のPR論文は、高次PFの負時間を含むtail絶対時間、比例配分、normalizationとsampling負担を解析している。[W1] SPRINTもnear-integrability、ランダム化、factorization等を統合したPF設計を扱う。[W2]

S1では、これらの既知の考えを越える「有限補正を使うことで選択が改善した」という証拠は得られていない。stage-only B1aを直すこと、negative-time実装が正しいこと、既知leading modelが一条件で成功したことだけでは、独立した方法論上の新規性にならない。

したがって、**現在のP-D主張は積極的な主研究開発を停止**する。コード・artifact・失敗機構の記録は残し、後続研究の検証基盤として使う。全条件で新規性があり得ないと証明された、という結論ではない。

B2の十分性を証明する新しい一般条件が得られる可能性は残る。しかし、それを示さずにTや許容誤差を動かし、B2/B4が違う点を見つけるまで探索することは勧めない。

## 7. 次の方向：R3を第一候補にする

### R3：固定DF表現の分割とアルゴリズム選択を予測する

推奨RQ：

> 固定DF表現で、tailの係数L1、分割依存PF誤差、決定論fragmentとランダムcomponentの実装費用を用いると、部分ランダム化する分割と決定論へ戻すべき条件を、詳細な全探索より少ない評価で予測できるか。

現在のS1はL_D=3だけであり、分割設計やtailなしendpointとの比較を解いていない。[S2] 有限補正の重要性が小さい今回の知見は、限定domainで簡易モデルを基盤に使う理由になる。H12や長RPEへ進む理由にはならない。

先行PRは既に優位領域と資源評価を扱うため、lambdaや精度の掃引表だけでは新規性不足。[W1] 狙う差分は、DF実装費用と分割依存誤差からの**未使用分割の予測、詳細評価が必要な候補の絞り込み、誤選択の仕組み**である。これは新規性が確認済みという意味ではなく、次に具体的な差分として監査すべき候補。

### R6：表現/抽出法は第二候補

「lambdaを減らす」「cost-aware importance sampling」は既知である。[W3,W4,W5] 実装費用を考えたR3選択で、表現が本質的制約と分かった場合に、一つの表現自由度へ絞ってR6-aへ進むのが自然。新規samplerへ先に全面移行しない。

R6-bを残す場合は、既存trajectoryの正のcost cから理想ISの比 `(E sqrt(c))^2/E c` を評価するだけで改善余地を診断できる。ただし経験分布の診断であって、未観測rare eventを含む真分布の保証ではない。この理想式自体は既知。[W4]

### R8：別の目的・制約を選ぶ場合

深さ上限はS1にない新しい問題設定である。研究として意図的に選ぶことはできるが、P-Dが止まったから自動採用しない。低深さ位相推定と最大時間/総資源の交換関係は既に研究されている。[W6] 特定hardwareの上限を勝手に仮定しない。

**順位：R3の研究設計 → 明確な表現上の障害が見つかればR6 → 深さ制約自体を目的として採るならR8。** 全方向を連続的にpilot実行する自動workflowにはしない。

## 8. いま追加するなら、既存artifactの再解析だけ

### 8.1 解釈を閉じる再解析

**研究判断を変える理由**：B1a由来の形式的な未決定と、主資源baselineの不確かさを区別し、Case Dの狭い実装条件による見落としを整理する。この結果によって、P-D追加精密化が必要か、それとも資源比較の再設計へ移るべきかが変わる。

**固定条件**：

- 上記commitとresult fingerprintを固定。
- 新規Hamiltonian生成、対角化、RTE sampling、compileを行わない。
- S1 v2は変更せず、事後解釈用artifactを別名保存。
- 主比較はB1b/B2/B4。B1a/B0はablation/診断。
- T、位相予算、候補集合、regret閾値5%、元のfeasibility規則を維持。
- native/nestedを別々に表示し、主モデルが選んだ点のregret、false acceptance、objective/shot-factor誤差、上限/下限hitを再掲。
- 最良B4値から5%以内の候補を別集合として固定し、その領域のモデル誤差・feasibilityを確認。全候補の誤分類とも区別する。
- B1aのm依存についてouter-stage値、internal work、deterministic/finite phaseを並べる。
- nested/nativeの目的値比をwork内訳へ分解し、formula差・構成差・誤差規則の差を別欄に保存。

**終了条件**：追加gridを開始せず、(a) 現S1の主baselineに未解決の選択問題があるか、(b) 構成差の意味、(c) 次テーマ候補をまとめて停止する。

これは既に見たデータの再解析であり、独立holdoutや元からの事前登録結果とは呼ばない。再解析前の計画を記録することと、元の結果の事前登録は区別する。

### 8.2 R3を採る場合だけの次の小規模検証案

**問い**：uniform component-actionによる分割選択と、実装費用を区別した分割選択が、意味のある資源差を生むか。

**判断が変わる理由**：差が再現し予測できればR3の中心仮説となる。費用補正後も同じ判断で既知モデル以上の情報が出なければ、このR3仮説を主題として追わない。

**最小の事前登録案**：

- 同じH4 snapshot、native構成、共通T=0.8、位相予算8.0e-7を起点にする。
- 分割を現在のL_D=3、未使用の中間prefixを一つ、tailなしendpoint L_D=12に限定する。中間prefixとその選定根拠は結果を見る前に固定。
- 全分割へ同じ既存PF familyと刻みgridを与える。tailなしではR=0、normalization=1の専用経路を用いる。
- 比較するcost proxyは、既存のuniform actionモデルと、同一compiler条件で独立に較正した決定論fragment/ランダムcomponentの異種費用モデルの二つ。後者もfull compiled total costとは呼ばない。
- 未使用分割を評価する前に特徴量と予測規則を固定。holdoutに対して規則を作り直さない。
- 一部の短い比較回路だけを、proxyによる選択の独立検査に用いる。H12、S2、長RPE全回路を追加しない。
- 主判定：共通詳細参照でのfeasibilityとregret。閾値は今回と同じ5%を継続する案とし、予測不確かさが差を覆う場合は未決定。変更する場合は新しい理由と数値を計算前に固定。
- 最良が探索端でも全域拡張へ進まない。未判定範囲を明記して一度停止。

この8.2は即時実行の指示ではない。R3の最も近い先行研究との差分を一文にした後に採否を決める。単に新しい分割で勝敗が変わるだけでは、新規性が成立したとはしない。

## 9. 文献

[W1] Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647（参照箇所：Appendix Aのpartially randomized product formulas、絶対時間・RTE配分・normalizationの議論）。PRX Quantum 7, 020332 (2026)。

[W2] Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741（SPRINT/GRADE）。

[W3] *Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization*, arXiv:2212.07957。

[W4] Cugini, Atif, Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1。Theorem 1の一般costに対する最適分布と資源比。

[W5] Kanasugi et al., *Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime*, arXiv:2603.22778v2（UWC、主対象はPauli-LCU）。

[W6] Ni, Li, Ying, *On low-depth algorithms for quantum phase estimation*, Quantum 7, 1165 (2023), arXiv:2302.02454。

文献の存在と関連箇所を確認したが、R3/R6/R8について全ての先行研究に対する新規性を確定したものではない。
