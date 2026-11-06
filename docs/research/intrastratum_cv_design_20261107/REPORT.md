# 層内Control Variates・層別＋CV：候補調査の報告

2026-11-07 JST。静的SOURCE調査・既知の数理式の適用・設計のみ。**新しい科学計算、保存結果の数値再解析、新control/推定器実装、人工test実行は全て0。** 新controlの正式採択・科学的GO/NO-GO・詳細科学レビューをしていない。

## 結論

**追加の量子測定なしに利用可能で、同じB1-OL層内で非定数となり、追加分散削減の可能性を持つcontrolは見つかった。**
第一候補はrotation labelのfragment群indicator。平均は保存actual Qとcatalogからの厳密な有限和で表現できる。h=1の非定数例を、既報support数とI/Z/ZZのSOURCE制約から示した。

ただし「非定数＋既知平均」は、bitとの共分散が非零であることの証拠ではない。実際の相関、有限係数の改善、学習費を含むRZ利得はUNKNOWN。feature値を取り出す量子費0と、係数を学ぶ量子費0を混同しない。

## 成果物

| 文書 | 内容 |
|---|---|
| [INTRASTRATUM_CONTROL_CANDIDATES.md](INTRASTRATUM_CONTROL_CANDIDATES.md) | conditional law、3候補、非定数/平均/費用/range、除外候補、調査上の優先順位 |
| [STRATIFIED_CV_DESIGN.md](STRATIFIED_CV_DESIGN.md) | 不偏性、層内分散と有限学習、Born floor、旧control相殺、配分/全費用/有限confidence、SOURCE再利用の限界 |
| [NEXT_SCIENCE_VALIDATION_DRAFT.md](NEXT_SCIENCE_VALIDATION_DRAFT.md) | h=1一特徴、train64＋paired全5層評価160＝224-shotの未採択案、独立性・会計・STOP |
| 本書 | 結論・出典・確認範囲・未解決事項 |

## 調査上の優先順位

1. **rotation fragment群indicator**：主なh=1で変化し、単一有界特徴とexact平均にできる。測定前のlawだけによる群balanceを提案し、稀な単labelの学習難を避ける方向。実際の群・mean・相関は未取得。
2. **rotationのsigned support**：既存sign/diagonal supportを使う。有限和で平均を作れるが、実conditional supportで異なる特徴値があるか、X/Y応答との相関は未確認。
3. **order-2 productsのlabel一致**：同一signed involutionの積がIとなる構造的理由がある。平均はexact平方和等。効くのは希少層で、学習費と全target寄与の釣合いが課題。

Taylor次数・event phase・基本rotation角は各層内固定。h=0はpathそのものが固定なので、任意のpath metadata controlが定数。価格だけとの相関や別basisのZ文字列一致を、測定信号との相関に置き換えない。Born/path信号/state/真variance/測定wordは特徴へ渡さない。

## 推定器と期待できる範囲

$$
\widehat\mu=\sum_h p_h
 \{\overline B_h-\widehat\lambda_h^\top(\overline u_h-\nu_h)\},
\qquad \nu_h=E_Q[u_h|h].
$$

独立trainでλ_hとquotaを凍結し、exact ν_hで中心化するfixed-nモデルでは不偏。層ごとのoracle追加削減はγ_hᵀΣ_h^+γ_h。pathに条件付けたBorn測定ノイズは残る。
現在の3-controlは u−ν_h=0 となるため、新推定器へ足しても層内削減は0。新しい自由label特徴により、層別が取り除いていないpath変動へ働く可能性がある。

同quotaのoracle分散差と、有限train/range/cal/整数配分を含む全期待費用改善は別。今回はいずれも実H4で数値評価していない。

## 固定資料と静的SOURCEの対応

対象：保存linear H4/1.00 Å/STO-3G/legacy DF12、8system qubits/全4電子sector、L_D3/m0/q1/δ=.02/r1/K2/coefficient_atol0、X/Y、N1 actual Q/selected full policy、N2保証ground、認証済みexact engine、理想gate限定。

- [理論8d8a290](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/8d8a290fa4d4cf24f968aea27953ad5422e418b3/docs/research/a1_b1ol_theory_20261107)：元controlの全分散・hybrid式と費用条件を参照。原文は変更していない。
- [752-shot結果3c8e55a](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3c8e55a2728f86a1eda17a2e0105050a637e7c86/docs/research/o2_a1_n3_pilot_admission_20261107/REPORT.md)：報告済みINCONCLUSIVE・zero-variance問題・費用/実行負荷の範囲を確認。bit/係数への新fitや新候補評価はしていない。
- [N1 744e1a0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/744e1a09661ebba6841fe5faf456f88c41c47780/docs/research/o2_a1_new_target_n1_20261106/REPORT.md)：保存finite law/正supportと固定a,b/5層の定義を確認。新しいν_h・Σ_hを実表へ計算していない。
- 正式共通指示はmainにあるresearch_project_instructions_unified_2026-10-25.mdを実際に読み、最新の静的限定指示を優先。研究概要・現状を全読し、VALIDATION_STATUS/manifestの既存証拠statusを確認した。
- 基点3c8e55aのRTEComponent/DFDiagonalPauliComponent、RTE._make_event、ConditionalSampler.draw/_pick、ControlBank.joint_mean/pattern_mass、fit_and_freeze/FrozenCV、production_mean、pilot_mean、calibrate/sufficient_shots、SavedPrices、manager・journalのSOURCEを静的に確認。
- 既存testsのconditional law、phase/順序、bit-only facade、frozen fit、range、独立性、全正層・価格保全を確認。testを新規実行していない。
- 一般の多重CVの最適係数/L²射影は、理論8d8a290が参照する一次文献[Glynn–Szechtman (2002) §2](https://web.stanford.edu/~glynn/papers/2002/GSzechtman02.pdf)の該当式も確認した。網羅的文献レビュー、新規性認定、漸近理論による小pilot性能保証はしない。

既存APIはglobal control平均・3-control/rank4・一つのmodel・旧method IDs等を仮定するため、層別＋CVが既に実装済みだとは扱わない。将来はconditional bank/cell別model/paired会計の最小adapterが必要。既存kernel/取得/価格/管理を置き換える必要はない。

## 次の最小案と未解決事項

提案はh=1の一scalarだけをX/Y各32 fresh trainで学び、各軸全5層(4,64,4,4,4)の共通fresh評価をB1-OL/SCVへpairedに適用する**計224 physical shots**。旧752データを混入しない。h=0,2,3,4のλ=0は設計段階で固定し、失敗後の救済にしない。
この案は追加層内削減だけを低次元で問う診断で、精度保証付き総費用比較ではない。

未解決は群balanceの可否、actual ν/Σ、応答との相関、train rank/係数誤差、range guard、全段階RZ利得、confidence/返却率、source接続の人工検証。群選択ルール・224shots・range guard・seed/streamを含め、全条件は提案まで。新しい実行契約の採択を代行していない。

## 保全・公開・STOP

独立worktree/専用branchで文書のみ追加し、元main/dirty/untracked、旧SOURCE/結果/STOP/科学契約/grant/ledger/marker、C1凍結を保全する。索引・研究概要・研究ノート・証拠manifestへの追記は未採択設計の案内であり、旧scientific statusを書き換えない。
SOURCE/入力・旧実行記録はbyte hashで確認し、new control実装・project module import・test/RNG/mean算出/Born/compile/保存結果再解析のcallは0。
remote SHA・公開本文/blob/treeを照合後mandatory STOP。新科学計算・実装・詳細科学レビューへ自動進行しない。
