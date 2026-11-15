# C向け引渡し：PR-DF LD1 F0 → 残差価格差

- **変換できた範囲**：13 round×X/Yの26軸。C−Sの直接の鋭い幅ではなく、CとSの別々の全support SOURCE envelopeから、保存F0の条件付きTV×幅をRZ単位へ換算した。m0約3.056e−10、m8約0.7642、m12約49,770.254 RZ。
- **実PCG64に残る仮定**：元normal samplerの固定seed/seed ensembleと、F0のIID一様53-bit格子joint lawとの接続は未証明。actual-PCG価格差は全26軸null。標準Qiskit1.3.0 SOURCEに従うcompiler動作も適用条件である。
- **次に最も必要な証拠1件**：対象seed確率測度を明示した、元PCG64 samplerのjoint-lawと条件付きIID-gridとの距離/couplingの根拠。追加compileの前に解決する。便宜的なseed数、IID仮定、PF2%を導入しない。

[REPORT](REPORT.md)にSOURCE別の導出と13行表示表、[C_HANDOFF.json](C_HANDOFF.json)に26軸の読取locator、[RANGE_EVIDENCE.json](RANGE_EVIDENCE.json)にexact分子・分母、[COMPILER_ROUTE.json](COMPILER_ROUTE.json)にcompilerのSOURCEと適用条件を保存した。今回の最終公開commitを相対path/selectorに添えて参照する。

## 受け取る式と根拠

同じround/axisで、μをF0の保存重み比によるpath law、νをF0仮定下のgrid path lawとする。同じlabelを同じword/angle/phase/scalar/wrapperへ写し、supportの和集合で同じf=C−Sを評価する。ideal正/gridゼロになったlabelも含む。

```text
U_S = q(2843+2082R)+5+1[a=Y]
U_C = q(10471+7682R)+16+5·1[a=Y]
-U_S ≤ f ≤ U_C
osc_union(f) ≤ U_C+U_S = q(13314+9764R)+21+6·1[a=Y]
|E_μ f−E_ν f| ≤ saved_F0_path_TV(m) (U_C+U_S).
```

U_Sは元Accountingのliteral展開とbasis/Z-foldが個数を減らすことから導く。U_Cはnew-ZZ/K2/8-mode Gaussian/head/wrapperのgate文法を、標準equivalenceの全basis gate数で抑える。BasisTranslatorのcost探索と、target errorなしopt1の全gate数非増加を使う。RZ数だけの単調性やC≤Sは仮定しない。詳細はREPORT §3–4。compiled RZを観測した上界ではなく、指定release algorithmを正常完了した場合のSOURCE上界である。

固定経路はAccounting 8457c443、Gaussian/compose 155529e2、provider c31ef424、context 1a87e533、new-ZZ/P2 fast SOURCE 39ddd59e。完全SHA/path/git blobはRANGE_EVIDENCE.sourcesにある。Qiskit1.3.0 upstreamは896d1c8b72888a52a46081a6d1f4de8c56db8aea。DのPF compiler調査dd435d07を再利用したが、PF周期閉包や価格をPR-DFへ移していない。installed Rust binaryの過去receiptとのbit単位同一性は主張しない。

## 読取と受入で分けるもの

最終公開commitから次の順に読む。新しいAPIやC worktreeの変更は不要。

1. C_HANDOFF.json.rows[i]の(m,axis,physical_id)とbase_axis_indexを照合する。950577f4の26軸対応、元C索引と同じIDを保持する。
2. range_recordから当該行のq/R、S_upper、C_upper、residual interval、oscillation、conditional_RZ_shift_upperのexact値を読む。
3. saved_F0_boundの固定{commit,path,selector}から元raw ec78fde5のROUND_BOUNDS.rows[m]へ戻る。event_count=qRとphysicalを照合する。X/Yは同じ保存TVを共有し、別々の価格関数/上界を持つ。
4. compiler SOURCE条件とF0の独立格子仮定が対象に合う場合だけ、**期待残差差の絶対値上界**として扱う。符号付き補正、モデル誤差、実PCG64・全law E[C]の受入とは区別する。

CのF0受入は259a0b2d:docs/research/pr_df_ld1_f0_admission_20261115/HANDOFF.mdで別途公開済み。今回の新しいrange/積をCが正式受入したとは扱わない。direct_residual_lower/upperはnull、actual_PCG64_joint_distance/price_shiftもnull。必要精度・信頼水準・許容誤差は未採択。

有限receiptはm0/m3/m6が較正、m4が独立holdoutを維持する。m4の約−0.544502%は各軸4 path平均に対する凍結予測差であり、全support残差幅ではない。P1/replayをpoolせず、観測min/maxを幅へ使用せず、m4を較正へ戻していない。保存E_I[S]、凍結正本3cac53da、m4受入9edfa5d3、F0 d749ef92、旧raw/STOPは不変。

## 件数・ロードマップ差分・STOP

共通SOURCE上界の導出1件、envelope/TV積26軸、保存TV再利用13件（26参照）。取得を必要とする価格receiptは0件。保留はactual-PCGの価格差全26軸等の未確定欄で、欠測を0に補完していない。科学的な入力/計算異常は0。静的AST/metadata/locator/exact算術/保全の確認はCHECKS.json。

全support rangeの不足を、**標準compiler SOURCE＋IID-grid仮定下の粗い上界**まで前進させた。m8まで1 RZ未満なのは単位比較であり、合否・十分精度の認定ではない。長roundでの価格精度、actual samplerのlaw接続、有限標本/モデル移送は未解決。G比較の対象ideal_finite_RTE_lawは変更せず、正式E[C]/CI/N/G/winnerはnull。ロードマップ本体は統合担当が更新する。

新path・H4 build/compile・F0再実行・rare探索・再fit・価格予測・CI・正式E[C]・Q3/N/Gは0。full-wrapper statevectorなど旧gateの未実施範囲を維持する。追加sampling/実行は別指示。指定branchへcommit・push・remote照合して**mandatory STOP**。
