# P0 HANDOFF：ユーザー／C／B向け

結果の正本は[REPORT](REPORT.md)と[run01/RESULT.json](run01/RESULT.json)。

- 実行SOURCE：`a287b8e6cb629c6a089c99b66b758751910ee15a`
- branch：`codex/d-compiled-residual-c-minus-s-20261113`

## ユーザー（判断事項）

1. **価格定義**
   - PR・Full-H RTEのcontrolled tailは、汎用control（CCX·CRZ·CCX）で生成されている。
   - P0で、安価な`CX·CRZ·CX`と**unitaryが同一**であること、compile後にZZ回転1回当たり**RZ +17〜19・CX +10**の差があることを確認した。
   - 回路を安価な形にするかどうかは価格定義の変更であり、PFとの公平性に関わる（実装監査R1）。
2. **C−Sの扱い**
   - 推奨は案B：既存のSとK0/K1を維持し、δ_ZZ·N_ZZを残差modelの一項にする。
   - 案A（S′）はK0/K1の再実行が必要。
3. **P1の実行**：既存の8 wrapper案に、pathごとのN_ZZ・N_Z・order-2数の記録を加えるか。

いずれも、別の明示承認が出るまで実施しない。

## Codex C

- K1のE[S]（`83d219cc`以降）は、**構造S（CX·CRZ·CX計数）**の期待値である。compiled側の汎用control形とは、ZZ回転1回当たり約17〜19 RZ異なる。
- 案Aを選ぶ場合の作業：
  - `h4_structural_kernel`の局所wordを、汎用形の文字どおりの展開へ置き換えたS′のkernelを作る。
  - K0 parityを再検証する。
- 案Bの場合は、K1をそのまま使える。
- E[N_ZZ]＝qR·P_ZZ（P_ZZ=0.9163、保存support確率の和）は、法則から厳密に得られる。

## Codex B

- 変更不要。構造SもP0の値も価格recordではない。
- 将来E[C]を受け入れる時は、C−S残差recordの`components`に、lowering項（δ_ZZ・N_ZZのSOURCE）と、その他の残差項を区別して持たせる。HANDOFF_B（`d_compiled_residual_c_minus_s_20261113`）の形式に従う。
