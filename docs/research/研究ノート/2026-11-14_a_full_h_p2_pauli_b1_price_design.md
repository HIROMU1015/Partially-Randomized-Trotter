# 2026-11-14：Full-H κ2 P-2の較正設計の修正と、PR-Pauli B1の価格検証計画（設計のみ）

## 作業の範囲

- Aが技術設計と人工testを行った。
- sampling、compile、実価格の評価・再解析、fitは0。
- 候補はいずれも提案で、正式な採択ではない。

## Full-H P-2（A1）

**現行案の欠陥**
- 現行のC/D案（18 compile）は、較正点がq=1だけである。
- そのため、\(E[C]=a+bs+c(q-1)\)のcが識別できない（rank 2）。
- 既存の`fit`も拒否するが、拒否するのは価格を取得した後である。

**識別の条件**：較正点の\((q-1,qR)\)が一直線上に並ばないこと。

**追加した検査**：新module `finite_price_design.check_plan`は、取得の前に次を拒否する。
- rank不足
- 領域の混在
- 較正とholdoutの混在
- wiring cohort

**改訂案**：
- **D7（推奨、24 compile）**：q8・R566の独立cellと、paired q2を加える。
- **D6（最小、22 compile）**：q8・R566の独立cellだけを加える。
- **人工SE**：cは7.6〜7.8σ。m12の相対SEはρあたり約0.4%。
- **paired q2の役割**：境界の加法性を直接検査する。

## PR-Pauli B1（A2）

**B0**：LD1 m0の1 pathで、RZ 1380。配線の証拠である。

**既存の価格の再利用**
- P3、Track AのPauli-JW、DF-tailの価格は再利用できない。
- B0は、LD1 m0 cohortのpath0として事前に宣言すれば、部分的に再利用できる。

**既存B1案（2 path）**
- LD2の配線と、m0でのばらつきの目安が分かる。
- model（rank 1）・law平均・外挿は検証できない。

**提案**
- B1-W：LD2の配線（2 compile）
- B1-CAL：辞書あたり24 compile。1 step cell（R_m6/m9/m12）、8 step cell、paired 2 step、holdout m5/m6
- B1-CALにより、\(a+hq+bs\)がrank 3で識別できる。

## 参照

[REPORT](../a_full_h_p2_pauli_b1_price_design_20261114/REPORT.md)、[HANDOFF](../a_full_h_p2_pauli_b1_price_design_20261114/HANDOFF.md)。

mandatory STOP。
