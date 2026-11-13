# P0：controlled ZZ loweringの実回路検証（Codex D、2026-11-13）

承認範囲は、P0の**小回路compile最大8件、sampling 0件**に限られる。実際に行ったのは次のとおり。

- 実行前に、角度・basis・回路8件をPLANとして固定し、SOURCEと一緒にcommitした（`a287b8e6`）。
- その固定SOURCEで、既存の`rte_compiled_cost.transpile_and_measure_cost`を使い、ちょうど8回compileした。
- P1以降、S′の実装、残差のfit、full-wrapper価格の取得、N/RMSE/Gは行っていない。元のS、K0/K1、保存結果は変更していない。

**結論：controlled ZZ loweringの差は成立した。**

- 実compiled側の`RZZGate(2θ).control(1)`は、Qiskit 1.3.0で**CCX·CRZ·CCX**に展開される。
  - compile後は、ZZ回転1回当たり**RZ +19・CX +10**（単独）、保存basis対に挟んだ場合は**RZ +17・CX +10**。構造Sの`CX·CRZ·CX`形と比べた値である。
  - 回転角（R2・R128）には依存しない。
- 2つの形は、制御位相・global phaseを含めて**同一のunitary**である。差は資源だけに現れる。
- 一方、保存basis対の内側では、compilerによる相殺・再合成でCが構造Sより**140小さく**なった（この局所文脈での値）。C−Sには符号の異なる複数の機序が混在する。

正本：
- 固定計画：[PLAN.json](PLAN.json)
- 実行結果：[run01/RESULT.json](run01/RESULT.json)。compile後の全gate列、wire別の数、operator照合を含む
- 要約：[P0_SUMMARY.csv](P0_SUMMARY.csv)
- [HANDOFF](HANDOFF.md)、[TESTS](TESTS.txt)
- 実装：[controlled_zz_p0.py](../../../src/trottertracks/controlled_zz_p0.py)／[runner](../../../scripts/run_controlled_zz_p0.py)／[test](../../../tests/check_controlled_zz_p0.py)

## 1. 固定SOURCE・条件

| 項目 | 値 |
|---|---|
| D基点 | `af6f27d7a6467bb355686421acd067544edcbbd2`（branch `codex/d-compiled-residual-c-minus-s-20261113`を継続） |
| 実行SOURCE | `a287b8e6cb629c6a089c99b66b758751910ee15a`（PLANと実装をcompileの前に公開） |
| C入力 | `83d219cc68c25540507a5539ebd2da94c29d06b3:docs/research/h4_pr_physical_closure_20261113/run02/RESULT.json`（sha256 `e6e3bef7…`）。324 component、9 registered basis、R2/R128 law |
| B参照 | `170047ef6e7a72aef5f4c6ff9c73f942097a694e`（今回のP0はB接続を使わない） |
| compiled側の生成 | provider `c31ef424`の`df_rte_qiskit._append_central_application`（`RZZGate(2.0*angle)`→`gate.control(1)`）と`_append_basis`（inverseが先、forwardが後）。worktreeのfileはproviderとbyte一致 |
| 構造S | C1 `Accounting`（`8457c443`の`h4_df10_price_v1.py`／`track_d_native_resources.py`）。byte一致を検査済み |
| compiler | `rte_compiled_cost.transpile_and_measure_cost`、`CompilerSettings(rz/sx/x/cx, opt1, seed17, '1.3.0')`、settings hash `1239304c…`（全8件で一致） |
| runtime | Python 3.11、numpy 1.26.4、qiskit 1.3.0 |

固定した設計（選択規則をPLANに記録）：
- **component**：ZZ支持で確率が最大のもの、`df-fragment-1:Z0Z1`（support [0,1]、符号−1、p=0.02569）。
- **basis**：そのcomponentのregistered full basis（`df-basis-451faeed…`、85 op：p・xx_plus_yy）。
- **角度**：order-0 rotation angle。gate角は2·符号·angleで、R2（m0）は−0.28436388126731005、R128（m6）は−0.004473363000491157。
- **回路8件**：形2（汎用control／CX·CRZ·CX）× 文脈2（単独／保存basis対に挟む）× law 2。9 qubit（system 0–7、ancilla 8）。

試験していないもの：support-restrictedのbasis選択、order-2のproduct（CZ）、tailの相対phase、head、wrapper。

## 2. 展開とunitary equivalence

**Qiskit 1.3.0のlowering（compile前、gate定義）**

| 形 | 定義 |
|---|---|
| 汎用control | `RZZGate.control(1)`→`ControlledGate 'crzz'`の定義は`ccx(anc,s0,s1)`、`crz(anc,s1)`、`ccx(anc,s0,s1)` |
| CCX | `cx 6, t 4, tdg 3, h 2` |
| CRZ | `rz 2, cx 2` |
| 構造S側 | `cx(s0,s1)`、`crz(anc,s1)`、`cx(s0,s1)`。C1 `Accounting`の`'crzz'`の分解 |

**unitary（9 qubit、512次元）**

| 照合 | 結果 |
|---|---|
| 汎用形と解析形 | 4組すべてで最大差**0.0**。global phaseを含めて一致 |
| 制御構造 | ancilla=0のblockが単位行列（誤差 ≤ 6.7e−16）で、非対角blockは0。制御位相の誤りはない |
| 単独回路と独立対角行列 | Qiskitの定義を使わずに作った \(|0\rangle\langle0|\otimes I+|1\rangle\langle1|\otimes e^{-i\theta Z_0Z_1/2}\) と、汎用形・解析形とも差0.0 |
| compile後と元回路 | 8件すべてで最大差 ≤ 1.95e−14（global phaseを含む） |

## 3. ゲート数（compile前後）

| 文脈 | 形 | 構造S（RZ） | compile後のRZ | CX | SX | X |
|---|---|---:|---:|---:|---:|---:|
| 単独 | 汎用control | 2 | **21** | **14** | 4 | 0 |
| 単独 | CX·CRZ·CX | 2 | **2** | **4** | 0 | 0 |
| 保存basis対 | 汎用control | 334 | **211** | **66** | 140 | 2 |
| 保存basis対 | CX·CRZ·CX | 334 | **194** | **56** | 136 | 2 |

R2とR128で全く同じ数になった（角度に依存しない）。

**差の発生箇所**（汎用−解析、compile後、wire別）：

| wire | 単独 | 保存basis対 |
|---|---|---|
| target `s1` | RZ +13、SX +4、CX +6 | RZ +11、SX +4、CX +6 |
| `s0` | RZ +4、CX +6 | RZ +4、CX +6 |
| ancilla | RZ +2、CX +8 | RZ +2、CX +8 |

**compile前の文字どおりの展開との比較**（定義からの算術であり、compileではない）：
- 汎用形の文字どおりの展開は、RZ 24（T/Tdg 7個×2、H 2個×2 RZ×2、CRZ 2）、CX 14。compile後は21で、CCX内のH/T列の再合成で3個減る。
- 解析形は文字どおりRZ 2・CX 4のままで、compile後も変わらない。

**相殺・再合成の影響**：
- 保存basis対に挟むと、汎用形の超過RZは19から**17**へ2個減る。target wireの1q runが、basis側のgateと連結して再合成されるためである。CXの超過（+10）は変わらない。
- 保存basis対の構造S（334）に対してcompile後は194で、**C−S=−140**。理由は2つある。
  - 回転が触れないwireでは、`B⁻¹`と`B`が隣接し、InverseCancellationと1q再合成で消える。
  - `xx_plus_yy`の文字どおりの展開が再合成される。
- Sはbasisの相殺を「同一op列の隣接inverse」だけで数える。この値は孤立した局所文脈のもので、実trajectory（隣接eventのbasis共有、head、outer）の値ではない。

## 4. 判定

**仮説は支持された（成立）。**

- 汎用control展開は、構造Sの`CX·CRZ·CX`より、ZZ回転1回当たりRZ +17〜+19・CX +10を加える。決定論的で、角度に依存しない。
- 保存support確率で重み付けると、event 1個当たりの期待値は次のとおり（設計用の算術であり、保存S/Cの再解析ではない）。
  - ZZ支持の確率：ΣP = 0.9163。
  - lowering差の期待寄与：**15.6〜17.4 RZ/event**。
  - 保存C1の凍結係数θL（H4、18.29/event位置）と同じ桁だが、これは検証ではない。
- 同時に、compilerによる負の寄与（basis対の相殺・再合成）も確認した。C−Sを「正の局所項だけ」と見なすことはできない。

## 5. 案A（S′）と案B（C−S残差）の比較

| 観点 | 案A：実loweringに対応する構造価格S′ | 案B：Sを維持し、lowering差を残差modelへ入れる |
|---|---|---|
| 定義 | `Accounting`の`'crzz'`を、汎用形（CCX·CRZ·CCX）の文字どおりの展開で数える別quantity S′ | D=C−Sに、項 δ_ZZ·N_ZZ（N_ZZ＝pathのZZ回転数）と、その他の項を持たせる |
| 計算量 | 単一pathはSと同じO(event)。閉状態kernelは状態に変化がある（CCX内のH/Tがwire状態とZ foldへ入る） | E[N_ZZ]=qR·P_ZZは法則から厳密に求まる（compile不要）。単一pathのN_ZZはevent metadataから数えられる |
| 13round拡張 | K1（閉状態）で可能。ただし新しいwordでK0の再検証が必要 | E[S]（既存K1）＋δ_ZZ·E[N_ZZ]＋他の残差。δ_ZZとその他の項の長q/R移送を検証する必要がある |
| 数値精度 | 整数の構造計数。kernelの区間評価はK0/K1と同じ方式 | δ_ZZは文脈で変わる（単独19、basis対17）。分散・文脈依存を校正し、holdoutで扱う必要がある |
| K0/K1の再利用 | **そのままは使えない**（新word）。h4_structural_kernelの実装は再利用でき、K0/K1を再実行する（科学承認が必要） | **そのまま再利用できる**（E[S]不変） |
| 残る差 | C−S′はcompilerの最適化だけになり、basis対の相殺など符号が両方ある項が残る | lowering項＋compilerの両符号項を、1つの残差modelで扱う |
| 実装・検証コスト | 中：counter変種、K0/K1の再実行、S′のparity検証 | 小〜中：N_ZZの集計（metadata）と、P1/P2のpaired DによるδとAの検証 |

**推奨**：当面は**案B**を採る。理由は3つある。
- K0/K1をそのまま使える。
- E[N_ZZ]が厳密で、P1のpaired Dで直接検証できる。

案Aは、δ_ZZの文脈依存が大きいと判明した場合の次の候補とする。

これとは別に、**回路そのものを安価な`CX·CRZ·CX`へ変える**という選択肢がある。これは価格定義の変更であり、PFとの公平性に関わる。前回の実装監査R1にあたり、ユーザーが判断する。P0は、2つの形がunitaryとして同一であることを確認した。

## 6. P1へ進む場合の推奨

- 既存の8 wrapper案（m0/R2・m6/R128、normal 2path、同path X/Y）に、**pathごとのN_ZZ・N_Z・order-2 product数**（event metadataの集計、compileなし）を加えて記録する。
- D−δ·N_ZZ（δ∈[17,19]はP0値）を**開発用の診断**として併記する。fitはしない。
- 規模の目安（設計用の算術）：
  - m6（8192 event位置）では、E[N_ZZ]≈7506、lowering差は約1.3〜1.4×10⁵ RZ/wrapper。
  - m0（2 event）では、lowering差は最大で約38 RZ。basis・head・wrapperの両符号項が支配的と予想される。
- 案Aを選ぶなら、P1の同じpathでS′も同時に数える（compile追加なし）。ただしS′の実装は別に承認を得る。
- 回路を安価な形へ変えるかどうか（価格定義の変更）は、P1より前にユーザーが決めておくと手戻りが少ない。変える場合、P1のCは新回路で取るべきである。

## 7. 技術テスト・保全

- [TESTS](TESTS.txt)：**34 PASS**（新P0 6件、残差adapter 13件、C `mean_compiled_price` 15件）。
  - P0 testの内容：設計の決定性、provider・counterの形、unitary照合、構造S、compile予算（偽compile関数で8件・超過時の拒否）、固定bytes。testの中ではtranspileを呼ばない。
  - これらは作者自身の確認であり、独立reviewではない。
- 実compileはrunnerによる8件だけ（`compile_calls`に記録）。sampling・fit・価格は0件。
- C/B branch、旧S・K0/K1・保存結果・STOPは変更していない。
- 公開後、**mandatory STOP**。P1へは自動で進まない。
