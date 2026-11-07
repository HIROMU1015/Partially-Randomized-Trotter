# N1：保存入力・SOURCE・算術の対応

2026-11-08 JST。[REPORT](REPORT.md)の固定条件と基準`0dbf1fe265176677e026ea982c0bd435987f4338`に限定する。**本書は数値実行仕様であり、以下の和・差・期待値を今回評価していない。** 入力のpath・SHA256・Git blobは[INPUT_REFERENCES.json](../../../artifacts/pr_reference_n1_preparation/2026-11-08/INPUT_REFERENCES.json)が正本。

## 1. 固定role価格

[ROLE_T_COUNTS.json](../../../artifacts/pr_reference_ct4/2026-11-07/ROLE_T_COUNTS.json)の`rows`を`name`で選び、`roles`を`role`で選ぶ。採用価格fieldは`account.T`（TとT†の**合計**）。`account.Tdagger`はその内数であり、もう一度足さない。

| 記号の入力 | arm／role | 保存account.T | 意味 |
|---|---|---:|---|
| D_B未融合 | Bprime / D_unfused | 8235520 | 固定D未融合価格 |
| B側D境界補正 | Bprime / D_boundary_fusion_difference | -420624 | signed台帳補正 |
| D_R未融合 | reference / D_unfused | 8174080 | 固定D未融合価格 |
| R側D境界補正 | reference / D_boundary_fusion_difference | -536760 | signed台帳補正 |
| Csw* | reference / CSWAP | 17472 | source固定16 CSWAP分の保存価格 |
| SB* | Bprime / scalar_or_vacuum_correction | 118 | 固定controlled scalar |

各armの`rare0_{arm}_X`,`rare0_{arm}_Y`,`rare1_{arm}_X`,`rare1_{arm}_Y`で上記値が一致していることを保存field同士で照合した。全8行は`projection_matches_literal_histogram=true`、`status=QUALIFIED_SIGNED_ROLE_ACCOUNTING`である。これは固定cohortの保存資格。全supportへの移送はCT-5B条件Cを残す。

後続計算では`D_a* = D_unfused + D_boundary_fusion_difference`とし、補正の符号を保持する。負の物理gate数・負の演算子誤差と解釈せず、絶対値に変換したり0でclipしたりしない。`K=D_B*−D_R*−Csw*+SB*`にCSWAPやscalarを重複加算しない。ReferenceのD roleはCSWAPを含まない。

CSWAPのnative分解が8 CXであることと、strict wordの採用T価格は異なる。16×7等の解析由来Tモデルへ置換しない。Referenceの`scalar_or_vacuum_correction`は固定scalarではなくtrajectory固有vacuum価格。保存値nonrare120／rare118は境界に代入しない。

## 2. 中央4 positive literal binary key

[KEY_REUSE_AND_SYNTHESIS.json](../../../artifacts/pr_reference_ct4/2026-11-07/KEY_REUSE_AND_SYNTHESIS.json)の`records`を`angle`と`exponent=-11`で一意選択する。precisionは一様10^-11、strict、pygridsynth2.0.0／seed17。以下は**保存値の引用**。

| 記号 | angle / float.hex | T+T† | positive wordのW | record key |
|---|---|---:|---:|---|
| f0（Ref full） | `0x1.7e3c19bcd3089p-6` | 118 | 0 | `b4e7ff5cb31c9073b31b97dd69dd0ae698d07a595d4c2503ce3ab7eb0a12c039` |
| h0（B′ half） | `0x1.7e3c19bcd3089p-7` | 112 | 5 | `00e5bc9412042a1f58389b34921b92dc1738ffb2cb36f41e8067eafad0ff1ac7` |
| f2（Ref full） | `0x1.fdaaba4d44069p-8` | 118 | 2 | `9e95458bb6e69e6a3bcdee5888bed2c1f023f242a255330c5cf28808f9a31533` |
| h2（B′ half） | `0x1.fdaaba4d44069p-9` | 116 | 5 | `9cc1d39756941e491661edf2cfa1c015f3f633a0bf3ddc74865d835fa68438a0` |

各recordは`guard_method=exact cyclotomic word matrix + outward interval strict Frobenius`、`positive_word_negative_actual_adjoint=true`、同一eの`strict_error_upper`と`sequence_sha256`を保持する。Wはglobal scalarの実装情報であり、controlled相対位相を保つため捨てない。各負角には**採用wordのactual adjoint**を使う。T合計は同じでもT†内数は変わる。

字面のkeyを固定し、floatからの再計算、理想pi snap、2pi縮約、近傍key、他precision、別seedの安いwordへ変更しない。Z/ZZ・符号の多重度はCT-5Bの中央対応を継承する。B′は各eventでhalf-angleが2物理出現、Referenceはfull-angleが1出現である。

## 3. 保存p2とlaw

入力は[B′ TOTALS_18ROUND.json](../../../artifacts/pr_reference_bprime/2026-11-07/TOTALS_18ROUND.json)の`rows[7].contributions.p_order2`。配列indexだけに依存せず、`m=7,q=128,r=1,bits=73,shots_per_axis=199`で一意選択する。

```text
642545340934082409/9444732965739290427392
```

これはFractionへ**保存文字列から直接**渡す。decimal／floatを経由しない。B′ SOURCEの`totals_and_stop`が`price_bounds(...,row['bits'])`の`parts`を文字列として保存する。`distributions(bits,r)`はfinite RTEの`order_probabilities`を`transfer_cost_bounds.distribution(...,bits)[1]`へ渡し、後者が`integer_cdf`の整数区間幅／2^bitsをFractionへする。`integer_cdf`は正規化した累積確率のfloorを使い、最後のCDFを2^bitsに固定する。

このSOURCE経路を静的に確認した。**common()、finite_rte_distribution、distributions、distribution、integer_cdfは今回呼ばない。** 後続N1でも既存p2の読取で足り、分布の再生成は不要。[V3 ANGLE_KEY_MULTIPLICITIES](../../../artifacts/pr_reference_rotation_v3/2026-11-07/ANGLE_KEY_MULTIPLICITIES.json.xz)のm7/両arm/X/Yはbits73・shots199。[CT-0 PRECISION_FEASIBILITY](../../../artifacts/pr_reference_ct0/2026-11-07/PRECISION_FEASIBILITY.json)もm7/q128/r1/bits73である。

p2は**eventごとのorder2確率**。保存rare/nonrare各1本を1/2ずつ平均しない。q128全体でorder2が発生する確率へ置き換えない。lawは候補／baseline共通で、未知ground情報を片armへ追加しない。

## 4. SOURCEと既存検証の再利用

| 既存資産 | N1での利用 | 実行しない処理 |
|---|---|---|
| [V3 SOURCE](../../../src/trottertracks/pr_reference_rotation_v3.py) `dparts`, `saved_path_parts`, `rotation`, `pair` | role意味、signed補正、固定16 CSWAP、half/full対応を読む。保存台帳を正本として引用 | `expected_parts/account_rounds/validate_cohort/validate_qpy/synthetic_tests`の科学再実行 |
| [B′会計](../../../src/trottertracks/pr_reference_bprime_accounting.py) `distributions`, `price_bounds`, `totals_and_stop` | 保存p2の生成来歴、Fractionから文字列への既存保存規約 | common/H/分布・全round再生成 |
| [CT-4 SOURCE](../../../src/trottertracks/pr_reference_ct4.py) `count_hist`, `role_account` | `T`の定義、actual adjoint、signed role価格とnormの分離を継承 | `attach/prepare/synthesis/validate/finish/main`と旧runner |
| [CT-0 SOURCE](../../../src/trottertracks/pr_reference_ct0.py) `validate_record` | angle/spec/word SHA/精度/route/T内数の既存metadata確認規則 | strict guard再評価／新合成。旧4 legacy欠損を許容する枝を新資格に使わない |
| [有限bit SOURCE](../../../src/trottertracks/track_c_h4_finite_bits.py)、[cost helper](../../../src/trottertracks/pr_reference_transfer_cost_bounds.py) | floor-CDFの意味と保存Fraction規約 | sampler／新確率計算 |
| [区間helper](../../../src/trottertracks/track_c_compiler_a_18round.py) `I/upper/lower` | 外向き区間表示の既存設計を参照 | heavy module importやcompiler/operator検証 |

数値実行時もread-only JSONと標準`int`／`fractions.Fraction`で十分。既存moduleのimportにはQiskit／環境・出力設定の副作用があるため、helper利用を理由に旧runnerやcache/output設定を起動しない。既存価格を再合成・再取得する実装は不要。今回、新runnerもテストsuiteも追加していない。

## 5. 後続の厳密算術

以下は**未実行の式**。入力価格は整数、p2はFractionなので科学的な浮動丸め誤差を導入せず評価できる。

\[
\begin{aligned}
D_B^\star&=D_{B,unfused}+D_{B,boundary},&D_R^\star&=D_{R,unfused}+D_{R,boundary},\\
K&=D_B^\star-D_R^\star-C_{sw}^\star+S_B^\star,\\
d_0&=2t(h_0)-t(f_0),&d_2&=2t(h_2)-t(f_2),\\
d_{\min}&=\min(d_0,d_2),\\
V_{threshold}&=K+128d_{\min},\\
U_{threshold}&=K+128[(1-p_2)d_0+p_2d_2].
\end{aligned}
\]

厳密値を結果の正本とし、Fractionは既約`numerator/denominator`で出力する。期待価格は整数とは限らない。pointwiseの整数T価格に対する厳密不等式と、期待値の有理数不等式を区別し、期待値に整数ceil/gcd下界や整数余裕を流用しない。

小数が必要なら、整数分子・分母から`decimal.localcontext`の指定桁数で`ROUND_FLOOR`／`ROUND_CEILING`の除算を別々に行い、閉区間として表示する。既存`I(numerator)/denominator`と`lower/upper`も同じ目的の候補だが、FractionをfloatへしてからIへ渡さない。判定は常に正本Fractionで行う。表示区間だけで十分条件を主張するなら、vacuum上界の**上端**が閾値の**下端**より厳密に小さい場合に限る。

## 6. 検算と資格

後続検算は[NUMERIC_EXECUTION_SCOPE](NUMERIC_EXECUTION_SCOPE.md)に限定する。欠損入力を0や推定値で埋めない。full supportへの価格移送は[CT-5B条件C](../pr_reference_ct5b_20261107/STRUCTURAL_CERTIFICATES.md)を前提として残す。

[PHASE_SUPPORT](../../../artifacts/pr_reference_ct0/2026-11-07/PHASE_SUPPORT.json)のm7 vacuum角域は、保存binary64生成手順の角度域である。角度の有界性はT価格上界ではない。frame未取得key、B′ b2 event、全vacuum取得／停止／価格、全support CT RPE成功、全18round/T-depthは**UNQUALIFIED**のまま。価格取消しは合成diskやcompiler diskの取消しを許さない。
