# CT-0：保存strict recordのcoverage

基準 `b3ff9d1b4d356ae67e8f55bf1e2cfd0715814cbb`。新A用precisionを採択せず、保存データだけを照合した。[REPORT](REPORT.md)の固定H4/S2/LD3条件に限定する。

## V3台帳のnumeric域

保存 `ANGLE_KEY_MULTIPLICITIES.json.xz` は662 token・108 arm/round/axis行。分類をやり直さず、各行のD融合のsigned差分を同じsigned angleで合算して非負を確認した。ゼロmultiplicityはrequired域から除いた。originは最初の登録ラベルであり、完全な出自集合と扱わない。

| arm | signed numeric key | positive key | recordあり | numeric key欠損 | symbolic key |
|---|---:|---:|---:|---:|---:|
| B′ | 494 | 400 | 153 | 247 | 1 |
| Reference | 484 | 384 | 154 | 230 | 18 |
| 登録14-CX（参考） | 496 | 401 | 154 | 247 | 1 |

B′/Referenceで共通のpositive keyは366、そのうちrecordありは153。これは同一角度のword共有機会であり、物理gate数削減でも要求precisionの資格でもない。符号・role・round・全物理出現は維持する。

V5補強の**実SOURCE inverse D**20 recordも補助照合した。174 signed／160 positive numeric keyで、V3 Reference域にないpositive keyは0。これは保存inverse Dだけの確認で、全support位相を補わない。

## precision別

表の分母は上記positive numeric域。値は「recordあり／guard方式も記載されたstrict再利用候補」。epsilonは保存recordの仕様であり、新Aの採択precisionではない。

| exponent e（epsilon=10^e） | 保存record数 | B′ | Reference | 登録14-CX |
|---|---:|---:|---:|---:|
| -13 | 948 | 153 / 150 | 154 / 150 | 154 / 150 |
| -12 | 951 | 153 / 153 | 154 / 154 | 154 / 154 |
| -11 | 954 | 153 / 153 | 154 / 154 | 154 / 154 |
| -10 | 954 | 153 / 153 | 154 / 154 | 154 / 154 |
| -9 | 954 | 153 / 153 | 154 / 154 | 154 / 154 |
| -8 | 953 | 153 / 153 | 154 / 154 | 154 / 154 |
| -7 | 949 | 153 / 153 | 154 / 154 | 154 / 154 |
| -6 | 940 | 153 / 153 | 154 / 154 | 154 / 154 |

要求値をまだ決めていないため、「precision未適合」は**未選択・対応未資格**を意味し、ある値で不適合と判定したわけではない。全recordのguardを再計算せず、保存された厳密上界の分数とepsilonを比較した。

## 保存recordの整合・provenance

raw archive SHA256を既存BUNDLESと照合し、全7,603件で以下を確認した。

- positive binary angle、precision exponent、`pygridsynth2.0.0:strict:seed17`を含むspecとkey hash、archive member名。
- PASS、sequence hash、許容gate alphabet、0≤保存strict guard≤epsilon。
- 保存T/Tdagger/global_W fieldがword文字数と一致。Tの合計・新価格・候補比較は算出していない。

7,599件には `exact cyclotomic word matrix + outward interval strict Frobenius` のguard方式が記載される。残る4件は旧recordで `guard_method` fieldがない。spec/hash/PASS/保存誤差は整合するが、**今回の保守的なstrict再利用候補から当該4 recordを外し、`GUARD_METHOD_PROVENANCE_UNQUALIFIED` とした**。旧結果の無効化・不正なguardの証明ではない。metadataの後付けや再guardは行っていない。

| positive binary angle | e | route |
|---|---:|---|
| `0x1.2d97c7f3321d4p+1` | -13 | pygridsynth |
| `0x1.921fb54442d18p-1` | -13 | pygridsynth |
| `0x1.921fb54442d18p+0` | -13 | fast path |
| `0x1.921fb54442d18p+1` | -13 | fast path |

読取検査の最初の試行はこの欠損fieldで停止した。schema差を認識して上記の未資格分類を追加し、同じ許可範囲の読取検査を完了した。歴史的SOURCE/recordを書き換えていない。cache missを合成へ送る `acquire`、旧runner、backend、optimizerは呼ばない。

## signed key・adjoint・未資格域

`abs(float.fromhex(key)).hex()`はlookup用。signed zeroはidentityに対応するが、他の符号や2π周期は消さない。負角はpositive保存wordの**actual adjoint**を使えば、strict spectral誤差上界は同じ。今回はそのwordを生成・回路化せず、保存alphabetと数学的adjoint可能性を確認しただけで、実emitter correctness検査は後続課題。Wを捨てるprojective置換は不可。

- **numeric key欠損**：一致するbinary角のrecordなし。0埋め・隣接float・理想π角で代替しない。
- **symbolic phase**：B′はinteger-pi event phase、Referenceはm=0…17のvacuum補正。V3 symbolic域をrecordありに数えない。
- **precision未資格**：存在するrecordも、新Aの予算・全support重み・採択precisionを満たすまでは合成済みとしない。
- **guard provenance未資格**：上記4 record。別precisionに方式が記載されていることは、その4件の証明ではない。

[PHASE_PRECISION_FEASIBILITY.md](PHASE_PRECISION_FEASIBILITY.md)のB′ event位相の有限supersetはV3 symbolic域を補助する**別の保存結果**であり、上のV3 coverage数へ混ぜていない。Referenceのvacuum全support keyは未列挙・価格未資格。

## データと再利用

[KEY_COVERAGE.json.xz](../../../artifacts/pr_reference_ct0/2026-11-07/KEY_COVERAGE.json.xz)に全signed→positive対応、role/round、precision別record key・保存guard・word hash・既存T field、missing key、symbolic一覧、inverse D補助照合を保存した。word本体は旧archiveを参照し複製していない。

C2-Bの保存record検証規則を再利用し、新Aの全supportや別PFの最適性は移していない。採用結果の規約は `verified_final` であり、初回/`verified` のT会計を使っていない。今回は旧総Tやallocationの再集計自体を行わず、C2-B下界を新Aのbaseline価格にもしていない。

**mandatory STOP。** precision・新規合成対象はGPT／ユーザーの次判断と独立した承認scopeで固定する。
