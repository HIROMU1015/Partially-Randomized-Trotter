# F1/F2を実装する場合の最小受入要件

**実装・test作成/実行は今回0件。** 以下は技術的完成条件であって、科学実行契約・採択済みschema・authorizationではない。W1/Bの試験群を再設計・実行せず、今回具体化した入口/P条件/費用へ限定する。

| ID | 必要な実装・保持情報 | 受入条件／失敗時 |
|---|---|---|
| A01 entrance | exact integer Rat、p table/label map、x/m/arm/σ、η/ρ、H/K_root、τ、source/input/policy identity、bit長 | p>0、正確sum=1、型/分母/domainが[preflight §1](preflight_precision_contract.md)と一致。SOURCEのFraction coercionを数学入力の認証としない。非法直接helperを入口で排除 |
| A02 identity | x=0をpositive sampler前に認識し、deterministic identityとその来歴を返す | 乱数/局所query/weight/angleを呼ばず、random zeroと区別。既存kernel/G7を無断変更しない |
| A03 preflight evidence | 与えられたK/Hに対するp_min/t_min/d_min、Δ/h、P0〜P5/Rのlhs/rhs、exact比較結果、source/入力/精度identity | 全到達parentの保証は[BIT]の一様下界に結び付き、全語辞書・全parent訪問に依存しない。固定defaultsだから成功とはしない。具体的selectorは別許可・別範囲 |
| A04 actual law | 同じ登録law、positive support、dyadic denominator、相対guard、bit integerとの対応 | SOURCEの実lawとpreflight証拠が一致し、law改変/再normalization/weight clipなし。canonicalの有限law条件もmodeに応じて満たす |
| A05 packet/weight | d区間/d_mid、acceptance/child law、実π/c̃/W̃、word/child/phase/σ、逆時間順、必要な親identity | πW̃=c̃のexact rational意味を保持。meanはM̃、係数biasは別。K helperのpure-parent規約とG7 child前提を混同しない |
| A06 failure | stage/reason、invalid対precision対実行failure、試行番号、parent/bit状態、P/guard証拠、成功prefix/資源/STOP | 正当なzeroはraw/coinのみ。B/Cをzeroにして継続せず、preflight成立後のguard違反は不整合として保全STOP。未完了を完了confidenceにしない |
| A07 angle | exact tangent、φ enclosure、代表値とτ、signed time、source/input/parent/precision identity | 既存[BIT]§6に対応するenclosure証拠がある。symbolic ratioだけならangle未取得とする。native合成/strict physical errorの証拠は別 |
| A08 budget/IID | existing G7 factor/bias/range/moment、U/B_o hi、positive remaining、fixed confidence、fresh-bit/測定モデル、全N完了 | 任意αへ黙って拡張しない。unknown B_RA/Z/true signalを予算に使わない。SHA traceや完走runの事後選別からconfidenceを認証しない |
| A09 cost/space | [F2](full_bit_complexity_accounting.md)のモデル、実L/m/bit長/K/H、raw/coin/accepted別prefix、root/draw/law/weight操作、外部取得/出力 | child_law再sumとcoin前child distributionを現SOURCE通り含める。最大/期待/全Nを分ける。実装を最適化する場合は旧固定版を保全し、再評価の範囲を別承認 |
| A10 output/runtime | object/binary/decimal等の出力形式、読書き費用、保存容量、bit供給/実行環境の制約、metadata | C_output/S_savedを0にしない。generator bitモデルとCPython実bytes/秒/serialization成功を分ける。未知値や中断の証拠を保存 |

今後、人工verificationの実装/実行が必要な場合も、その許可範囲をユーザーが別途指示する。今回は数値・formal・sampling oracleやtest例を新設していない。実分子/保存値評価、native/回路、backend、DF/PR adapter、P_Uへ自動拡張しない。

既存W1数学保証に矛盾する挙動が見つかった場合、旧SOURCE/結果を修正して消さず、固定identity・反する条件・影響を保存してSTOP。数学保証の変更、新retry、failure budget、conditional confidenceは別の判断と承認が必要である。

**次の担当はユーザー**：この仕様を使う通常実装/限定人工検証を許可するか判断。許可された範囲のCodexが実装を行う。今回の資料はその開始許可ではない。**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
