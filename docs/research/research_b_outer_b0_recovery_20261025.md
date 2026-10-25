# 研究B B0：技術復旧と新attempt

旧SOURCE `8d0cb96df9d1f6229f207550634c7e11aff2851c` と[旧失敗原本・STOP](../../artifacts/research_b_outer_b0/20261025/native-pilot-once/)は保全する。今回の修正・同条件再実行はユーザーが明示的に指示した範囲。承認証明・grantは作らない。

## 将来attemptの限定変更

科学fixture／sector／state、6候補・両T・4代表、全48 trajectory・96 compile要求、complex128、operator 2-normのabsolute atol1e-12／rtol0／phase fitなし、compiler profileは[既存契約](research_b_outer_a0_b0_execution_preparation_20261025.md)と同じ。旧attemptを上書き・resumeせず、各attemptはfixed clean SOURCEと排他的fresh outputで実行しSTOPする。今回許可された技術復旧に限り、原因と修正理由を確認して新attemptを開始できる。原因不明の反復・科学的条件の変更には拡張しない。

まず `--diagnose-first` で旧失敗identity B3／T=+0.18／代表0／Xだけを検査する。入力5境界、config6、event／event operator各6、RTE sequence12、D block18、独立exponential9、system／controlled／reference各1、operator gate4、expectation2、compile1。費用、全mean、他caseは起動しない。Metadata scopeに登録した全境界を照合する。

FAIL時には数値差・actual／reference matrix、raw／compiled回路をgzip+base64 QPYでpartial_results内に保存する。成功したgate・meanの数値をeventsへ保存し、wrapper rowをcompiled gateより前にpartialへ登録する。既存EvidenceのJSON／総出力枠は維持する。保存失敗もSTOP。共通認証・汎用frameworkを追加しない。

診断で明確な実装問題を確認した場合だけ最小修正・関連テストを行い、新科学SOURCEをcommit固定し、同じ科学条件の全B0をfresh outputで実行する。全gateを費用96要求に先行させ、数値許容差を緩和しない。技術的に解決できないcorrectness FAIL・識別力不足は保存してSTOPする。

## 実績

診断SOURCE `2721d69a98213a35facf4dc7fafa08ab1f0b40ea` で最初の失敗caseを再現し、spectral誤差1.791447313e-11、raw wrapper誤差1.155206482e-14とQPY／行列を保存した。保存回路だけのpass診断（script SOURCE e82e99558）では、最適化前のofficial-definition展開ですでにrawから1.792369935e-11ずれ、global phaseが2045.1768174869376 radに累積していた。

修正はB0からだけ指定する `compiled_wrapper(...,stable_phase=True)`。Gateの公式definitionをそのまま辿り、global phaseの項を `math.fsum` で一度集計し2πでreduceしてから既存transpileへ渡す。Operatorから位相を推定・fitしない。Native basis、optimization1、seed17、layoutその他profileは維持し、loweringの数値実装名だけ `official_definitions_fsum_phase_v1` と記録する。旧研究の既定compiler経路は変更しない。Cache identityにも新loweringを含める。

関連人工検証はB0 opaque11件／共通10件PASS、compiler人工5件PASS（旧3件＋多数phase項・controlled公式definition2件）。修正版は最初のcaseを同じtolで確認後に同条件全pilotを行う。旧契約や旧結果を遡及変更しない。A0再実行、A1包含、MC、shots、新PF・grid、全RPE評価は対象外。

V1 SOURCE `c7cde1868a6b38f9505ecc4f0858b98b49c506e8` の限定診断はcompiled誤差4.386058909e-13でPASSしたが、同SOURCE全体attempt corrected-1は16 wrapper通過後、D1／+T／代表0／Xで2.318114585e-12となりSTOP。保存D1回路の段階診断では、lowering直後1.585565770e-12、BasisTranslator後4.635958878e-12であり、U-to-native展開と位相の大きな和の剰余処理にも丸めが残っていた。

V2では定義由来binary64 phaseの和と2π剰余をFractionで正確に処理し、最後に一度float化する。Uの既知ZSX分解とP=phase(theta/2)RZ(theta)を明示し、scalar phaseを同じ和へ含める。これでBasisTranslatorへ大量のUを渡さず、native gate列へ同じoptimization1・seed17を適用する。Operator評価や位相fitをcompilerへ導入せず、許容差は固定。Lowering名は `official_definitions_exact_phase_native_v2`。追加の人工U/P identity検証もPASSした。
