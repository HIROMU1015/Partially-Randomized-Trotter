# 既存Clifford+T資産：新Aへの適用可能性

固定参照 `a095a225ceae383027b300e1f31ec5ccbb1522fc`、2026-11-07 JST。静的SOURCE・schema確認のみ。以下の関数・runner・testは今回実行していない。採用判断と後続scopeは[REPORT](REPORT.md)を参照する。

## 関数・保存資産の対応

| 既存資産 | 入力・出力 | 新Aへの適用 | 必要な差分 | 保証・制約 |
|---|---|---|---|---|
| [track_c_h4_synthesis](../../../src/trottertracks/track_c_h4_synthesis.py)：`exact_word_matrix/strict_guard` | word、実binary RZ角→strict Frobenius上界の有理数 | 核を再利用可能 | 新AのRZ target・精度・joint埋込を対応付ける | exact cyclotomic word評価＋外向き区間。W保持、phase fitなし。新A全回路の誤差保証ではない |
| 同 `acquire` | positive angle/e/version/strict/seed→word、T/T†、depth、guard、hash | 新規取得が別途承認された場合のbackend候補 | 独立出力、明示したfinite key、保存recordの検証と読取reuseを分離 | cache missで合成とfile書込を行う。旧OUT/runを再開しない。fast path PASS≠exact型認定 |
| C2-B [raw_synthesis.tar.xz](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a095a225ceae383027b300e1f31ec5ccbb1522fc/artifacts/track_c2_b_pilot/2026-11-06/raw_synthesis.tar.xz)、`BUNDLES.json` | 保存word/precision/guard/provenance | 条件一致部分だけ再利用可能 | 新Aのnumeric angleと採択precisionをspec単位で照合 | 保存auditは7,603 record／963 positive angle key。完全coverage未確認。Full Y4/LD12の総Tを移さない |
| [track_c2_b_strong_baseline.load_inputs](../../../src/trottertracks/track_c2_b_strong_baseline.py) | bundle/word/request→検証済み辞書 | loaderの照合パターンを再利用 | 固定path・件数7603・72 case・旧capsのassertを新Aへ転用しない | status/hash/guard≤epsilon/T数を確認。import経由の旧module OUT設定にも注意。今回は呼ばない |
| [track_c2_b_pilot.angles/items_for](../../../src/trottertracks/track_c2_b_pilot.py) | native回路／Y4 block→positive key/count/誤差weight | key形式を参照。`items_for`は直接不適合 | PRのD/RTE/frame/product/phase/CSWAP/wrapperを保存SOURCEから対応付ける | Y4/LD12の決定論的compositional多重度とB/D roleを前提。新Aの期待個数・unknown phaseを持たない |
| [track_c2_b_verify.lower_native](../../../src/trottertracks/track_c2_b_verify.py) | native block、allocation、record、role→明示Clifford+T回路 | emitterの核を再利用 | new A role/round/axis・wire・measurement/barrier/resetを保持するadapter | wordはwritten productの逆順にappend。負角はactual adjoint、W phaseを保持。戻りbasisは合成回路の実inverse。対応していない演算は現在raise |
| 同 `synthetic_checks` | 保存word→adjoint/発行gate数/全空間共役bound検査 | 後続の最小検査例として再利用 | 新A wrapper・native target・phaseの範囲を追加 | 保存wordの実adjointと共役条件を検査する例。科学性能や全supportの証明ではない。今回未実行 |
| [track_c2_b_pilot.exponent/allocate](../../../src/trottertracks/track_c2_b_pilot.py) | count、誤差weight、budget、実T価格→一様/貪欲非一様配分 | 有限選択と有理feasibilityの考え方を再利用 | precision/capは新Aで別判断。有理期待countとpair制約に適合させる | 旧2 precision・2 capを自動採用しない。Tが精度に単調と仮定せず、実recordの削減量を使う。貪欲解は最適性保証なし |
| [strong_baseline.optimize_case/exact_fractional_price_lower](../../../src/trottertracks/track_c2_b_strong_baseline.py) | 登録2精度knapsack→実行可能解・有理下界/gap | 強baseline設計の参考、直接移植不可 | 有理期待Tを目的にする場合のgcd/ceil・solver提案解検証を再定義 | C2-Bは整数count/T saving格子。新Aの有理期待価格を整数へ切り上げる下界は使えない |
| 同 `safe_alpha/evaluate/reoptimize_alpha/global_alpha_lower` | round価格/rho/beta/alpha→整数shot総T、dual下界 | 同等なalpha自由度・下界の考え方を再利用 | 新Aの幾何・有理期待T・対称性・有限候補範囲を資格する | 旧cap/alphaとinteger-total ceilはそのまま使えない。LP/dualは実現baseline価格ではない。大域最適は未保証 |
| C2-B `verified_final/`、`track_c2_b_verify.run`、strong `geometry` | 補正済みdisk→rho/beta/N/全T | raw単位を合わせた一度のphase変換を再利用 | 旧γ4096やrho基準・追加compiler allowanceを新Aへ機械的に移さない | 初回補正前を採用しない。`run`は回路/数値処理と出力を伴うので今回呼ばない |
| [pr_reference_rotation_v3](../../../src/trottertracks/pr_reference_rotation_v3.py)：`token/profile/expected_parts`、保存angle台帳 | signed key/role/有理multiplicity→3方式の保存内訳 | 保存データ形式と構造式を再利用 | symbolic phase、非負物理count、block/pair/wire対応を必要範囲で補う | exact分類とbinary実角を区別。`TOKENS.setdefault`は全origin一覧ではない。融合差分はsigned補正。今回再実行・再分類なし |
| [pr_reference_bprime_accounting.distributions](../../../src/trottertracks/pr_reference_bprime_accounting.py) | bits/rと固定SOURCE→floor-CDF有理確率 | 固定lawの会計資産として再利用可能 | T価格・誤差weightを接続する前にkey coverageを資格 | 保存expected資源とworst-path/mean誤差の保証を区別する。今回確率の再評価なし |
| 同 `error_and_shots`、[transfer_error.qualify/phase_bound](../../../src/trottertracks/pr_reference_transfer_error.py)、[transfer_accounting.account](../../../src/trottertracks/pr_reference_transfer_accounting.py) | 固定誤差/phase/law→全support disk、rho/beta、十分N | 新Aの誤差・位相・alias・成功条件の基準 | U_N対U_CTのoperator errorをrawへ一度追加。必要時だけN再評価 | 元H表現への無条件保証ではない。vacuum算術誤差と合成phase誤差を混同しない。旧Nは合成込み未資格 |
| [track_c_h4_synthesis.t_metrics](../../../src/trottertracks/track_c_h4_synthesis.py) | 全native列/precision→T/CX/Clifford/合成後gate depth | explicit word単位の課金方法を再利用 | recordを読取供給する分離、resetと新Aの全wrapper、非一様precision、期待価格 | 内部で`acquire`を呼ぶ。出力depthはT-depthではない。native opt2前提コメントをcompiler Aへ自動移送しない |
| [pr_reference_depth_v5.transfer_matrix/apply_ops](../../../src/trottertracks/pr_reference_depth_v5.py) とV5補強 | native列→wire frontier/span/境界 | 将来のword DAG検査の核候補 | CT word順序・T-depth定義・新境界・非線形期待値の資格 | 旧native span/δはword置換後に不変とは限らない。40.995952%をCT/T-depthへ流用しない |
| [SP-0.5 economics](../../../src/trottertracks/algorithm_codesign/synthesis_placement/economics.py)：`angle/key/sequence_matrix` | rational rad/π、written word→値・key・行列 | exact provenance表現・word順序の参考 | binary keyとsymbolic keyを明確に分ける | `exact_sequence/error_guard/synthesize`はπ/4周期・projective witnessを含みstrict cacheへ直接不適合 |
| [c1_h4_full_synthesis](../../../src/trottertracks/c1_h4_full_synthesis.py) などC1誤差module | 行列segment→数値Frobenius/norm-growth診断 | 別資産として仕様確認、主経路には使わない | 新Aの保証/価格providerの代用にしない | 名称のsynthesisだけでCT word/T価格実装と判断しない。別model・数値pilotで、rigorous bound未取得を明記したmodule |
| [check_track_c2_b_pilot.py](../../../scripts/check_track_c2_b_pilot.py)、[check_track_c2_b_strong_baseline.py](../../../scripts/check_track_c2_b_strong_baseline.py) | 保存bundle/人工case→hash・配分・総計/STOP検査 | test資産の入出力を参照 | 新Aに関係する最小検査だけを将来のscopeへ限定 | 旧全record/総T再集計・optimizerを含むため、今回は読取のみで未実行 |

## key・phase・recordの再利用条件

cache照合は、実numeric RZ角、positive key、要求precision、backend/version、strict/seed、PASS、word hash、guard上界とepsilonの関係が揃う場合に限る。roleとmultiplicityはcache identityとは別に保持し、同一wordを何回実行するかを失わない。負角のwordは正角wordの実adjointとし、符号が逆でも異なるprecisionを無条件に同一pairへ与えない。

source controlled gateはcompiler前にcontrolを含め、lower後の通常native RZを合成する。合成後に新しくcontrolを付けたり、uncontrolled gateのprojective errorをcontrolled相対phase保証へ読み替えたりしない。Wはexact matrixのphase bookkeepingであり、物理T gateでも無料の相対phase補正でもない。

未確定分類でもnumeric binary角ならstrict targetを定義できる。一方、V3のtrajectory-dependent vacuum/event phase placeholderはそのまま`float.fromhex`へ渡せない。未知部分を0・π・平均位相へ置換しない。今回required-key一覧やmissing-key件数は生成していない。

## 現在の適用範囲

- **再利用できる核**：strict単一RZ guard/backend、actual adjoint emitter、保存word/provenance、fixed-law multiplicity、raw diskと整数shotの既存基準。
- **限定拡張が必要**：新Aのwhole role/wrapper/reset、cache coverage、symbolic phaseの全support価格、precision/誤差配分、非負の有理期待T、T-depthのwire配置と期待値保証。
- **直接使えないもの**：別PFの総T・予算/normalization、projective guardをstrictとする扱い、整数価格専用のceil/gcd下界、旧native depth改善率のT-depthへの移送。

必要な追加検証・precisionの採択は未実施。**mandatory STOP。次担当はGPT／ユーザー。** 総T・T-depthの優劣や新規性の認定はしていない。
