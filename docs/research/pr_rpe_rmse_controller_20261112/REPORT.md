# PR–RPEのRMSE型総資源controller：技術実装（2026-11-12）

既存B branchのC1受入済みHEAD `7dd61438b64a4b015ce0b7a7db81187d9867d8a2`を再利用し、**共通κで全roundを結んだconfiguration生成・RMSE/N/C適格性判定・供給値のexact積和・M/κと外側条件の有限menu比較**を追加した。新17件＋既存54件＝71技術testがすべてPASSした。新しい実N/C/G、normalization、保存科学値の数値再解析、fit/sampling/compile、実資源最適化は行っていない。

## A/Cへの先行共有

実装前に[共通仕様V1案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d562bc38d3b63df51f775aa4eb0998afbf1873e2/docs/research/pr_rpe_rmse_controller_20261112/INTERFACE.md)とJSONをcommit/pushし、固定URLを本チャットへ提示した（`d562bc38d3b63df51f775aa4eb0998afbf1873e2`）。A/Cは同じrepositoryの固定仕様を使って並行開発できる。別チャットやworktreeへ書込み・エージェント起動はしていない。現行の[INTERFACE](INTERFACE.md)・[JSON仕様](INTERFACE.json)・[人工I/O](SYNTHETIC_IO_EXAMPLE.json)は実装fieldまで補足している。

原論文の非固定URLは現在v2であること、v1も存在することを確認した。参照した式は[arXiv v2 Appendix E.3](https://arxiv.org/pdf/2503.05647v2#page=42)の回転数とRMSE予算である。Bはその式のfinite-K・signed-piece・現在のstate/actionへの適用を正式採択していない。仕様は技術案であり、科学実行許可ではない。

## 実装と既存機能の再利用

新規libraryは[`pr_rpe_rmse_controller.py`](../../../src/trottertracks/pr_rpe_rmse_controller.py)一つで、科学runnerは追加しない。

| 機能 | 動作・再利用 |
|---|---|
| `generate` | 明示されたL_D/ΔPF/M/共通κと固定h/formulaを1つの全scheduleへ変換。既存`p.structure`/m5固定weights・ABS_TIME整数配分と`p.identity`を再利用 |
| `paper_event_mapper` | A/B共用の候補整数化案。全roundのκ定義/value/SOURCEを固定。未知A mapperは明示callback/sourceがなければNOT_EVALUATED |
| `assess` | 生成済みconfigurationの改変、N/global RMSE/finite-K、C種別/domain/measurement/law/cohort・欠測を確認。Gは計算しない |
| `cost` | assessが適格とした供給済みN_X/YとCを積和。既存`exact_scalar/exact_value`を再利用し、巨大Nとbinary64価格をexact rationalで保持 |
| `select` | 外側(method,L_D,ΔPF)ごとのM/共通κ最小と、その外側候補間の点最小を比較。whole scheduleのみ選び、異なるκのroundを合成しない |

既存`rpe_r_selection`は固定h/Mのround別R選択、`rpe_provider_integration`は現行高確率型N batch、`rpe_r_price_intake`はC1の固定domain/finite-cohort受入を引き続き担当する。これらとN/C provider・旧会計のSOURCEは変更していない。高確率型の`CONDITIONAL_ROUND_PASS`やalpha予算を仮造してRMSEを旧集計器へ通さない。新たに必要な積和は既存exact primitiveを使う小さなpolicy中立ループに限定した。

## scheduleとidentity

configurationはH/state/snapshot/decomposition/SOURCE、split/順序/L_D、PF公式・要求幅、h/M、κ kind/value/定義、mapper SOURCE、finite-K/law、RMSE policy/error contract、compiler/control/測定wrapper/P0を含む。各rowはT_m=h·2^m、qRPE、qPF、実PF幅、signed weights・配分/event数を持つ。h/Δ/κ/時刻/実幅はint/binary64/exact-rational入力からexactに正規化する。

PRの整数化案は要求総event目安をqPFごとに均一outerへ丸め、全roundで同じκを使う。signed formulaの最小piece数と既存剰余tie規則を保全する。κとeffective event数の等号や、finite-Kでのnormalization保証は主張しない。異なる定義には名前/SOURCEを持つA callbackが必要で、共有κのecho・配分整合を検査する。

PF/PRのqPFはceil(T/ΔPF)、実幅はT/qPF。実幅がround間で変わる場合を保存し、同じeffective Hamiltonianの冪と仮定しない。hとΔPF、paperの基準unit、control/時間半減の整合はAの全schedule誤差契約で資格する必要がある。Bは時間や誤差予算を無断で半減・変更しない。

IDはspecと全scheduleを含む。Mやκが変われば、同じmのrowも別identityになる。旧固定R IDの再ラベルをしない。変更したR/allocationや他κのN/C recordを混ぜると不適格になる。

## 適格性・費用・不確かさ

RMSE policy familyは`PR_PAPER_RMSE`で、旧`HIGH_PROBABILITY_RPE`と別。A供給の全schedule RMSE boundと同じ目標を比較するだけで、N solver、二乗和・bias合成、normalization計算は実装していない。Aのerror_contract、state/target・alias/prior/decoder/全round、finite-K/signed-piece/κ対応の資格が必要。科学値ではpolicyの`EXTERNALLY_QUALIFIED`も外部から必要であり、Bは発行しない。人工evidenceは`SYNTHETIC_ONLY`で科学資格と分離する。

Nは各軸の量子shot数で正の整数。Cはfull identity/law/source/measurement/単位と明示cohortに対応する1 recordを各軸に選ぶ。DIRECTとMODEL、finite cohortと期待値を分け、MODELの適格性/domain/q変数/校正検証SOURCEを確認する。純RTEモデルの評価変数はevent_countで、PF/PRはqPF。未検証q/Rモデル・null価格・逆符号測定・cohort混合・forced cohortは適格にしない。

既定はqualified expectation。有限cohortは明示scenarioだけに限定し、sampler距離を新しく資格しない。normalizationや古典標本数を掛けない。不明なcross-κ/cross-round共分散・systematic・joint CIはnullとし、価格recordの既知uncertaintyはcopyで保持する。

生成成功やassessの適格性だけではG評価済みにならない。selectはcostが未取得ならpendingとし、欠測候補を除いた最小は`AVAILABLE_QUALIFIED_SUBSET`と表示する。正当なRMSE不成立だけをINFEASIBLEとして除外し、foreign Nの不成立値では候補を刈らない。有限menu外は常に未探索として残し、同値・価格policy・fixed h/formula/K/κ定義・SOURCEを保全する。解析κ最適値は計算も採択もせず、外部seedとして与える場合もcompiled最適値と区別する。

PF・PR・全H純RTEを同じH/state/h/RMSE/価格profileで比較できる。純RTEはone-bodyを含む全非scalar Hをランダム化するラベルで、B3 one-body保持型を拒否する。PFはfull-H scopeを要求する。PF/純RTEのcoverageが欠ければ方式比較completeにせず、全方式が有限menuにあってもPF baselineのglobal最適性やPR優位を宣言しない。

## 保存成果とテスト

[LEGACY_COMPATIBILITY](LEGACY_COMPATIBILITY.json)は、A-P1公開`71ba49bb28b295aa28e51101bc3b1bf07d500767`の12 N recordを元menu/request/law/比較policy/alpha付きround statusへ照合した記録。値は再計算・転用せず、高確率型のまま保持した。C1の32価格record・16モデルの`qualified_for_candidate=false`、finite cohort限定も保存metadataで確認した。A0診断は既存受入のまま、RMSE N/Cへ昇格していない。

[人工テストSOURCE](../../../tests/check_pr_rpe_rmse_controller.py)は17件。共通κ、M変更・全identity、signed配分、未知mapper、生成/資格/G分離、RMSE不成立/不安定、高確率型拒否、欠測/null/重複、C1型MODEL false、SOURCE/law/compiler/unit、有限cohort、巨大整数N、whole-schedule選択、外側/内側menu・未探索・同値、3方式/B3区別、数値backend非起動を確認する。人工例では独立roundのhybrid費用4に対して、合法な共通κ scheduleの最小は20となり、hybridを選ばない。これは人工例であり実分子の改善結果ではない。

詳細：[TESTS](TESTS.txt)／[HANDOFF](HANDOFF.md)。既存会計15・pipeline12・R選択12・C価格受入15も回帰して全PASS。作者自身の技術確認で、独立科学レビューではない。

## 未確定事項・次の実行範囲

1. **A/ユーザー**：論文版、RMSE error contract・state/target/ground仮定、h/ΔPFとM、共通κの種類とfinite-K/signed-piece変換、端点/丸め/時間半減、誤差項の関係を科学的に確定する。Bのdraftは採択の代替ではない。
2. **C/ユーザー**：新しいsplit/ΔPF/M/κ/scheduleに対する価格domain、モデル採用・必要精度、期待値/finite-cohort policy、rare/全law・不確かさの必要範囲を決める。C1のfalseを緩めない。
3. **ユーザーの別途明示許可後にA/C**：有限の固定tuple/snapshot/SOURCE/precisionで必要なN/RMSE証拠とC校正・独立検証を供給する。PF baseline/純RTEも同じRMSE目標で扱い、必要な探索範囲を明示する。
4. **B**：公開されたN/C recordを本interfaceへ照合し、実G/最適化を行う場合は、その有限menuと価格policyを含む別の科学実行許可を受ける。

本担当は技術実装と人工testまで。既存SOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOP、A/C worktree/process/cacheを保全し、共有環境を変更していない。commit/push・remote/主要成果物照合後mandatory STOPし、新しい科学計算や実資源最適化へ自動進行しない。
