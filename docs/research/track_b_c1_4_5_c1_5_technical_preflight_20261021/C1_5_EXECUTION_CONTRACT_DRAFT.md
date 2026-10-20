# C1-5実行前契約草案 — 未採択・実行権限なし

`C1_5_CONTRACT_DRAFT_READY`は文書の準備状態だけを表す。科学実行readinessは`C1_5_REQUIRED_CERTIFICATES_MISSING`。下記の採択・証拠・新SOURCE/authorizationがない状態でH4計算、native取得、N、shots、Pilot/Full-N、全round RPEへ進まない。

## 共通target候補と比較

候補A：R1 exact signed involutionモデルの同一m=5 finite target。microstep P5(-ixR)、独立step meanのordered積、outer S2 signalを区別し、どこまで比較するかをユーザーが採択する。

候補B：保存DF Hamiltonianの指数signal `<psi|exp(-i T H_DF)|psi>`。候補AからのDF/R1 binding・PF・Taylor差を認証して初めて同task比較となる。旧R1 model_bindingの数値残差をrigorous norm certificateへ格上げしない。

有限target比較familyはOrdinary、Partial Return、Closed P3+tail、General Full、Closed P5。共通p/label/sign/ordered word/basis/scalar、許容誤差、state/access、measurement、provider、相殺規則を揃える。Full/P5の同理想accepted ensembleを保持し、digital lawやN/費用の一致は要求しない。全方式に合法cache・前処理・selector・DP・basis共有を同等に認め、その費用を記録する。

既存correctness条件T=4/5、q_outer=1、r_RTE=4、m=5、sigma=+1、outer S2は参照値であり、C1-5の性能最適条件として未採択。R1 exact p/新partitionは旧PR-2と別identityである。

## 20実行前条件

| # | 固定する契約 | 現状態 / 次の担当・必要承認 |
|---|---|---|
| 1 | 共通targetと評価signal | 未採択。ユーザー・GPTがfinite限定/保存DF指数と比較層を選ぶ |
| 2 | R1 exact modelとDF binding | MODEL_BINDING_UNCLOSED。Codexのcertificate回収/技術接続scopeを明示。新計算は別承認 |
| 3 | state preparation/access | 物理準備・準備頻度・誤差・費用UNKNOWN。ユーザーがaccessを採択し、Codexが実証拠を結ぶ |
| 4 | T/q/r/m/sigma・各方式schedule | m=5維持、他は参照値のまま未採択。Short-step、全deterministic half/scalar、timeを固定 |
| 5 | finite-bit仕様 | root/probability/angle、positive support、actual π/W、canonical各方式certificateが必要。Full P0-P5の流用禁止 |
| 6 | PF/model/Taylor bias | 既存式の適用条件は確認。実入力PF定数とbinding不足。誤差ledgerへ全層別計上 |
| 7 | native/controlled error | uniform strict operator norm、actual inverse、signed CQ/angle/scalar証拠未取得 |
| 8 | readout error | 測定規約のみ接続。実backend校正/誤差/費用UNKNOWN |
| 9 | Re/Im confidence | axis配分・bias控除後positive残余・M2/range・IID/測定条件未固定。新CI/Nなし |
| 10 | familywise failure | rows/axes/resource-tail/technical abortのfamilyを事前固定。alpha=0.05は未採択候補 |
| 11 | whole-path estimator | 人工最小実装済み。実H4接続・state/schedule/measurement/永続Evidence transactionはIMPLEMENTATION_REQUIRED |
| 12 | 全trial Nと終端 | N未計算。全予定ID完走・valid zero分母・failure terminal・prefix非正式を固定する新実行承認が必要 |
| 13 | full-support provider | 観測key成功を全supportにしない。実coverage certificate/unknown key policy未取得 |
| 14 | native price | 到達層（uniform strict上界/認証期待値等）をユーザーが選ぶ。native取得はtool/profile/新scopeと別承認が必要 |
| 15 | classical/setup/storage | 全方式のexact arithmetic/root/selector/cache/replay/Evidence/終端費用、単位、取得方法を固定。UNKNOWNを0にしない |
| 16 | 強いbaseline | 下表の指数task対照を取得済み扱いしない。対象/representation/精度/最適化機会を採択 |
| 17 | SOURCE/input/runtime | 新SOURCE、exact model/partition/snapshot/label/basis、compiler/options/runtimeと出力identityを固定 |
| 18 | one-shot/retry/STOP | 新scope・新one-shotとretry/resume規約・failure保存・STOPを明示承認。旧grant再利用なし |
| 19 | grant/claim/ledger/marker保全 | 旧SOURCE/raw/失敗/authorization/STOP・非公開原本・他worktreeを変更しない |
| 20 | 実行後の判断対象 | 同target誤差/費用/強い対照/完走を確認してユーザー・GPTが科学判断。資源優位・新規性・投稿GOを自動認定しない |

## error/confidenceと費用の契約

`epsilon_complex=0.05`、`alpha_familywise=0.05`は**未採択候補**。complex精度からaxis精度への配分、全row/axis/resource-tailのfailure family、deterministic certificateと確率的failureの区別を先に決める。root/angle/native/state/readout、Taylor/PF/model差を区別し、bias控除後の残余が正である場合だけ既存Bernstein/log/ceil式を適用する。今回数値十分N・実shot/Monte Carlo・CI・性能ランキングは0。

whole-path全trialはunique ID、実πとW、ordered product、raw/coin zero、測定±1、全N完走を保持する。全pathを量子前に先決定した場合だけzero quantum回避を認める。prefix実行済み費用は保持する。failure・保存障害・precision不足をzeroへ変換せず、未完了prefixにconfidenceを移さない。

費用はlogical、strict native、event期待価格、全trial総量子費用、classical/setup/online/storage/terminalで分離。uniform保守上界とexact期待価格を別の目的量として採択する。native compiler/synthesisを使う場合は新tool/profile/strict誤差/coverage取得範囲の別承認を必要とする。

## 指数taskの強いbaseline

| 対照 | 既存部品 | 残る契約・状態 |
|---|---|---|
| DF Ordinary PR | 既存DF RTE/outer S2 source | R1新exact入力、scalar/state/誤差/confidence/native費用へ再結合。旧PR-2を同一入力扱い不可。UNKNOWN |
| deterministic symmetric S2 | `df_partial_s2.py`のdeterministic blocks/forward-reverse構造 | 同保存DF/state/T/精度で合法step・strict native/準備/readoutを認証。UNKNOWN |
| Pauli CTS/SCU | 既存D/G系列のsynthetic CTS部品、LCU/測定契約の参照 | 実H4 representation/access/取得・Pauli展開/前処理/provider/誤差/費用が未接続。H4比較SOURCE未作成、UNKNOWN |

強い未取得対照を省いて一般資源優位を主張しない。新H4比較SOURCE・実行は今回作成していない。[旧baseline草案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/068f0ae8771fccf28c529bda12a84abe8b7e7648/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/C1_4_NATIVE_BASELINES.md)

## GO/STOP

GOは上の必須項目が採択/結合され、UNKNOWNを仮定で補わず、凍結SOURCEと実行前必須検査がPASSし、ユーザーが新scopeの科学実行を明示承認した場合だけ。草案完成・人工PASSは承認を代行しない。

SOURCE/input不一致、law/target/誤差定義の変更、新非自明証明の不可避化、mandatory技術gate不成立、coverage/価格未取得、positive残余なし、実H4/native作業の未承認要求、保全対象変更の必要、technical failureなら依存実行を停止する。無関係な静的不足整理は続行できる。

新証明が必要かどうかは、既存保証で結べない具体的gapを特定した時だけ判断する。今回`PROOF_REQUIRED`確定箇所は0。非列挙exact期待価格や最良word相殺の一般保証を要求するなら、その必要性・scope・既存法での閉鎖可能性を別途検討し、承認なしに新定理/cost oracleを作らない。
