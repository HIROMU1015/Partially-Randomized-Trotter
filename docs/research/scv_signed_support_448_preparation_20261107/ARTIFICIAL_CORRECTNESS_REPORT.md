# 人工correctness結果

2026-11-07 / SERVER1。**新規42＋既存159＝計201件PASS**。約18秒のローカル人工unit/mock実行であり、実H4の科学的改善・CI・独立再現の証拠ではない。

正本は[ARTIFICIAL_TESTS.log](ARTIFICIAL_TESTS.log)と[test別結果・source hash](TEST_RESULTS.json)。再実行commandは同JSONに保存した。Python3.12.3、共有opt2 venvは読み取り専用、cacheは本作業専用の/tmpだけ。

## Fixtureと確認範囲

既存 `test_scv_signed_support.fixture` の人工3qubit LAW/catalog（a,b,cとzero-grid label）、dyadic質量、signed値−1/2,+1,−1,+1/2、手作りbit列、mock RZ=9等を使用する。H4 LAW/catalog/state/gateを人工処理へ投入していない。人工stream mapping検査だけはseed20261110でtoy wordを生成するが、実H4科学streamは作成/消費していない。

| 追加検証 | 結果 |
|---|---|
| Saved control | 通常moment構築を禁止してtyped復元→bank→fit→paired。hash/commit/path/schema/feature/Fraction/field数/LAW/catalog/H/preparation/sign/support/qubit/identity/duplicate JSON拒否。immutable tupleとsampler変異検出 |
| Fit/guard | 複数signed値、intercept OLS軸別各一回、binary64→Fraction freeze、rank不足/nonfinite STOP、D=6両符号PASS、D>6拒否、clipping/refitなし |
| Stage | 両trainをfitより先に取得、X→Y fit、両PASS前eval拒否。X失敗時128 trainと費用を保持しY fit/evalなし |
| Quota/paired | mock448 physical、全5層、trial ID固定二分、6集計、同じbit、共通eval320一回課金、train128込み |
| Stream | 12 domain/24子mapping、path/bit・X/Y・train/eval分離、重複claim拒否、64+1 word境界、旧PinnedSamplerの追加PRNG禁止検査を再利用 |
| Finite call | 各固定capのschema照合と境界縮小mockによるcap+1のbody拒否、955 gateの次の適用拒否。静的capと観測counter分離 |
| STOP競合 | 同時2callがlock外で進行→STOP後新START拒否→既存RETURN回収。fork childとの共有受付、lock所有childのSIGKILL後STOP、worker再attempt拒否 |
| Worker/pool | 既存group→ManagedPool→2 fork workerを人工法則/確率/価格で実行、bit/価格fsync・exit0回収。価格失敗後bit保全、planned未開始/価格欠測UNKNOWN、STOP先行cancel/drain |
| Cost | 全cache-hitでもRZ課金。人工train1152＋eval2880＝physical4032 RZ、baseline参照evalだけ、SCV参照合計。参照費を二重加算しない |
| Activation | 人工parent全448、restore/decode/stream各1、独立fit2。resource failureはrestore/decode/RNG前STOP。foreign manager/restart/consumed・SOURCE/seed/profile/compiler混入拒否、最小cron環境で受動help |

## 再利用suite

`test_scv_signed_support`、`test_scv_scalar`、`test_scv224_pilot`、`test_scv224_activation`、`test_a1_control_variates`、`test_a1_n3_connection`、`test_a1_n3_campaign`、`test_a1_n3_managed_runtime`、`test_a1_n3_resource_admission`、`test_scv_signed_h4_metadata`を再利用し、新 `test_scv_signed_pilot` の追加境界と合わせた。

新規suite単独42件、全体201件。復元の人工テストはmomentの再計算を禁止し、実保存H4の復元は未実施。quota mockはphysical448、provider呼出し境界は小規模人工job/縮小capで検査し、実H4を448回走らせて上限を検査したものではない。全427840 gate・29120 wordを実科学で消費していない。

manager経路は既存qualified実装をmock接続した。今回新しいcron/SSH断・本番manager・実科学workerは起動していない。既報の持続manager人工qualificationを再認証したとはしない。host全体のkill、storage failure、実H4の長時間Fraction増大・時間/RSSは未検証。

全テストPASSはSOURCE候補の技術correctnessを示すもので、CV相関・分散削減・学習費回収・期待RZ改善の科学的結論はUNKNOWN。実科学GO/NO-GOや保証付き本検証を採択していない。
