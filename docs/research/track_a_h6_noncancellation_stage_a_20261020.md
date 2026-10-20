# H6 noncancellation段階A：保存値再解析と終了監査

2026-10-20。基点`6cfc4579223b0ab4e54adbdacbc3b16a88ecf382`。科学SOURCE **`85e250302d15b427731a50ddbe321fc280ec5f1a`**。
ユーザーが段階AのSOURCE/契約固定、one-shot解析、必要shot、数値検証、保存、終了監査まで明示承認した。段階B、詳細科学レビュー、最終evidence freeze、新規公開pushは承認対象に含めていない。

**12候補×10条件の120 endpointすべてで非相殺適格性が確定した。適格34、不適格86、未解決0。成分closureは12/12 PASS。**
11候補のsigned成分を保存signalから構成し、p15/q4の1候補は回収済み公開成分をそのまま使用した。必要analytic shotは適格34 endpointで保存した。以前のnullを新しい結果へ対応付けたのであり、旧SOURCE・旧会計・旧STOPを上書きしていない。

## 物理条件・指標・入力

Linear H6、隣接1 Å、STO-3G、tol-only DF19、12 modes/sector400、T=.8、P=0。B0I S2、prefix5/10/15、q1/2/4/8（δ=.8/.4/.2/.1）。元のsymmetric_directional/full cosine-sine Hadamard wrapper系列に対応する保存signalであり、新規回路は生成・compileしていない。
ε=.05/.01/.005/.001/.0005、allowance倍率1/10、α_axis=.025を保持した。shotは元の対称軸・経験的allowanceに条件付けたanalytic sufficient shot数で、実量子shot実行、PRのMC trajectory標本数、n本の古典取得合計費用とは異なる。

Hamiltonian/state/input等は[保存bindingの転記](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_a/2026-10-20/input_binding_reference.json)を参照。元12rawは調査packetとV1 input receiptのbytes/SHAに一致し、candidate/prefix/q/T、Hamiltonian/DF/state/input/compiler/providerのmetadataを照合した。p15/q4は固定δ公開資料のsignal・reference・allowance・log_Bが元rawと完全一致することも確認した。保存signalの再生成、SCF/DF/state/reference/evolution、位相再回転、NPZ decodeは0。

## 判定結果

下表の各セルは**倍率1 / 倍率10**。適＝適格、不適＝不適格。120は候補12×精度5×倍率2で、独立した120実験ではない。

| prefix / q | ε=.05 | .01 | .005 | .001 | .0005 |
| --- | --- | --- | --- | --- | --- |
| 5 / 1 | 不適/不適 | 不適/不適 | 不適/不適 | 不適/不適 | 不適/不適 |
| 5 / 2 | 不適/不適 | 不適/不適 | 不適/不適 | 不適/不適 | 不適/不適 |
| 5 / 4 | 不適/不適 | 不適/不適 | 不適/不適 | 不適/不適 | 不適/不適 |
| 5 / 8 | 不適/不適 | 不適/不適 | 不適/不適 | 不適/不適 | 不適/不適 |
| 10 / 1 | 適/適 | 不適/不適 | 不適/不適 | 不適/不適 | 不適/不適 |
| 10 / 2 | 適/適 | 不適/不適 | 不適/不適 | 不適/不適 | 不適/不適 |
| 10 / 4 | 適/適 | 不適/不適 | 不適/不適 | 不適/不適 | 不適/不適 |
| 10 / 8 | 適/適 | 不適/不適 | 不適/不適 | 不適/不適 | 不適/不適 |
| 15 / 1 | 適/適 | 不適/不適 | 不適/不適 | 不適/不適 | 不適/不適 |
| 15 / 2 | 適/適 | 適/適 | 適/適 | 不適/不適 | 不適/不適 |
| 15 / 4 | 適/適 | 適/適 | 適/適 | 適/適 | 不適/不適 |
| 15 / 8 | 適/適 | 適/適 | 適/適 | 適/適 | 適/適 |

[全120行のMarkdown表](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_a/2026-10-20/endpoints120.md)、[CSV](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_a/2026-10-20/endpoints.csv)、[JSON](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_a/2026-10-20/endpoints.json)に、軸別shot・headroom・状態・全会計fieldと元raw SHAを保存した。候補別状態表は`eligibility12.csv`。
例えばp15/q4、ε=.05、倍率1はcosine `7153`、sine `7242`、合計 `14395` shot。これは元結果の転記で、費用単価を掛けた総RZやPR/PF比は計算していない。
不適格endpointの必要shotはnullのまま。元関数が保存する軸単独の中間診断がある場合も、endpointの必要shotへ補完しない。

## 算術・closure・数値検証

- 11候補：`discard = saved exact_truncated − saved full-H reference`、`PF = saved corrected − saved exact_truncated`。full-H referenceは保存された不回転の値を使う。
- p15/q4：既存公開のdiscard/PF成分をcopyし、本解析の入力成分を再構成値へ置換しない。数値監査側では保存差との一致を別に確認する。
- 元SOURCE2261a3adの`track_a_v1_contract.signal_envelopes`をそのまま呼ぶ。signed成分の和と元total_signedを元のrelative/absolute1e-12で照合し、各軸の絶対値和envelopeを得る。12/12 PASS。
- 元の`d0_saved_value_reanalysis_v1.accounting`と`d0_frozen_shot_rule_v1`を変更せず120回使用。allowance/log_B/log_B_margin、倍率、α、headroom/normalization/log-domain/ceilのbinary64演算順、u/headroom guardを保持する。未確定や不一致はその行を保存して契約をSTOPする経路とし、補完・retryなし。
- one-shot子プロセス内の監査で、Fractionによる保存差・envelopeの別表現とclosureを12候補確認し、120 endpointのheadroom、元の整数shot順・ceil区間、状態、合計shotとpolicyを照合した。primary会計の再実行・再試行ではなく、封印済みの別検証予算である。

人工14件＋既存shot-only回帰9件、計23件PASS、fail/error/skip0。実保存値の数値解析前にSOURCEをcommitした。wrong reuse/closure FAIL/未確定/非finite/予算超過は拒否し、入力不変、軸絶対値和、10 endpoint、元のshot算術との一致を人工値で確認した。
post-run検査は保存された132個の数値監査記録、120行CSV/JSON、結果SHA、入力/SOURCE/runtime/旧証拠、native exit/reapを照合するだけで、新たな数値解析を行っていない。primary agent自身の検査であり、別担当による独立科学レビューではない。

## 実行契約・call予算・終了監査

新しい[sealed contract](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_a/2026-10-20/sealed_contract.json)、grant、ユーザー指示receipt、one-shot/launch marker、ledger、consumption receipt、STOPを独立packetへ保存した。旧grantの再利用・返却・再開は0。

| 科学処理・監査の単位 | 予約 / 完了 |
| --- | ---: |
| 保存signalからの成分構成 | 11 / 11 |
| 既存公開成分copy | 1 / 1 |
| 元関数によるclosure/envelope | 12 / 12 |
| endpoint適格性・必要shot会計 | 120 / 120 |
| 別表現による候補数値監査 | 12 / 12 |
| endpoint数値整合性監査 | 120 / 120 |

新signal/phase rotation/SCF/DF/state/sampling/compile/model prediction/fit/weighted cost/PF selection/ranking/PR statistics/stageBは全0。
元Python pathを使用し、実際に読み込むstdlib・native library等99 fileを今回のruntimeへsealした。旧provider/venv全34,273fileや過去compilerのbinary完全同一性を再認定したとはしない。

子PID `455040`、親PID `455022`、native exit `0`。同じ起動親が`os.wait4`で回収し、残存owned science worker0。stderrは空。数値本体・前後gateのchild内時間 `0.24718240089714527`秒、起動/入力検査/終了回収を含むchild wall `0.45126055367290974`秒、CPU `0.44676299999999997`秒、peak RSS `49307648` bytes。
保存12件と12 envelopeを一度indexし120条件で共有する小さいstdlib処理のため、単一science childが効率的だった。CPU/RAM/workerの独自上限、affinity変更、共有設定・他ユーザー/processの変更は0。既存OS制御は維持した。

## 保存SHA・保全

| 保存物 | SHA256 |
| --- | --- |
| sealed_contract.json | `b037de21b102761133770dc09e1a705e1bb566564df11b9de9990ac787b62526` |
| components.json | `420030920a03a43f949f594cb021db06ca03ecbafb19d5d92ebb5f9028b5da93` |
| endpoints.json | `723095aac104d9a5a1314242ce9944c330e8fd52249c1c3955fd50a858f936dd` |
| endpoints.csv | `1c5c0be2b5a5084908588e3bf2ddf23172c14b1a25abb0117cef954186c7a59b` |
| numeric_validation.json | `746d1377ae6ab02b43a03600695f3d802fa02ff9019d5d97b8e61ac89399e0ab` |
| summary.json | `da1a15e1c5d2b7b389f0b9016ae3eeef8a89d85e93a94917c42f93252b8af028` |

12元raw、旧数値SOURCE3file、新SOURCE4file、公開入力7file、前回調査packet・70件packetの55file、今回の算術runtime99fileを実行前後・終了監査で照合してPASS。既存SOURCE/raw/grant/ledger/marker/STOP・70件の比較結果・v2本文・論文主表/Figure2は不変。新規の科学成分/適格性/shotを新packetへ保存し、旧`eligible=null`や古いSTOPを遡及更新していない。

## 残る限界・段階Bの必要事項

今回の対象だった12候補/120 endpointの数値未確定は0。元30scopeを止めていた**この適格性欠測**は新結果により解消した。一方、30scopeの総費用・PF最良・PR/PF比・順位比較はまだ未実行で、完成した比較結果と呼ばない。
経験的bias/envelope・allowanceは厳密なbias certificateではなく、PR母平均や未観測希少event、全域の費用優位を保証しない。H6のprofile差、model/runtime同一性未証明、S4仮説、初回最終gate欠測、旧profile48軸等の費用比較の制約は解消していない。
後続Dのtruthiness欠測集計問題は旧SOURCEを変更せず保全した。今回の120行が確定したことを理由に旧Dのunresolved空配列や旧費用結論を遡及認定しない。

[段階B引継ぎ](../../artifacts/resource_applicability/track_a_h6_noncancellation_stage_a/2026-10-20/stage_B_handoff.json)はmetadataだけで、実行契約や有効grantではない。適格34 endpointの両軸に元単価/置換単価fieldが存在することを照合したが、乗算・順位・比は計算していない。

次担当**ユーザー**が、H6 noncancellationの30scope（重複を除く10 endpoint）について、新たな基準費用の構成と実測置換・PF選択・PR/PF比較を別途明示承認する必要がある。次担当**Codex**は承認後に次を行う。

1. 新120会計を旧null120行だけへ来歴付きでjoinし、他PF・保存PR/SE/cohort・候補集合32/96/136は保持する。`eligible=None`を明示STOPする新adapterを使い、旧Dのtruthiness経路を流用しない。
2. 元から確定した基準費用は保存値へ完全一致させる。今回新しくshotが得られた34行の基準費用は未定義だったので、新承認下で構成し、旧保存winnerの再現と偽らず新来歴へ分ける。
3. 元ALL_SAVED458の単価政策と条件を固定した新SOURCE/入力/runtime/契約/grant/ledger/marker/STOPを準備する。今の段階A grantは消費済みで再利用しない。追加signal/compile/sampling/予測/refit/shot再計算・PR統計は許可に含めない。

新規公開範囲は未承認でlocal commitのみ、push0。詳細科学レビュー・最終evidence freezeは未開始。**段階A完了後mandatory STOP。段階Bや追加解析を自動実行しない。**
