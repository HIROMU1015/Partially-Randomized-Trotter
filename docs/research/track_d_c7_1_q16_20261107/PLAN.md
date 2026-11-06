# Track D C7-1：q16独立whole-path探索的paired価格 — 実行前PLAN

2026-11-07 JST。基点[C7-0](../track_d_c7_0_structural_preparation_20261107/REPORT.md) `8b422c28c22666a32103fdae33a3c9b9512ef248`、専用branch `codex/track-d-c7-1-q16-exploratory-20261107`。ユーザーのC7-1指示により、新しいq16 whole-path16本・256 occurrence・8192 events、raw correctness、各path PASS後だけ最大64 P0 wrapper、postcheck/pairedχ、価格統計/CX残差/凍結モデル診断を一回実施する。**SOURCE・本PLAN・seed・モデル/幅・人工結果をsampling前にcommit/公開し、remote SHA/tree/公開bytesを照合する。** 新grant・承認証明は作らない。

## 科学条件・SOURCE・既存STOP

linear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1、q16のみ。元Hamiltonian Hの厳密ground仮定・状態準備費除外。ground vector、ground solve、DF fitは使わない。

元C science `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`、packet SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、law記述 `c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5` を保持。signed atom、scalar/identity、normalization/attenuation、位相・順序・測定規約を変更せず、lawのweightをunitaryへ掛けない。

C6a校正結果 `e9d4e2f602d5f9e014507ca7e391719bab017b88`、科学SOURCE `6b6667026ec8b670f610e41141d195883a9eb513`。C6b最終 `fed5f458be2f625de4e1b8c98b8cc5da6e551b14`、当初SOURCE `23725fc6f6d6d0c2bec93e31e32074711804e250`、承認済み引継ぎSOURCE `44a58b9fba5d3aba7a31d0a6b12e3aa25be9def4` の生成証拠・旧新STOPは保持する。今回のq16は新しい許可scopeで、C6bのresumeでも引継ぎ許可の流用でもない。C5 `9a6920b8355643701ab7436a4fd3b34862e9e3bd`、C3完了 `cc554cf6daea08df6c20ea8e1e92405259ce49b5` 等の旧SOURCE/input/raw/result/grant/ledger/marker/STOPを変更しない。適用AGENTS/PROJECT_MAP・正式共通指示2026-10-25を遵守する。

## Samplingの事前固定

cohort `track_d_c7_1_q16_exploratory_paired_price_v1`、q16_sample00..15、各occurrence0..15を昇順生成する。各step32 events、総256 canonical occurrence/8192 events、同一pathを両方式・X/Yへ一度ずつ渡す。

新domain `track_d_c7_1_q16_exploratory_paired_price_v1:independent_holdout:q16`、master2026110701、master tuple `(2026110701, domain, '8b422c28c22666a32103fdae33a3c9b9512ef248')`。既存 `trajectory_seed(master_tuple, 'proposed', endpoint, 4, 'C7_1_PAIRED_PRICE_ONLY', sample, occurrence)`、SHA256(repr(tuple))先頭8 bytes/little endian、NumPy1.26.4 PCG64。[完全seed表](../../../artifacts/track_d_c7_1_q16/2026-11-07/technical/seed_plan.json)に256 seedと順序を固定し、C6a96・C6b256既使用seedと非重複を確認する。旧q8 domainの`unused`文字列を再利用しない。非重複はideal joint-IIDの証明ではない。

canonical `configure_rte → sample_occurrence → iter_sample_rte_events` を再利用する。全16本のevent列を一回だけ生成してローカル保存し、record/event hashを固定してからCAR/回路/価格へ進む。価格によるpath/seed/cut選択、除外、差替え、追加標本なし。旧cohort巡回、prefix再利用、短path反復をしない。

## 凍結予測・構造候補・探索的統計

[凍結モデル](../../../artifacts/track_d_c7_1_q16/2026-11-07/technical/frozen_model.json)はC6a保存q2/q4 affine係数と全48係数共分散のidentityを保持し、q16の方式別/paired予測・幅をsampling前に固定する。再fitせず、C6b q8をfit点へ追加しない。

一次familyはRZ/CX count、RZ/CX depth、total depth、sizeの各X/Y、計12量。方式別絶対価格・全資源/方式/axis共分散も保存するが、絶対価格の確認的精度PASSは対象外。peak qubitsは別の幅。X/Yを32独立whole-pathへ数えず、n=16、SDはddof1とする。

以下は**未検証q16の事前予測**であり、新しい分子回路の観測結果ではない。X/Y各々同じ保存係数で予測する。

|資源|strong予測|Reference予測|paired予測|τ=1+.05×strong予測規模|
|---|---:|---:|---:|---:|
|RZ count|81224.75|80733.8125|−490.9375|4062.2375|
|CX count|31293.375|27917.375|−3376|1565.66875|
|RZ depth|19573.0625|9546.375|−10026.6875|979.653125|
|CX depth|15322.6875|5891.25|−9431.4375|767.134375|
|Total depth|45686.5625|21139.625|−24546.9375|2285.328125|
|Circuit size|165987.625|162312.6875|−3674.9375|8300.38125|

名目的alpha=.05、二側normal/Bonferroni、z=Phi^-1(1−.05/(2×12))。e=観測paired平均−凍結予測、SE=s16/√16、一次は `|e|+z SE≤τ` によりEXPLORATORY_WITHIN_WIDTH/EXPLORATORY_OUTSIDE_WIDTHを記録する。予測−観測も符号を明示して保存。一次は凍結予測値に条件付け、calibration varianceを重複加算しない。

別population-line欄は `|e|+z√(SE²+vcal)≤τ`。C6a保存sample covarianceからCov(meanq)=Sq/16、q間擬似独立を条件として、q16予測の重みw2=−6/w4=7を用い `vcal=36 Var(meanΔ2)+49 Var(meanΔ4)` とする。方式間共分散を保持し、保存係数共分散と同じ線形伝播を使う。校正不確かさ、holdout変動、model bias、q-domain移送、finite-bit lawを混同せず、別欄を新たな確認的familyとして認定しない。

CX構造候補 `208−224q` のq16予測は整数−3376。各path/axisで `rCX=観測ΔCX+3376` とexact-zero判定、平均/標本SD/min/maxを保存する。整数候補の一致と5%価格幅内は別評価。全観測path一致ならCX_CANDIDATE_MATCHED_OBSERVED_PATHS、一つでも非零ならCX_CANDIDATE_VIOLATED。どちらも全path/全qの定理の認定にはしない。

C7-0の固定/bulk/boundary＋P0残差、RZ/sizeの固定負担と累積差、Reference lane/critical pathは事前の解釈候補。q16結果を見て特徴/係数を変更し、同じ標本を新モデルの独立holdoutと呼ばない。PCG64 joint-law、normal/tail/support、モデルbias/domainは未資格なので、全体の統計資格はSTATISTICALLY_UNQUALIFIED、確認的PASS/有効95%同時coverageはfalse。

## 最小実装・再利用・人工検証

旧C6aはq2/4・n16、旧C6bはq8/cut4・n32に入口/集計/診断が固定されている。そのguardと旧ファイルを変更せず、q16/cut8/n16専用の薄いrequest/event接続・価格資格・完全標本集計を追加する。集計/診断はC6bの同じ手順とC6aのdescription helperを用い、q-domainと標本数だけを新scopeへ束縛する。

回路組立はC3 `prepare_steps/physical_segment/mixed_head`、CAR/head/signed atom cacheは既存B2/C6a、Gaussian/strong baselineは元StrongBuilder、Reference actual inverse/χは既存adapter、wrapper/価格検査はC5、raw checkerはStage A、P0/postcheckとpairedχはC2/C3/C6aを再利用。新sampler、Gaussian合成器、compiler、state simulatorは作らない。

新workerだけで既存C6b `BatchedStateActions` を選択し、同じofficial local-matrix変換/Aer double/native callback、copy削減・入力batch・宣言位相復元を使う。価格対象の回路/P0は変更しない。新q16人工7件＋既存C6b/C6a/C5/C3/C2/高速化の計41件PASS。[人工記録](../../../artifacts/track_d_c7_1_q16/2026-11-07/technical/synthetic_tests.json)。2-mode toy q16のAB/actual inverse/vacuumχ/X/Y、seed/domain/旧guard、mock canonical16 calls、完全標本/covariance、CX反例の保存、校正共分散伝播、raw PASS後だけ4compileの投入を確認した。人工PASSは分子価格・成功保証ではない。

## Whole-path・強い対照・correctness

時間順U0..U15、V=U15…U0=A B、B=前8 step、A=後8 step、cutはcomplete step8後に固定する。whole VにReference gadget一つ、A†は同じAのactual inverse、path全体analyticχ、二つのCSWAP cascade、追加reference register、scalar/identity/signを保持する。

strong symmetricはuncontrolled Gaussian、directional中心、head relative fusion、tail同basis共有、15の合法half-sweep step境界融合、scalar/identity位相集約。Referenceは内部14境界融合、global cutの1境界融合損失、追加8 qubits・16 logical CSWAP・χ/readoutをfull wrapperへ含める。弱い対照、短gadget反復、事後cut変更、別回路の節約値加算なし。

各pathのraw correctnessで独立CAR、whole積/AB=V、actual A†、unitarity/vacuum ray/χ、scalar/identity/sign、strong各branch、complex coherence、X/Y誘導observable・global phaseを照合し、**同path raw PASS後だけ4 wrapper compileへ進む**。失敗pathのcomplete-case平均を作らない。

P0：Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation_degree1.0/qubits_initially_zero=false/topology-free/位相保持。component abs1e−10、observable数値Frobenius/signal2e−10、rtol0、phase fitなし。compile後CAR/raw/paired X/Y/χ/reference leakage/宣言位相を同規約で検査。全256 system入力列と全出力振幅を使い、17-qubit全Operatorを構成しない。長q不一致を閾値緩和でPASSにしない。RZ/CX count・各depth・total depth・size・peak qubitsを別々に記録する。

## 資源・一回実行・公開・STOP

実空きCPU/RAM・適用quotaを確認し、独立path/compile並列、raw PASS後pipeline、CAR/head/atom cache、高速化部品を再利用する。初期memory admissionは保存C6bの完了済みraw/Reference/direct工程別peakを回路長比2で拡大した見積り。配列次元は一定であり、この拡大は上界・性能予測・資源上限ではない。実空きmemoryと照合して投入し、CPU余りを重いstate-actionに配分する。歴史的worker数を恒常上限にせず、共有OS/venv/scheduler/他ユーザー・他jobを変更しない。正常jobの不要な中断/再起動、実行中SOURCE hot-editなし。

出力 `artifacts/track_d_c7_1_q16/2026-11-07/run01/` はfresh専用、retry/resumeなし。元event/QPY/NPZ/reference配列・conditions/fusion・内部資源/log/cacheはlocalで保全する。公開はSOURCE/PLAN、科学scalar checks、価格/paired統計/共分散/CX残差/凍結モデル診断、seed/event hash索引、STOP。raw payload・内部絶対path・CPU/RAM/runtime/thread観測・資格情報を追加公開しない。

SOURCE/input/law/endpoint/seed/event hash、cohort混入、順序/cut、AB/actual inverse/vacuumχ/scalar/identity、raw/compiled correctness、strong公平性、P0/precision/scope、証拠の破損/欠落、未許可call、非干渉を維持できない状態では依存科学処理をSTOPしてpartial保全。無断retry/resume・閾値/seed/cut/model変更なし。

**モデル幅外やCX候補反例だけなら有効な結果として保存し、correctness/契約が成立する範囲で予定評価を完了する。** 不利pathを除外せず、モデルを修正して同じ標本で再評価しない。

B4-2 `07d18bfe1d5145d6f8de026c0d994d5cb1c783c8` のideal条件付き14738は参照のみ。B5 `0a8b953aea1be20b110784960763d7f36ebccac8` のlaw/joint-law/全path全round実装誤差・actual共通N不足は残る。q2/4/8追加calibration、n16超のsampling、q≥32、新fit/ground/DFfit、N/actual成功保証/全13-round費用、routing/T合成/noise、詳細科学レビューは禁止。

完了または不成立時にSOURCE/input前後照合、全予定IDと欠損/失敗、partialを含む結果とSTOPを保存。専用branchへ公開し、remote SHA/tree/公開bytes照合後**mandatory STOP**。次担当ユーザー／GPTがq32以降の構造モデル/追加検証/校正の必要性を判断し、Codexは自動進行しない。
