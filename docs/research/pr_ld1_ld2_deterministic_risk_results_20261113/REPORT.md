# 保存H4 PR-LD1/LD2：決定論的RPE riskの初回2診断

2026-11-14 JST。利用者が採択・明示許可したLD1/LD2各1whole-shot vector、計2件を完了した。SOURCE系列名に合わせてdirectoryの20261113は保持した。**追加配分、Monte Carlo、signal再計算、価格・G、最適化は実行していない。**

## 結果

両構成で`VERIFIED_FINITE_BINARY64_SUPPORT`が成立し、条件付きbinary64-decoderモデルMSE上界が得られた。以下は表示用の丸め値。厳密な有理数は[LD1原本](run01/RESULT_LD1.json)／[LD2原本](run01/RESULT_LD2.json)を正本とする。RMSE欄も母集団riskの点推定ではなく、このモデル上界の平方根表示である。

| 構成 | モデルMSE上界 U (Ha²、近似表示) | sqrt(U)表示 (Ha) | P(any unsafe) | P(any zero) |
|---|---:|---:|---:|---:|
| LD1 | 2.398951760583e-1 | 0.4897909513846459 | 2.127156415673e-4 | 9.998365097859e-8 |
| LD2 | 2.396879588138e-1 | 0.48957936926894513 | 2.125319012131e-4 | 9.969347644789e-8 |

- LD1：未解決確率は約`[9.998365097859e-8, 2.127156415673e-4]`、zeroなし全体tie確率の保守的範囲は約`[0, 2.126156579163e-4]`。
- LD2：未解決確率は約`[9.969347644789e-8, 2.125319012131e-4]`、zeroなし全体tie確率の保守的範囲は約`[0, 2.124322077366e-4]`。
- 両構成とも各roundの**reference-oracle tie確率はexact 0**。これは前推定状態に依存する実全RPE tie確率の厳密値ではない。実全体tieの厳密値はUNKNOWNで、上記の範囲のみを与える。
- unsafeは宣言した安全集合外であり、誤branch又は未解決そのものではない。未解決の範囲は`P(any zero) ≤ P(unresolved) ≤ P(any unsafe)`、zero優先則からzeroなしtie上端は`P(any unsafe)−P(any zero)`である。

## 固定された条件と計算

H4 linear 1Å/STO-3G/tol-only DF10、sector(2,2)、元H/state/snapshot、S2・各L_D=1/2、元ΔPF=h=.1（binary64由来のexact値）、K2、M12・元共通κ/prior/decoder/branch/alias/shift0を使用した。targetは保存solver energy `−2.1663874486347634 Ha`を**条件付きproxy**として別入力に採用し、元bundleのproxy=false/ground未資格/mean UNQUALIFIEDは変更していない。

X/Y各軸の全13round vectorは`64,60,56,52,47,43,38,34,30,25,21,17,12`。各構成に別allocation ID/SOURCEを付け、元canonical configuration IDを保った。旧S3のRMSEやN資格は移送していない。

`EXACT_AFFINE_OF_BINARY64_RAW`でrawからp=(1+raw)/2を有理数計算し、保存ideal finite-K lawに条件付くfresh IID shots/axes/roundsとして全countを列挙した。ρ_m=1.0 rad、count上限4225/round、隣接組上限55,043,173/構成。実検査は各2,318,425組で閉包が成立した。二項確率・loss和はexactで、tail切捨て・MC・統計confidence区間は使用していない。

採択した未解決loss familyは`[0,L_U]`、`b=nextafter(pi/float(original_h),+infinity)`から`L_U=max((b−E_proxy)^2,(−b−E_proxy)^2)`。今回のL_Uは約1127.771812556 Ha²。未解決energyを補完せず、単一のcompleted estimatorを新定義していない。

上界は既存式`U=all-good終端寄与 + P(any unsafe) × L_bad`。今回L_bad=L_Uである。all-good終端寄与はLD1約4.714089885985e−7、LD2約4.713567760668e−7 Ha²だが、これは全riskでも条件付きMSEでも純shot誤差でもない。保守的なunsafe loss項を含む上界と区別する。今回の2上界の差でLDの優劣や最適配分を決めない。

## 全roundの確率

以下も丸め表示。0はexact 0である。reference-oracle tieは全26行でexact 0のため表の列を省略した。全体のactual tieが0という主張ではない。

| m | N_X=N_Y | LD1 unsafe | LD2 unsafe | LD1 zero | LD2 zero |
|---:|---:|---:|---:|---:|---:|
| 0 | 64 | 3.530694831587e-17 | 2.500698420303e-18 | 9.293417532041e-33 | 2.533716483693e-36 |
| 1 | 60 | 5.148949647870e-13 | 1.724185671406e-13 | 1.469991628653e-23 | 5.626109774802e-25 |
| 2 | 56 | 6.592001139222e-12 | 4.331212442418e-12 | 6.332079892804e-18 | 3.203427778323e-18 |
| 3 | 52 | 1.357998146919e-15 | 1.384564534055e-15 | 1.291371201023e-28 | 1.363031693665e-28 |
| 4 | 47 | 1.427531688869e-11 | 1.441734120446e-11 | 0 | 0 |
| 5 | 43 | 1.999136302020e-9 | 2.011401979712e-9 | 0 | 0 |
| 6 | 38 | 2.892444597961e-10 | 2.923777451965e-10 | 5.672036603106e-19 | 5.847559936989e-19 |
| 7 | 34 | 1.027033377042e-7 | 1.026195249599e-7 | 8.246215014951e-13 | 8.228898632491e-13 |
| 8 | 30 | 3.716173987004e-7 | 3.712602656398e-7 | 2.974158385061e-12 | 2.968440285161e-12 |
| 9 | 25 | 3.336468018788e-6 | 3.334671523010e-6 | 0 | 0 |
| 10 | 21 | 7.158627450268e-6 | 7.150965001974e-6 | 0 | 0 |
| 11 | 17 | 1.523127398974e-4 | 1.521621570120e-4 | 0 | 0 |
| 12 | 12 | 4.944094861022e-5 | 4.941766315757e-5 | 9.997985219218e-8 | 9.968968511433e-8 |

## UNKNOWNと科学的な限界

- **今回のモデルMSE**：UNKNOWN理由なし。指定domainで全count・good遷移検査・loss条件が成立した。
- **実全体tieの厳密確率**：前状態・全履歴の分布を厳密伝播していないためUNKNOWN。reference-oracle tieやunsafeと同一視しない。
- **元energy taskの資格**：保存solverのground資格、保存state、raw/targetの数値allowance、actual sampler law、理想実数decoderとの差の包含は未取得。exact有理数計算はこれらを認証しない。
- **元Minkowski・十分N/global RMSE**：今回のrawモデル診断から元予算各項へ分離・接続する証拠はなく、精度達成/未達成の正式判定はしていない。上界が大きいだけでは実riskの大きさや不可能性は決まらない。
- **G/winner**：対象外。正式N・価格は供給しておらずnullを保持した。N最小化やLD順位付けも未実行。

元full-H proxyに対するriskにはPF・finite-K・normalization等の影響も含まれ得る。純shot統計誤差と同定せず、AのH_eff由来δE又はδE²を加算・控除していない。元Minkowski契約やPR finite-Kの科学的資格も変更していない。

## SOURCE・実装再利用・確認

- 実行caller/入力を先にcommit固定：`43898ada12e405eacda27b5bccf1d464a0362ed2`。[INPUTS](INPUTS.json)／[EXECUTION](run01/EXECUTION.json)。
- 数値評価器：B `d71721f30c7b7fc13ba0789eefffd6238aca3ebc`の既存D.evaluate。既存K.prepare/count/R3・全vector検査を再利用。src/testsの変更なし。
- A bundle：`91349e703e0d09ef702ccd108745dfaf1acc29f7`。元config a9abf8de、数値SOURCE fefd548e、raw carrier16210932と照合。全IDとround SOURCEは[前回のmetadata](../pr_ld1_ld2_deterministic_application_preparation_20261113/SOURCE_AND_CONDITIONS.json)参照。
- C接続は、既存test helperが抽出する固定5cc7c07eの純metadata関数だけを再利用。C全provider graphを実行した結果ではない。科学演算は実装済みDが実rawに対して行った。
- 独立2taskを2workerで実行。全体約7.165秒、各worker約6.792/6.909秒、peak RSS 74,648/73,864 KiB。共有環境・ライブラリ・thread設定変更なし。
- [VERIFY_RESULTS.py](VERIFY_RESULTS.py)で保存SOURCE/mean/nullとexact質量・MSE恒等式を確認しPASS。確率/decoder/riskは再評価していない。[CHECKS](CHECKS.txt)。

元SOURCE/input/raw/結果/ledger/marker/KEEP/STOPは保全した。再実行・配分追加・上界改良等は今回の2件に含めず、別指示まで行わない。commit/push・remote照合後は**mandatory STOP**。
