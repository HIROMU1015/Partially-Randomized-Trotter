# Full PF・Truncated B0I p7 shortlist：不足48 meanと全13round packet

2026-11-15 JST、担当Codex A（主agentのみ）。**予定上限48 candidate-roundを48件実行し、48件PASS、FAIL/UNKNOWN 0件。保存S2 mean 30件は再計算せず再利用し、6候補それぞれ全13roundを完成した。Bの既存intakeも6/6 PASS。** MSE/risk/N/価格/G・rankingは実施していない。

[全体実行結果](run01/RESULT.json)／[78round表](ROUND_SUMMARY.csv)／[6packet索引](PACKET_INDEX.json)／[B受入](B_INTERFACE_CHECK.json)／[SOURCE](SOURCE_INDEX.json)／[B引渡し](HANDOFF_B.md)／[C引渡し](HANDOFF_C.md)。科学SOURCEは **fb9a1878b677b7ee116b52e804e5a01cdaa96004**、raw packet carrierは **06a15ee45beb177c018f6260ea4a5ac8cf66cfe6**。

## 1. 引き継いだ条件と証拠

A217eab647のstagingとc212cee85の710 record auditを引き継いだ。過去の候補一覧・packet化を調査し直していない。保存Linear H4/1.00Å/STO-3G、tol-only1e−8 DF10、sector(2,2)/36、保存Full DF-H target、同H/state/snapshotを維持する。h=.1、M12、元prior/decoder/誤差policy、P=0。値の正本は既存baseline manifest c5466946とcanonical configuration。

- H hash：`c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0`
- state hash：`9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b`
- snapshot：`12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96`
- Hamiltonian constant c=2.29310124732 Ha、B0I p7 tail identity=4.5302064712666724e−5 Ha。

Full PFはcを総Tに一回。B0Iはcとp7 tail identityを元kernelのphase因子でそれぞれ総Tに一回適用し、合算scalarと代数的に同じ規則を保持した。保存weighted-Hermitian completion、signed weights、fragment順、stateを変更していない。新H生成/SCF/DF/state solve/固有系/reference計算は0。

## 2. §10.5：実行前の重複照合

記録：[preflight/DUPLICATE_CHECK.json](preflight/DUPLICATE_CHECK.json)／[固定48件・保存30件・10step](preflight/PLAN.json)。元710保存record auditのSOURCE・物理ID・actual幅・gate証拠を再利用し、実行までの関連公開HEAD差分・同family実行jobを追加確認した。

| 対象 | 照合したHEAD | 判断 |
|---|---|---|
| A作業branch | 217eab647 | staging正本と一致、clean |
| B total/intake | 9c0e12d87 | 指定SOURCEと一致 |
| C最新候補対応 | 0b4b2d941 | 855573064から候補対応表/metadata追加だけ、新mean/数値codeなし |
| T2 | 5fb2c32e3 | 指定SOURCEと一致 |
| A旧signal/Full PF | 714247619 | 前段SOURCEと一致 |
| D | acea1118a | 保存価格binding資料、今回meanの追加なし |

確認範囲で新しい受入可能な同等meanは0件、同じmeanの進行中jobは0件。したがって新48件、保存30件の区分は変わらなかった。S2Lの4件は旧meanがあるが現行step/power/norm/reference pair証拠が不足するため、新しいdirect計算で置き換えた。旧判定・rawは保存したままで、旧S2L結果そのものを昇格させていない。

m0のFull S2/B0I S2の2件は、ユーザー指示に従い、同一physical roundの保存値として再利用した。50件へ拡張していない。確認できない別serverの非公開runまで「未実施」と断定していない。調査範囲と元資料はpreflightに明記した。

## 3. 実装と数値計算

既存 `I.plan→C.prepare→I.produce_signals→C.execute` を再利用し、provider `baseline_signal_numerics.evaluate` は無変更で実行した。追加は保存stepを供給する最小context、48件のcall範囲/first-failure STOP、6候補の並列caller、B intake確認だけ。

### 保存中間値

Full PFのscan/follow-upとT2に保存された**10個のstep operator**を一回ずつ読み、全48件で共有した。保存CELL_SCOPE/resultとcanonicalのformula/prefix/actualΔ/signed scheduleを照合し、固有値対角化・Schur・expm・step再構築は0。元stepのq乗、state actionと内積、既存数値gateが今回の科学callである。

actual幅は表示shortlist値ではなく各roundのcanonical値を使用。たとえばS2・16h/7のm0/m1/m2/m3/m4はactual幅 .1/.2/.2/.2/(16h/7)、q=1/1/2/4/7となる。Y4のearly roundも元scheduleのまま。exact有理数と表示binary64を混同せず、[ROUND_SUMMARY](ROUND_SUMMARY.csv)に両者を記録した。

### 保存30件とSOURCE統一

stagingの保存30 meanを `C.execute(..., saved=records)`へ渡した。raw/corrected/reference/normalization/数値gateは元値のままで、再計算していない。X/Yは同じ複素meanの実部/虚部として扱い、96 meanへ二重計上しない。

packetの単一SOURCEは今回の統合実行SOURCEに統一したが、元のproducer/数値SOURCE、元cell/round/file、scalar規則は `provider_binding.input_source`に保持した。**統合packetのSOURCEと元保存値のproducerは別の役割**である。元ファイルを書き換えたり、過去の結果を今回新規に計算したと呼んだりしていない。全30件のlocatorは [SAVED_MEAN_SOURCES](run01/SAVED_MEAN_SOURCES.json)。

### 並列化・資源

64 CPU affinity、観測MemAvailable約512GBの環境で6候補を6workerで並列実行し、科学batchは14.2922秒。shared state/stepはread-only、候補/round出力は分離、frozen関数importは開始前に初期化した。同cell二重投入0、sampling0。失敗時は全候補の新call投入を止め、開始済み結果を保全する設計。今回失敗・retry・backend変更・閾値緩和・再正規化は0。

## 4. 6候補の取得結果

| 候補 | configuration ID先頭 | 新規PASS | 保存再利用 | 完成round | B intake |
|---|---|---:|---:|---:|---|
| Full PF S2・16h/7 | 11cf8ca6 | 9/9 | 4 | 13/13 | PASS |
| Full PF Y4・16h/3 | a799697f | 8/8 | 5 | 13/13 | PASS |
| Full PF M5・16h | 70f7b317 | 7/7 | 6 | 13/13 | PASS |
| B0I p7 S2・16h/7 | a6e55777 | 9/9 | 4 | 13/13 | PASS |
| B0I p7 Y4・16h/3 | defc5e88 | 8/8 | 5 | 13/13 | PASS |
| B0I p7 M5・16h | 424eecbf | 7/7 | 6 | 13/13 | PASS |
| 合計 | | **48/48** | **30** | **78/78** | **6/6** |

不足・未起動・FAIL/UNKNOWNは0件。各複素mean、physical/canonical ID、actual幅/q、SOURCE・gateはraw/packet/CSVに保存。決定論PFの全78件でraw=corrected、B=1/logB=0/attenuation=1。旧保存m12は再計算していない。

## 5. 数値gateと技術テスト

既存step/power unitarity・norm gate1e−9、reference pair1e−10を維持。新48件での最大観測値：

| 診断 | 最大保存値 | 対象 |
|---|---:|---|
| step unitarity残差 | 3.384861750989796e−14 | Full PF M5/m5以降 |
| power unitarity残差 | 5.810744870765344e−12 | B0I Y4/m11 |
| norm誤差 | 3.33510996597397e−13 | Full PF Y4/m11 |
| reference pair差 | 4.767096847331804e−12 | m11 |

nonfinite・source/input不一致・numerical failure0。source freeze後に元input bytesの一致も確認。Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1。gate PASSはbinary64実装の局所診断で、rigorousなenergy RMSEやground-state資格を意味しない。

新人工6、保存staging回帰9、B既存人工13の**28件PASS**。[TESTS](TESTS.md)。BコードはA所有の一時detached checkoutで無変更実行し、B/C/Dのactive branch/worktree/process・共有venvを操作していない。作者による検査であり独立レビューではない。

## 6. Bへの技術的受入

raw packet commit06a15ee45を固定してから、B9c0e12d87の `T.create→T.bind_signal` を6候補に実行し、すべてPASS。全13round、canonical/physical ID、single-source provenance、provider binding、law、normalizationを受理し、mean値の保持を確認した。B code変更0、risk/DP/MSE/N solver呼出し0。

B total-viewは元configurationを別名で破壊的に変更せず、元候補への明示的bridge IDを生成する。元configuration IDとB view IDの対応・packet/configuration locatorは [PACKET_INDEX](PACKET_INDEX.json) と [B_INTERFACE_CHECK](B_INTERFACE_CHECK.json)。**intake PASSは信号の技術的受入であり、必要N・精度保証ではない。**

## 7. 不足資格とロードマップ影響（§10.6）

解消した課題は、価格shortlist6候補の信号欠測44件とS2Lの現行direct証拠不足4件、全13roundのsingle-source packet不足である。精度側と価格側を同じ物理候補で接続できる信号入力が揃った。

残るのは、Bの採択された推定器/誤差契約の下でのMSE/risk・必要N、state/ground・law・全RMSEの科学的資格、C/Dの同候補・全roundの資格あるE[C]、比較policyとG会計。signal局所biasやH_eff shift、1shot価格からwinnerを決めていない。N/MSE/十分N/global RMSE/C/G/winnerは未評価/null。

統合担当がロードマップの「Full PF全13round信号の照合/不足」「Truncated shortlistは価格と信号候補が不一致」「48 meanの取得待ち」を本成果へ更新し、依存関係を **B packet受入済み→別承認のMSE/N→同候補E[C]とのG接続**へ変更できる。ロードマップ本体はAが直接編集していない。

次：ユーザーがBのMSE/N・C/D価格等の次の科学scopeを判断・承認し、各担当がその範囲で進める。Aの今回の48件取得とpacket完成は終了。成果をcommit/pushしremote SHA/tree/主要ファイル照合後、**mandatory STOP**。追加候補/round・新PF幅・RPE/N・compile・Gへ自動進行しない。
