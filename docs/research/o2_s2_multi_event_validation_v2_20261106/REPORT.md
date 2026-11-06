# O2-S2 複数event有限table評価 v2：六cell完了

2026-11-06 JST。専用branch：`codex/o2-s2-multi-event-validation-v2-20261106`。

**固定synthetic入力のT主指標では、境界価格IS qB/Bは元law P0/Bを改善しなかった。** qB/Bは加法価格IS qA/Bより約0.00134476%低いが、P0/Bより約0.01296807%高い。同じBernstein十分shot数まで含めると、sampling変更の追加利益は元lawに対して残らない。

科学SOURCEは [4c9f3fb32deef5c50fd98517ccbcd03311f0b059][EXEC]、設計正本は [5eb90e8241e69e85a1ca3e79938a367700c24a48][DESIGN]。前回の[途中結果][V1]とSOURCEを保全し、ユーザーの明示再実行許可を受けて転送修正を適用した。許可証明・grant・承認文保存は追加していない。

## 1. 固定条件と科学範囲

D2 Closed P5／D1保存row 10。3 system qubits、`R=(1/5)Q0+(3/10)Q1+(1/2)Q2`、`Q0=Z0`、`Qi=Vi† Zi Vi`、`V1=R_XX01(π/4)`、`V2=R_XX12(π/4) R_ZZ01(π/4)`、`R_P(θ)=exp(-iθP/2)`。degree5、r=2、τ=5/7、x=5/14、σ=+1、state=`000`、ancilla 1。有限targetは `[P5(-i(5/14)R)]²`、時間順i→j、行列順VjVi。

geometry、basis、DF rank、split L_D、δ窓は非該当。この入力は既知synthetic development入力であり、分子の元H厳密ground、DF/PF halves、全RPE round/energy保証を持たない。G9/G10 x=5/7のIR・価格は使っていない。

ε_axis=1/200、α_axis=49/24000、probability bits160、root bits256、η=ρ=10^-12、primitive error10^-6、interval dps指定100。全cellに同じ旧D2 biasを適用し、D3の別保証/precisionへ変更していない。bias≈0.00009858726477358945、remaining Δ≈0.004901412735226411。六cellは同じaxis条件の代替forecastで、新しい同時familywise実験ではない。

## 2. 再利用、修正、correctness

D1保存入力commit `fc96571881f301435a655f057709df4d6c1678e0`、D2境界SOURCE `28fc3ca17ec51bb7656a498e3eea9b72966eb3fa`、Parent/RA SOURCE `4ccc1832f63cd323b23fc05623f7e0b58cf34f37` を再利用。D1誤差会計のSOURCEは `874ca12d347f8f35e5d8d64f65f24abc4a2df523`。各path・入力hash・再利用moduleの対応は [SOURCE.json][SOURCE] にある。

接続処理は保存bindingのordered pairと価格をD2 kernel／Parentの `reweight_events/event_statistics` へ渡す。元lawは保存P_iの積P0、係数Aij=a_i a_j、weightは実qに対するAij/q。符号・phase・actual saved sequence/adjointを保持し、非可換交換、angle fusion、新synthesisを行わない。価格依存normalizerは全pairを使う既存root enclosure／dyadic規則で取得する。既知ISの適用であり、新しいsamplerや最適化原理ではない。

今回のコード修正は、既存 `integer_io_limit(1_000_000)` の範囲をProcessPool全体へ広げ、明示forkでprivate childへ継承させたことと、出力先v2の分離だけ。整数の文字列変換方針であり、科学的なbit精度や有理数値を変更しない。共有環境・他者processの設定は変更していない。

実行前の人工39件はPASS（既存boundary19、Parent proposal13、pair接続6、巨大Fraction転送1）。登録入力での実行後には、全3 lawの3,969 pathでsupport・正規化・係数=実q×weight、ISの160-bit lawを確認した。pair表とP0 lawはv1と同一bytes、P0のM2/L/N/費用も同一。P0の加法・境界期待価格は旧D2集約値にT/CX/1Qすべて厳密一致した。記録は [result_checks.json][CHECKS]。

## 3. 六cellの十分shot数と量子資源

下表は両axis合計の既知gate費用。controlled保存IRにXのH,H、YのH,Sdg,Hを加え、system状態準備を除外した。wrapperはT/CX追加0、1QはXで2・Yで3。Tは1Qの内数であり、二重加算しない。表示のみ丸め、正本は [RESULT.json][RESULT] の厳密有理数、一覧は [summary.tsv][SUMMARY]。

| Cell | N/axis | T費用 | CX費用 | 1Q費用 | 共通hの係数 |
|---|---:|---:|---:|---:|---:|
| P0/A | 773,966 | 429,626,133.047707 | 23,509,517.974704 | 1,111,501,995.747995 | 1,547,932 |
| P0/B | 773,966 | 427,870,692.302325 | 19,998,636.483940 | 1,105,697,758.230249 | 1,547,932 |
| qA/A | 774,100 | 429,681,472.405874 | 23,477,066.753507 | 1,111,631,897.319490 | 1,548,200 |
| qA/B | 774,100 | 427,931,933.535182 | 19,977,989.012123 | 1,105,844,037.176643 | 1,548,200 |
| qB/A | 774,093 | 429,683,776.356192 | 23,489,240.448050 | 1,111,635,819.290071 | 1,548,186 |
| qB/B | 774,093 | 427,926,178.861741 | 19,974,045.459148 | 1,105,822,928.427993 | 1,548,186 |

未知のancilla準備/readout等の共通費が残る場合、資源別に右端×h_metricを加える。測定時間をgateへ換算せず、hを黙って0にしない。qA/qBは既知T価格で作った固定lawであり、未知hについて最適化済みとはしない。

| Law | M2 | L | B上の期待path T（wrapper追加T=0） |
|---|---:|---:|---:|
| P0 | 1.346090538198331 | 1.160211419612103 | 276.414398243802 |
| qA | 1.346120336876094 | 1.222242887759806 | 276.406106145964 |
| qB | 1.346113624831538 | 1.220755566593518 | 276.404888599781 |

各lawで実finite-bit qから `M2=ΣAij²/qij`、`L=max|Aij/qij|` を求め、共通の
`N=ceil(log_upper(2/α_axis)*(2*M2/Δ²+4*L/(3*Δ)))` を使用した。P0だけに旧保守的Nを流用していない。真のsignal、未知meanを使ったvariance縮小、実測shot数は使っていない。

## 4. 追加利益と境界効果を分けた判断

同じB上で、qB−P0の既知T費用差は **+55,486.559416**、qB−qAは **−5,754.673441**。共通T費h_Tを残すと、それぞれ `+55,486.559416 + 254*h_T`、`−5,754.673441 − 14*h_T` となる。従って固定law・共通非負h_Tの範囲では、qBはP0より悪く、qAより良いという向きは変わらない。

qB/Bの期待path TはP0/Bよりわずかに低いが、M2とrangeが増え、N/axisが127増えた。この増加が価格低下を上回った。qA/BよりはNが7少なく価格も低いため、小さな追加節約がある。それだけでは元lawを含む強い比較集合でのT改善を支持しない。

固定lawごとのA/B比較では、すべてBのT費用が低い。P0では429,626,133.047707→427,870,692.302325である。これは既存境界相殺の効果であり、samplingの新利益へ合算しない。今回のT六cellではP0/Bが最小。CXの既知成分ではqB/BがP0/Bより低いが、主指標を変更して成功扱いにはしない。未知h_CX等が異なれば補助資源の順位も条件付きとなる。

小差でも、保存した実finite-bit law・整数native価格・厳密有理数会計内の差の符号は丸めに依存しない。未知のpath-dependent wrapper価格、別compiler/sequenceや別入力に対する頑健性は未検証で、その範囲へ順位を外挿しない。

## 5. Normalizer、古典setup、メモリ

ideal score和のenclosureはqAで約0.06964375203140004、qBで約0.06978595082521004。両端の厳密値をRESULTへ保存した。表示で同じ値に丸まっても幅0と扱わない。実qはenclosureを使った既存dyadic生成後の値で、理想qへ置き換えていない。T=0、support脱落、precision不適格は発生しなかった。

| 処理 | qA | qB |
|---|---:|---:|
| normalizer certificate wall秒 | 0.016506 | 0.016424 |
| proposal（内部normalizationを含む）wall秒 | 10.814984 | 10.145882 |
| M2/L/N wall秒 | 1.244908 | 1.498688 |

normalizer certificateは同価格の係数をまとめて既存root enclosureを使う診断で、Parent内部normalizationと同じscoreを対象とする。上表の各処理と全体wallを重ねて加算しない。pair表取得0.093702秒、3 law並列部分12.424460秒、記録された全体wall12.655222秒（最終RESULT書き出し前）。親CPU0.599829秒、終了child CPU合計24.956016秒。これらはこの一回のsetupで、独立benchmarkや速度倍率ではない。

CPU affinity32、開始時MemAvailable52,102,868 kBを確認し、IR/suffix cacheを再利用して独立3 lawを3 processで評価した。GPU driverは利用不可で、純粋な有理数CPU workloadにGPUは使っていない。親peak RSS98,424 KiB、終了childの最大peak RSS108,260 KiB。後者は同時総RSSではない。恣意的な資源/wall上限は追加せず、共有環境や他jobを変更していない。

pair表7,948,641 bytes、P0 law7,207,348、qA law3,069,003、qB law3,069,145 bytes。全pair価格とnormalizerを取得する明示table法の費用であり、非列挙DP/samplingの効率を示していない。オンラインのtable走査、bit生成、weight lookup、IR接続等は `NOT_MEASURED`。RNG・大量drawの独立benchmarkは実施していない。量子gateと古典秒・memoryを根拠なく換算/合算しない。

## 6. 保証、主張の範囲、次の判断

これは同一補正target・共通bias/Δ・axis confidenceに対するBernstein**十分shot数forecast**で、最小shot数でも実量子shot実績でもない。量子測定のランダム誤差の実験比較はしておらず、六cell同時familywise保証も新設していない。数値丸めは表示だけで、科学値は厳密有理数と既存outward sqrt/log enclosure。q丸めは実qによるweightで補正し、biasへ二重計上していない。

結論はこの固定synthetic有限tableのT主指標に限定される。既知ISの新原理、最適law、全RA/CTSに対する優位、非列挙samplerの効率、古典費込み全実行時間、分子PR＋RPE全roundの費用改善は示していない。system状態準備除外を理由に、存在しない分子PF/roundを価格0へ置き換えていない。

**次はユーザー／GPTが、このT不改善結果を踏まえ、本S2案を拡大するか判断する段階。** 拡大の前には、強い元lawを上回る機構と、必要なnormalizer/online生成・full-wrapper条件を具体化する必要がある。今回の結果だけを理由に別r/degree/time/precision、regularizer、別主指標、分子入力を自動探索しない。追加の科学評価・重要レビューは別許可を要する。

Codexによる今回の許可範囲は六cell評価・保存・公開照合まで。科学的採択/GO判定は行わず、公開・取得確認でmandatory STOPする。前回v1のSTOP/結果、旧D1/D2/D3/Parent/C1等の固定SOURCE/raw/契約/grant/ledger/marker/STOP、元dirty checkout・main・旧branchを保全した。科学signal matrix、H/state/reference生成、RNG/量子shot、build/compile/synthesis、新optimizer/samplerは実行していない。

## 7. 出力と公開確認

[SOURCE][SOURCE]、[RESULT][RESULT]、[pair table][PAIRS]、[P0 law][P0]、[qA law][QA]、[qB law][QB]、[表示TSV][SUMMARY]、[人工correctness][CORRECT]、[結果接続確認][CHECKS]を保存した。研究概要・検証status・manifest・関連索引・研究ノートへv2完了と主指標不改善を追記し、v1の途中停止履歴は残した。

文書/JSON/公開内容・固定SOURCEの確認後、専用branchへcommit/pushし、remote SHAと公開成果物の取得内容を照合する。外部の独立再現やCI結果とは区別し、公開確認結果は最終チャットに記す。

[EXEC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/4c9f3fb32deef5c50fd98517ccbcd03311f0b059
[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/5eb90e8241e69e85a1ca3e79938a367700c24a48
[V1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8ee0e4c3c72e8d4de50c922189d814c3bb7afa1d/docs/research/o2_s2_multi_event_validation_20261106/REPORT.md
[SOURCE]: ../../../artifacts/o2_s2_multi_event_validation/2026-11-06/v2/SOURCE.json
[RESULT]: ../../../artifacts/o2_s2_multi_event_validation/2026-11-06/v2/RESULT.json
[PAIRS]: ../../../artifacts/o2_s2_multi_event_validation/2026-11-06/v2/pairs.json
[P0]: ../../../artifacts/o2_s2_multi_event_validation/2026-11-06/v2/law_P0.json
[QA]: ../../../artifacts/o2_s2_multi_event_validation/2026-11-06/v2/law_qA.json
[QB]: ../../../artifacts/o2_s2_multi_event_validation/2026-11-06/v2/law_qB.json
[SUMMARY]: ../../../artifacts/o2_s2_multi_event_validation/2026-11-06/v2/summary.tsv
[CORRECT]: ../../../artifacts/o2_s2_multi_event_validation/2026-11-06/v2/correctness.json
[CHECKS]: ../../../artifacts/o2_s2_multi_event_validation/2026-11-06/v2/result_checks.json
