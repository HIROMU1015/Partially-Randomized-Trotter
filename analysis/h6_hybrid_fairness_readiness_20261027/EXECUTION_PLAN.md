# H6 Hybrid公平性補完：Phase Bの非実行計画

2026-10-27。**Phase Aのみ完了。Phase Bは未承認・未実行。** 本文は実行CLIや承認の証明ではない。
固定候補は[FIXED_CANDIDATES.csv](FIXED_CANDIDATES.csv)の4件に限る。追加prefix/q/r/K、H2/H4、純RTE再拡張、旧成功cohortの再取得は対象外。

## 科学条件・入力・実装

Linear H6/1.00 Å/STO-3G/中性singlet、保存918非identity Pauli項＋constant、同じraw ray/full-H reference/exact norm、保存abs係数降順・label tie-break。S2/T=.8、finite paired Taylor RTE、central-rotation、full cosine/+Im sine、P=0。
Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、追加topology/routingなし。主会計は仮想ρ=0、ε_sig=.05/.01/.005/.001/.0005、α_axis=.025、`G=N_cos*C_cos+N_sin*C_sin`。実ρ・母平均tailは未保証のまま。
入力ref・SHAと再利用codeは[SOURCE_INPUTS.json](SOURCE_INPUTS.json)。inputは元Git blobをそのまま復元し、`bind_saved_inputs`で元bytesを確認する。新しいHamiltonian/state/reference生成・再正規化を行わない。
候補configは固定4件専用adapter、lawは元`h4_pauli_pr_search.law_descriptor`のprivate抽出でvalidatorだけ接続する。元RTE kernel、tight L2、D2 control、wrapper、compiler、shot会計は変更しない。

将来出力は`artifacts/h6_hybrid_fairness_extension/2026-10-27/`、cacheもその配下の新namespace。旧実行directory・grant・marker・STOPは再開しない。Phase Aの無効計画とSOURCEを上書きせず、新しい実行SOURCE/runtime/input・batch条件を固定する。

## B0：承認後、sampling前に固定するもの

ユーザーのPhase B実行許可を得てから、保存統計を使う数値設計と候補固有のlaw/meanを進める。同じ許可範囲内の通常段階で再承認を挟まない。

1. 元入力bytes・旧成功価格・STOPを確認する。候補ID、SOURCE、runtime、精度、科学call上限、停止条件を固定する。
2. 新lawを各候補1回構築・保存し、signal/価格両方へ再利用する。B・CDF-grid mass・角・保持/tailを旧r8や純RTEから代用しない。
3. **tight L2：scalar128bit/vector中心binary64（53bit）、各候補最大1状態作用、計最大4状態作用。** 同一作用でcosine/sineを出力する。reference作用/新生成0。精度未確定を残し、自動昇精度・成功計算の再実行をしない。
4. 元のsigned bias/u、B一回、exact norm、1% numerical guardと任意精度整数shot十分条件を適用し、各候補×5精度をELIGIBLE/BIAS_INELIGIBLE/NUMERICALLY_UNDETERMINEDへ分ける。技術例外はTECHNICAL_FAILUREとして別に保存する。N/G=nullを0/∞へ変えない。
5. 以下の標本設計を数値的に評価して、n_E・n_E上限・n_Cの決定規則/下限/上限・総trajectory/compile上限を**最初の探索sampling前**に固定する。これらは現在すべて`NOT_FIXED`。未充足のままsamplingを開始しない。

科学callの固定部分は新law最大4、mean状態作用最大4、reference/state/H生成0。価格標本の数値上限はPhase Aで計算しないため、まだ完全な実行契約ではない。

## 標本設計の方法（今回数値代入なし）

旧価格標本n48は基準資料であり、新nの既定値ではない。[SAVED_COMPARATOR_STATISTICS.csv](SAVED_COMPARATOR_STATISTICS.csv)は既存2 Hybrid anchorと純RTE最良の保存G/SE等の転記15行。旧Hybridのmetadataには両軸SEとpaired価格配列があるが、分散/共分散の別fieldはない。**今回は配列から分散・共分散を再計算していない。** その復元・再会計もPhase Bの許可範囲に含める必要がある。

候補j・精度eのtrajectory費用を `Z_je=N_cos,je*C_cos,j+N_sin,je*C_sin,j` とする。
必要な標本分散は `v_je=N_cos,je²*v_cos,j+N_sin,je²*v_sin,j+2*N_cos,je*N_sin,je*cov_j`。両軸を独立としない。
同じ価格標本を5精度で共有し、精度ごとに新trajectoryを取り直さない。

- **探索n_E**：承認後に旧の保存paired価格と新lawのevent確率を調べ、既存SE・希少event・検出したいG差を踏まえた共通初期n_Eと上限n_E,maxを数値固定する。r比例の価格/分散proxyを実測値としない。未知候補の事前分散は参考に留め、探索標本で確認設計を更新できる形にする。Phase Aではn_Eを算出しない。
- **確認対象**：5精度のうち少なくとも1精度でjoint ELIGIBLEとなり探索価格を得た新候補を、原則すべて独立確認する（最大4）。これにより、費用が高かった候補も近傍/境界の対照になる。元のr8/RTE成功価格は再取得しない。
- **nominee**：探索内の適格・有限G最小を精度ごとに選ぶ規則とtie処理を、探索sampling前に固定する。IDは探索終了後・確認前に凍結する。確認対象全体の観測minとは別列に保持する。
- **目標精度案**：既存研究と同じ、名目95%半幅が参照Gの1%以内を設計の目安にする。これは母平均tail・逐次/多重選抜を含む厳密な保証ではない。最終の比較は両側SEを保存し、記述的2 combined-SEで近接ならINCONCLUSIVEを許す。
- **旧比較側SE floor**：独立な新価格と旧価格の差の分散は `v_new/n_C+SE_old²`。旧SEは固定され、新候補だけ増やしても消えない。例えば目標差半幅をD、名目係数をcとすれば、`n_C >= v_new/(D²/c²−SE_old²)` の分母が正の場合だけ有限設計を得る。分母非正なら、旧cohortを無断でtop-upせず、その目標は未達として残す。今回はこの式へ保存数値を代入しない。
- **確認n_C**：新探索で得たpaired vを使い、固定した相対半幅目標・各精度・旧r8およびRTEの比較側floorを考慮して必要n_Cを求め、事前に定めた下限/上限へ適用する。名目係数cの定義、整数丸め、SE=0/欠測/非有限時の処理も探索前に明文化する。候補ごとの最終n_Cとseed/index範囲は**確認開始前に一度だけ固定**する。
- **確認後top-upなし**：確認価格を見て恣意的に増やさない。今回案は確認を一回の固定n_C blockとし、確認結果で再設計しない。上限で差が解けない場合はINCONCLUSIVE。旧結果の再samplingや追加候補へ自動移行しない。
- **希少event**：全allowed orderをsignal meanへ含めるが、価格は有限標本。未観測order/componentを0寄与とせず、SE=0を母平均不確かさ0としない。希少eventが支配する恐れ・探索分散が不安定・数値nが非現実的なら上限で止め、母平均tail未保証を明記する。新しいtail証明は行わない。

最大価格取得量は記号的に `Σ_j(n_E,j+n_C,j) ≤ 4*n_E,max+4*n_C,max` trajectory、両軸compileはその2倍。数値capは承認後の設計で、sampling前に固定する。science retry/旧STOP解除の許可はこの計画から生じない。

## Seed・実行・比較・終了

元H6のSeedSequence address `[20261025,6,L_D,q,r,K,cohort_tag,index]`（探索101、確認202）を維持する。4候補はいずれも旧Hybridにないrを持つ。Phase Aではaddress引数をmock検査しただけで、実H6 seed一覧の生成・RNG replayはしていない。Phase Bのsampling前に計画addressとseedの重複を確認する。数学的なPRNG独立性の証明としない。

独立候補/trajectoryを元free-CPU dispatchとfresh native workerで並列化する。共有venv/cache/OS設定・他ユーザーprocessには触れない。law/tail preparationの重複を減らし、QPY/cacheは新namespace内だけで使用する。旧科学workerを止めない。
各trajectoryはbody→D2 central→両axis wrapper→元compiler。event/support/phase、QPY/fingerprint、RZ/CX/depth、native exit/reapを保存し、代表native/compiled correctnessを検査する。元price_job/QPY圧縮・復元・paired処理を優先して再利用する。全compiler形式証明を新必須条件にしない。

一次比較はp78/r16,32,64対旧p78/r8、各新候補対保存RTE q8/r64/K2、p684/r16対旧p684/r8。ε=.05でp684を含める。新集合min、探索nomineeの確認結果、確認後minを分け、5精度の共有cohortを記す。
結果は別表へ保存し、元14/15・旧SOURCE/raw/grant/ledger/marker/STOPを書き換えない。登録上限r64に新最良が残ってもr128を自動追加しない。

入力/SOURCE不一致、重大なcorrectness不整合、実OOM/継続不能、科学call上限、技術的失敗では新結果を保全して停止する。失敗科学callの無断retry、数値政策変更、比較側top-upは禁止。局所的な未確定は方式の不適格/0費用とせず記録する。
Phase Bの固定比較・保存・native回収・公開/remote照合を完了したらmandatory STOP。論文中心claim採択、詳細科学レビュー、研究全体の再結論へ進まない。
