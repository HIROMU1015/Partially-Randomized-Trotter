# Exact engine：高速化候補と保証の比較

2026-11-07 JST。**設計案であり、実装・採用・性能実測ではない。** 条件、SOURCE記号E/N/C、保存timingの範囲は [ANALYSIS](EXACT_ENGINE_HOTSPOT_ANALYSIS.md)。次回の手順は [PLAN](EXACT_ENGINE_VALIDATION_PLAN.md)。全候補のwall時間削減率、memory増減の実測値はUNKNOWN。

## 1. 保証区分とcertificateの依存関係

候補は次の三つに分ける。区分1でも新コードの基本correctnessとSOURCE固定は必要である。

| 区分 | 意味 | 旧証拠との関係 |
|---|---|---|
| 1：既存exact経路の再利用 | 検証済み不変metadataや同一exact計算結果を重複利用し、算術自体を変えない | 再利用・cache identityの正しさを確認すれば、数値誤差項を増やさずに済む可能性 |
| 2：数学的同値な新実装 | 疎contraction、整数dyadic、別exact rational等 | 新実装の数学的同値性とcorrectnessが必要。算術誤差0の再確立後に保存局所誤差評価を再利用できる可能性 |
| 3：新しい数値保証 | float/GPU、interval、誤差認証付きmixed precision | 新roundoff・正規化・Born/threshold保証、適用科学条件の確認が必要 |

固定 [ENGINE_ERROR_CERTIFIED](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fdf00cee3c87d804845497afc89ac6a71f3b3f59/artifacts/o2_a1_new_target/2026-11-06/engine_qualification_run01/ENGINE_ERROR_CERTIFIED.json) は、保存H4/1.00 Å/STO-3G/legacy DF12/8 system qubits/全4電子sector/`L_D3,m0/q1,δ.02,r1,K2,atol0,X/Y`、N1 actual Qとselected policy、N2保証state、理想gate・保存DFモデル限定である。

その論理は `b_engine=b_state+4E+2/2^53`、`E≥(1+δ*)^Lmax−1`。各項の意味を保つ。

| 要素 | 再利用の必要条件 | 変更時の追加作業 |
|---|---|---|
| N2 state bound | 同じ保存state bytes、H/sector/ordering/理論基準 | 別stateへ転用不可。旧state solveは再実行しない |
| 局所δ* | 同じcertified binary64 matrixの有理数値と理想exact-angle gate、support対応 | matrix値・角度・gate定義を変えるなら局所認証の見直し |
| Lmax / 一様support | 同じ有限gate集合、順序・path/axis、列長cap、位相意味論 | 融合gate・別path集合へ無断転用しない |
| 適用/Bornの算術誤差0 | exact contractionとexact平方和・比・range検査 | 新exact kernelの同値性、または新roundoff bound |
| 初期・最終正規化 | 初期sqrt(norm)は比で相殺、rounded積の最終normは明示評価 | norm=1省略や無認証再正規化不可 |
| grid項 | exact ceil、同じuniform53-bit測定wordモデル | 近似thresholdなら別bias/誤差評価 |
| SOURCE/runtime lock | 旧kernel・依存・input identityへbindされている | 新SOURCEに別の適合説明と固定が必要 |

**数学的な上界の材料を再利用できることと、旧SOURCEへ発行されたcertificateが新実装を自動認証することは異なる。** `inspect_certified_bundle`は旧profileのSOURCE hashを、`verify_runtime`は依存環境を照合する。旧lockを上書きして新kernelを通さない。旧engineはそのまま残し、新候補を独立module/namespaceで比較する。区分1/2で全前提が維持されれば、局所interval認証を再実行せず保存δ*/E/state/gridを使う説明が可能になり得るが、今は認定していない。

## 2. 実装候補の評価

### O1：exact零に基づく対角・疎contraction（区分2、第一候補）

| 評価項目 | 内容 |
|---|---|
| 改善対象・根拠 | `E.apply_exact`のdense d×d積和。対角/疎gateでも全entryを計算するSOURCE事実 |
| アルゴリズム・効果の機構 | certified matrixから一回だけexact非零entryのrowリストを作り、元j順に積和。対角なら各成分に対角要素のみ作用、exact1なら参照再利用、−1/±iならexact符号交換を候補とする |
| 演算数 | 対角gateのcomplex積をdense `D*d`からDへ、一般疎gateを`(D/d)*nnz(matrix)`へ削減できる構造。wall速度比ではない |
| 現SOURCEへの適合 | sdg/cz/p/rz/crz/crzzは対角、xx_plus_yyは|00>,|11>を保つ2×2混合block、Hはdense。rzzもALLOWED/理想式にはあるが、読んだ保存tableに同名entryは見つからず、実在を前提にしない |
| Exact性・認証 | 同一Fraction値だけで演算。exact零の項だけを除き、tiny非零を維持すれば理論上等価。旧local matrix誤差は同じ対象を評価するが、新contractionは未認証 |
| 再利用・最小差分 | 既存table、`multiply/add`、dense oracleを再利用。独立`apply_exact`候補と疎row前処理を追加し、dense fallbackを残す |
| 検証 | 人工matrix全entry/stateの厳密一致、ordered support/controlled/Y符号、zero/identity/invalid shape、original matrix不変。理想三角関数をoracleにしない |
| リスク | nameだけの誤分類、tiny entry切捨て、qubit順、対角でも必要なphase削除。nameをhintにしてもmatrix構造のexact検査を行う |
| 次の実行範囲 | 別途許可された人工実装・人工性能測定から開始可能。実H4 Born/build/compileは別明示承認が必要 |

保存xx_plus_yy代表行には極小の非零虚部があり、RZ(π付近)にも極小非零実部がある。理想式の「ほぼ実」「ほぼ零」で省略する案にはしない。保存matrixを新しいsin/cos評価で作り直さない。

### O2：ordered support別index計画（区分1、O1に続ける）

| 評価項目 | 内容 |
|---|---|
| 改善対象・根拠 | `E:145–148`のmask/base/indexの反復生成。SOURCE上の反復であり実時間不明 |
| 機構 | `(N,tuple(qubits))`ごとにmask、ordered offsets、有効base/indicesを一回作成。不変tupleを後続gateで再利用 |
| Exact性・認証 | 同じindex列を渡すためFraction算術は不変。全supportのrange/重複検査を前処理で維持 |
| 再利用・最小差分 | 現index式を移したprivate plan作成器＋call-local/engine-localの小さい辞書。公開gate/key/orderingは変えない |
| 検証 | 昇順/逆順/非隣接supportでindex列一致、全出力を一回ずつ更新、N違い/重複/範囲外を棄却 |
| memory・リスク | offsetsだけはO(d)、全block indexは1 supportあたりO(D)、全observed support数SならO(SD)。Python int/tuple overheadも測る。実observed supportの遅延cacheとし全組合せを事前列挙しない |
| 次の実行範囲 | 人工test/benchmarkのみをまず許可対象にする。実科学入力は別承認 |

### O3：認証済みgate descriptorの事前bind（区分1、O1/O2後の候補）

| 評価項目 | 内容 |
|---|---|
| 改善対象・根拠 | `E.probability`のkey→`matrix_record`→hash→table参照。捨てられるFraction pairsの再生成がSOURCEで確認できる |
| 機構 | まずhash-only検証経路で不要pairs生成を省く小変更を検討。次に同一builder/runtimeで生成したgateを最初に完全検証し、immutable descriptorへbindし再利用 |
| Exact性・認証 | stateに渡すcertified matrixを変えない。lookup回数削減だけ。ただしgate名/angleだけを信用して検証削除すると旧identity保証を失う |
| identity条件 | certificate bytes/hash、SOURCE・依存profile、gate class/定義/control/angle hex、raw matrix hash/shape、ordered support、builder/policy versionをbind。gate objectを外部へ公開・再利用変異させない。mutable入力は毎回検証か拒否 |
| 再利用・最小差分 | `gate_key/matrix_record/attach_saved_proof`、既存hashとsigned-zero witnessを維持。旧関数は変更せず別の前処理/descriptorを追加 |
| 検証 | 同じkeyでmatrixを変える、class/definition/control/params/supportの変異、certificate/SOURCE/profile不一致、signed-zero違いをfail-closedに棄却。warm hitでも改変が隠れないこと |
| リスク | TOCTOU、無制限cache、gate object id再利用、数値的zeroとraw signed-zero混同。metadata hashを毎回作る費用自体も未知 |
| 次の実行範囲 | 人工gateだけで最初の検証が可能。実保存gateの再構築・新verification処理を走らせるなら今回外、次回scopeに明記 |

「検証一回」は任意の後続gate objectを同じkeyだけで通す許可ではない。不変descriptor自身を使う設計に変えない場合、matrix検証を残す。

### O4：固定D列・scalar・wrapper・同一trial回路の再利用（区分1）

| 評価項目 | 内容 |
|---|---|
| 改善対象・根拠 | `selected_circuit`が全shotでD列を作り、`SavedPrices.price`が同じpath/軸をもう一度build。cache hitもbuild後に判定 |
| 機構 | 不変のliteral gate descriptor/templateをprivateに保存。同一trialではengineとpriceが同じ検証済みlogical descriptorを使い、価格側はfresh copyへreadoutを追加 |
| Exact性・認証 | 角度のbinary64評価順、scalar/event phase、basis cancellation、gate順・global_phase・capを維持すれば同じ行列積 |
| 再利用・最小差分 | 保存UNIFORM_ERRORのD/scalar literal記録は設計照合用に再利用可能。ただしそれだけでfull builderを置換しない。既存selected builderをoracleにtemplate/cache adapterを追加 |
| 検証 | 人工path全列でgate名/angle hex/ordered support/順序/global_phase/軸/価格readout一致、price側のmutationがengine descriptorへ戻らないこと |
| リスク | shallow copyやmutable Qiskit objectによる汚染、policy/path/axis衝突。compile cacheのactual circuit fingerprintも維持 |
| 次の実行範囲 | 人工builder照合は次回技術検証に限る。実gate/circuit build/compileは別明示承認 |

### O5：不要なlist copyとBorn平方の重複削減（区分2）

| 評価項目 | 内容 |
|---|---|
| 改善対象・根拠 | `out=list(v)`は参照copy、Pの平方はT/Pで重複。SOURCE根拠、時間は不明 |
| 機構 | 全要素上書きが保証できる新規出力buffer、またはblock snapshot付きbuffer再利用。Bornで各成分の平方を一度だけ計算しP/Mへ分配 |
| Exact性・認証 | 同じFractionの状態とP/(P+M)を得れば等価。nonpositive norm/range/threshold検査は維持 |
| 再利用・最小差分 | 現dense関数と`norm_squared/born_probability/require_probability/threshold_count`をoracleとして小関数一個ずつ比較 |
| 検証 | input非変異、output全書込み、blockのread-before-write、zero norm、非単位norm、全ancilla位置、exact p/t一致 |
| リスク | in-place aliasで旧入力を失う、cacheされたprefixを上書き、small成分の省略。効果はcontraction全体より小さい可能性もあるが未測定 |
| 次の実行範囲 | 人工kernel test/benchmarkを別途許可後。実H4 Bornは別承認 |

### O6：同一prefix・path確率のprivate cache（区分1、後順位）

| 評価項目 | 内容 |
|---|---|
| 改善対象・根拠 | state/probability cacheは現Cにない。H/scalar/forward Dの共通prefixや同一path反復はSOURCE上可能だがhit率未測定 |
| 機構 | 同一initial stateから同一gate列prefixのexact stateを不変snapshotとして保存。path cacheならexact p又はthresholdだけをprivateに保持 |
| cache key | 新SOURCE/kernel/環境identity、certificate hash、H/N1/preparation/law/policy、state bytes/hash＋正規化規約、N/ancilla/ordered support、prefixの全gate key＋matrix hash＋**順序**、scalar/identity/event phase。path版はorder/ordered labels/軸/δ/r/K/qもbind。prefix版で軸を省くなら共通prefixの同一性を証明 |
| Exact性・認証 | 同じexact state/pを使い、shotごとのfresh path・measurement wordを通常通り消費すれば数学的targetは同じ。測定bitのcacheは禁止。suffixは入力stateが違えば結果共有不可 |
| 再利用・最小差分 | `raw_state/table`、literal gate列、既存private facade境界。新しいbounded private辞書のみ。初めは全path cacheより単一共通prefixを検討 |
| 検証 | key各項の変更でmiss/STOP、prefix切断位置のoff-by-one、copy変異、p/tの厳密一致、同じword tape/bit、oracle情報がCVへ出ないこと |
| リスク | 大きいFraction stateがworkerごとに増える、低hit率、cross-stage/oracle漏洩、共有cache lock。bit/pの公開を増やさない |
| 次の実行範囲 | 人工入力から。現科学契約のone exact call/trial・call accounting・fresh cache規約とは別途適合確認が必要。旧752/224のprobability/state/cacheは取り込まない |

数学的には等価でも、旧科学runの呼出し回数・fresh cache条件を黙って変更しない。新しい科学契約で採用する候補である。

### O7：dyadic整数表現・共通分母（区分2、第二段階）

| 評価項目 | 内容 |
|---|---|
| 改善対象・根拠 | state/matrixはbinary64由来、加減乗算はdyadic閉包。Fractionの一般gcd/オブジェクト生成が負担候補 |
| 機構 | `integer*2^(-exponent)` 又はblock/state共通2冪分母でcomplex演算。加算はshift、積は整数積、簡約は2因子の除去。Bornの最後だけ一般整数比とceilへ戻す |
| Exact性・認証 | bit落とし・丸め・overflowがなく、保存各entryをexact decodeすれば同一有理数。可変長整数必須 |
| 再利用・最小差分 | `multiply/add/apply_exact`とexact p/tをoracleに、別moduleで表現と局所contractionを実装。matrixをfloatへ往復しない |
| 検証 | 人工dyadic/non-dyadic、large exponent/subnormal由来/zero/符号/相殺、分母制約の検査、全振幅をFractionに戻して等値確認。非dyadic入力は拒否又は宣言済みexact fallback |
| リスク | 最大指数共通化で不要に大きい整数、指数/shiftバグ、cache/snapshot形式増加。一般Fraction APIを置換するより差分が大きい |
| 次の実行範囲 | 人工prototypeと測定を別途許可後。O1/O2後もgcd/object費が残ることを確認してから検討。実科学比較は別承認 |

### O8：GMP/gmpy2又はC/C++ exact kernel（区分2、第二段階）

| 評価項目 | 内容 |
|---|---|
| 改善対象・根拠 | Python演算dispatch/tuple/Fraction生成、大整数演算の実装定数。現内訳はUNKNOWN |
| 機構 | GMPの`mpq`又は専用native loopで同じinteger/rational積和を実行。既にCPython intもnativeなので、native化だけで速いとは断定しない |
| Exact性・認証 | numerator/denominatorのexact変換、canonical form、符号、zero/除算例外が必要。floatへ変換してから戻す経路や有限bit overflowを禁止 |
| 再利用・最小差分 | まず独立gmpy2 complex-pair adapterで既存dense/sparseループと比較。native loopは効果確認後。依存は専用venv/buildだけへ追加 |
| 検証 | 人工general rational/dyadicを元Fractionと全振幅/p/t比較、例外/serialization/入出力ownership、複数process/forkとSTOP挙動を検査 |
| リスク | boundary変換が利得を相殺、canonicalizationは依然必要、ABI/依存再現性、fork/native thread/GIL挙動が未確認 |
| 次の実行範囲 | isolated人工技術検証の別許可が必要。共有venvへinstallしない。実科学runは別承認 |

[GMP公式manual](https://gmplib.org/manual/Rational-Number-Functions)は有理数算術のcanonical入力・結果簡約を規定する。[gmpy2公式mpq仕様](https://gmpy2.readthedocs.io/en/latest/mpq.html)は整数比等のexact構築を説明する。これは機能上の適合候補であり、今回version採択・install・速度評価をしていない。

### O9：NumPy/float/GPU（区分3、初回候補にしない）

| 評価項目 | 内容 |
|---|---|
| 改善対象・機構 | native/vector/parallel contractionでPython loopを削減する仮説。小stateの転送・launch費や大きいexact整数との比較は未測定 |
| Exact性 | 通常のfloatでは同じ有理数結果を保持しない。保存binary64 local matrixが同じでも**state適用**の算術誤差が新規に発生 |
| 追加保証 | gate列のroundoff蓄積、underflow/overflow/FTZ、FMA/reduction順・backend差、normの正下界、Born比誤差、threshold/grid biasをuniform scopeで認証する必要 |
| 再利用・最小差分 | 旧Fraction engineはoracleとして維持。新kernel・数値enclosure・backend/profile記録が必要で、小変更とはいえない |
| 検証 | 近いstatevectorや平均bit一致だけで合格しない。局所・全path構造に適用できる上界と、threshold境界を含む保証が必要 |
| リスク・承認 | 未認証のfloat精度を旧arith0へ代入不可。理論設計/人工検証後、認証scopeと実科学計算を別明示承認。今は実装・採用しない |

### O10：interval / 誤差認証付きmixed precision（区分3）

| 評価項目 | 内容 |
|---|---|
| 改善対象・機構 | 必要な精度でpを包含し、thresholdを確定できる場合だけ低精度結果を使用する仮説。多くのpathで確定する割合はUNKNOWN |
| 数値保証 | 丸め方向を保証したreal/complex区間演算、matrix入力enclosure、全演算伝播、Tの正下界、division/range保証が必要。通常floatの差分から区間を経験的に作らない |
| threshold方式 | `p∈[pL,pU]`を保証し、`ceil(M*pL)==ceil(M*pU)`なら元exact pと同じt。異なるなら精度追加又はexact fallback。任意に中点を選ばない |
| 再利用・最小差分 | 旧exact kernelはfallback/oracle。既存mpmath intervalは**局所gate認証用**で、長いstate/Bornのruntime kernelとして適合済みではない |
| Exact性・認証 | 同一thresholdまで認証できれば同じwordでbitは一致し得るが、state/p自体は区間。旧算術0証明の単純転用ではなく、新しい包含・fallback correctnessが必要 |
| 検証 | exact grid境界・境界の両隣、p=0/1、零/極小norm、区間膨張、相殺、refinement不収束、fallbackのword二重消費0 |
| リスク・承認 | intervalがdense Fractionより重い、ほぼ全件fallback、区間libraryの信頼条件、誤差予算変更。人工設計検証と将来の実科学scopeを別承認。最初には採用しない |

## 3. 数学的kernelと分けるworker scheduling

752-shotの127 workers/128 task groupという境界は保存事実である。`Campaign.group`はまずworkers件を投入し、完了した件数だけ補充し、group全完了を待って戻る。128件目は最初のworkerが空くまで待つ。常に全taskを同時実行しているわけでも、各層を別poolへ分けているわけでもない。

| 論点 | 現SOURCE/記録 | 将来案と条件 |
|---|---|---|
| worker数とtask数 | admissionはaffinity内idle比率のfloorを二窓観測して小さい方。保存では128 allowed/127 workers | 128へ固定・切上げしない。実際の利用可能CPU/memoryとgroup task数を併せて選ぶ。127を新しい普遍上限にも使わない |
| 負荷不均衡 | quota順にcellを並べ、そのcellのtrial順。pathによってgate列/bit長/価格missが違い得る | bitを見ないmetadataに基づく投入順や動的queueを人工mockで比較。負荷予測の効果は未知 |
| 投入順とRNG | 親`prepare_trial`がclaim・path/word tape・journalを作ってからsubmit。workerは再drawしない | 順序変更はdomain stream内のword割当を変え得る。まず旧順でticket/path/wordを固定し、投入だけ並べ替える。STOP時の未投入tailも保全 |
| group境界 | fit/freeze等の依存と全完了barrier | 依存のないタスクだけpipeline化を検討。train/evalの先行条件、早期STOP、未使用の余分なshotを発生させない |
| 起動/終了 | activationで一poolを作りgroup間再利用、最後にdrain/reconcile。forkでbundleを継承 | pool再利用は既存機能。groupごとの再起動削減を新機能と誤認しない。cold起動とwarm定常を別測定 |
| 不変入力共有 | decode済みstate/table/registryをfork継承 | immutable化してcopy-on-writeの増加を抑える候補。Python refcount/allocatorでpage共有が崩れ得る。RSSだけで共有効果を判定しない |
| cache/lock | worker-local LRU＋campaign-private SQLite WAL。path/axis lockが価格側を直列化、bitはその前に保存 | immutable descriptorはprocess-localかfork前共有から。新たな共有state cache/大きいlockは初回に導入しない |
| memory/oversubscription | exact中間stateとcacheがworker別、保存RSSはhigh-water | 起動費・worker memory・共有/私有page・実空き資源・native thread数を次回観測。worker追加が常に速いとはしない |

新SOURCEのmock taskで「128件を127 workers相当に投入」のqueue/word/journal検査は可能だが、今回実行していない。実行中224-shotのpool/manager/worker/cache/STOPへ変更しない。共有OS・scheduler・cgroup・venv・他ユーザーprocessを変更して資源を確保しない。新しい人工的な低いCPU/RAM/worker上限は設けず、実際に使える資源と非干渉に合わせる。

## 4. 優先順位

1. **O1を単独比較し、O2を追加して比較する。** 同じFractionと保存matrix値を維持でき、消す演算がSOURCEで具体的に特定できる。初回はcopy/Born/template/manager変更を混ぜない。
2. **O3、O5、O4を個別に検討する。** 重複処理はあるが、identity保持又は小さい効果の切分けが必要。内訳を見て選ぶ。
3. **O6、O7、O8は次段階。** cache hitとmemory、一般gcd削減やnative化の必要性を人工計測で確かめてから進む。
4. **O9/O10は新しい数値保証を設計する別経路。** 精度保証の負担を理由に現時点では初回候補にしない。

これは科学的意味論、差分の小ささ、明白な計算重複、追加認証負担による**技術的提案順**である。保存shotの再ランキングや実測速度順位ではない。
