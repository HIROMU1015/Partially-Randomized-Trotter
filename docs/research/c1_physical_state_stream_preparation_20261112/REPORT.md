# C1状態閉包とStreaming構造計数の技術検証（2026-11-12）

**実C1の平均S、さらに平均compiled価格を全event列生成なしに評価できることは、依然として未確認である。** 今回確認したのは、C1の状態・flush依存、単純な状態縮約が成立しない人工反例、元の計数wordを変更しないStreaming consumerである。新9＋既存3＋既存sampler API人工1＝**13テストPASS**。実H4/H6のS・平均価格・保存値再解析は0で、C1の係数/価格資格を変更していない。

実装SOURCE `c1135fdb9a021e85ffabdbf4ef56098ec3ac97e6`、基点`2a882fc9d17ed5e32f5581d0f220514daad0235a`、既存C branch `codex/r-variable-compiled-price-design-20261111`を継続。[実装](../../../src/trottertracks/mean_compiled_price.py)／[新テスト](../../../tests/check_c1_structural_stream.py)／[STATE_AUDIT](STATE_AUDIT.json)／[TESTS](TESTS.txt)／[既存3件](REUSED_TESTS.txt)／[sampler API](SAMPLER_API_TEST.txt)／[HANDOFF](HANDOFF.md)／[次の科学案](SCIENCE_PLAN.json)／[保全確認](CHECKS.json)。

## SOURCEと適用範囲

| 対象 | 固定SOURCEと確認箇所 |
|---|---|
| C1本実行・結果 | `1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7`／公開`2e6a0cdf5a647058cc7afeeb5741bfafa24c974b`、係数凍結`7eeabf0ae482ee610d667e28470692d209f2bb21` |
| H4 counter | `8457c443554af5fadaa3ba36f97f6316ca3c5050:src/trottertracks/h4_df10_price_v1.py`、`Accounting._native/basis/tail/finish`。現在のfileも同blob |
| H6 counter | `ae44283d169b61deace7e1c0d18c4538d4a3fc0c:src/trottertracks/h6_df19_price_v1.py`、同API。NQ=13/control12、H4はNQ9/control8 |
| C1 wordと一括保持 | `r_price_phase1_science.structural`、`Backend.price_requests/acquire`、元`h46_long_time_native.sample_requests/compose`。既存production SOURCEは不変 |
| sampler | `c31ef424175c02379eafebe8fc6c7548c1d9aabe:src/trotterlib/rte.py`の`iter_sample_rte_events/sample_rte_events/_make_event`。現fileと3関数のAST一致を確認 |
| 既存平均法 | `2a882fc9d:mean_compiled_price.MeanKernel/full_wrapper_mean/local_residual_mean/combine_mean`、`native_df_gate_model.expected_rte_occurrence`、旧connected-clusterの`_iid_sequence_form/_sequence_form` |

将来の物理対象はLinear H4/H6・1 Å・STO-3G、保存tol-only DF10/DF19、weighted completion・元fragment順・保存H/state/snapshot、H4 S2/δ.1/p1/K2、H6 m5_best/δ1.6/p7/K2・元11 signed piece/負時間/整数allocation・R≥11。symmetric-directional controlled PR、測定付きfull Hadamard wrapper、P=0、元Qiskit1.3.0/rz,sx,x,cx/opt1/seed17。今回は分子入力・state・raw eventをloadしていない。H6 counterを読むことと、H6分子の価格を計算することを分ける。

## C1が保持する状態

| 保持情報 | SOURCE上の役割・寿命 | 平均計数での問題 |
|---|---|---|
| `pending_basis=(operations,inverse,role)` | `basis()`が全文字列/数値パラメータ/qubit/順序の一致と逆向きを調べる。gateがflush。piece/outerを跨ぐ同一Accountingに残る | frozen basis registryが有限なら候補数を固定できる可能性。ただしbasis IDだけでは不十分で、実到達word/basis-planの証拠が必要 |
| `pending_z[wire]` | native RZを同wireへbinary64で逐次加算し、非RZが触れるwireだけflush。literal `angle!=0`ならfolded RZを1個加算 | W個のregisterという固定memoryと、値の状態数が小さいことは別。異なる非ゼロ角は将来の相殺で区別される |
| tail内`active=(basis_id/hash,operations)` | 同basis applicationを共有。identityで閉じ、合法tailの末尾でも閉じる | tail内の継続情報。transport chunkごとにtailを呼び直すと同じsource wordではない |
| tail内`phase` | eventのatan2 phase、負productのπ、identity rotation等を元順で加算。tail末尾で非ゼロなら相対Pをemit | r_j回の履歴に依存。Pの有無がpending basis flush/Zにも影響。単なるwrapper固定費ではない |
| `global_phase` | loweringやuncontrolled diagonalで加算し、finishで報告 | 現counterのS分岐には読み戻されない。価格observableだけの状態から除ける可能性と、phase付き全report/作用の保存は別。今回省略していない |
| `native/folded` count・depth、feature/role Counter | 履歴の加算・wire到達時刻。Sは最後にfolded RZ数を読む | count/depth値そのものを全Markov状態へ列挙する必要はないが、Python整数の桁数は長さとともに増える。今回source counterを置換していない |
| q/J/allocation・signed blocks | `structural`がhead forward→tail→reverse headを元順に繰返す | 同じouter law/wordを反復できる条件と、状態継続の両方が必要 |
| scalar/wrapper/terminal | scalar `−cT`は全体で一回、sineだけSdg、H/measure、最後にfinish | 内部checkpointでfinishすると将来の相殺を変える。wrapper費は独立に足せるとは限らない |
| compiler残差 | C1はSとは別にθ0+θq q+θL qRを使う | Sの閉包だけでは実compilerの期待残差・外挿・全law資格は得られない |

`tail()`のactive/phaseは合法tail単位で初期化・終端処理される一方、`pending_basis/pending_z`等はAccounting object上で継続する。この二つの寿命を混同しない。全pieceのphaseを一括で足す、chunkごとにtail終端を作る、outerごとにcounterをresetする変更はいずれも採用していない。

binary64値は理論上有限bit集合なので「どんな有限状態表現も数学的に不可能」とは結論しない。しかし、固定register数だけでは小さな遷移行列を作れることにならない。実数的に見れば累積角は履歴依存し、binary64でも到達する異なる値がq/Rとともに増え得る。実H4/H6の合法primitive grammarが毎回有限windowでflushを強制するのか、あるいは価格に必要な同値類へ縮約できるのかは未検証である。

## 確認済み・未確認・成立しない条件

| 分類 | 結論 |
|---|---|
| 確認済み | supplied closed-state modelの既存期待値合成、人工全列挙との一致。今回のlazy consumerは元C1計数wordを直接呼び、人工入力で全result dictionary・gate呼出し列・phase/featureが一致 |
| 確認済み | tailへのgenerator入力、合法tail内のchunk連結、piece/outerを跨ぐ同一counter、terminal一回、signed 11-piece順序、scalar/wrapper一回。下位の既存PCG64 iterator APIもtoy testでtuple版と一致 |
| 未確認 | 実H4/H6での実用的な小さい閉状態・厳密/近似平均S、全体のStreaming入力接続、平均compiled価格・長q/R誤差・全law期待価格保証 |
| 成立しない一般的縮約 | accepted C1 native primitive上では、basisだけ、またはbasis＋Z非ゼロflagだけでは将来価格を決められない。人工prefix .25/.5に同じ−.5を足すとterminal Sは1/0 |
| 成立しないsource変換 | C1 literal countは角度0と2πを同じにしない（人工S 0/1）。物理unitaryの周期同値をそのまま現在のSへ導入できない |
| 成立しないsource変換 | finishを途中に入れる、tail phaseの加算をchunkで再括弧化する操作は一般に元Sを保存しない。人工例で差を確認 |
| 成立しない全体価格の主張 | Streamingの一path SをE[S]やcompiled価格とすること、RTE内部κだけをfull価格とすること、Sへraw κ(C)を重ねること |

これらの反例は**人工native/event metadata**であり、そのprefix/角度組が保存H4/H6のsampling lawで実際に到達するという証拠ではない。したがって、単純縮約の一般的な正しさを否定するが、特定のfrozen物理supportに対する有効な別の閉包を否定したものではない。

## 必要最小限の拡張：元wordへforward-only入力を渡す

`stream_structure(Accounting, blocks, row, scalar, axis, batches, allocation=...)`を既存mean moduleへ追加した。`_ForwardTailBatches`が元`structural`の昇順indexアクセスをone-pass iteratorへ変換する。**旧counter、tail実装、head/tail/wrapper/finish loopをコピー・書換えず、そのまま呼ぶ。** 同じtailは一つのevent iteratorとして最後まで消費し、合法tail終端だけを元sourceに任せる。

allocationに対して不足/過剰event、tail不足/過剰、逆順access・未消費tailは例外となり、priceを返さない。既存MeanKernel/MeanPiece/full_wrapper_mean、local residual、MeanPriceProviderの関数本体は変更していない。B controller/candidate/総費用は追加していない。

返値は`quantity=SINGLE_PATH_STRUCTURAL_PROXY`、元finish出力とconsumerの消費数。`physical_state_closure/mean_structural_price/mean_compiled_price`はnull、qualification=false。実入力を渡すpath計数自体は科学計算なので別の実行許可が必要。今回入力したのは人工fixtureだけである。

consumerはevent列をmaterializeしないが、**upstream producerのmemoryを保証しない**。listや`json.load`の結果を渡せば、その保有memoryは残る。逐次Sは全E=qR eventを生成/読取して処理するため、列挙生成そのものを避ける平均法ではない。

## 既存producerとの接続範囲

既存`rte.iter_sample_rte_events`はper-eventにorder→componentをPCG64でdrawしてyieldし、`sample_rte_events`はそのtuple化wrapperだった。3関数（iterator/tuple/_make_event）の本体はC1 provider SOURCE c31ef424と現treeでAST一致。`df_rte_circuit`にも`iter_sample_events`が既存であり、新samplerを実装する必要はない。旧toy testを再利用してtuple/iteratorの順・sample一致を確認した。実分子のsamplingは実施していない。

一方、`make_df_partial_s2_step_request`とoccurrence dataclassは全eventsを持ち、`__post_init__`でlen/normalizer/angle/order/phase/probability/controlled/ancilla等を検査する。`h46_long_time_native.sample_requests`はqJ requestをlist化、C1 backendはto_dict列とraw JSONを再度一括生成し、compile側はQC/fingerprint/DAGを全体で構築する。consumerだけ差し替えてこのpipeline全体がstreamingになったとは扱わない。

将来S専用leafへ既存iteratorをつなぐ場合も、元piece順・per-piece seed/trial binding、signed time/τ、basis registry/operation照合、イベント確率とnormalization、application順・countの検査を維持する。旧dataclassのlen検査を単に削除しない。保存rawのincremental readerも元JSON順とbinary64値を変えず、tail内は一つのiteratorとして供給する必要がある。今回その実分子用leaf/readerは実装・実行していない。

## 三方式の比較と再利用

| 方式 | SOURCEへの適合・処理量 | 正確性・未解決部分 |
|---|---|---|
| 閉状態期待値 | MeanKernelを再利用。d_sが固定ならO(d_s³[Σlog r_j＋log q＋J])、保持O(Jd_s²)等。全event生成を避けられる候補 | 実C1の状態同値類/遷移・報酬、piece/outerのlaw継続、zero/phase/終端の証拠が未取得。compiled残差は別 |
| Streaming S | 同じAccounting/旧word、O(qR＋qJ×head量)。consumerはO(W＋現在のbasis/event＋固定head metadata)、整数counterのbit長はlog Eに依存 | これはpath Sのsource会計。人工入力の一致確認済み。全pipelineのmemory改善・E[S]・価格精度の資格は別 |
| 局所期待値/connected-cluster | 既存 `_iid_sequence_form`、条件付きκ/window多重度を再利用。K2/cluster≤3はpiece当たり最大14 order項で、component/basis校正費は別 | 内部RTE、PF–RTE、piece、outer、wrapper、compiler残差を区別。閉包/打切り/境界systematicを未取得のまま保つ |

旧connected-clusterの校正/予測/holdout、SOURCEのdirty provenance、H4 DF12/H5 DF9・uncontrolled中央RTE・L4/6/8等の範囲は[前段調査](../mean_compiled_price_preparation_20261112/REPORT.md)を再利用し、同じ数値検証を繰り返していない。現C1 full-wrapperの係数へ移植しない。Sを基準とする新残差は、同じ局所定義でのpaired C−Sと全境界を対象にし、C1 affineとlocal残差の両方を足さない。新たな科学近似は採用していない。

## 人工検証と保全

新9件は、H4/H6用counterへの人工イベント/非空head、両軸、H6人工11piece/負角、piece・outer継続、terminal/axis/scalarの正確な呼出し列、chunk phase反例、不足/過剰入力、状態縮約/周期角反例を確認した。量子operator・実compiled回路の検証ではない。

consumerのlive-object試験では、人工E=2048と8192で新規event objectの同時生存数がいずれも3以下、終了後0だった。len/random accessを拒否するproducerで検証し、元S resultを得た。これは入力producer/consumerの当該人工条件の保持構造で、peak RSSや実分子のmemory量・高速化率は測定していない。

既存3件はclosed-state人工全列挙、B run_missingでのmissing/再呼出し抑止、二重計上/欠測管理。さらに既存sampler APIのtoy1件を再利用した。全13件PASS、作者自身の技術確認で独立科学レビューではない。A/Bのprocess/worktree/cache/未公開変更や分子rawを操作していない。C1/C2、前段mean成果、旧SOURCE・input/raw/result/cohort/grant/ledger/marker/KEEP/STOPを保全した。

## 次の具体的な科学候補

最小の次案を**S0a：保存4セルのnormal path index0、X/Y計8件のS parity診断**とした。H4 q8/R8・q16/R8、H6 q1/R32・q2/R32、計4保存path、stream Sを新たに8件評価して保存Sと比較する案（旧counterの新再計数0）、新sampling/compile/fit/N/Gは0。ただしSの新評価・保存値の比較は科学再解析なので、今回行っていない。新mean方式のdevelopment診断として扱い、C1の原roleを保持し、fresh holdoutとは呼ばない。[SCIENCE_PLAN](SCIENCE_PLAN.json)にcandidate ID/入力selector/不足実装を固定した。既存S0の32 path案へ自動拡張しない。

実用的な閉状態には、frozen support/basis/angle/phase/flush footprintと合法event grammarに対する状態遷移・reward同値性、元のsampling law、binary64/source-orderへの対応が必要。有限pathでのstate観測やS一致は、そのまま全supportの閉包証明ではない。平均Sを近似する場合は、既知developmentデータと別のfresh検証を用意し、局所/boundary打切りと統計誤差を分ける。

新compiled価格が必要なのは、Sのsource一致ではなくC−Sの新domain/新法則/長q-R・wrapper/controlled条件を検証する場合。同一条件の保存価格/残差が目的を満たす部分を再compileしない。Bの実RMSE/M/κ physical候補・必要価格精度が定まってから、q-only/R-only/jointの有限点、校正/独立cohort、最大call数を選び直す。旧C2の40/16、前段のfresh32案は自動実行・最適案扱いしない。

次：**ユーザー**がS0a等の具体的な科学範囲を判断する。**Codex C**は必要な保存reader/検査leafとSOURCEを固定し、明示許可範囲だけを実行する。**Codex B**は最終統合を担当し、Cからは将来の価格/資格/欠測を受ける。今回S・価格・N/GをBへ新供給したことにはしない。commit/push・remote SHA/主要成果物照合後mandatory STOP。
