# O2-S2：最小科学検証の実行仕様案

2026-11-06 JST。**設計のみ。未決定事項のある提案であり、完成した実行契約・実行許可ではない。** 入力と固定SOURCEは[対象選定](S2_MULTI_EVENT_TARGET.md)、全体報告は[REPORT](REPORT.md)。既知ISをwhole-pathへ適用すること自体の新規性は主張しない。

## 1. 一件に限定する科学的な問い

D2 Closed P5の `R=Σp_iQ_i, p=(1/5,3/10,1/2), σ=+1, τ=5/7, r=2, x=5/14, degree=5, state=000` を用いる。有限targetは `[P5(-i(5/14)R)]²`、実装meanは保存係数と保存native sequenceによる二stepの積。geometry/basis/DF rank/`L_D`/δ窓は非該当。

問うのは、**同じ実装meanと誤差保証を維持し、既存literal境界相殺の非加法価格を使ってproposalを変えると、共通Bernstein十分shot数×期待full試行価格が強い対照より減るか**である。旧D2はlaw・N不変で価格だけを変えた研究なので、この新joint-law評価の許可を含まない。

次回の最小科学評価として提案する範囲は有限table上の決定論的な価格・law・moment・forecast会計。RNG、量子shot、信号matrix、回路build/compile、新synthesisは不要。sampling生成器は既存方式との接続を設計するが、online実測を追加するかは別途選択する。

## 2. 入力とSOURCEの再利用

| 資産 | 固定版／再利用点 | そのまま使わない部分 |
|---|---|---|
| D1 events・cache・resource map | `fc96571881f301435a655f057709df4d6c1678e0`、[対象選定のpointer](S2_MULTI_EVENT_TARGET.md#3-推奨入力と固定pointer)。row 10の正しいx、native IR/価格/係数/実proposal/weightとstrict cache | G9 x=5/7のIR/価格への置換、未保存angleの新取得 |
| D2 `boundary.py` | `28fc3ca17ec51bb7656a498e3eea9b72966eb3fa`、[固定SOURCE][BOUNDARY] の `gate/inverse/cancellation_depth/native_price`、suffix価格cacheという既存手法 | `aggregate_pairs` の出力は集約値のみ。これを個別価格tableとみなさない |
| Parent `parent_allocation.py` | `4ccc1832f63cd323b23fc05623f7e0b58cf34f37`、[固定SOURCE][PARENT] の `digital_proposal/reweight_events/event_statistics` | allocation探索や既存Parent研究の再開 |
| Return Aggregation有限bit規則 | 同Parent SOURCEの [return_aggregation.py][RA]、`root_interval/dyadic_distribution/dyadic_index` | Closedの群DPを価格依存pair-normalizerと同一視しない |
| Bernstein会計 | 同SOURCE [Parent design][DESIGN] のscalar式、[g8_bounds.py][BOUNDS] の `log_upper/ceil` | `ledger` 全体はr=1 Taylor/biasを使うためr=2へそのまま呼ばない |
| D1誤差会計 | `874ca12d347f8f35e5d8d64f65f24abc4a2df523`、[accounting.py][D1A]・D1 row 10の既存bias/remaining | `whole_budget/whole_cost` の独立law・加法価格の式を新joint law/境界価格へ流用しない |
| wrapper | preflight固定版の [Hadamard wrapper][WRAP] のgate配置を静的参照 | builderを呼ばない。主RPEの±1 Hoeffdingをweighted測定へ直接転用しない |

既存[boundary tests][TEST]にはphase/adjoint、cascade、wire/key、no commutation、unconditional zero-fill等がある。今回実行していない。後日の必要な人工検査では、この適合範囲を再利用し、既存科学runnerや保護されたone-shotをtest経由で起動しない。

必要な拡張案は一つの薄いtable接続だけ：保存bindingの順序付きpair IDに、係数・元確率・加法価格・境界価格を結び、既存IS/統計関数へ渡す。既存kernelを再実装せず、新しいsampler、optimizer、normalizer算法、pipeline管理層は作らない。**今回はその接続も実装しない。**

## 3. Joint lawと同meanの条件

step iの保存recordから、`coefficient`=\(a_i>0\)、`proposal`=\(P_i>0\)、`weight`=\(a_i/P_i\) を取得する。符号・複素phaseは保存 \(V_i\) に含める。このClosed P5のcomplete supportを使い、元の二stepは独立なので

\[
P_0(i,j)=P_iP_j,\quad A_{ij}=a_i a_j,\quad W_0=w_iw_j.
\]

\(P_0\) は保存された実digital lawそのもの。`digital_proposal(mode='canonical')` で作り直すと有限lawが変わり得るので、元lawの代用にはしない。

候補qはpair全体に対するjoint tableであり、一般にstepごとへ因数分解できない。時間順i→jに対応する測定 \(Y_b(i,j)\in\{-1,1\}\) に対し

\[
X_b^{(q)}=\frac{P_0(i,j)}{q_{ij}}W_0(i,j)Y_b(i,j)
=\frac{A_{ij}}{q_{ij}}Y_b(i,j)
\]

とする。補正を単に `P0/q` にしない。全正係数pathでq>0、実finite-bit qとweightの厳密対応、同じcontrolled sequenceとphaseが必要。これにより同じ \(\sum A_{ij}V_jV_i\) のcos/sin平均を保つ。理想係数への置換、未知のpath信号によるproposal/variance選択は行わない。

Full等の拡大対照では元positive supportがsubprobabilityの場合がある。各方式の実joint lawとany-zero→whole-zeroを維持し、zeroを全trial分母へ含める。accepted-only再正規化や同一marginalからのjoint推定は不可。主案Closed P5でも、価格0の非zero-amplitude pathをzero-fillとして落としてはならない。

## 4. 回路価格を取得する方法

各pairの保存IRを時間順 `IR_i + IR_j`、行列順 `V_j V_i` とする。元IRがinternally reducedである条件を維持する。既存inverse判定とsuffix価格を使い、相殺depthをdとした各native指標aについて

\[
C_{\rm add}^{(a)}(i,j)=C_i^{(a)}+C_j^{(a)},\qquad
C_{\rm bnd}^{(a)}(i,j)=C_i^{(a)}+C_j^{(a)}-2C_{\rm suffix(i,d)}^{(a)}
\]

を取得する設計である。これは保存回路に対する**新しいpath価格評価**なので、次回科学実行許可の対象となる。D2保存平均から逆算しない。full pair IRを保存する必要はなく、pair ID・価格vectorと元bindingへのpointerで足りる。

合法なpassは隣接literal inverseだけ。controlled構成、signed Rのactual saved string/adjoint、相対global W phaseを保持する。合成sequence内部、任意のbasis交換やcommutation、新angle、最適compilerは含めない。Gaussian/basis共有一般は未検証で、この入力の保存basis gateのliteral相殺だけが含まれる。

価格vectorはT/CX/1Qを別々に保持する。Tは1Qの部分なので足さない。RZ数・depth・wall timeをこれらから換算しない。欠落cache key、異なるprecision/sequence、wire/phase不整合ならSTOPし、新synthesisで補完しない。

## 5. 既知IS、normalizer、実生成への接続

選択された主資源aの**同じ価格境界**を用い、加法価格から作るqAと境界価格から作るqBを比較する。正価格の場合の理想score/normalizerは

\[
s_\omega=A_\omega/\sqrt{C_\omega^{\rm full,(a)}},\qquad
Z_C=\sum_\omega s_\omega,\qquad q_\omega^*=s_\omega/Z_C.
\]

元weightが一定とは限らないため、`P0/sqrt(C)`への無条件置換はしない。Parent `cost_aware` へ係数Aと価格を与え、既存の固定root enclosure→normalization enclosure→dyadic規則で実qを得る。weightは**実q**による `A/q`。root/probability丸めを理想qへ置換せず、fixed resolution不十分・support脱落ならSTOPする。

D2登録metadataの3,969 pairを明示的に取得する案である。setupには全価格取得、sqrt enclosure、normalizer和、dyadic table作成が必要。時間・CPU・peak memory・table bytesを計上する。初回setupを各比較方式に公平に帰属させ、再利用時は償却回数を明示する。既存取得のsunk costと新規IS setup費を混同しない。

既存Closed P5のreturn/群DPは係数構造を利用するが、suffix-dependentな境界価格を含む \(Z_C\) の安価な計算法ではない。因数分解やDP高速化は今回発明しない。列挙有限table検証の結果を非列挙scalabilityの証拠にはしない。

実生成は既存 `dyadic_index` でpair indexを選び、保存pair IDへ戻す接続が候補。`dyadic_index` はtableを順に走査するため、selectionをO(1)・無料とみなさない。bits消費、tableアクセス、weight取得、pair IRの再利用／境界処理・serializationをonline費の項目に含める。実測しない項目は `NOT_MEASURED` とし0にしない。

**zero-price分岐：** `cost_aware` は正価格を要求する。境界後T等が0になるpathの有無は今回未評価。発見時にはそのanalytic候補をSTOPし、既存 `regularized` と正のλを採択するかユーザー／GPTへ戻す。λを実行費へ加算せず、hidden floor、path削除、無断のCX主指標への変更はしない。新regularizer探索も自動追加しない。

## 6. 六cellの公平な比較

| 固定joint law | 加法回路 A | 境界相殺回路 B |
|---|---|---|
| P0：元digital product law | P0/A | P0/B：D2と同じlaw・相殺の対照 |
| qA：加法full価格で作る既知IS | qA/A | **qA/B：既知ISにも同じ相殺機会を与える強い対照** |
| qB：境界full価格で作る既知IS | qB/A | qB/B：候補 |

主案はClosed P5という既存Return Aggregation方式に固定するので、RAの情報とdictionaryは全cell共通である。boundary効果は**固定lawのA/B**、samplingの追加効果は**同じB上のqB対P0およびqA**で区別する。qB/Aもcross controlとして残す。全cellで同じcache、合法なpass、wrapper、bias、precision、confidence、価格情報へのアクセスを与える。改善率を足したり掛けたりしない。

他の適合RA（Full/partial/Closed P3等）とCTSも既存対照として適用可能性を確認したが、最小一arm案では再会計しない。方式間の最良性を判断するなら同D2条件の必要armを別途採択し、law/zero-fillを保持して同じ情報・IS・境界機会を与える。本案だけでbest RA、CTS超越、分子全体の優位を主張しない。

## 7. 共通の誤差・confidence・十分shot数

D2旧保証の再利用を提案する。保存row 10のstate-uniform biasを共通に保持する。契約の記号で

\[
\epsilon_T=3x^6/720,\quad e=8\rho+16(1+\rho)\epsilon_{\rm primitive},
\]
\[
\beta=2\{[(1+\epsilon_T)^r-1]+[(1+\epsilon_T+e)^r-(1+\epsilon_T)^r]\},
\quad \Delta=\epsilon_{\rm axis}-\beta.
\]

\(\Delta\le0\) ならSTOP。D3鋭い保証を使う場合は全cell共通に採択し直し、旧保証baselineと混ぜない。reweightingは固定係数meanを維持するので、q丸め自体を新しいmean biasへ二重計上しない。回路・係数が変わればこの前提を再確認する。

各実joint lawについて、符号/phaseを上記の通り回路側へ保持した実係数から

\[
M_2(q)=\sum_\omega A_\omega^2/q_\omega,\qquad
L(q)=\max_\omega|A_\omega/q_\omega|
\]

を取得する。Y²=1の測定を前提とする。未知のmeanでvarianceを無料に縮めない。既存scalar Bernstein式の共通使用案は

\[
N(q)=\left\lceil\log_{\rm upper}(2/\alpha_{\rm axis})
\left(\frac{2M_2(q)}{\Delta^2}+\frac{4L(q)}{3\Delta}\right)\right\rceil.
\]

baseline P0にも同じ有限tableの正確なM2/Lと同じ式を使う。旧D2の保守的production Nは過去anchorとして区別し、新候補だけtightにした比較をしない。新joint lawで旧N、independent moment積、旧accepted-tail/hard-cost保証を据え置かない。Nは十分forecastで、実行shot数・最小shot数ではない。

同じ半径Bと残余精度/αでM2=B²を入れるだけのBernstein移植では、Hoeffdingに正のrange項が加わりshot削減にならない。主RPEの単位半径Hoeffdingとは測定範囲が異なる。元H厳密ground仮定もweighted測定/PF信号の半径1を保証しない。

旧条件は `epsilon_axis=1/200, alpha_axis=49/24000, probability_bits=160, root_bits=256, eta=rho=10^-12, primitive_error=10^-6, interval_dps=100`。**再利用候補であり今回正式採択しない。** 複数law/axisのconfidenceを同時campaignとするか、同一精度での代替forecastとするかを先に決める。旧familywise/resource-failure allowanceを新六cellの同時保証へ無断転用しない。

## 8. Full試行価格と古典費

\[
G_Q^{(a)}(q,K)=\sum_{b=X,Y}N_b(q)\sum_\omega q_\omega
C_{K,b}^{\rm full,no\text{-}prep,(a)}(\omega),\quad K\in\{A,B\}.
\]

保存native IRはcontrolled event処理を含む。上式には両step本体と境界、必要なHadamard wrapper、ancilla準備/readout等の採択された費用を含める。system状態準備だけを除外する。未知の準備/readoutのaffine parameter hを全て0にしない。

最小wrapper案は既存配置の `H→controlled evolution→[Sdg for Y]→H→measure`。このClifford-only wrapperを採択し、T-countのgate指標を使う場合に限り、外側H/Sdgの追加Tは0と定義できる。CX/1Qでは各gateを対応する別価格で加える。準備/readoutの未知費はaffine hとして残す。追加の角度処理/補助操作が必要ならその価格を先に定義し、都合よく除外しない。

qA/qBのscoreにも、主指標について採択された同じfull価格を使う。hが未定なら固定lawでのaffine比較に限定し、h=0専用proposalを全hで最適としない。量子gate数と古典秒/メモリは別座標で報告する。共通の実測換算モデルがない限り加算しない。

古典費は \(T_{\rm setup}(q,K)+\sum_bN_bT_{\rm online}(q,K,b)\) とmemory/table量を分ける。元P0の既存生成経路にも比較可能なonline費を用いる。tableをbaselineにも渡す比較と、既存factorized generatorの実装比較は区別する。online未測定なら「総古典費込みで改善」を結論しない。

本synthetic入力には分子のdeterministic PF halves、全RPE round/branch、energy成功条件がない。これらを既知価格0へ入れず、最終 \(\sum_{m,b}N_{m,b}\mathbb E[C_{m,b}^{\rm full,no-prep}]\) は未評価とする。

## 9. 次回許可後の最小出力と判断材料

承認された場合だけ、薄い接続の必要な人工検査後、固定bindingから全pair価格を一度取得し、qA/qBのtableと実weightを既存関数で作り、共通統計・六cell費用を評価する案。pair順序/ID、係数・元/新q・weight、T/CX/1Q価格、normalizer enclosure/bit条件、M2/L/N、full価格境界、setup/未測定online費とSOURCEを追跡できる最小出力にする。新承認台帳は不要。

境界価格が異なってもqB/BがP0/BやqA/Bに勝たない、range/ceilで利益が消える、setup/online負担が量子節約に見合わない、normalizerの列挙が支配する場合は拡大理由が弱くなる。同価格、zero-priceでanalytic候補不成立、bias不適格、support/phase/価格対応不備も結果・STOP材料になる。逆に有限table上で追加利益が残るなら、次に非列挙生成と分子wrapperへ接続する理由になる。閾値・改善率・採否は今回設定/判定しない。

負荷実行前にはCPU/RAM/GPU/実quotaとbit演算・pair価格取得のボトルネックを確認し、suffix cache・共通データ・独立pair処理の適切な並列化を選ぶ。scienceのprecision/seed/比較条件を変えず、共有環境に干渉しない。旧D2の人工worker/RAM/wall制限を新案へ自動転用しない。今回の読解からspeedup倍率は主張しない。

## 10. 実行前の選択・許可とmandatory STOP

| 担当 | 採択・許可が必要な事項 |
|---|---|
| ユーザー／GPT | 第一候補D2 r2 Closed P5、六cell、旧D2保証/bit/精度を再利用するか。比較を代替forecastとするか同時保証とするか |
| ユーザー／GPT | 主指標をTとする旧文脈の再利用案、CX/1Q併記、wrapper・未知hの境界。別指標を選ぶ場合はそのprice input |
| ユーザー／GPT | zero-price発見時のSTOPを維持するか、別途既存regularized法とλを指定するか。未発見の段階でλを無断固定しない |
| ユーザー／GPT | 古典online実測を含めるか。含めるならbenchmark条件・draw数/seed等を別途固定。含めないなら総古典費込み結論を留保 |
| ユーザー | 必要最小限のtable接続実装・人工検査、および新path価格/normalizer/q/M2/L/N/費用再評価の明示許可。研究方向の採択とは別 |
| Codex（許可後） | 上記で固定された範囲のSOURCE/入力/出力を具体化し、必要な実装と評価を行う。新angle/信号/compile/RNGを勝手に追加しない |

現在は実行仕様案の公開・取得照合でSTOPする。後日評価を許可された場合も、入力/target/phase/key/positive support/precision不整合、bias exhaustion、未許可条件が必要になった時点でSTOP。retry、別candidate/namespaceでの続行、旧grant/marker再利用で回避しない。重要科学レビューは必要な場合に独立した開始許可を得る。

[BOUNDARY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28fc3ca17ec51bb7656a498e3eea9b72966eb3fa/src/trottertracks/algorithm_codesign/rte_development/boundary.py
[PARENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4ccc1832f63cd323b23fc05623f7e0b58cf34f37/src/trottertracks/algorithm_codesign/parent_allocation.py
[RA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4ccc1832f63cd323b23fc05623f7e0b58cf34f37/src/trottertracks/algorithm_codesign/return_aggregation.py
[DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4ccc1832f63cd323b23fc05623f7e0b58cf34f37/src/trottertracks/algorithm_codesign/parent_experiment/design.py
[BOUNDS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4ccc1832f63cd323b23fc05623f7e0b58cf34f37/src/trottertracks/algorithm_codesign/g8_bounds.py
[D1A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/874ca12d347f8f35e5d8d64f65f24abc4a2df523/src/trottertracks/algorithm_codesign/rte_development/accounting.py
[WRAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/79253e0aa761c06ca9439b795d5619c7865f860c/src/trotterlib/rpe_hadamard_interrogation.py
[TEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28fc3ca17ec51bb7656a498e3eea9b72966eb3fa/tests/tracks/algorithm_codesign/test_d2_key_mechanisms_boundary.py
