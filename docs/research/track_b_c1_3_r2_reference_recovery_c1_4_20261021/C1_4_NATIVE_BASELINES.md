# C1-4：native費用・強い対照の契約草案（未採択）

今回取得したものはideal logical circuitのselected correctnessであり、strict native価格・認証期待費用・性能比較は`UNKNOWN`。C1-4ではSOURCEを静的に読んだだけで、費用oracle、新合成、benchmark、Nやresource winnerを計算していない。

## 非列挙経路で既にある部品と不足

| 部品/層 | 既存SOURCEが与えるもの | 今回の境界・必要な追加事項 |
|---|---|---|
| D1 inventory/acquisition | `collect_pending`が`reference_events`をlist化し、`freeze_inventory`が全eventのratio集合を作り、評価側も全eventを走査。[D1] | production samplerが非列挙でも、この取得経路は全列挙。H4へそのまま流用しない。全列挙を別工程に移して回避と呼ばない |
| G8 on-demand | accepted eventのratio keyでmiss取得、memo/LRU、strict RZ列、actual-adjoint結合。zeroでcache/provider/readoutを呼ばない。[G8] | 観測keyだけのcoverage。provider_deltaは仮定、`quantum_measurements=0`。全supportの取得成功・期待価格・終端保証ではない。旧capacity/bytes数値を新契約の上限へ自動継承しない |
| Closed P5 | root、ordered pair、長さ4 first-label群。群/angle構造O(L^2)、completion DPはO(L)状態・現処理O(L^2)。群normalizerを取得する。[P5]、[M§5] | 全長4 word表は不要。ただしDP/群/有理root/bit演算/選択器の前処理は有料。現group indexの線形scanはO(L^2)になり得る。全word依存のbasis cancellation/native価格まで群angleだけで決まらない |
| General Full | 局所parent query、ordinary envelopeとU、棄却、実weight。全normalizerを事前取得しない。[M§3,4] | 全child係数/child law構築・sort・F3再query/Evidence費用は現SOURCEに残る。未実装の案A/B最適化を取得済みと扱わない |
| Uniform CQ / rotation | G8にuniform CQ誤差deltaを仮定したevent誤差式がある。[G8B] | 新direct DFのbasis/sign/中央Z・ZZ/rotation/scalar/actual adjointの全support strict certificateが必要。6 eventのfloat一致を一様native保証へ格上げしない |
| 共有basis/cancellation | 既存DF builderはactual inverseを使う。新adapterは保存basisを接続する | 無条件basisとcontrolled中央操作の境界、event/step間cancellationを全方式共通の規則へ固定する。D1の未相殺加法会計やG7 helper単価を自動移植しない |
| Key/Evidence | 既存D1はangle kind/ratio/precision/tool/strict phaseをinventory keyへ結ぶ。[D1] | DFではmodel/basis/central gate/control/sign・実angle変換・compiler profile・誤差・native IR/cost/証明へのsource-bound結合が必要。未観測keyはUNKNOWN、0ではない |

### 費用として分ける5層

1. **Logical size**：新builderのparentは`|u|`個のsigned-Q積と1個のchild rotation、parent/scalar phase。有限basis catalogと操作列から論理構造を数えられる。raw instruction数はnative T/CX/depthではない。旧DF RTE上界APIのTaylor orderをreduced word lengthへ型偽装しない。
2. **Strict native T/CX/1Q/depth/workspace**：採択toolchainと誤差profileのcertificateが必要。system-onlyのup-to-phase実装を無料controlled providerへ置換しない。G7 helperと今回のdirect DF（workspace beyond system=1）は別経路。[M§9]
3. **Event依存期待native費用**：実全trial proposal pi_eとc(e)から`sum pi_e*c(e)`を取得/認証する必要がある。観測sample平均を母期待値や認証上界としない。full-supportのuniform価格上界なら保守的上界にできるが、exact期待費用とは別であり、未取得のuniform oracleを仮定しない。
4. **全trial・量子call総資源**：採択された十分Nと、whole-path law/zero判定・受理率上界を結ぶ。期待accepted数、accepted-tail cap、hard capは別。古典zeroにも費用があり、準備/readoutをzero後へ遅延できる条件を明示する。[A]、[M§10]
5. **Classical setup/online/storage/終端**：全方式でinput取得、exact rational/bit演算、root/angle、群/selector、cache miss、source照合・監査/requery、Evidence・bytes・終端処理を含める。CPU秒・rational operation数・bit complexity・量子T/CX/depthを明示価格モデルなしで加算しない。[M§4,10]

既存G8 `accepted_cap/acceptance_upper`は条件付きで全normalizerなしのresource上界へ使えるが、未知c(e)を埋めない。D1 `whole_cost`は独立step・whole-path quantum execution・未相殺加法価格という条件付きで、event価格と受理率の会計を示す。新DFの境界相殺・全support native・期待価格が得られない場合はその範囲をUNKNOWNに残す。[A]、[G8B]

## 同じfinite Taylor targetでの強い対照

| 方式 | 既存構成/有利な最適化も認める範囲 | 今回未取得の結合 |
|---|---|---|
| Ordinary RTE | 共通p/x/mによる既存generator。cache/selector/basis共有を同等に認める | R1 DF provider、実finite-bit/error/whole-path/native会計 |
| Partial Return | 既存canonical低次数+tail構成。合法なprefix/cacheを認める | 同上。Full F3のP0-P5を代用しない |
| Closed P3 + tail | closed P3群・tail、最適化された選択/再利用 | 同上。m=5全体のnormalizer/rounding/補正と取得費用 |
| General Full Return | local Green query/envelope/rejection、合法cache/selector | 今回はFullの6 eventまで。全trial測定・native・古典生成性能は未取得 |
| Closed P5 | O(L^2)群、completion DP、群normalizer、cache/selectorを含む強い対照 | R1入力でのgroup-law有限bit保証・DF provider/native。今回instantiate/取得しない |

これらは同じ`P5(-i*x*R)`を理想meanに持つ既存構成である。[GEN]、[P5]、[M] 共通p/符号Q/ordered word・time/scalar・測定/準備・最終bias/confidenceを合わせる。precisionは共通精度予算を満たす合法な選択を全方式へ認め、特定方式だけcache・grouping・basis reuse/cancellationを禁止しない。現在のH/Kを他方式に無条件十分としない。

FullとClosed P5は**同じ理想accepted ensemble**を生成する。共通providerなら同じeventの理想量子単価は共通。一方、Fullの棄却/ordinary envelopeとP5の群normalizer、全trial proposal/weight、finite rational root、実digital moment/range、十分予算、setup/online/保存費用は一致を仮定しない。理想ensemble一致をdigital bytes/予算/総native費用の一致へ移さない。[M§5,10]

## 同じexponential taskの追加対照

- **DF Ordinary PR**：既存DF経路があるが、今回のR1 exact model/finite mean・scalar・state・準備・誤差・confidenceへの結合と新native費用はUNKNOWN。旧PR-2の135係数差を無視して再利用しない。
- **Deterministic symmetric PF**：同じ保存DF/state/T・総精度へ合わせ、必要step数とstrict native/準備/readoutを含める必要がある。現在UNKNOWN。
- **Pauli CTS/SCU**：Hamiltonianの合法な別representation/accessを認め、その取得/Pauli展開/前処理/LCU/測定費用も課金する。旧3-qubit I1 fixtureの取得をH4完成providerと扱わない。現在UNKNOWN。

全方式に同じHamiltonian・state・target・許容bias/confidenceと正当なprovider/access機会を与える。有限Taylor対照とexponential対照を同じ未補正残差でランキングしない。強いUNKNOWN対照を省いて一般資源優位を宣言しない。旧G9/G10/D1〜D3/PR-2/Pilotは新C1性能結果ではない。[M§9,10]

次の契約で、対象をlogical上界/strict native上界/期待nativeのどこまで求めるか、uniform coverage、全support終端、価格取得、baseline、相殺、古典/保存の座標を固定する。全列挙の代替cost oracleや新非自明証明が必要なら別のSOURCE/証明承認へ返し、今回は作成しない。C1-4草案完成はC1-5の実行GOではない。

[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/rte_development/pipeline.py#L51
[G8]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g8_pipeline.py
[G8B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g8_bounds.py
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g9_p5.py
[GEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g7_generator.py
[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/rte_development/accounting.py
[M]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md
[M§3,4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L109
[M§4,10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L321
[M§5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L336
[M§9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L651
[M§10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L689
[M§5,10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L394
