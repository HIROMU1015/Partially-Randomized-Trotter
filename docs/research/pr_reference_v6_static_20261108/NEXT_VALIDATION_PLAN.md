# V6：次の限定科学検証案

[総括](REPORT.md)／[候補と条件付き証明](PHASE_IMPROVEMENT_OPTIONS.md)。2026-11-08。

**今回許可された静的作業は本計画の公開まで。以下の科学数値診断・保存値再解析・phase provider実装／検証は未実施であり、実行許可も受領していない。** 新しいSOURCEとcall範囲を明示したユーザーの実行承認が必要。承認証明書や新grant管理は不要で、元SOURCE・結果・STOPを保持する。

## 1. 推奨する判別順序

現行SOURCE不変更案A1は、実際の時系列prefixに沿う全support証明が閉じれば最小変更である。ただし現在はM_iのSOURCE対応が未完成で、新上界を投入できない。追加の静的証明を先行させる選択肢は残すが、旧γ_L Mから根拠なく回数やbasis項を削った数値実験は行わない。

算術変更候補B1には、既存の片側exact-prefix上界Mとnormalization上界B_normから **B_norm＋2uM** という明示された条件付き保証式がある。そのため、最初の科学scopeは次の**V6-P：保存m17上界による条件付きscreen**を推奨する。phase providerや長round回路の実装前に、保証を実現した場合の整数Nと全18round総RZに利益の余地があるかを判別する。B1が実装済み／資格済みであるという比較ではない。

複数precision、和の順序、periodic reduction、q/r、αを探索する診断にはしない。Pの結果が不利でも自動でfsum・A1・他候補へ移らない。

## 2. V6-P：固定入力・規則・最小call範囲

### 2.1 科学SOURCEと入力

科学的基準は **f2638a7845685a119619382b850441f5d8d33713**。以下のそのcommitのblobだけを用いる。実行用adapterを将来作る場合、その追加commitも実行前に固定する。新しいadapterは未作成であり、「準備済みrunner」とは呼ばない。

| 固定入力／SOURCE | 使用する情報 |
|---|---|
| [transfer_error.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/pr_reference_transfer_error.py) | 旧phase_boundのM・normalizationの意味、raw disk→ρ/β、旧算術モデル。phase_bound／qualify／commonは呼び出さない |
| [STAGE3_ALL_SUPPORT.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE3_ALL_SUPPORT.json) | m17のphase_arithmetic.exact_prefix_absolute_upper、normalization_vs_controlled_source_upper、旧phase_error_upper、shared disk、Reference native raw disk、nominal radius、算術前提 |
| [原ROUNDS.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/ROUNDS.json) | `CN_S2_h0.02_ld3`のm17にあるPF_phase_upper、元raw_radius_lower／raw_extra_disk_upper、q/r/bitsと継承SOURCE／proposal条件 |
| [STAGE4_RPE.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE4_RPE.json) | 保存Reference N、m0〜m16の固定schedule、RPE成功規約 |
| [B′ ERROR_RPE.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/artifacts/pr_reference_bprime/2026-11-07/ERROR_RPE.json) | 保存B′626、native/shared disk、外向き端点と整数丸め規約。kernelも誤差も交換しない |
| [B′ TOTALS_18ROUND.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/artifacts/pr_reference_bprime/2026-11-07/TOTALS_18ROUND.json) | 保存両armのRZ単価区間とexact Fraction、全18round総計。旧14-CX価格は使わない |
| [D0 RESULTS.json.xz](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/artifacts/pr_reference_cost_structure/2026-11-08/D0_run01/RESULTS.json.xz)・[D0 checker](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/scripts/check_pr_reference_cost_structure_d0.py) | 保存36セルの単価・N・9role、恒等式と区間会計の再利用。D0 campaign自体は再実行しない |
| [B′ STRUCTURAL_APPENDIX](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/docs/research/pr_reference_bprime_20261107/STRUCTURAL_APPENDIX.md) | 固定compiler contextでRZが取消対象外、symbolic P→1RZ、scalarの扱い。単価不変という条件付きscreenの根拠 |

対象は **m17、q=131072、r=32、保存bits、B′／Reference×X/Y**。保存値の引用に基づく固定で、他q/rを試さない。全independent finite-bit event選択を覆う旧MとB_normを使い、新trajectoryをsampleしない。E0、gap、groundの古典記述、common.pklからの物理再計算は不要。

入力が不在・欠損・重複・schedule不一致ならSTOP。Mやnormalizationを0／推定値で補完しない。旧STAGE3のbinary64/no overflow/underflow等の算術前提を受け入れられない場合も、その前提を変更して診断を続けない。

### 2.2 比較する三つの行

1. **保存B′**：N=626を含む現在の資格・単価をそのまま参照する。
2. **保存Reference**：N=630と旧phase上界を参照し、旧十分条件の経路を算術検算する。
3. **条件付きReference-B1**：F/G・native context・価格を固定したまま、二laneのbinary入力をexactに集約し最後に一回RNできた場合のphase上界だけを `B_norm＋2uM` に交換する。これは仮定付きの新方式screenであり、実行したphase値や実装済み新回路を価格付けしたものではない。

u=2⁻⁵³、B_normとMは保存された上向き端点を用いる。実際の位相SやΦを計算しない。δ_newを恣意的に0、旧値の一定割合、観測誤差、B′のdiskへ設定しない。旧normalization項を削除せず、最終一回丸めの構成と二回＋最終加算の構成を混同しない。

B′に同じproviderを架空に挿入しない。共通上流scalar/event/global-phase処理は両armとも不変。共通変更が必要になればこのscopeで続行せず戻す。

### 2.3 算術とRPE再評価の範囲

既存192-bit directed intervalのI／upper／lower helperと、`error_and_shots`の**純粋な算術規則**を再利用する薄いadapterを将来追加する。現在のmoduleはcommonやcatalogue等へ接続するため、runner全体を呼ばない。新estimator、sampler、compiler、optimizerは不要。

- 旧δで保存Referenceの630と両axis条件への接続を検算する。新δは候補式を外向き評価して保存する。
- shared SOURCE/event/proposalのdiskとnative raw diskを**別の保存上向き成分から**組み立てる。丸め済みtotalからphaseをfloat減算してD_otherを作らない。未評価成分を0扱いしない。
- 成分の保存端点を再利用すると旧の内部intervalから端点化した分の微小な保守性が増える可能性がある。それを隠して旧内部値のbitwise再現と呼ばない。旧Nの再現・保存ρ/βとの整合を検算し、相違が整数Nや比較結論を変える場合はSTOPして理由を報告する。端点を内向きへ丸めて一致させない。
- nominal radiusとPFは原ROUNDSの同じ式から再構成し、ρ、βを外向き保存後に `ceil(4 log(1440)/(ρ² sin²β))` を取る。disk単位を混ぜず、旧compilerの二重加算を禁止する。
- 正の半径・統計余裕、同じenergy精度・36 axes×α=0.05の条件を保つ。α、PF、finite-bit law、rを再配分しない。
- **新Nの再評価はReferenceのm17のみに限定**。X/Yに同じ保存worst-case boundを適用することを確認し、他17roundは固定。B′626をtargetや代入値にしない。Nが1でも減るか、総RZがB′より有利かは別判定。

この計画は必要δや新Nの探索gridを含まない。単一の閉じた候補式に対する診断であり、今回は一度も数値代入していない。

### 2.4 条件付き全18round RZ再会計

同じF/Gと1個のPが固定compilerの同じsymbolic loweringを使う限り、P角の変更だけではRZ個数を変えないという既存構造を前提とする。新しい量子価格は取得しない。その前提が成立しない実装へ変える場合、このscreenの価格を転用せずSTOPする。

保存単価区間をμとし、m17の両axisで `(N_new−N_saved) μ_Ref` をexact Fractionで会計する。他17round、B′全18roundを保存値のまま足し、Reference−B′の直接差区間を得る。9roleの和とwhole価格の一致、D fusionのsigned会計、区間端点の相関規約をD0から再利用する。D0の未変更診断を再実行して新結果と呼ばない。

出力は少なくとも旧／新候補phase bound、残るD_other、ρ/β、整数N、価格不変の条件、全18roundの条件付きRZ比較を持つ。CX・depth・qubitは別資源として既存資格を参照し、今回はその新総計を作らない。T-countやT-depthの評価は0。

### 2.5 科学call、検算、STOP

**一つの固定campaign**：保存m17上界による候補δ評価、旧条件の検算、m17の条件付きN評価、既存18round単価へのN差分再会計。新phase生成0、quantum simulation0、trajectory sampling0、回路build/compile0、新native価格0、strict guard0、合成0、T評価0。CPU/RAM/workerの独自上限は作らず、この小さい保存値算術を重複させない。

必要な検算は、入力coverageと科学条件、δの単位と内訳、旧N再現、外向き端点、整数ceil、role合計／whole総計／直接差の一致。人工算術fixtureと科学入力の検算を区別し、新しい承認証明・大型audit基盤は作らない。

判定は以下を区別する。

| 結果 | 報告と停止 |
|---|---|
| candidate boundが小さくならない／Nが減らない | その候補式では整数shot削減を支持しない負結果。自動で別方式へ移らず完了STOP |
| Nは減るが全18round総RZでB′より不利／区間が重なる | 保証／整数Nの条件付き改善と、全体優越の未成立・未判定を分離。完了STOP |
| 総RZ差区間がReference有利 | **一回RNと価格不変を実現・資格化できた場合の条件付き利益**。実装済みの優越や総T改善とは認定せず完了STOP |
| 入力不整合、旧N不一致、算術前提不成立、未評価誤差、区間処理不備 | 科学条件を変更して救済せずSTOP。確認できない結果をPASSにしない |

結果・SOURCE・固定入力参照・検算・STOPを独立領域に保存して公開する。旧成果物を上書きしない。campaign完了後はmandatory STOP。Pが有利でも次節の実装や全18round最適化を自動開始しない。

## 3. V6-Q：Pで余地が残った場合の実装資格案（P範囲外）

**別途のユーザー判断と実行許可が必要。今回の実装指示ではない。** 最小変更はB1のbinary入力読取・exact和・合計一回RNのproviderと、旧 `vacuum_phase(F)+vacuum_phase(G)` の一か所の合計callsite。F/G builder、actual inverse、Qiskit setter、PF、law、compilerは触らない。

検証は次の境界で固定できる。

1. **純粋算術fixture**：既存float→ratioと最終float変換を再利用し、0和、正負の大きな相殺、normalization済みg、半角のexact scaling、F/G別丸めと合計一回丸めの差、unsupported familyのrejectを人工入力で確認する。最終floatとの残差はexact有理数から独立に囲む。NaN/∞や入力変換の丸め直しを受け入れない。precision探索やscientific値でのfsum比較には広げない。
2. **SOURCE対応の有限fixture**：既存transferのq1/r1とq4/r1、rare/nonrare各j0の保存eventを再利用する4構成に限定する。既存保存witness／入力を照合し、旧laneを変更せず同じgと実literalを一度ずつ読むことを確認する。保存trajectoryを新しく抽出し直したり、q128やm17 whole buildを追加したりしない。必要な保存入力が取得不能ならfixture scopeを勝手に置換しない。
3. **correctness**：各構成のSOURCE vacuum-ray条件、F/G順、actual inverse、scalar/event符号、X/Yに渡るP角の意味を確認。有限回路検証ではexact-ground情報をproviderへ渡さない。既存局所P(Φ)→RZ(Φ)解析資格とcontextがそのまま使えるならcompileを省く。新しいcontextを要する場合は、全回路compileへ拡張せず、必要な局所PとX/Y経路を事前固定して追加承認scopeへ戻す。
4. **全support保証**：有限fixtureの誤差観測だけでは資格にしない。exact入力和・最終RN・旧M・旧normalizationからの条件付き論証を実providerへ接続し、read/copy/composeで角度を変えていないことを確認する。CPython最終変換のruntime前提が閉じなければUNQUALIFIEDでSTOPする。

上記4構成の科学的vacuum評価／simulation／回路buildを実施するには、その固定SOURCE・保存入力と対象を明示した許可が必要。今回Qのrunnerや回路は作成していない。Q実行を決める場合は保存sample識別子とarchive内pathを実装前に固定し、欠損を推定しない。この未固定pathを理由にPの保存上界screenを拡張しない。

Qが成立しても、Pで得た条件付き価格／Nを実装済み保証へ昇格できるか照合し、結果・不足を公開してSTOPする。新guard、新word、m17 CT、総Tを同時再開しない。Qでの新しい科学callの具体的な許可はPと独立である。

## 4. A1・他候補を次に選ぶ場合の境界

A1のSOURCE不変更案は、全support prefix grammarと伝播式が静的に閉じて初めて、その導出式を固定した数値診断へ進める。M_iを実測で補完せず、q1例外・融合境界・最終加算・normalizationを落とさない。A1だけで旧上界より鋭くならなければ負結果でSTOPし、SOURCEを改変して救済しない。

fsumを選ぶ場合は、joint streamか二つの和か、CPython／platform条件、最終加算、overflow範囲を先に固定する。B1の式を無条件転用せず、実装したアルゴリズム自身のboundが必要。解析block集約、periodic reduction、高精度p探索はこの最小P/Qへ追加しない。

## 5. 今回の完了と次担当

今回作成したのは3文書のみ。科学的なSOURCE・結果・STOP、保存626／630、全18roundの費用値はすべて不変。新しい科学計算・保存値再解析・位相実装・回路・compile・N再計算は未着手である。

**公開・remote照合後mandatory STOP。** 次担当はGPT／ユーザー：A1の追加証明を先行させるか、V6-Pの保存上界・ceil・条件付き総RZ診断を許可するかを判断する。CodexのP実行、Q実装／科学検証、他候補への移行は、それぞれ選択された固定scopeへの許可があるまで開始しない。
