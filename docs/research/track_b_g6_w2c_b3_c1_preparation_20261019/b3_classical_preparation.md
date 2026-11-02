# B3：古典費用の技術準備と最小実装差分

[入口](README.md) / [契約候補](execution_contract_candidate.md) / [条件対応](cost_evidence_matrix.md)。旧[F2][F2]・[F3][F3]・[G7][G]の意味論は変更しない。

## 1. Canonical入口をどう揃えるか

旧F3のFull非zero P0〜P5、SOURCE/hash/evidence/trace/event/angle再queryはそのまま残す。旧Fullのuniform lower-boundをcanonicalへ流用しない。新補助SOURCEは以下を実装した。

| 補助API | 最小機能・費用の所在 |
|---|---|
| `root_witness` | 供給されたgroup radicand/rootについて、外向きdyadic grid・enclosure・旧canonical相対幅条件をexact比較。rootを新規生成しない |
| `law_witness` | 供給されたlawの和1、正support、H-bit grid、旧`dyadic_distribution`の相対guardをexact検査。lawを新規生成しない |
| `canonical_snapshot` | 固定canonical objectのgroup/label/root/Bを取得し、`distinct`のfirstと全excluded conditional、`fixed_first`の実excluded、`root_collected`のfrozen law builder witnessを有限個検査。新setup preflight費用へ計上 |
| `canonical_state` | 実際に保持されるparameters/groups/roots/B/law/root_childだけを取得。各sample guardでconditional lawを再構築する変更はしない |
| `AuditParity` | 全4方式に共通のSOURCE照合、input/policy seal、保持law/state hash、出力seal、terminal failure。各armの内部処理はそのまま呼出callbackへ残す |
| `event_witness` | πW=c̃、raw/reduced word、逆順time-word、元degreeのphase、child/label/provider呼出metadata。canonicalの合法なraw還元をFullのraw zeroへ変えない |

このadapterは**新しいcanonical samplerや独立group係数証明ではない**。group式・branch/selectorの意味は固定G7 SOURCEと既存保証に依存する。root/law checkerはその既存postconditionの再検査である。snapshotを外部から任意に与えてgroup公式まで正しいと認定しない。将来driverでは、実importしたSOURCEの場所・bytes/commitを固定し、そのobjectからsnapshotを取得する。checker自体は静的/人工検査済みだが、実G7入力での成立は今回確認していない。

Fullの既存F3とcanonical補助入口の手順数が同じという主張はしない。共通に必要な証拠基準を揃えた上で、Full固有P0〜P5/Green/packetとcanonical有限group/conditional witnessの**実追加費用をそれぞれ測る**。canonical overheadを0とせず、Fullのhash・evidenceを消した仮想版を使わない。Fullにも共通overlayを適用する場合は、その費用を明示し二重課金を避ける。

## 2. 実行driverへ接続する手順（未実行）

1. 保存raw/契約/SOURCE bytesと引用値のidentityを確認し、各方式のNを保存値から読み込む。`budget`再評価・reference再列挙は0call。
2. A層（旧generator）とB層（Full既存F3＋canonical補助入口）を別workloadとして固定する。科学対象が同じでも処理量は同一ではないので、層を混ぜて順位を出さない。
3. canonicalは固定constructor→有限witness→held state seal。Fullは旧prepareを省略せず、そのcertificate/stateを共通overlayへ接続する。angleは未採択のτ/K_phi/Jが固定されてから同じenclosure/誤差規約へ結ぶ。
4. 量子前zero、event、typed failureを別recordにする。canonical sample/eventの実law再構築を置換しない。accepted eventの既存再queryも費用に残す。
5. driver側の測定clockを`StageLedger`へ注入する。現在の人工検査はfake clockのみであり実費測定ではない。実driverは候補条件の採択・SOURCE固定後に接続/初回correctness確認が必要で、今回「実G7で実行可能性確認済み」とはしない。

## 3. 計測項目と二重計上防止

S=setup/budget/preflight/取得、μ=全trial online、Ω=未計上のoutput/finalizeとしてW2c-A CL条件へ渡す。setupを何回共有するかはscheduleで固定する。候補は「arm/layerごとに一度構築し、X/Yで共有可能なstateを検証して再利用」。これは未採択であり、既存G7の実行を変更しない。

| prefix・項目 | 将来の記録 | 解釈 |
|---|---|---|
| setup、budget値load、preflight、SOURCE読込/hash | process CPU ns、monotonic wall ns、call数、input/state/source hash | budget再計算なし。load/照合も無料にしない |
| R：raw zero | order/label selector、bit供給、reduce、zero evidence/write | 正当zero。追加quantum呼出なし |
| C：coin reject | raw＋parent/child、root、law/sort、Parent.s再sum＋coin | **child law/sortはcoin前**で支払う |
| A：accepted | 上記＋child selector、proposal/weight/event、必要なangle/guard/旧event再query | acceptedだけをμの分母にしない |
| canonical各mode | group/label/first/conditional/root-child再構築・draw・event処理 | raw/coin rejectionを追加しない。Green費用も付けない |
| F3追加 | guard、全SOURCE/state hash、seal/serialization、trace、angle時再query | 旧F3を削除しない。共通overlay費用と内部F3費用を別欄 |
| cache、bit供給 | cold取得、lookup/hit/miss/eviction、実H-bit数、key identity | fixed SHA診断とfresh-IIDを区別、cache共有機会は両側同じ |
| 保存・output | bytes、write/hash、buffer/manifest、failure prefix、flush/finalize | online recordとΩを二重加算しない。全evidence保持/streamingはscheduleで明示 |
| memory | baseline/peak RSS等の観測、Python traced allocation、live buffer/cache、output bytes | allocator観測とF2 bit領域を別座標。peakをN倍しない |
| bit operation | 既存F2記号上界、選択したcall数・operand bit幅の診断 | Fraction内部GCD/isqrtの正確bit演算数を測定済みとはしない |

`StageLedger`はexclusive stageを要求し、nested加算を拒否する。細分化profileのobserver overheadは独立に記録し、非instrumented totalとnested値を足し合わせない。旧SOURCEは変更せず、将来instrumentationは専用processのcallback/適用範囲を固定したobserverに限定する。native/key収集や既存ledgerへの書込は行わない。

## 4. 実CPU順位と期待費用の違い

有限のCPU実測を数学的な母平均μのcertificateにすることはできない。特に旧固定SHA 128-trial traceはIID cost推定ではない。候補には次を明示的に分けた。

- **第一の技術候補**：既存seedによる固定全N workloadを一度実行し、観測総CPU・memory/outputを記述的に比較する。途中prefix×Nの外挿をしない。これで分かるのは当該schedule/環境/bitstreamの実費であり、母期待費用や再現分散の保証ではない。
- **期待費用を判断したい場合**：fresh-IIDのcost estimand、runtime変動の扱い、sample数・confidence/停止規則が未採択。G7のα_axisをCPU期待値のCIへ流用しない。新しい統計保証を今回作らない。

上記のどちらを目的にするかは科学的判断として残す。短いprofileだけでFull全N費用優位を宣言する経路は設けない。

## 5. 最適化・保全・限界

保存B1/B2/cache・静的identityを再利用し、同じ値の再取得を避ける。独立arm/layer profileは各専用processへ分離可能だが、共有CPU/IO競合が比較を歪めない実環境・call budget・aggregate memoryを見て並列度を選ぶ。人工的なCPU/RAM/wall上限は新設しない。共有設定/他processへ変更を加えない。bit順序・sampling law・精度を変える高速化は未承認。

新補助SOURCEは人工テストまで完了。実G7への統合runner、angle policy、cost estimand/scheduleの採択が残る。SOURCEファイル照合やsealは来歴guardであり悪意あるprocess全般のattestationではない。OS kill時の永続保存はdriver側の課題として残す。**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[F2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/full_bit_complexity_accounting.md
[F3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aea9b8c228c5b940f2fcba0f94493b3ffdec46e/docs/research/track_b_g6_f3_preflight_angle_20261018/implementation_contract.md
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
