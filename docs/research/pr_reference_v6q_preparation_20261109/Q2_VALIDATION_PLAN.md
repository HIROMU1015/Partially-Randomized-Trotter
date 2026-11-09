# Q2：保存4構成の限定correctness検証計画

[REPORT](REPORT.md)／[実装・証明義務](B1_IMPLEMENTATION_AND_PROOF.md)。**準備だけを完了し、科学実行は未承認・未実行。** 添付指示§2・§8に従い、以下のH4 inputを新providerへ通す前にユーザーの明示承認が必要である。

## 1. 固定SOURCEと実在する入力

元SOURCE・科学入力は **f2638a7845685a119619382b850441f5d8d33713**。候補providerと人工testはこの準備を公開したcommitへ固定する。旧pilot／transfer／B′／compiler／DF／RTE builderを変更せず、[Q2_INPUTS.json](../../../artifacts/pr_reference_v6q/2026-11-09/preparation/Q2_INPUTS.json)のsource_bindingsを用いる。

archiveは次の二つ。省略名は下表のmemberを識別するためだけの名前で、新しいsample IDではない。

- **pilot**：`artifacts/pr_reference_interference_pilot/2026-11-06/raw_prices.tar.xz`、Git blob `2b5f2878b860b6597456896903dae160ab2d5b78`。
- **transfer**：`artifacts/pr_reference_transfer_qualification/2026-11-06/raw_prices.tar.xz`、Git blob `715d6fc777e2200739ef82c0f2cbdad5be5d3d14`。

| q/r | rare | j | 保存seed | archive内の正確なmember |
|---|---|---:|---:|---|
| 1/1 | false | 0 | 4216572573356262833 | pilot：`prices/q1_rare0_00.json` |
| 1/1 | true | 0 | 4841711708811669773 | pilot：`prices/q1_rare1_00.json` |
| 4/1 | false | 0 | 7349747399721085684 | transfer：`raw/q4_r1_rare0_0.json` |
| 4/1 | true | 0 | 12931531744631036724 | transfer：`raw/q4_r1_rare1_0.json` |

4件のmember bytes・archive SHA256はmanifestに保存済み。pilotのrはrecordに存在せず、**旧pilot SOURCEのoccurrence_rte_steps=1と旧saved_samplesのr1 adapter規約**で固定する。欠損を任意に推定したものではない。event fieldはpilotが`event_records`、transferが`events`。対象member以外を選ばず、未知のsample／pathを生成しない。

共通入力は同commitの`artifacts/pr_reference_interference_pilot/2026-11-06/common.pkl.xz`。圧縮／uncompressed pickleのbyte hashをmanifestへ固定し、既存pilot INPUTS.jsonのhashとの一致を確認した。**今回unpickleしていない。** 実行時は自分のfresh outputへ読み出し、旧artifact/cacheへ書かない。保存ground／DF／Hamiltonianを再利用し、新solve／SCF／DF生成を禁止する。

## 2. 実行前に固定する接続

Q2を実施する場合、四つの保存recordを読む薄いdriverが必要。**そのdriverは今回作成・実行していない。** 新provider、manifest、旧SOURCE、下記callと検算を使い、許可scope内の実装を終えてから実行SOURCEをcommit固定する。追加の認証・grantは作らない。

| 再利用関数／資産 | 接続規則 |
|---|---|
| B′ `reconstructed(row)` | component IDと保存Taylor orderから既存`_make_event`へ接続。保存angle・phase・normalization等のschema対応も照合し、seedから新samplingしない |
| transfer `assemble(q,r,cfg,dist,events)` | 同じeventsを使う旧F/Gとcut。q1の特別分割・q4の二lane fusionを保持する |
| B′ `build_saved(row)` | 同じ保存recordの4-CX B′をcorrectness参照として再利用。価格取得／最適化を呼ばない |
| pilot `reference_core`／`axis_signals`／`coherence`／`ground` | F.inverse・2 cascade・relative P・X/Y・保存stateとbit順を保持。groundは検証入力だけで、providerへ渡さない |
| candidate `exact_binary_phase` | 同じ実F/Gから一回だけ呼ぶ。q1/q4は局所fixtureであり、m17 wholeを実行したと扱わない |
| 既存V2 q1／q4 builder witness・compiler symbolic資格 | SOURCE構造の参照。witnessを新providerのPASSの代用にしない |

`common()`は保存cacheを読む経路だけを使う。cache/hashを確認し、fresh cacheを設定した後で呼ぶ。欠損時にprepare_common等へfallbackして科学入力を作らない。旧initialize／attach_cache／runnerを呼んで、過去STOPを解除・迂回する運用にしない。

## 3. 一つのQ2 campaignに含む最小call

**4構成、各1回、固定順：q1 nonrare、q1 rare、q4 nonrare、q4 rare。** 4 independent strataの新標本取得ではなく、保存された有限correctness fixtureである。lawの平均・失敗率・RPE成功を四つの頻度から推定しない。

各構成で次を行う計画とする。

1. 保存recordからcfg／dist／eventsを再構成し、旧assembleからF/Gを1組得る。`trajectory`／`events_for`／旧sample runnerを呼ばない。assemble内部が返す旧登録baselineは価格比較のarmへ採用しない。
2. 同じrecordから既存B′を1回buildし、そのcoherenceをcorrectnessの参照zとする。これは強baselineの費用を変更する操作ではない。
3. 新providerを**1回**評価。独立の整数common-grid参照和とRN判定を照合し、取得したg／θのbinary値と順序が同じであることを記録する。exact参照はproviderの`_lane_sum`自身を正解にしない。
4. 旧phaseと新phaseで旧reference_coreを各1回build。F/Gのgate／wire／global_phaseの読取前後を照合し、coreは最終Pの角度以外が同じであることを確認する。新P parameterのhexが返却floatを保持することも確認。
5. 両coreのX/Yを既存axis_signalsで評価し、B′参照zと比較する。phase fit、別々のglobal-phase fit、事後renormalizationはしない。
6. 各laneのvacuum ray、Fのactual SOURCE inverse、scalar／signed event／D／basisの保持を確認する。既存statevector／小さいlane Operatorを再利用し、**17-qubit whole Operatorは作らない**。数値matrix評価は有限correctness診断で、全support operator certificateとはしない。

4回のF/G assembly、4回のB′ build、旧／新Reference core計8回、新provider計4回がこのpositive scope。旧assemble内部の小回路生成も同じ4構成の再利用に含まれる。X/Y評価は旧／新各2 axes×4構成の16回。vacuumは2 lane×4、F inverseは4構成の局所検証。新しいq/rへ拡張しない。

### 固定negative witness

**q1/r1 nonrare j0だけ**で新Reference coreのcopyへancilla Zを1個追加し、X/Yを2 axes評価する。coherenceが新coreの−zへ反転することと、正しいzへの一致条件を満たさないことを確認する。これはrelative位相をcommon scalarとして消せないことの意図的な診断で、候補回路や新価格armにはしない。zが小さすぎて区別不能ならnegative witnessをUNQUALIFIEDとし、別sampleや別角度を探索しない。

Q1のunsupported gate／nonfinite／subnormal reject fixtureは既存人工結果を参照し、実H4へ新しいRXや別Hamiltonianを挿入して再試行しない。

### Compile・価格の境界

予定compile **0**、新native価格 **0**。旧symbolic P→1 RZと固定contextの構造論証を先に使う。SOURCEのgate型／family／wire／pass前提が一致しなければ、局所数が似ているという理由で価格不変を認定しない。新context／新native correctness compileが必要なら、その最小局所scopeを事前固定する別判断へ戻し、このQ2で自動compileしない。

## 4. 閾値・判定・保存

旧pilotの`TOL=1e-9`とH4 A2／transfer adapterの絶対誤差規約を維持する。toy A2の1e-11やCTの別guardへ読み替えない。

- 入力hash、event ID/order/angle/phaseの対応、wire・SOURCE変化の許可範囲、整数exact和・RNの検査はexact／構造一致。型不適合をfloat変換で救済しない。
- vacuum leakage／vacuum action、SOURCE inverse、旧／新のX/Y複素信号と参照zの差は、それぞれ対応するnorm／absolute errorで**1e-9未満**。X/Yを別々に通して複素差の大きさを隠さない。
- RNはtoyの数値epsilonで比較せずexact有理数の中点と偶奇で判定。数値exp／matrix診断の誤差はRN certificateと分離する。
- negative witnessは−zとの誤差<1e-9、正しいzとの差>1e-9。境界／区別不能は未資格で、thresholdを緩めない。

出力はfresh `artifacts/pr_reference_v6q/2026-11-09/Q2_run01/`等の独立領域。driver／provider SOURCE、四recordの来歴、binary入力対応、exact和とRN証拠、有限correctness結果、残る証明義務、STOPを保存し公開する。旧SOURCE・raw・STOPへ書かず、失敗後の自動retry／入力置換をしない。

Q2有限PASSの名称は、例えば`FOUR_SAVED_FIXTURES_PASS_ALL_SUPPORT_PENDING`。m17全support、626 shots、総RZ優越の実装済み認定へ昇格しない。入力／復元／provider reject／RN／phase／correctness／context不一致はFAILまたはUNQUALIFIEDで停止する。

## 5. 全supportについて別途必要な判断

本providerは非零subnormal exact和を拒否する。Q2の有限4例では、全canonical finite-bit supportでこのrejectが起きないことを証明できない。入力型・finite literal・global_phase生成／normalization・exact和のnormal域を覆うSOURCE論証が必要である。必要なsaved-literalの型／dyadic lattice数値監査は新規再解析なので、実施するならSOURCE・有限対象・判定規則を別途固定し、明示承認を受ける。今回その監査・ℓ算定・m17 phase計算は0件。

単にrejectを解除したり、旧B_normを0にしたり、未証明項へ新allowanceを追加したりしてV6-Pへ接続しない。`phase_for_round`のm17選択を実saved scheduleへ結ぶconsumer契約も必要。他17roundはlegacy defaultのままとする。

V6-PのN／総RZ結果は既存引用に限定し、Q2で再計算しない。classical長round overhead、m17 CT、新合成、word policy、総T、A1／fsum／別precisionはこのscope外。

## 6. 承認と終了

次に必要なのは、**この4保存構成と上記positive／negative witnessに限るQ2のphase評価・回路build・state simulation・局所inverse検証**へのユーザー明示承認。新compile／新価格／RPE再算定／全18round総RZ再計算を含めない。必要になった場合は実行前に戻す。

Q2のdriverを実装してSOURCE固定後、一つのcampaignで検算・結果公開を完了したらmandatory STOP。科学的な保証不足を認識した時点でも対象を拡張せず停止する。**今回はQ1・入力固定・Q3静的整理の公開後に準備STOPし、Q2は開始しない。** 次担当はGPT／ユーザーである。
