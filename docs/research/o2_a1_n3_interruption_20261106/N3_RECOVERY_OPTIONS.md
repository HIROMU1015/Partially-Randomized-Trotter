# A1 N3：限定再開と新規campaignの比較

2026-11-06 JST。**両案とも今回未実行・未採択。** 技術上の比較であり、科学条件の変更や重要科学レビューではない。
固定SOURCE03928・中断結果d97e・[事前判定基準a53c](references/N3_RESULT_DECISION_FRAMEWORK.md)を保持する。
対象は保存H4/1.00 Å/STO-3G/legacy DF12/8qubit/全4電子sector/L_D3、m0/q1/δ=.02/r1/K2/coefficient_atol0、N1 actual Q/selected policy・N2保証ground・認証済exact engine、X/Y。
ε/α、3control、quota/range、compiler価格範囲の新しい採択はしていない。

## 案A：保存397件を利用する限定再開

### 再利用できるものと条件

- 保存397bitと397価格をlogical identityごとに一回だけ利用する。全SOURCE/input/law/policy/state/engine/compiler/profile、domainとordinal、path、scopeを照合する。価格だけをcache hitとして無料にしない。
- Xの320件と既に凍結した係数/rank/rangeはそのまま保持し、Xを再測定・再fitしない。Yの77件は部分trainとして保持し、五層各64件が揃う前にfitしない。
- Yの残りはlogical quotaとして243件。うち77件はpathと測定word固定済みの未完了trial、166件は未投入。B1 pilotsと全方式cal/prodは未着手として、元の固定順序・quota・十分N/cap/STOPを維持する。
- 既存word tapeはaudit/復元用で、CV入力にしない。Born/state/E0/gap/真varianceやpath別信号を学習・配分へ渡さない。

### stream位置は現在未保証

元STREAMSは60domain/120spawn keyとmaster seedを保存したが、PCG64内部stateのcheckpointはない。
元sourceは各wordを`PCG64.random_raw() >> 11`で取得する。domainごとのpath整数棄却を含む**全消費word数**とmeasurement wordを保存tapeから照合し、固定NumPy/PCG64/SeedSequence実装でprefixを再現できれば、次の位置を定義できる可能性がある。
再構築時は、終了済397だけでなく固定済474trialの全消費wordを反映する。未完了77件のpath/measurement wordを新しいwordで置換しない。未投入分だけを次位置から取得する。
今回は実RNGの再生・新word生成を禁止されているため、**prefix一致と復元後の位置は未検証**。理想独立53-bitモデルを、PRNG独立性の証明へ変更しない。

### 未完了77件とexactly-onceの限界

77件にはBorn入口/bit/価格のdurable記録がない。しかしcall入口だけのjournalでは、最後のselected buildの戻り・gate適用進捗・signal終了時点を確定できない。
これらを再処理すれば、logical trialは同じでもengineの**追加attempt**となる。旧one-shot/no-retry契約に自動的に適合するとは言えない。
明示許可を受けた場合に限り、同じ固定path/wordで未完了logical trialを再処理する仕様、attempt ID、未知の旧試行と費用の扱いを先に固定する必要がある。
成功した397件をもう一度取得することや、失敗箇所へ新seed/fresh代替trialを選ぶことはこの案にも含めない。

durable bitがあるが価格だけない将来の復旧では、再測定せず保存bitに価格を追加する。durable bit+価格があれば再取得も再課金もしない。
exactly-onceを保証できるのは検証された**保存logical ledger**の範囲である。ログ外の物理実行・消失したattemptの古典費まで一回と保証できない。今回の範囲は理想engine simulationで、実backend再実行の保証へ移さない。

### 統計保証・費用・承認

元の試行prefixを変えず、全quotaを保ち、train後にλを固定し、cal/prodの未消費独立domainと元range/confidence式を保てることが必要。
未完了prefixを捨てる、都合のよいbitを残す、途中結果からcontrol/精度/予算を変える、prod結果で候補を選び直すことは許容しない。
元証明が要求する独立性と停止事象の扱いを復旧仕様に対応させるまでは、元のα保証を自動継承しない。完走したことに条件づけたcoverageも主張しない。

旧397件の1,935,700 RZは再利用してもcampaignの全段階費へ一回含める。旧の中断attempt古典費UNKNOWNは残り、追加attempt/復元/監督の古典費を別記する。
新しいvariance/N/総費用は計算していない。保存データの再利用で計算を節約できる可能性はあるが、期待費用改善や完成費用は保証しない。

旧STOP・marker・ledgerは消さず、parent_run=d97eを参照する新しい復旧directoryで管理する。
必要な許可は「旧one-shot/no-retryの限定例外」「77追加attemptの扱い」「残り固定stageと有限call scope」「prefix復元の実RNG照合」「新SOURCE/運用」の明示採択と実行許可である。
元runnerにはresume機能がなく、現時点でそのまま実行可能な復旧実装はない。

## 案B：新しい独立N3campaign

旧experimentは未完了のまま固定d97eと今回診断を保全する。旧397bit・係数を新train/cal/prodへ混ぜず、旧を独立した外部再現や完走campaignとも呼ばない。
N1/N2/engine証明・同じ科学条件・選択policyは再利用できるが、再発防止adapterを含む新SOURCE/環境を人工確認後に固定し、実入力前に公開照合する。

新しいcampaign ID、独立domain/streamとseed方針を**outcome前**に明示採択する。ここでは新seedを選定・生成していない。
master seedを変える場合も、旧結果を見て有利なseedを探索せず、一回だけ固定する。旧domain/tapeを新streamの一部として再利用しない。
law・selected policy・state・exact kernel・ε/α・control/予算・compiler価格範囲を変えない技術改修と、独立experimentとしてのstream変更を区別する。
float kernel、probability cache、control削減などを再発防止の名で混ぜない。

費用は旧中断experimentの既知RZ/古典UNKNOWNと、新campaign全段階の費用を別列にし、研究全体の支出を報告する場合は両方を含める。
新campaignはtrain/pilotから取得し直すため再利用案より重複計算が増え得る。新N/価格/総費用は未評価で、実行時間や完走を保証しない。
旧の中断費用を除いた新campaign費だけで、研究全体が安くなったと主張しない。

必要な実行許可は、新しい一条件・4方式×X/Y・一campaign、採択済み条件の継承、新stream/seed、有限train/cal/prod/cap・実state/Born/必要compile、manager/STOP/記録方式、新しいSOURCEとfresh outputである。
N1/N2/engine再計算や追加campaign、期待費用保証、科学レビューは自動で含めない。

## 比較と推奨

| 項目 | 案A：限定再開 | 案B：独立campaign |
|---|---|---|
| 旧397件 | 同じlogical trialとして再利用、費用も一回継承 | 新estimatorには使わず旧experimentとして保存 |
| 再現性 | 未保存PCG64 stateのprefix復元、attempt解釈、legacy ledger統合が必要 | 新SOURCE/streamを開始前固定でき、experiment分離が明確 |
| 統計条件 | 旧独立domain/全quotaと復旧選択を対応させる追加確認が必要 | 新独立domainで元の固定統計経路を使う。完走・CV改善は未知 |
| 計算/費用 | 保存分の再取得を回避できるが復旧実装と検査が必要。古典UNKNOWNは回復しない | 保存分も新experimentで取得し、追加費用が増え得る。旧費用を消さない |
| 元STOPとの関係 | one-shot/no-retryの明示的な限定変更が不可欠 | 旧STOPはそのまま、新規campaignへ別許可 |
| 現在の準備状態 | prefix/77attempt処理は未検証、resume driverなし | 持続managerと観測adapterの配備検証が未完了 |

**技術上の既定推奨は案B**。現在は旧stream位置と不確定attemptの扱いを保証できず、独立experimentに分けた方が再現記録と契約の対応が明確なためである。科学的優劣の判断や採択ではない。
計算再利用を優先するなら案Aも候補だが、先にprefix復元・idempotent ledgerの技術仕様と旧契約の例外を明示し、未保証点を解消する必要がある。
どちらを選んでも[持続process管理と観測](N3_RECURRENCE_PREVENTION.md)を先に整える。今回の人工PASSだけで次の実科学起動を許可しない。

**両案未実行。公開照合後mandatory STOP。**
