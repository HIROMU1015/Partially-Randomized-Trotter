# N5-B0終了後：同じword・emitterの最小検証案

2026-11-08 JST。[REPORT](REPORT.md)。**提案・未実行。今回の静的許可で数値実行を開始しない。** 新しいprecision／辞書／emitterを既定案にしない。

## 1. 最初の判断と対象

**GPT／ユーザー**：既存strict上界を保持して鋭くする経路Sと、pure common joint scalarをraw信号保証へ分離する経路Gを区別し、どの保証を次に資格するか判断する。数学的なscalar測定不変性と、実consumer全経路への適用資格、m17の実行・成功資格は別である。

推奨する最初の技術scopeは、**m17のB′／Reference×X/Yへ共通に使うword具象化経路の局所適合性確認**。巨大なm17回路や物理groundのsimulationではなく、既存wordを使う固定小fixtureとSOURCE意味論を対応させる。科学的保証の採択はこの技術検証から自動で行わない。

## 2. 固定入力

[NEXT_DIAGNOSTIC_INPUTS.json](../../../artifacts/pr_reference_n5b0_static/2026-11-08/NEXT_DIAGNOSTIC_INPUTS.json)をそのまま候補集合とする。

- C0各W residueでpositive literalの文字列順最小の既存record、全8件。Actual archive member／record key／sequence hashをJSONへ固定済み。これはdiagnostic対象の選択で、word policyの変更ではない。
- 既存frame `0x1.08ba32d420f2cp+2` とb2 `0x1.921fb54442d18p+2`。後者はW=72、residue0で、W数とresidueを分ける例。
- 正負2方向。負側は同じpositive word全体のactual inverse。
- Prefix phaseは旧N3 fixtureのzero／one_tenth／prev_tau／tau_via_negative_subnormal。
- 固定2-qubit native技術fixture内でwordをCXの前又は後へ置く。Wordにcontrolを後付けしない。対象angleは保存recordのliteral、guardを評価するための新targetではない。
- 比較用のnamed gate列・W metadata、元native phase、qubit/clbit／reset/readoutの対応は既存SOURCE規約を維持する。

完全なrecord集合は上記JSONで凍結済み。m17の全到達集合、新しいbasic key、別precision、別seedを探索しない。基本的なW loop／copy境界／standalone wordについては、同じsequence・経路の既存N3/N4検証があれば再実行せず引用する。

## 3. 担当・再利用・具体的call

**Codex**が明示実行承認後に限定wrapperを作り、次だけを実行する案とする。

1. 固定SOURCEとrecord hashを照合し、[EMITTER_BOUND_OPTIONS](EMITTER_BOUND_OPTIONS.md)の因子分離の前提を型付きで追跡する。Named gate列・wire順・actual inverse、global_phase欄、測定までのsinkを明示する。
2. 既存N3-B1b-0 helperのword_test／emitter／exact_word_matrixの必要な純粋部と、変更しないlower_native／N4 consumerを再利用する。Strict_guard／acquireは読み込まず実行しない。旧runner.mainも呼ばない。
3. 未検証の固定record／prefix／CX位置の組だけで小さいemissionとOperator等の技術診断を行う。Named列・位相fieldを直接照合し、必要なら同じ小さいfixtureのX/Y observableについてscalar不変性と実経路を対応させる。Physical H4 simulation／samplingはしない。
4. 比較に使うscalarはexact Wと実phase fieldからSOURCEで指定し、targetに合わせた数値phase fitを使わない。旧diagnostic toleranceは実装適合の目安に限定し、m17の10^-16級strict certificateとして使わない。

新規基本合成・word取得・strict guard、native PF build/compile、provider拡張、m17のerror和／radius／shot再会計、価格集計はこの候補scopeに含めない。Toy fixtureのbuild／局所数値診断への承認と、後続の科学的保証採択・RPE再会計承認を混同しない。

## 4. 判定とSTOP

| 結果 | 後続の分岐 |
|---|---|
| 同じnamed列でpure common scalarと適合 | SOURCE条件下のsignal因子分離候補をGPT／ユーザーへ返す。M17 PASSにはしない |
| 相対位相・word順・inverse・wireの不一致 | 不一致を保全してSTOP。新adapter／回路の自動修正や新規合成へ進まない |
| Strict operator保証の改善には未取得のpi／prefix上界が必要 | その具体的な数値評価だけの新scopeを別途固定する。小さい誤差を仮定しない |
| 経路のSOURCE／runtime意味論が不足 | 未資格としてSTOP。Finite Operator一致だけで保証しない |

固定技術scopeの完了又は上記欠損でmandatory STOP。Call集合を増やしてPASSを探さない。必要なら旧検証の再利用でcallを減らせるが、未検証の追加recordやphase条件へ広げない。

## 5. その後も別に残る課題

Factorization／emitter検証が成功しても、m17のlogical guard・量子化・域は別途扱う必要がある。次の新仕様はまだ選択していない。

- 同じwordのより厳しいguard資格が必要なら、対象literalとwordを固定した別のguard／保証scope。
- b／basic precision／指数域／provider域を変えるなら、先に選択規則・有限key集合・停止条件を固定する別scope。
- 実行可能policyが固定された後に、m17四条件だけの保存RPE再会計。626／630・88 bits・alphaを維持し、無断のshot／precision最適化なし。
- T価格はその後。M17では626／630の差があるため絶対frame等も含む公平な会計が必要。M7 anchorの流用、全18round総Tへの自動拡張なし。

**現在はmandatory STOP。次担当はGPT／ユーザーで、保証経路の採否と、必要な限定技術実行の明示承認を判断する。** 統括兼Dのnative／q64／13-round adapter研究を上記scopeへ追加しない。
