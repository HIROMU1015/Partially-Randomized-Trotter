# 新A S1/S2：保存template誤差・有限guard/T frontier診断

2026-11-08 JST、SERVER 2。実行SOURCE／scope **c96e0a3a31fb0db5f1ae9a8903567012e21b78f9**、科学入力base **800554fbb64c21e584bb29a1a8f52490ab4163e0**。Branch `codex/pr-reference-noncentral-s1s2-numeric-20261108`。

**承認されたS1/S2を一度実行し、結果と指定検算を保存してmandatory STOP。**

- **S1：UNQUALIFIED。** 両variant×B′／Reference×X/Yの8集計行は全てnull。保存catalogの未対応templateを、SOURCE alphabetへ限定する前に全体で拒否する準備SOURCEの判定により、RTE限定版も評価されなかった。上界改善・悪化を判定できない。
- **S2：FINITE FRONTIER COMPLETE。** 375 literalで保存guard／離散Tの非劣候補を取得。現wordの組は332 literalで非劣、43 literalで支配された。Policyは変更せず、総T・RPE優越は未評価。

## 1. 固定条件・実行SOURCE

[計画§1](../pr_reference_noncentral_reuse_20261108/NEXT_VALIDATION_PLAN.md)の入力と有限候補だけを使用した。[準備済みrunner](../../../scripts/diagnose_pr_reference_noncentral_saved.py)のSHA256は`6346c2591d8e22d1d15571493c8796a2567f4fdefdfcc586be22e72eb068598c`、[再利用helper](../../../scripts/audit_pr_reference_noncentral_reuse.py)は`b4fdd5cac3989b8084d6846ffc6b55914c4f00cf54d4479aa526b549547685fc`。両方ともc96e0a3a…のbytesと一致し、実行後も変更していない。Metadata audit mainや旧V0/V1／fixtureを再実行していない。

Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外、Partial S2 LD3 h=.02 finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、compiler A／Qiskit1.3.0。既存B′4-CX／Reference F/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、D融合・basis共有・scalar/event相対位相・X/Yを維持した。M17 q131072/r32/bits88、B′626／Reference630 shots/axis、alpha1/720、failure.05、phase cap.4、energy1.5936e-4 Haは不変更。

固定metadataのSHA256 `396f11326d2a5bc86316d0997c744a315b60a856ce5912a1fab064676f285027`、7保存入力のbytes／hash、m17／N対応を確認した。SOURCE component alphabetがcatalogに含まれることもrunner内で確認した。

## 2. S1：未資格になった箇所

Runnerは保存lowering内のkey別guard和から各template weightを作った。保存済みweightの件数とnullは次のとおり。

| 保存weight集合 | Template数 | Null数 |
|---|---:|---:|
| B′ | 37 | 4 |
| Reference | 33 | 4 |
| Actual SOURCE inverse D | 5 | 0 |

未対応は、両armの次の2 basisについてforward／inverseの4 templateずつだった。

- `df-basis-2164b5a8682b4bf589af7f18d27a4d9ca19340c790562127576a3f12d9c227f1`
- `df-basis-7218fbb90d527b0ece1345f3ceda01a0fb5aa8f756e9bcdc4ab2873efe39703f`

これらのIDは、保存STAGE1_STRUCTUREのSOURCE component basis集合（7 ID）に含まれない。停止後の[COVERAGE_DIAGNOSTIC](../../../artifacts/pr_reference_noncentral_saved_numeric/2026-11-08/COVERAGE_DIAGNOSTIC.json)は、保存instructionのliteral集合とguard key集合の読み取り照合だけで、この関係とmissing keyを記録した。Guardやweighted sumを新たに計算していない。

`weighted_parts`は、引数のいずれかのtemplate weightがNoneなら、要求alphabetを見る前にNoneを返す。そのため、

1. **全catalog basis版**は、実際に含めた2 basisのguard欠損により未資格。
2. **SOURCE RTE basis限定版**は、使用しない2 basisの欠損にも妨げられ、集計自体が実行されていない。

準備SOURCEのcoverage判定がRTE限定版には広すぎた。これを、SOURCE componentで使用されるbasisにも必ずguardが欠けるという科学的結論にしない。8行の非悪化flagもnullであり、True／Falseや「改善なし」へ置き換えない。D／frames／products／CSWAPを手で再集計して判定を迂回していない。

[計画§1のSTOP条件](../pr_reference_noncentral_reuse_20261108/NEXT_VALIDATION_PLAN.md)は「coverage不足」「固定SOURCE／入力不一致、数値前提の不整合」でmandatory STOPとしている。原SOURCE／raw／STOPを保持し、今回は判定を修正して再実行していない。Missingを0で補完したり、追加recordを取得したりもしていない。

## 3. S2：保存guard/Tの有限frontier

対象は固定metadataのD／frames literal集合と現word＋exact-literal recordだけ。PASS・strict spec・guard method文書付きrecordを使い、保存logical guardをFraction、保存Tを整数で比較した。Composite現wordは保存Q＋Gを使い、emitterを別帰属とする規約を保持した。

`g_a<=g_b`かつ`T_a<=T_b`で少なくとも一方がstrictならaがbを支配する。375 literal中、現wordが非劣frontierに残るのは332、支配されるのは43だった。

| Role別subset | Literal数 | 現wordのguard/T組が支配される数 |
|---|---:|---:|
| B′ D | 159 | 0 |
| Reference D | 160 | 0 |
| B′ frames | 216 | 43 |
| Reference frames | 216 | 43 |

Role subsetは重複し、両armのframesも共通literalなので、件数を合算しない。「Dで0」は、より小さいguardのrecordがないという意味ではない。T増加とのtradeoffを持つ非劣候補も残る。現wordが非劣であることも、全体最適性の証明ではない。

Frontierはrecord ID単位で保存し、同一sequence／同一guard/Tの複数provenanceを残した。各集合の非劣ID数の分布は、1 ID:40集合、2:19、3:163、7:1、8:42、10:19、11:91。Distinct word数とは異なる。

この結果は、保存候補に現wordより局所logical guard／保存Tの組が劣らない選択肢があることを示す。Literal共通overlay、actual adjoint、strict W、emitter／shell、SOURCE→native C、consumer／terminalの資格は別に必要。**Recordを選択・適用していないため、m17／全18roundのT削減やRPE成功は認定しない。** 両armへ同じ保存候補を与え、C0優先は不変更。新しいprecision・価格・wordを追加していない。

## 4. 検算・成果物・限界

[RESULTS.json](../../../artifacts/pr_reference_noncentral_saved_numeric/2026-11-08/RESULTS.json)は元runner出力をそのまま保存した。8集計行、全template weight、375 offer集合と非劣IDを含む。Genericなrunner statusは`S1_S2_FINITE_SAVED_DIAGNOSTIC_COMPLETE`だが、S1の科学的資格は上記のとおり未取得である。

[CHECKS](../../../artifacts/pr_reference_noncentral_saved_numeric/2026-11-08/CHECKS.json)は`SCOPE_COMPLETE_S1_UNQUALIFIED_S2_COMPLETE`。保存frontierを、guard順のprefix最小Tによる別の判定手順で全375集合について検算した。全て一致した。これは同じ有限保存候補の自己検算であり、外部独立科学レビューではない。Global Tや新しいpolicyの計算ではない。

[SOURCE_INPUTS](../../../artifacts/pr_reference_noncentral_saved_numeric/2026-11-08/SOURCE_INPUTS.json)、[ENVIRONMENT](../../../artifacts/pr_reference_noncentral_saved_numeric/2026-11-08/ENVIRONMENT.json)、[STOP](../../../artifacts/pr_reference_noncentral_saved_numeric/2026-11-08/STOP.json)も保存。GPUA100-gleap02／Python3.12.3・stdlib、64 CPU affinity、起動前MemAvailable約474 GB。既存single processで約1.557秒、CPU／RAMの独自capや共有設定変更はない。

旧V0/V1の負結果を保持し、中央precision探索を繰り返していない。B′ scalar域外、Reference vacuumのdomainと実到達角、event到達性、中央／emitter／shellを未資格のまま残した。RPE式・raw disk・半径・位相・整数shotを再計算していない。新guard／合成／word生成／provider／consumer／build／compile／simulation／samplingは0。S2で行ったのは承認済みの保存guard／T比較だけで、価格取得・総T・optimizerは起動していない。

旧SOURCE／raw／result／grant／ledger／marker／STOPと他Track・他ユーザーを不変更。正式共通指示2026-10-25、親／worktree AGENTSと研究資料の優先順位を適用した。

## 5. 次の判断とmandatory STOP

**Mandatory STOP。次担当はGPT／ユーザー。** S1のcoverage判定を「実際に参照するtemplate／alphabet」に限定する最小訂正と、同じ入力・word・norm・q/r/NのS1だけの再評価を許可するか判断する。元SOURCE／今回の結果／STOPは保存し、訂正版は別来歴にする。今回完了したS2を再計算する必要はない。

S2の43 frame literalについて有限overlay候補を検討するかは別の研究判断である。S1集計が未資格の段階で、合成・S3共役pair・domain・RPE／全18round総Tへ自動進行しない。
