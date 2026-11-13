# T2 scalar STOP後の最小科学案（未承認・未実行）

今回T2はscalar完全一致gateで停止し、normal sampling/compileは0だった。actual scalarは例外に保存されていない。元固定SOURCE・input/expected scalar・STOPを維持し、科学的な無断retryやguard緩和を行わない。

## 次の最小案：preparation-only一回

| 項目 | 提案する有限範囲 |
|---|---|
| 対象 | A a9ab固定PR-LD2/m12の同H/state/snapshot・prefix2、元λ/scalar規則 |
| SOURCE | 8291f16ceb6b88cc8a43dbe508db419b9a26f373の固定入力loader/D runtimeを基準。例外時の診断出力だけを足した新caller commitを実行前に固定する |
| 科学call | input復元1回、同じNativeRuntime.context/prepareを1回だけ |
| 上限 | sampling0、生成event/path0、native evolution/full wrapper build0、compile0 |
| 記録 | ham.constant、prep.extracted_identity_coefficient、その元順binary64和、expected scalarのdecimal/hex、差/ULP診断、実λ・fragment/preparation/tail/component/basis hash・元scalar定義/SOURCE |
| 実装 | 既存guardの例外直後、同一呼出しのframe値又はguard前の限定観測で保存。二度目のprepareを呼ばず、scalarを上書きしない |
| 成功条件 | 不一致の実値・定義差を記録できること。科学的同値性や価格資格の認定ではない |
| STOP | 一回の記録完了又は任意の不一致/失敗でSTOP。tolerance化、先行sampling/compile、別prefix/roundへの変更なし |

これは今回unused compile枠の継続ではなく、ユーザーによる**別の明示承認**を要する新しい有限診断案である。今はSOURCE比較の静的読解までに留め、保存arrayからscalarを再計算したり、prepareを再実行したりしていない。

差が小さいと判明しても、直ちに無害としない。元scalar一回会計、科学的な回路定義、full-wrapperの価格/数値簡約への意味を区別し、A/C/Dの元出典とのbridge又は条件修正が必要かユーザーが判断する。現時点で特定の丸め原因・tolerance・新scalar値を採択しない。

## その後に必要な別scope

scalar整合条件を解決した後、ユーザーが明示的にT2再試行を選ぶなら、同LD2/m12/normal fresh一path/index0/X,Y/最大2 compileを新SOURCEと独立run directoryに固定する。再samplingのrole/cohort/seed扱いは、その時の有限指示で固定する。旧run01・消費済みattempt・STOPを書換えない。今回は一度もsampleしていない事実も保持する。

平均価格の取得へ進む場合は、ユーザー＋B/Aが同target/decoder/全事象lossのN資格又は有効N集合、候補比較margin・許容regret、価格confidenceと同時familyを選ぶ。C/Dはtarget ideal law対sampler law、同候補のS適合性、rare/model/systematicの未知項と、今後取得する実測費用を提示する。今回の4.7秒は巨大価格取得費用を表さない。

その入力に応じ、必要セル・normal fresh n・軸・SOURCE・最大compile数・mean intervalの前提を具体的有限範囲へ固定し、ユーザーが別途明示承認した場合だけ進む。旧2%/95%・n8・全52軸を自動採択せず、今回の一pathを独立holdoutとdevelopmentの両方へ流用しない。

巨大compileの実行可能性は今回未検証である。Sや未証明outer-windowを正式Cの上下界・m12 meanへ昇格しない。**ユーザー**が次の準備診断の有限範囲を判断・承認し、**Codex D**が承認後だけ同context一回の診断を実施する。**A/C**は元scalar出典・bindingを照合し、**B**は価格未知を維持する。現在はmandatory STOP。
