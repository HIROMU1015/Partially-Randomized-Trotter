# scalar quantity取り違えを解消する最小案（未実装・未採択）

[REPORT](REPORT.md)／[DIAGNOSTIC_RESULT](run01/DIAGNOSTIC_RESULT.json)。原因は、Aのtail identity scalar s_RをDのtotal scalar c_H+s_Rのexpected欄へ渡していたこと。実s_RはA descriptorと0 ULP差で、c_H=2.29310124732、元contextのtotal=2.515477156730702。診断1回で停止済み。以下は修正提案であり、今回の新値はruntime/requestへ採用していない。

## 最小変更の候補

1. `ld12_price_requests._request`の境界でquantityを区別する。元A `tail_descriptor.scalar` はtail-onlyとして保持する。
2. 同じ保存H/snapshotに結び付いたHamiltonian定数を明示SOURCE付きmetadataとして供給し、**元と同じ順序 `c_H+s_R`** でexpected total scalarを作って既存guardへ渡す。今回得た数値を理由なくhard-codeする修正はしない。既存のpassive request生成に実preparationを埋め込まない。
3. 型名/欄名/出典の短い記述でtail identityとtotalを明示する。必要ならtailの照合欄を分けるが、guardの無効化、tolerance化、scalar再分配、新しいcompiler/会計層は不要。
4. 人工非zero c_Hでtail-onlyをtotalへ渡すケースを拒否し、正しいtotalとsource/context key、scalar一回、X/Y quantityをmetadata/mockで確認する。Cのscalar転送も正しいquantityへ結ぶ。旧C5ccのbinding差分を直すだけでは、今回のD selectorの誤接続は解消しない。

これが最小の案だが、今回は実装しない。共通selectorはLD1にも使われるため、修正の技術対象範囲は明示する。今回LD1のpreparationや別roundの科学実行はしておらず、そこへ新しい数値を移植・実測済みとしない。

## 科学契約・価格への影響

元native composeのscalar会計はc_H+s_Rで一回であり、今回それを変更していない。修正案はその意味にexpected欄を合わせる接続の訂正で、PF分割/fragment順/q/R/K/law・new ZZ/compiler・full measured wrapperを変える案ではない。

ただし、歴史的T2のrequestに固定されたexpected値と、後続runのprice input/SOURCE/context keyは変わる。これは既存runの後付け訂正ではなく、**新SOURCE・新requestとして固定すべき契約入力の変更**である。古い52軸coverage/PLAN、旧T2 run01、今回の診断原本/STOPを上書きせず、必要な選択requestだけを新たに生成する。B canonical scheduleを自動で再定義しない。

s_Rだけをnative totalとして使う案は採らない。それはc_Hのcontrolled phaseを落とし、ancilla分枝の相対位相を変える。小さい丸め差だから無害という修正でもない。source-boundなtotalの接続を直しても、巨大回路の実行可能性・compiled価格・全law期待値・信頼区間は未取得のままである。旧P1/P2の価格を再解析・再ラベルしていない。

## T2再実行の条件

- ユーザーが上記のquantity/source接続修正を承認する。適切な既存codeと今回の診断値/元H sourceから対応を固定し、人工確認後に新実行SOURCEをcommitする。
- ユーザーが**科学実行を別途明示承認**する。PR-LD2/m12、同H/state/snapshot/S2/ΔPF=.1/K2/元exactκ/P0、normal fresh1 path/X,Y/最大2 compile等の有限範囲、cohort/index/seed規則、出力先を新runとして指定する。今回はこの承認を得たとは扱わない。
- 旧scalar完全一致gateを維持し、入力/λ/fragment/preparation/sourceを照合する。失敗時の値は同一呼出しから保存し、無断retry/別seed/別round/別候補へ進まない。
- Cへの結果bindingはtail/total quantityと新request/context keyを対応させ、path Cとmeanを分ける。T2が成功してもmean/CI・fit/外挿・N/RMSE/Gへ自動進行しない。

**ユーザー**：修正とT2新runの有限科学scopeを別途判断。**Codex D**：承認後だけ接続の最小修正と、明示された科学範囲を実施。**A/C**：descriptorのtail-only定義とsource-boundなtotalの扱いを共有し、**B**はcompiled価格未取得を維持する。今回の作業は公開照合後 **mandatory STOP**。
