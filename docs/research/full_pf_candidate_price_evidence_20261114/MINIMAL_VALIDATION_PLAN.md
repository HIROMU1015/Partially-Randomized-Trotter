# Full PF3候補：保存価格を優先した最小検証案

**未承認・未実行。** 今回はmetadata接続だけ。価格値の受入数値分析、fit、holdout予測/残差、direct compileは別の明示科学実行承認後。S2/.2、標準Y4/.4、M5 best/.8を全て比較対象として残す。

## 1. 対象domainと保存価格の受入

候補・全roundの正本はA3ae055faのC_HANDOFF、consumer configuration IDは [COVERAGE](COVERAGE.json) の3個。元H4 DF10/全one-body＋fragment/元順序、同H/state/snapshot、h=.1/M12/P0、同scalar、full controlled measured X/Y wrapperを維持する。

最小追加件数は、D e7dca5f7で確定済みの元native lowering/CompilerSettingsを選び、保存5round/10軸を受け入れられる場合のもの。DのRECEIPTS/source_checksを再利用し、consumer側のidentity/law/profile_extension/scalar/sourceを明記する。保存価格のcopy・適格性確認・校正用数値利用のscopeを承認対象に含める。旧signalのeligibilityやreference pair欠測を価格のRMSE資格へ混ぜない。

別のloweringを選ぶ場合は、同compiler hashでも旧価格を控除しない。Dが同じnative処理を使うとSOURCEで示せる場合だけ移送を検討する。証拠がなければ下表の校正/holdout全9round・18 wrapperが新domainでの識別案となる。8件案から18件案へ自動拡張しない。

## 2. 同nativeを使う4新round・最大8 wrapper案

formula/actual幅ごと、X/Yごとに独立の `a+b qPF` を検証候補とする。2つの異なるqで2係数は識別可能だが、全qの正確性は導けない。calibrationとholdoutの役割は取得前に固定する。

|公式/固定actual Δ|校正入力（保存）|追加校正セル|独立holdout（新規）|
|---|---|---|---|
|S2/.2|m3 q4、m4 q8|なし|m5/T3.2/q16 `987ba8031fd5b790cf59`|
|Y4/.4|m3 q2、m4 q4|なし|m5/T3.2/q8 `0de9142bd399c82598c8`|
|M5/.8|m3 q1|m4/T1.6/q2 `8cf3ab565283bebfbbba`|m5/T3.2/q4 `aca80e6e14487cde159e`|

全4セルはX/Y各1回の決定論full-wrapper compileで最大8件。PR samplingは0、量子shot Nは0。Full PFでは統計的trajectory cohort数nの設計は不要で、固定seed_transpiler=17と両軸のreceiptを保持する。cache hitやX/Y同値による件数減少は事前に仮定しない。偶然同じ整数価格でも軸を結合しない。

科学実行時の順序案：

1. SOURCE/input/preparation/scalar/full-wrapper/profileを照合し、保存5roundの数値受入を行う。欠測・不一致はSTOPし新compileへ自動fallbackしない。
2. M5 m4のX/Yを元 `h46_long_time_native.compose/cost_trial` と既存検査・compile/cacheで取得する。native SOURCEはD e7のSOURCE_CATALOGから固定する。新SOURCE/callerをDが固定し、既存SOURCEを上書きしない。
3. 各formulaの校正2round×両軸で計6 axis-modelをfit。保存係数を変更せず、別モデルとして係数/domain/誤差指標を凍結する。q0=0や原点通過を強制しない。
4. 凍結後にm5の3セル×X/Yを取得し、6軸の予測との差・相対誤差を既存処理で比較する。holdoutをfitへ混入させない。
5. 不一致、未知precision、native正当性gate不成立では結果を保存してSTOP。採択thresholdを緩和したり、再fit・別q・direct全roundへ自動移行しない。

fit・6軸のholdout予測/比較は上記を許可する次の科学scopeに含める必要がある。全78軸の新モデル価格評価、正式採択、N×C/Gはこの初期案に含めない。

## 3. この初期検証で埋まらない部分

- **early actual幅**：要求幅より狭い6round（S2 m0、Y4 m0/m1、M5 m0/m1/m2）は別domain。q1でも価格不変とはしない。必要なら該当6round/12 wrapperのdirect又は別の角度移送根拠を有限scopeとして検討する。自動追加しない。
- **同actual幅の低q**：S2 m1 q1/m2 q2、Y4 m2 q1は校正q範囲より下。m5上方holdoutでは検証しない。短い3round/6 wrapperを別候補にできるが、必要性はBの利用範囲/誤差budgetで判断する。
- **長q**：m6〜m12は未検証。m5の一点一致からq2048/1024/512へ資格を移さない。SOURCEによる境界閉包の証明、又は必要domainの独立directと有効な誤差包絡がなければUNQUALIFIED。m12を最初からcompileする必要はない。
- **全13round**：モデルなしdirectだけで完全coverageにするなら現保存5round以外の34round/68軸が不足する。これは差分表であり、全件compileの提案/承認ではない。

他qのreceiptでもactual幅が違えば同じa/bへ入れない。固定Tでqだけ変える検証は幅も変わるため、固定actual幅のq検証と区別する。P0 compilerの角度依存最適化・signed piece/outer境界・scalar/wrapper固定費をmodel外で二重加算しない。

## 4. 必要精度と正式Gの成立条件

Bが同エネルギーtaskに対して候補別の適格N又は許容N集合、比較family、許容regret/識別marginを供給するまで数値価格精度はUNKNOWN。3候補のNを共通と置かない。PRの統計αを価格CIへ転用しない。

有効な同時price区間 `[L_i,U_i]` と非負Nがある場合、既存B costは `[Σ N_i L_i, Σ N_i U_i]` を使える。点最小と区間優越は別。モデル誤差の同時上界b_iがある場合だけ `|Ghat−G| ≤ Σ N_i b_i`。差の識別には候補双方の誤差を考慮する。同じ価格変数を本当に共有する場合のみNの差で重み付けできる。ここではこれらの数値評価はしない。

Full PFの保存direct整数価格は固定compile条件では決定論値であり、n=1のcohort標準誤差を付ける対象ではない。一方、q-affineのmodel誤差は系統的で、校正点の一致や単一holdout誤差を全roundのb_i/CIと認定できない。XY・formula・qの共有モデル誤差を無根拠に独立と置かない。

正式Gには全対象round/axisで採用可能なdirect又は適格model、SOURCE/lowering/scalar/wrapper/P0の統一、BのN資格が必要。価格のないdomainを0埋めしたり、signed piece数を価格下界と呼んだりしない。

## 5. 将来のA追加幅

Aから同形式のconstructor・全round physical/canonical ID・exact requested/actual幅・q/T・formula weights SOURCE・H/state/snapshot/scalar・profileを受け取り、既存receiptを再照合する。要求幅labelでkeyを作らず、同じphysical回路＋全input/source/profile/lowering/axisでのみ再利用する。

新要求幅でも同physical roundが得られれば旧consumerと新consumerのIDを両方保持し、元receiptを新結果へ付け替えない。actual幅がroundごとに変わる場合、同じq-affineへまとめない。8次PFや新候補は自動追加しない。

## 6. 次の担当とSTOP

ユーザー：元native価格scenarioを使うか、別loweringか、保存受入/fit/予測/最大8 compileという初期科学scopeを実行するか判断する。必要精度はBの条件と合わせて確定する。

D：明示許可された場合だけ、元回路/科学gateを再利用し選択セルの価格とreceiptを取得する。C：許可された保存値受入・fit/holdout比較に既存モデル処理を使用する。B：別途許可scopeでN・価格の資格を照合し、全round入力が揃った場合だけGを評価する。

今回は全て未実行。公開・remote照合後mandatory STOP。
