# RPE D限定実行：幾何・理想branch・literal energyは成立、保存α基準は未確認

2026-11-04 JST / LOCAL。ユーザーの別途明示実行許可後、公開SOURCE
[`6942e5176d2dec115ac3c271e48f84377abda915`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/6942e5176d2dec115ac3c271e48f84377abda915)
を無変更で**160 bit・1 pass**実行した。exit0、`COMPLETED_MANDATORY_STOP`。

**全18roundの正半径・共通literal E_refへの位相条件、理想nearest-branchの全17隣接条件、
alias/lift・literal energy目標は成立した。一方、保存αのexact和が保存全体予算.05を僅かに超えたため、
固定SOURCEの総合判定は`UNKNOWN/STOP`である。** 丸めによるPASS化、再実行、予算変更はしていない。

## 対象・固定入力・再利用

Linear H4／隣接1.00 Å／STO-3G／保存legacy DF12、Partial L_D=3、単一δ=.02、
8qubit・4電子sector70、保存physical stateのnormalized ray、m0..17の保存q/r/K/N/α/β/ε。
共通保存E_refは−2.1663874486347616 Ha。δ・μ₀・E_ref・β_RPE・energy目標は元SOURCEのbinary64が表すexact rational、
ε・αは既存包含で使った保存decimalのexact rationalとした。

既存36軸のideal categorical包含・局所Hoeffding判定をそのまま再利用した。
旧結果commitは`d2e7a7db683e116243eae061c0e4c87865054e9e`、旧包含SOURCEは`f4905fff592351d33ba07b927892fb9532c99711`。
旧m17は`34840ce9bf81d985b32764d9bc9d77645ed52e59`の2軸を再利用した。
[固定計画](../rpe_branch_energy_preparation_20261104/LIMITED_EXECUTION_PLAN.md)と
[入力commit/path/blob/SHA256](../rpe_branch_energy_preparation_20261104/fixed_inputs.json)の10ファイルを照合済み。
SOURCEの5ファイルも公開commitと一致し、科学実行前・実行後を通して無変更である。

既存のArb helperと準備済み純関数を再利用した。新たな科学実装やdecoder変更はない。
`U_cat`は既にεに含まれるため再加算していない。保存β_PF/β_RTEも直接評価した系統位相差へ再加算していない。
physical stateをPF有効固有stateへ置換せず、保存中心の半径をPFの厳密単位半径とみなしていない。

## 結果

| 検証対象 | 保存結果 |
|---|---|
| 保存中心の正半径 | 全18roundで成立。最終m17の下端表示は0.572701393748848 |
| 共通E_refへの位相誤差b_m<保存β_RPE | 全18roundでstrict成立。事前固定した円板式と矩形4頂点式を評価し、全roundで矩形式を選択 |
| 理想branchのπ−2b_(m−1)−b_m>0 | 全17隣接対でstrict成立 |
| 最終円周位相誤差の上端表示 | ≤0.00000277358400720240 rad |
| literal referenceの±π境界までのmargin下端表示 | ≥3.09826490461709 rad。最終代表元のlift条件も成立 |
| literal E_refに対する最終energy誤差の上端表示 | ≤0.000138679200360120 Ha。保存暫定目標0.00015936001019904 Haよりstrictに小さい |
| 保存decimal αのexact和 | **0.050000000000000004626684** |
| 同union boundによる成功下界 | **0.949999999999999995373316** |
| 保存alpha_total=.05に対する判定 | `UNKNOWN/STOP` |
| 総合mathematical_literal_reference_decision | `UNKNOWN/STOP` |
| 実装Float64 decoder・physical ground energy | いずれも`UNKNOWN/STOP`、一様丸め上界とη_refはnull |

表の包含上端は原rational endpointを15桁で上向き、下端は下向きに表示した。
これは保存結果の表示整形だけであり、別precisionでの再評価や新しい科学再解析ではない。
全18roundの表示・選択法・各隣接margin・原本selectorは[ROUND_RESULTS.tsv](ROUND_RESULTS.tsv)。
正式なexact endpoint・判定は[原RESULT.json](../../../artifacts/rpe_branch_energy_connection/2026-11-04/run01/RESULT.json)を参照する。

### αのstrict比較と成功確率の意味

保存decimal経路のexact値は次のとおり：

    Σα = 12500000000000001156671 / 250000000000000000000000
    1−Σα = 237499999999999998843329 / 250000000000000000000000
    alpha_total = 1/20.

科学条件・幾何・branch・alias/liftの下では、各軸の座標成功事象Gから
**数学的decoder・literal E_refに対するenergy精度成功**への決定論的含意を接続できた。
その成功確率の保証下界は上記のexactな1−Σαである。
保存95%条件をstrictな固定算術で確認するには不足するため、SOURCEの総合判定を昇格しない。
この十分下界の僅かな不足は、実際の失敗率が5%を超えたという証明ではない。

元binary64 αの診断exact和も
`1888946593147858248857/37778931862957161709568`であり、
binary64 budget `3602879701896397/72057594037927936`との判定も`UNKNOWN/STOP`だった。
両経路の外向きbinary64表示は`0x1.999999999999bp-5`。decimal主判定を別の解釈へ切り替えていない。
V2の保存中心点に対するfloat64二項確率は使っていない。
union boundにround間・軸間の独立性は不要だが、各軸Hoeffdingのshot内IID・有界性は条件として残る。

## 実行・資源・保全

18幾何セル、17隣接branch条件、1 alias/energy集約、decimal36項和とbinary64診断36項和の各1回を実施した。
new mean包含、H/DF/state/operator生成・decode・solve、power/expectation、CDF/law replay、
二項PMF、sampling/RNG replay、回路build/compile、cost評価は全て0。
Float64 decoderを実H4保存平均へ新たに実行していない。
retry/resume、精度追加、条件変更、別候補・別namespaceでの追加科学実行も0。

Python3.11.0rc1／NumPy1.26.4／python-flint0.9.0／FLINT3.6.0、IEEE binary64を確認した。
CPU affinity32、事前MemAvailable 53,895,972 kB。GPU driverは利用不可だが、固定経路はCPU scalar Arbである。
q個のcandidate列挙・二項格子・行列生成を避けた小さいO(18)経路を用い、不要なworker poolは起動しなかった。
人工CPU/RAM/worker/wall-time上限や共有環境の変更はない。実行時MemAvailableは原RESULTに記録済み。
scalar評価は約0.01436秒、process wallは0.32秒、maximum RSS50,280 kB。
これは比較実測によるspeedupの主張ではない。

原RESULTのSHA256は`a80ce0376ff2e29241911fee901f327a49bfb7a33495d340caf7236370712b68`、48,099 bytes。
RESULTを編集・再生成していない。stdout/stderrは原byteを保存、stderrは空。
resource公開記録は私有interpreter絶対パスを省いた[resource_summary.json](../../../artifacts/rpe_branch_energy_connection/2026-11-04/run01/resource_summary.json)。
原`time -v`記録は私有領域にそのまま保全した。入力・SOURCE・原結果hash、scope・inventoryの技術確認は
[VALIDATION.json](VALIDATION.json)へ保存する。

固定SOURCE準備時の人工24件＋既存V2人工10件PASSを再利用し、無変更の同じ人工テストは再実行していない。
実行後は18round・17marginの欠測なし、全phase条件、alias/energy判定、α判定、null/global status、
固定入力・SOURCE・原RESULTのbyte保全と表示表の転記を照合した。科学式の再計算はしていない。
LOCAL固定SOURCEの主agentによる技術照合であり、immutable CI・外部再現・独立科学レビューではない。

## 未保証・次の担当・mandatory STOP

実装Float64 decoderの全座標成功域・全候補に対する一様angle/wrap/root/distance誤差は未保証。
η_ref、真のsector ground energy/stateの同定・精度、外部alias priorの物理的正当性、
実state preparation、実PRNGのIID、全18round実回路・compiled/noise bridgeも未保証である。
literal energyの上界をこれらの保証へ移送せず、経験的PF誤差をrigorous上界へ昇格しない。

原checkout、Codex C、他branch/worktree、SERVER2 H8、旧SOURCE/raw/grant/ledger/marker/STOPには操作していない。
既存科学SOURCE・結果は保全し、新しい報告・概要追記・研究ノート・evidence statusだけを更新する。

次の担当は**ユーザー**：保存α基準の扱いと、残るFloat64/reference条件のうち次に扱う範囲を指定する。
**Codex（別指示後のみ）**：指定された範囲の技術準備を行う。予算変更・追加科学計算・重要レビューは自動開始しない。

`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、
元仮想測定bridgeの`U_virtual_IID_upper=null`、`U_exec_upper=null`を維持。
**結果公開・remote SHA/公開成果物照合後mandatory STOP。**
