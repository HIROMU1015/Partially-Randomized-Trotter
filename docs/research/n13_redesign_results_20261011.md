# N1技術停止・N3方向保証・N2固定S修正：限定batch報告

2026-10-11 JST。**package状態はTECHNICAL_STOP、科学実行1回、再実行0回。**
N3とN2は完了し、N1は保存入力のdecode不備で停止した。
N1未取得を科学的な否定結果と呼ばない。中心テーマは未採択、次段の実行認可false、mandatory STOP。
旧GPTレビューを再実行せず、以下の新結果のみGPTへ戻す。

| 項目 | 今回の到達点 |
|---|---|
| N1 | synthetic検証は通ったが、実際の保存入力で技術停止。科学selector、PF上界、basis-onlyとの差、総費用の判断は未取得 |
| N3 | 小gap例で方向付き係数保証のみactive5を受理。同じU/cのnorm-onlyと旧scalarはactive6へ戻る |
| N2 | 固定Sのweighted-L1 fitは.0021以下。Frobenius fitのentry-L1は.0036で不適格 |
| 保存監査 | 独立保存算術660 checks、保存diagnosticの別JW/有理数再構成411 checks PASS |
| 注意 | immutable CI・外部科学再現・compiled回路の厳密機械certificate・量子資源改善の認定ではない |

## 固定条件とsource

Branch: `n13-redesign-preparation-20261011`。
準備commit: `dbffe64d4c9f7cc7cf27707f5d7f8604376848f8`。
science source/execution commit: **`bbcb0176d43bfd1f7a10a87fa8f38e200d61ec46`**。
旧基点 `ae7434452450e80ec007d65417335e18a063fa1e`、旧science `ed335a007bee6966a37f3026ac188534e526b5b9` は保持。

[元scope](n13_redesign_limited_scope_v1_20261011.md)、[execution plan](n13_redesign_execution_plan_20261011.json)、
[実行前数学確認](n13_redesign_execution_20261011.md)、[利用者認可](n13_redesign_inputs/execution_authorization_20261011.txt)。
planに179 source/input file hashとlibrary/compiler/resource条件を固定した。
科学sourceは実行前commit済みで、その後修正していない。
[一次保存](../../artifacts/n13_redesign/2026-10-11/run1/)、
[失敗](../../artifacts/n13_redesign/2026-10-11/run1/failure.json)、
[run audit](../../artifacts/n13_redesign/2026-10-11/run1/run_audit.json)、
[実行log](../../artifacts/n13_redesign/2026-10-11/run1.log)を保存する。

## N1：入力形式の事前確認が不足していた

対象は固定synthetic JW4、ν2、sector dimension6、signed DF rank2、weights(.7,−.4)、
geometry/化学basisなし、DF-prefix L_Dは該当せず全fragment S2。
T=.6、q=1/2/4/8、delta=.6/.3/.15/.075、εH=.02、εcomplex=.04、二軸failure合計.05。
44 factor候補、178 pair、gauged/ungaugedとexact/cluster/shift/cutoff対照を増やしていない。

保存packの `ungauged_eta` は、4値のplain listではなく
`{"real":[...],"imag":[0,0,0,0]}` という複素vector表現である。
新sourceはこれをlistとして `eta[k]` で読んだため、`conjugated` 内で `KeyError: 0` になった。
実際の保存schemaと異なるlist形式の合成fixtureを用いたこと、およびmetadata precheckでvector形式を確認しなかったことが原因。
最初のGaussian一体回路を構築した後で停止し、factor certificate、科学PF評価、shot selector、native compileへ到達していない。
三contextのpool taskはいずれも同じdecode不備に遭遇した。科学的な改善/悪化/上界の緩さは判定できない。

18件のsynthetic PASSはこの入力形式への対応を保証していなかった。
初期のAPI/interval処理の失敗と修正も[preexecution履歴](../../artifacts/n13_redesign/2026-10-11/preexecution/test_history.json)に残す。
今回の科学失敗後はsource修正・再実行をしていない。
将来修正するなら、real/imag vectorの明示decode、imaginary成分/長さ/finite検査、
保存packの形式をそのまま使うfixture、およびlaunch前のschema全域検査が必要。
**この修正・再実行は次の認可対象であり、今回自動的には開始しない。**

今回native IR数0、科学N1 rows0。ordinary control/global phaseの検証はsynthetic testで行った範囲に限り、
科学domainのcompile絶対作用が検証済みとは述べない。
新しいN1 selectorがbasis-onlyと何を違えて選ぶか、exactに対する総資源差は未解決。

## N3：同じU/cでも方向情報が採否を変えた

固定JW6、ν2、初期active[0,1,2,3]、sector dimension15。
非負density `.3 n0 n1`、hopping h12=.2、h03=.1、h24=.08、h35=.04、h45=.05。
係数は宣言したliteralのbinary64値を正確な有理数へ変換。
δgoal=1/250=.004、固定λgrid五点、外部2本では旧scalar、外部1本だけ方向式を使う。
真のa/E0/C/gap/vectorは保証やpromotionに使わず、全selector recordのhashを閉じてから診断した。
これは基底エネルギー幅のtaskであり、時間発展TδやQPE/active solverの量子費用評価ではない。

| input | diagonal e / 変更 | 旧scalar active数 | 同じU/c norm-only | 方向付き |
|---|---|---:|---:|---:|
| original_gapped | (−.2,−.1,.1,.25,6,8) | 4 | 4 | 4（旧scalar段階で受理） |
| original_small_gap | (−.2,−.1,.1,.25,.4,.6) | 6 | 6 | **5** |
| strong_external_coupling | small-gap、h35=.16、h45=.20 | 6 | 6 | 6 |
| near_external_energy | small-gap、e5=−.1 | 6 | 6 | 6 |

小gapで返したactiveは[0,1,2,3,4]、discardedは5。
係数trial U=−.25、外部下界c≈.36、β²≈.0041。
norm-onlyの右辺β²/(c−U+δgoal)≈.00667752443は.004を超える。
一方、固定gridのλ≈−.27、L=2λ≈−.54、κ≈.008258558932、
t=κ/(c−U+δgoal)≈.0134504217について、t(U−L)≈**.003900622297≤.004**。
採否は丸めた表示値ではなく、保存された有理数の厳密比較による。

BB†=dΓ₂(vv†) と BB†≼κ(A−LI) が、activeの低エネルギー側の結合を最大normへ潰さず評価する。
Uとcを共通にしたnorm-onlyが不受理なので、今回の追加受理をU/c改善だけに帰属しない。
λ=−.30相当および−.24相当の有効点は採択条件を満たさず、その他二点はSPD不成立。
結果を見てλ点を追加していない。

選択後の真値診断ではactive a≈−.294087596219、full E0≈−.294349788497、
差≈**.000262192278**で保証幅.004内。
保証は真値より約15倍保守的だが、5軌道を係数だけで受理できた。
数学的なSchur証明の成立と、この一件の採否改善を区別する。
新規定理、一般分子での有効性、中心テーマ採択は認定しない。

強結合ではpromotionが5を戻し、active[0,1,2,3,5]から4を除外する。
方向条件の右辺は有効二点で約.07057/.07477、norm-onlyも約.11777で不受理。
当該5軌道の実際の幅≈.009760339071も.004を超えた。
低外部energy例はc≈−.34≤U=−.25でSchur条件が成立しない。
当該active5の実際の幅≈.044337838704も目標を超えた。
これらは**同じpromotion規則が選んだ5軌道**の結果であり、全ての5軌道選択が不可能という結論ではない。
gap下界の粗さだけが原因とは扱わず、今回の候補には真の幅超過もある。

取得費用は最大5×5の有理数LDL/solveを五点、ν2の最大10 determinantsと接続pairの係数trial評価。
many-bodyの真の最低値を解くselectorへ置換していない。
一般fillingでのtrial集合増大やbit complexity、一般二体・多外部への拡張と実用性は未検証。

## N2：許容表現の回復まで

固定S=[[1,0],[1,1],[0,1],[-1,0]]、K0=[[.3,−.1],[−.1,.2]]、
J=SK0Sᵀの03/30に+.001。固定一件、LP solve1回、追加S scan0、compile0。
upper triangleのdiag weight1/offdiag2でentry-L1を最小化し、εH=.0021で判定する。

| interpretation | LP primal L1上界 | dual下界 | Frobenius fitのentry-L1 | 判定 |
|---|---:|---:|---:|---|
| intended rational | .002（1/500） | .002 | .0036（9/2500） | L1適格、Frobenius不適格 |
| stored binary64 | .0020000000000001683 | .0020000000000000018 | 約.0036 | L1適格、Frobenius不適格 |

意図された有理数ではprimal=dualで最適性を証明。
binary64ではgap=1.6653345369e−16のprimal/dual sandwichを記録し、decimalの最適値を無条件コピーしない。
非負残差変数・dual符号・係数等式・weighted slackを有理数で検査した。
別の保存監査でweighted-Frobenius正規方程式を有理数で解き、L1不一致を確認した。
charge数・S疎性だけで効率性は成立せず、bit幅、compute/uncompute、workspace、phase synthesisの費用比較が必要。
今回は許容表現の回復であり、signed-charge回路やFT資源の改善結果ではない。

## 監査・性能・保存状態

preexecution [18 synthetic tests](../../artifacts/n13_redesign/2026-10-11/preexecution/final_synthetic_tests.txt) PASS。
[独立保存監査](../../artifacts/n13_redesign/2026-10-11/saved_audit.json)は660 checks、N3 records29、N2 interpretations2。
[独立diagnostic監査](../../artifacts/n13_redesign/2026-10-11/independent_diagnostic_audit.json)は411 checks、
別のexplicit JW、CAR符号、trial期待値、class下界/coupling、matrix dominance16件、保存caseのspectral checks64件を照合。
後者は保存係数と既に閉じた採否の照合であり、新しい科学条件・候補scan・compile・LP solveは0。
N1 IR0を「科学native検証PASS」と誤読しない。

科学run wall≈2.724秒、parent CPU≈1.293秒、children CPU≈3.094秒。
parent peak RSS354365440bytes、child最大357187584bytes、一次出力84644bytes。
CPU spawn最大2、thread各1、worker AS4GiB、aggregate8GiB、wall/CPU capなし。
N1がcompile前に停止したため、主負担の実測比較やspeedup率は得ていない。
各上限内で終了、科学retry0、量子shot実行0、GPU0、新規分子load0。

全体manifest checkerは既存の `relevant_commits[7]` description欠落でFAILED。
新entryだけのvalidatorと既存87 result entries保全を別監査する。既存不整合は修復しない。
rootの既存dirty/untracked、旧Track A/B、旧representationのsource/input/results/STOPを保全する。
結果commitとremote取得receiptは公開後のpublication記録で報告する。

## GPTへ戻す判断

1. N3の小gap一件での方向情報による追加受理が、研究継続の根拠になるか。
2. 適用class・有理数取得費用・固定trialの保守性を考慮し、方法論として何を追加検証すべきか。
3. N1は科学判断不能。入力decodeと実保存形式のfixtureを修正した新versionの一回再実行を認可する価値があるか。
4. N2は補助修正として留め、回路資源の主張へ拡大しない判断が妥当か。

**mandatory STOP。中心仮説null、次の科学batch認可false、他Track統合なし。**
