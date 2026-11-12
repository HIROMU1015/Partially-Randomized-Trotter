# H4 PR：1 preparation・R2/R128の物理閉状態診断（2026-11-13）

**元C1の構造RZ数Sに対象を限ると、wire別の有限価格状態を構成する十分条件を、固定H4の実supportで確認した。** 共有active basis・phase flagと各wireの正確なpending Zを保持する方法であり、非ゼロflag/mod 2πへの置換ではない。一方、実physical kernel・全境界map・元Accountingとの遷移/報酬parityは未実装なので、**実用的な平均S評価器としての成立はUNKNOWN**。実平均S/Cや価格モデル資格を取得したとは扱わない。

ユーザーが承認した`A_PHYSICAL_CLOSURE`だけを実施した。実trajectory sampling、full-path S、平均S、build/compile/transpile、fit、平均compiled C、N/G、S0a再実行、直接価格pilotは全て0。原C1/C2/connected-cluster、S0a run01失敗/run02成功、旧SOURCE/raw/cohort/結果/STOPは不変。

正本：[run02局所診断](run02/RESULT.json)／[状態縮約の検査](REDUCTION.json)／[構成的論証](STATE_PROOF.md)／[実行計画](EXECUTION_PLAN.json)／[引渡し](HANDOFF.md)／[最終10技術test](TESTS_REDUCTION.txt)。これは作者自身のSOURCE分析と有限検査で、独立科学レビューではない。

## 固定条件・SOURCEとB接続

Linear H4/1.00 Å/STO-3G、保存tol-only DF10、sector(2,2)の保存数値state、S2/L_D1/ΔPF=.1/K2/h=.1/M12/P0。保存DF-Hがtargetで、stateの厳密ground資格は付加しない。元weighted completion、fragment/basis順・signed time・scalar一回、symmetric_directional、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、測定付きfull Hadamard X=cosine/Y=sineを維持する。

| 項目 | 固定値 |
|---|---|
| A正本 | `7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75`、S0 CONFIGURATION |
| C基点 | `d458b5e4f596d628d923263f3b2f1678960f2ffa`、固定PRICE_REQUESTS/SCIENCE_PLAN |
| B API | `af284d853890155bacd19e61850c196bd1d5c660` |
| B正式接続 | `75b5c89bddf8a1ffa4a8ac99b2c06d6e05fa012d:docs/research/pr_rpe_s0_v2_connection_20261113/metadata01/V2_CONFIGURATION.json` |
| B configuration | `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。前回Cの暫定metadataとm0/m6の全identity・physical candidate・law_bindingが一致 |
| H | `c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0` |
| state | `9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b` |
| snapshot | `12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96` |
| preparation | `b99720d6189a9975a46f064ba895d2f83514671e668ce42c34a4d71e834568d7`。元C1/H4 Mq provenanceの値と一致 |
| compiler | `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。今回compileなし |

共通κはA正本から次をcopyした前回requestをそのまま参照し、近似値から再生成しない。

```text
13112614441153360648422410673503886836581176429267589536059228160/539496659426909122096437389293855970805348475396548904610992201
```

| round | physical ID | T | qPF | R | piece/allocation | event位置数 |
|---|---|---:|---:|---:|---|---:|
| m0 | `53e00d8b07d1547d19a7` | .1 | 1 | 2 | S2/1 piece/[2] | 2 |
| m6 | `2a8c6718220eec59d419` | 6.4 | 64 | 128 | S2/1 piece/[128] | 8192 |

bind内T=.8は継承された参照metadataで、実Tは上表candidateを正本とする。Bとの照合結果は価格recordではない。元V1 packetやC1価格IDの付替えはしていない。

## 実行経路と保全した初回エラー

初回SOURCE `c5e2e5046f674277778b224329301c04a52eec8f`は元law APIへ`seed=None`を渡し、最初のdescriptor生成前に停止した。[初回STOP](run01/STOP.json)・[実行log](EXECUTION.txt)を保持する。元APIはmetadataでも整数を要求するため、既定値0へ修正し、RNGを起動せず整数APIの人工testを追加した。[TECHNICAL_FIX](TECHNICAL_FIX.md)参照。

修正版SOURCE **`eeccdf18ee839ccddb824afbc5c8d0aa8b52b7b6`**でrun02を完了した。元S0aの`Context/no_circuits`を再利用し、元C1 input hash/preparation/H/fragment/source照合と、QuantumCircuit生成/transpile禁止を維持した。既知C1保存path/input領域には完全なstandalone preparation objectがないため、保存snapshot/receiptと元loader/prepareから既存descriptorを復元した。SCF/DF/Hamiltonian/stateは新生成していない。

対象は同じpreparation hashの**1個**。初回エラーのため復元処理は2回起動したが、成功したlaw descriptorはR2/R128の**2個**だけである。保存event列・S0a path・compiled cacheは読まない。Counterの元native展開を使う局所tracerは`_native`を収集用に置換し、full structural word/finish・実S計数を呼ばない。

その局所結果を固定Git objectへ保存後、SOURCE **`2b3b0ee7ffea92de25a2c0c549d64f258c34a611`**の`reduction_evidence`で十分条件を解析した。使用した正確なanalysis SHAは[REDUCTION](REDUCTION.json)の`analysis_source`を正本とする。追加のpreparation復元・law生成・実価格評価はない。run02の当初UNKNOWN判定を改変せず、後段の縮約論証と分けて記録した。

## 実supportと2lawの結果

復元したrandomized tailは**324非identity component・9 basis**。各basisに36 componentが属する。component全体には負符号180、正符号144があり、1/2wire Z/ZZ support、full basis operations・qubit順・coefficient/probabilityを保存した。`extract_identity_phase`を維持するので、人工identity rotationを物理反例にしない。

λ_R=2.8629570945298686、tail hashは`172c0572fc14a56d285439ebd01820494d04bf35cd45cb55583667df4eaaf91e`。以下は**今回承認されたlaw metadata診断値**で、平均価格ではない。表示値を入力へ戻さず、JSONのbinary64値/hexを正本とする。

| law | τ=λ_R(.1/R) | P(order2) | rotation angle order0 | rotation angle order2 |
|---|---:|---:|---:|---:|
| R2 | .14314785472649344 | .010051742947234783 | .14218194063365502 | .04767978754827417 |
| R128 | .00223668523010146 | .0000025013685907457728 | .0022366815002455784 | .0007455616052239744 |

元`make_rte_config/finite_rte_distribution`のK2をそのまま使った。合法eventはrotationとorder個のproductを同じcomponent集合から選び、products→rotationの順で作用する。ideal descriptorのevent確率は元`weight[k]/Z × ∏(p_i/fsum(p))`。全324³のorder2 eventや全event列は生成していない。実samplerはPCG64/float64 CDFのままで、ideal lawとの距離・独立性・全law期待値は新たに資格していない。

## 状態依存性と判定

| 寄与 | 実supportで確認したこと | 縮約の扱い |
|---|---|---|
| pending basis | 9 basis×正逆の18wordはいずれも全system8wireを非RZでreset | full operations/inverseを保持。literal逆対で消えた場合はZを変更しない |
| pending Z | 合法applicationは各wireでreset又は完全untouched。resetなしZ加算なし | 各wireの正確なbinary64 pending値を保持し、元順加算/literal zeroを使う |
| tail phase | order2のπ＋負productのπのみ。R≤128では非負有限和 | 0 iff order2未出現。tailを分割せず、phase flagを全wire marginalへ共有 |
| phase境界 | 最後のrotationはcontrol Zをreset。reverse headはcontrol RZより先にcontrolをreset | phase正値はRZ1件としてflushされ、その大きさは以後Sへ戻らない。moduloは使わない |
| forward/outer | forward最後のDF diagonalは全systemをreset。reverse各directional diagonalは全9wireをreset | tail entryとouter出口が固定。q1/64を展開せず有限な境界mapとして扱える条件 |
| scalar/XY/terminal | −cT一回、Y sdg、H/measure、finish一回という元順 | 固定outer出口に対する別suffix。途中flushを挿入しない |
| full report/compiler | depth/role/global phaseの保存reportと実compilerはSだけとは別 | 縮約対象外。全report/compiled Cの同値性を主張しない |

SのRZ発行rewardはwireごとに加算できる。native multiwire命令は触れた各wireをflushするが、Sの分岐は他wireのZ値に依存しない。basis/phaseの共有情報を残したwire marginalなら、wire間相関を保持したまま期待rewardを足せる。独立性を仮定して分散やCIを合成してよいという意味ではない。

構成的な全合法continuationの論証は[STATE_PROOF](STATE_PROOF.md)。application境界のpending Z集合は、固定tail entry、basis inverse出口、touched application出口の有限集合で閉じる。実データ由来の集合サイズはsystem wire順に5/5/5/5/5/5/6/4。共有basis＋entry/phase flagを含めた保守的tail state enclosureは**100/100/100/100/100/100/120/80**。これは到達state数・完成kernel全体のサイズ・性能実測ではない。entry/close/head/outer/terminal mapとcontrol rewardは別途必要である。

判定を分ける。

- **確認済み**：この固定support・R2/R128の元S observableに対する有限wire-marginal価格状態の十分条件とSOURCE上の構成的論証。人工identity反例や有限float registerという一般論から導いていない。
- **UNKNOWN**：実physical transition/reward kernel、全boundary map、元Accountingとのparityと確率mass/精度、実用性能、実E[S]。今回Markov行列や実平均Sは作っていない。
- **対象外/未資格**：full Accounting report、compiled C、長q/R一般化、sampler対ideal距離、compiler残差、全law価格・共有covariance/CI、N/G最適化。

## 人工テストと技術実装

新moduleは固定B照合・local native tracer・wire reset/suffix依存性・十分条件の検査だけを追加し、元Accounting/MeanKernel/stream_structureは変更していない。`projected_step`は元順のwire別Z-fold遷移を表す小関数で、人工テストでのみrewardを評価した。

最終**10 test method PASS**。literal binary64/zero、basis inverse/順序、controlled product/rotation reset、元Accountingのpending値との一致、metadata用整数seed、正式Bとの一致、実診断未呼出し、2πを0へ同一視しないphase barrier、相関eventのwire marginalを既存MeanKernelへ渡した人工全列挙一致を確認した。初回7件、seed修正8件の記録も保持。元C1/S0aの既存実装parityは保存証拠を再利用し、重複実行していない。

計算は324 component/9 basisと固定K2のlocal alphabetに対するもので、qRの全event列を保持しない。将来はevent kernelをorder0 rotationとorder2のproduct²→rotationの混合へ分解する候補がある。ただし今回確認したのは状態縮約条件と人工接続だけで、実H4の速度向上や平均計算完了とはしない。

## 次段階とSTOP

次は同じ1 preparation・2lawで、物理wire-marginal kernelとentry/close/outer/XY suffixを構成し、元Accountingの局所/境界遷移と完全照合するK0が有効である。現在のdescriptor/local効果を再利用し、sampling/compile/fitは不要。実supportでのtransition/reward行列作成・数値parityは次の科学範囲として固定する必要がある。

次回の承認案は、K0と、K0通過時に限り**実平均Sをm0/m6×X/Yの4値だけ計算するK1**を一括scopeにする方法である。lawの対象・binary64演算/確率伝播精度・failure基準を固定し、Sとcompiled Cを区別する。今回はK0/K1を開始しない。ユーザーが両段階を一括許可した場合、K0成功後に同じscopeの許可を取り直す必要はない。

平均compiled Cには同条件のE[C−S]がなお必要であり、C1のq8/16,R4/8係数をR2/R128へ移さない。前回の直接価格D（2候補×normal2path×XY、最大8 wrapper）は代替のdevelopment案として保持するが、今回は不要な追加実行をしない。選ぶ場合は新SOURCE、seed/cohort・元sampler/law・同path S/C・記述統計・STOPを固定し、別途sampling/compile等を許可範囲に含める。n2を全law平均の精度保証としない。

今回の結果はBへ**診断metadata**として渡し、平均compiled価格recordとして供給しない。C1 MODEL false、mean S/C/G、law距離・共有covariance/CIはnullのまま。成果をcommit/push・remote SHA/tree/主要成果物照合後**mandatory STOP**。

次の担当：**ユーザー**がK0とK1等の次の有限科学範囲を判断する。**Codex C**がそのSOURCE/指標/STOPを固定し、明示許可内だけを実施する。**Codex B**は固定V2の下で診断metadataを参照し、価格値を欠測に保つ。
