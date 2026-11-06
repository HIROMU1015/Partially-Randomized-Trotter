# N3結果の事前判定方針

2026-11-06 JST。**結果未閲覧で固定する判定文書。新しい科学条件の採択・実行指示ではない。**

## 1. 固定資料と適用範囲

設計のbaseは `a167effd4a1192d9806ea0bde8e0f7af71a1ef55`。
同commitの[N3草案](../o2_a1_n3_connection_20261106/N3_EXECUTION_CONTRACT_DRAFT.md)は未採択時の履歴である。
正式条件は、ユーザー指定の公開済みcommit
`03928ce55bc5a1bfd070aa9f0a6bf8a990a0ff25` の
[EXECUTION.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/03928ce55bc5a1bfd070aa9f0a6bf8a990a0ff25/docs/research/o2_a1_n3_science_20261106/EXECUTION.md)。
取得したGit blobは `251eedffa2d55785adafcfec6e730fa19d230a34`、
[本文の保存コピー](references/N3_EXECUTION_PUBLISHED_03928ce5.md)は原文のまま保全する。

現在の実行結果・途中journal・marker・lock・STOP・ledgerは読まない。
本書はN3開始前の登録とは称さず、**並行実行中に結果を知らずに固定した方針**とする。
公開時点でN3が完了していたかは調べず、N3完了前の登録とも断定しない。
結果受領後は本版を上書きせず、この版のcommitを引用して別の結果報告を作る。
欠測や不利な結果を理由とするε、α、比較対象、費用範囲、判定閾値の変更はしない。

## 2. 正式契約から確認した条件

対象は保存linear H4、1.00 Å、STO-3G、legacy DF rank12、8 system qubits、
全4電子number sectorのN2保証ground、分割 `L_D=3`、`m0/q1/δ=0.02/r=1/K=2`、
`coefficient_atol=0`、X/Y、理想logical gateと保存DFモデルに限る。δ掃引ではなく一点である。

| 項目 | 採択済み内容 |
|---|---|
| 再利用する入力 | N1 `744e1a09661ebba6841fe5faf456f88c41c47780`、N2 `5e9bd08068738cdf1070bdc43bee8194b113ae33`、engine `fdf00cee3c87d804845497afc89ac6a71f3b3f59` |
| 一次target | N1 actual finite-grid Q、selected full列、N2保証groundによるattenuated複素meanのX/Y座標。w=W=1 |
| 共通精度 | ε_X=ε_Y=1/8。energy、元の未近似Hの信号、18-round RPEの保証へ移さない |
| 共通confidence | 4方式×2軸を同時にα_diag=1/20。各方式軸はα_diag/8、calとprodへ半分ずつ、各H正層へ均等分割 |
| A1 | 固定3control、5層×64 conditional train/axis、intercept付きOLS一回、rank4、D_CV=2+Σ|λ_j|≤4。cal/prodはunconditional Q |
| B0 | unconditional Qのraw mean、train/pilot 0 |
| B1-O | order n0/n2の全正層。各正層64 pilot/axis、exact質量で合成 |
| B1-OL | 固定5正層の直接条件付き取得。各正層64 pilot/axis、exact質量で合成 |
| cal/prod | 各評価単位512 fresh cal、cal/pilotからJ=8配分と十分Nを凍結。prod合計≤8192/method/axis |
| 数値誤差 | 保存ENGINE_ERROR_CERTIFIEDの厳密な有理数b_engine。丸め表示や未採択1/32で置き換えない。exact νなら追加centering項0 |
| 乱数モデル | 独立uniform53-bitモデル、PCG64/SeedSequence seed20261106、path64word/trial・別bit word1。PRNG独立性の数学的証明とは称さない |
| 主価格 | 同selected full no-prep Hadamard interrogationのcompiled RZ。Qiskit1.3.0、basis=(rz,sx,x,cx)、opt2、seed17、backend/coupling/layout/routing指定なし |
| 実行順・中断 | A1 X/Y train→B1 pilot→各方式軸cal/計画/prod。最初の失敗でcampaign全体を中断し、他方式の救済実行をしない |

同targetの成立にはH/partition/preparation/tail/Q/policy/state/sector/order/phaseとcompilerのbindingを要する。
N1のidentityはbaseのmanifest該当entry、正式SOURCE/profileと突き合わせる。値の再生成はしない。
actual countによるQ(ω|h)、全正層、X/Y readout、bit0→+1/bit1→−1を保持し、event phaseをbitへ二重適用しない。
学習はconditionalでも、A1評価をbalanced 5層平均へ変えてよいことにはならない。

## 3. 精度保証と観測一致を分離する

方式軸ごとに、固定λ、独立fixed-n cal、所定range、一様engine誤差、外向き丸めの十分N、
全正層のquota完了、stream分離が保存証拠で成立するときだけ `CERTIFIED_RETURN` とする。
層別の場合、各統計幅e_hに対しΣp_h e_h≤ε−b_engineを確認する。
同時cal保証の下でprodのBernstein/Hoeffdingを適用する既存経路を用いる。
配分候補の選択は同時に成立するvariance上界とprod前の情報だけを使うので、prod値を見た選び直しは許さない。

これは「返された推定値についての確率的誤差保証が契約から成立する」という判定である。
真のμを今回新たに計算して実誤差を認定する判定ではない。方式間の点推定が近いことだけでも証明にならない。
全方式両軸が返るときに、条件付きモデルの下で同時座標保証を主張できる。
複素絶対誤差へ表現し直すなら座標の二乗和による幅を明記し、同じ1/8だと言い換えない。

STOPを含む一般の実行では、保証の意味は
`P(返却され、かついずれかの保証幅を外れる) ≤ α_diag` である。
`P(完走 | ...)` や `P(幅内 | 完走) ≥ 1−α_diag`、必ず返却する確率を自動的に保証しない。
有限word capやguardのため、完走seedだけを選別したcoverageも主張しない。

## 4. 全実現RZ会計

各方式mについて、X/Yを足した次を主費用とする。

\[
C_m^{obs}=\sum_{b\in\{X,Y\}}\sum_{s\in\{train,pilot,cal,prod\}}
\sum_{i\in\mathcal I_{m,b,s}^{executed}}c_b(\omega_i).
\]

pilotは既存型ではtrain stageに入る。意味上のpilot内訳を別列に示すが二重計上しない。
A1のconditional train、B1のpilot、全cal、全prodを含め、B0に不要なtrainを加えない。
同pathの再出現・cache hitでもshotごとに同じRZを課金する。compile回数はcache miss等の古典実費として別記する。
状態準備だけを量子指標から除外し、D half steps、scalar/identity/event phase、制御、ancilla、readoutを含む。
shot数が違うので、状態準備除外を「共通だから相殺」と説明しない。

中断時はparentとworker末尾を含む全開始・取得bit・価格・未回収callを保存証拠から区別する。
unpriced bitがあれば総RZはUNKNOWNであり、既知価格の和は非負費用の**既知部分の下限**にすぎない。
未取得shotは実現費用へ足さず、予定quotaは別列に置く。中断で省略された他方式を0費用の完了方式にしない。
古典generation/fit/cal・配分・補正/engine simulation/compile/cache・lock待ち/setup、wall/RSSは別vectorとし、秒とRZを足さない。
実engineシミュレーションのCPU費を量子online費へ混ぜず、欠測はUNKNOWNとする。

## 5. 事前判定表

判定は `(契約整合, 返却/精度状態, 全費用の完全性, 比較結果)` の組で保存する。
主比較は**A1対B1-OLの両軸合計**。B0/B1-Oは補助比較、軸別値は内訳とする。
契約どおり全4方式両軸の保証が揃うまでは、正式なcampaign主比較を確定しない。
部分的に両方式だけ揃った値も未完了campaignの内訳として保存できるが、主結果へ昇格しない。

| 状態・条件 | 事前の分類・許される解釈 |
|---|---|
| 全保証・全価格が揃い、C_A1^obs<C_B1-OL^obs | **有利（当該campaignの実現RZ）**。差・比・train/pilot/cal/prod内訳を併記 |
| 同じ成立条件でC_A1^obs>C_B1-OL^obs | **不利（当該campaignの実現RZ）**。期待費用・一般的アルゴリズム劣位までは含めない |
| 同じ成立条件で同額 | **同額**。有利/不利を作らない。0費用分母なら比は未定義 |
| A1がB0/B1-Oを下回るがB1-OLを下回らない | 補助baselineに対する実現改善。強いbaselineに対する改善の達成とはしない |
| 保証済み精度は揃うが価格欠測 | 精度の達成を報告、費用比較は**判定不能** |
| 十分Nが8192を超える/正層quota不能 | **未完了（固定scope内の十分条件不成立）**。真の必要最小shot数の下界、推定器の不可能性、アルゴリズム劣位とはしない |
| rank不足 | **未完了（固定学習の識別性guard）**。minimum-norm解の出力を学習成功としない。control削減/再fit禁止 |
| D_CV>4 | **未完了（採択range guard）**。CV不偏性の否定とは別で、許可された保証経路では返却しない |
| 実bit/CV値が理論range外、stream/target/価格profile不一致 | **判定不能（契約/保証前提不成立）**。影響した方式軸の保証を付与しない。原因が不明なら断定しない |
| word cap・資源/環境・価格失敗・途中STOP | **未完了**、費用は既知/UNKNOWNを併記。整合性も失われていれば**判定不能**を併記 |
| A1 trainでSTOPしbaseline未取得 | **campaign未完了・方式間比較不能**。実行順による打切りであり、baseline勝利/A1劣位とはしない |
| 片軸だけ完了 | 当該軸の返却保証を限定記載。複素target/両軸費用の達成は未完了 |
| 保証前提は成立し、返却値が別途承認済みの真値検査で幅を外れる | その検査のtarget/誤差を明示して失敗を記録。1件だけで確率定理の反証とも、成功seedだけの再実行理由ともしない |

本版では非ゼロの最小改善率を新設しない。厳密整数RZの差の符号だけを実現比較の規則に用いる。
会計に不確かな共通費や欠測があれば符号を確定せず、成立する下限等だけを記す。
「有利/不利」は上表の限定した意味であり、実用的重要性や期待費用の検定結果ではない。

## 6. B1-OLが強い対照である理由

固定5層内でg1/g2/g3は定数である。各層の条件付きbit meanをμ_h、varianceをσ_h²とすると、固定λで

\[
\mathrm{Var}_Q[B-\lambda^T(g-\nu)]
=\sum_h p_h\sigma_h^2+\mathrm{Var}_{h\sim p}[\mu_h-\lambda^T(g_h-\nu)].
\]

B1-OLの独立層別推定のvarianceはΣp_h²σ_h²/n_h。
5層の既知質量を使った平均に層内定数CVを足しても、合成後のCV補正は相殺する。
この恒等式はB1-OLの費用込み一様優位を証明しない。整数quota、価格、全段階費、cal上界の保守性が異なる。
A1対B1-OLを測らず、Raw/order-onlyへの改善をCV独自の資源削減と認定しない。

## 7. 1campaignからの主張と固定する報告形式

許されるのは、当該固定H4/一点δ/一round/一seed/一compiler・理想モデルでの返却保証、
guard/十分配分の成立状態、実現全RZと観測差である。CVの真variance削減、期待価格/期待総費用の改善、
seed間成功率、他分子・rank・L_D・δ・round・native実機への移送、最終RPE総費用の優位性は未評価。
calのsample varianceとV_U、選択十分Nを真variance/最適Nと呼ばない。

後日の結果表は `方式, axis, contract/source, 予定/実施quota, rank/λ/range,
b_engine/centering, cal/V_U, 選択N/route/配分, return状態, STOP理由,
段階別既知RZ/unpriced/完全性, 古典vector` を持つ。
比較表には本版の固定commit、全方式完了gate、主差C_A1−C_B1-OL、主比、分類と主張範囲を付ける。
**今回はこの空の報告仕様だけを固定し、N3実データの填入・判定は行わない。**

期待費用は[別計画](EXPECTED_RZ_COST_GUARANTEE_PLAN.md)で扱う。公開照合後mandatory STOP。
