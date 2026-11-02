# PR-2 matched-accuracy M1-B1 bounded compile契約 v1

作成日：2026-09-30
M1-A result commit：`3c1831e326c27c5f679b3820997f27916d26ed9f`
M1-A result SHA-256：`1f960d7a33296e2dcb74d497e360572b26409dc9aeae01522335e7b91ed81086`
M1-A result fingerprint：`422f898bba1e3849d0f45830082b76d4f42da436e2b49796e562cd79fc716c9e`
status：`M1_B1_BOUNDED_COMPILE_CONTRACT_FROZEN_EXECUTION_NOT_AUTHORIZED`

## 1. 位置付け

M1-AはH4 linear 1.00 Å、STO-3G、DF rank 12、sector 8 qubit、`T=0.8`で210候補を評価し、
206候補をaccuracy適格、うちB2/B3 random候補194件を全件適格と判定した。一方、64件のproxy
frontierから16件を選ぶ旧selectorは52件の未選択proxy非支配候補を残し、契約どおり
`SELECTION_LIMITED`、compile 0で停止した。

この結果は失敗として削除または再解釈しない。「`(shots, n_det, n_rand)` proxyだけではactual compiled
costの候補を16件へ安全に圧縮できなかった」という監査結果として保存する。本契約は外部reviewの
`PROCEED_BOUNDED_COMPILE_EXPANSION`を受け、selectorを改良せず、M1-Aで固定済みの有限gridを32 trajectory
だけ直接compileするM1-B1を定義する。

本書、実装、schema、test、zero-compute planはM1-B1科学計算を認可しない。実行には、planをbyte固定した
後の別result-prior execution authorizationと実行前reviewが必要である。

## 2. claimとprior-art gate

[M1前最終amendment v2](pr2_matched_accuracy_m1_preexecution_amendment_v2.md)のprior-art gateを変更しない。
Cugini--Atif--Subasi (2026)のcost/variance共同最適化と、Kanasugiら (2026)のsingle-ancilla QPE、部分
randomization、化学系end-to-end resource estimateは既知として扱う。

残す問いは、importance-sampling分布の最適化や一般的resource optimumではなく、canonical finite-RTE
samplingのもと、固定DF-prefix split、discard、full deterministic、random-dominant endpointを同じ
complex-signal accuracyへそろえたとき、fixed-`q`方式判断がnormalization、analytic shots、状態準備除外の
full measured Hadamard wrapper compileを戻して維持されるかである。

## 3. candidate集合の完全固定

入力はM1-A resultのbyte identityだけである。B1でsnapshotを読んでsignal、bias、normalization、shot、
accuracy eligibilityを再評価しない。

- random：M1-A signal recordで`method in {B2,B3}`かつ`accuracy_eligible=true`だった194 fingerprintを
  全件そのまま用いる。B2は145件、B3は49件である。
- deterministic/discard baseline：M1-AのB0全12件とB1全4件、計16 fingerprintを用いる。
- B0の4件はaccuracy不適格だが、baseline completenessのため1 wrapper/axisをcompileする。ただし
  matched-accuracy `G` frontierまたはwinner比較へ入れない。
- B1でcandidateを追加、除外、置換せず、signal値を根拠に再選抜しない。
- 旧16-cell selectorのselected/unselected集合は監査情報であり、新しい候補制限として用いない。

## 4. wrapperと上限

random cellごとに32 trajectoryを固定し、各trajectoryを同一のrandom drawのままcosine/sine二軸の
full measured Hadamard wrapperへ使う。state preparation、backend execution、quantum shotsは含めない。

\[
194\times32=6{,}208\quad\text{random trajectories},
\]

\[
194\times32\times2=12{,}416\quad\text{random full wrappers}.
\]

B0/B1は16 cellそれぞれcosine/sine一件ずつで、

\[
16\times2=32\quad\text{deterministic/discard full wrappers}.
\]

総上限は

\[
\boxed{12{,}448\ \text{full wrappers}}
\]

である。process workerは最大6、各workerのBLAS threadは1とする。この上限は将来のexecution
authorizationが明示的に許可した場合だけ使用できる。

## 5. seed、task、cache/checkpoint identity

random trajectory seedは`candidate_fingerprint`、`trajectory_index=0..31`、固定master seedからSHA-256で
導出し、axisを含めない。したがってcosine/sineは同じtrajectoryを共有する。各tail occurrenceのseedも
candidate、trajectory、outer step、tail occurrence、RTE stepだけから導出し、step/occurrenceごとに独立な
座標を持つ。

wrapper cache/checkpoint identityは少なくとも次を全件含む。

1. M1-B1 source commit。
2. compiler identity/fingerprint（Qiskit 1.3.0、basis `rz,sx,x,cx`、optimization level 1、seed 17、
   coupling/backend/layout/routing指定なし）。
3. candidate fingerprint。
4. axis。
5. trajectory indexとtrajectory seed。deterministic/discardは両方null。
6. frozen M1-A result SHA-256。

一項でも異なるcache/checkpointは再利用しない。特にcandidate fingerprintが異なるcell、partial split、axis、
trajectory、compilerまたはsource commit間でcacheを流用しない。既存checkpointは全identity一致と内容hash
検証に成功した場合だけresumeできる。

## 6. M1-B1の終了点

M1-B1は32 trajectoryのactual compiled resource mapを作成した時点で必ず停止する。次を行わない。

- 追加96 trajectory、128 trajectory精密化、adaptive extension。
- held-out候補確定、H4 1.30 Å path/file access、transfer。
- 最終winnerの精密化、S3、別geometry、別分子、別PF、H12。
- 結果後のthreshold、grid、candidate、seed、compiler変更。

B1終了後は自動継続せず、次のいずれかへ研究判断を戻す。

- `CONTINUE_RESOURCE_STUDY`
- `NARROW_TO_TECHNICAL_NOTE`
- `STOP_DUPLICATIVE`
- `COMPILE_RESULT_INCONCLUSIVE`

判断材料はmethod/split別actual frontier、fixed-`q=8`とmatched-accuracyの方式判断差、intermediate partialの
frontier残存、`q,r,K`集中、state-preparation `P` lower envelope、および旧proxyとactual compiled costの
対応である。B1単独で投稿可能性、一般的resource optimality、held-out transfer成功を主張しない。

## 7. zero-compute gate

zero-compute planはM1-A artifactをidentity確認のため一回読むだけで、development/held-out NPZ、molecular
calculation、signal、trajectory sampling、occurrence sampling、circuit、compiler、quantum shot、GPUを全て0に
固定する。194+16 cell、12,448 wrapper cache key、source/compiler/candidate/axis/trajectory identityを機械的に
生成し、plan fingerprintを固定する。

plan作成後、専用testと索引・manifest更新をcommit・pushして実行前reviewへ渡す。ここまでのstatusは
`M1_B1_BOUNDED_COMPILE_CONTRACT_FROZEN_EXECUTION_NOT_AUTHORIZED`であり、M1-B1本実行は未承認である。
