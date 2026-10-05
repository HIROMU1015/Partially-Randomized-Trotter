# Track A PM-2 保存値による精度と資源境界の解析結果

2026-10-05 JST。固定済みsourceと利用者の明示指示により、PM-2を一回実行し、保存値の再計算を照合した。
statusは `PM2_PRECISION_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW`。
新しい科学計算ではなく `POSTHOC_SAVED_VALUES_ONLY` のlocal evidenceで、利用者指示によりresult commitへ収録する。
実行時のlaunch HEAD・source commitと、結果を収録するGit commitは別identityである。
mandatory STOP、next_stage_authorized=false、research_decision=null。研究価値の判断は人のreviewへ戻す。

## 対象とidentity

[固定契約](research/pr2_pm2_precision_resource_contract_v1.md)・[解析実装](research/pr2_pm2_precision_analysis_implementation.md)は不変。
source commitは `324435d77b6642dbd44e8d1f178420daf62e77ed`、
launch HEADは `bea4cf00f08e1c2d5fba3649c6f58cc31490fd1e`。
基準evidenceは `194cc604b90c56a0e7e949b91b064a4bcfc846da`。M1-A/M1-B1/PM-1/M2の明示4 JSONがそのblobと一致する。

developmentはH4 linear 1.00 Å、STO-3G、DF rank12、8 system qubits、T=0.8、
二次DF-prefix PF、L_D=0/3/4/5/6/9/12、q=1/2/4/8、delta=0.8/0.4/0.2/0.1、登録r/Kの全218候補。
transferは使用済みH4 1.30 Å、同じbasis/rank/Tへの元M2固定5構成だけで、held-out再探索ではない。
費用は保存Qiskit1.3.0 opt1、basis rz/sx/x/cx、seed17、topologyなし、状態準備なしのfull-wrapper compiled metrics。

ε=0.005〜0.1の301対数点＋正確な0.05、計302点。α_real=α_imag=0.025。
全67,346 candidate-ε行、適格境界223行、P envelope4,693行、method代表2,334行を保存した。
元ε=0.05で223候補のeligibility・整数axis shotsが完全一致、6指標matched workもrel1e-12/abs1e-6で再現した。
不適格行のworkはnull/MISSINGであり、0費用やmethodとしての実現不可能性を意味しない。

## P=0での点最小領域

以下は固定表示点でのprimary compiled RZ点最小である。行間の連続εの厳密境界や統計的winnerではない。
method間の新しい10% GO基準は導入していない。

| domain | 最初〜最後の表示ε | 表示点数 | primary点最小構成 |
|---|---:|---:|---|
| development全218 | 0.005〜0.0054158189504 | 9 | B2 rank3 q4 r4 K4 |
| development全218 | 0.00547017101788〜0.0235054643684 | 147 | B2 rank3 q2 r4 K4 |
| development全218 | 0.0237413604716〜0.1 | 146 | B2 rank3 q1 r4 K2 |
| M2元5構成 | 0.005〜0.00667938131366 | 30 | B3 rank0 q8 r32 K4 |
| M2元5構成 | 0.00674641423837〜0.1 | 272 | B2 rank3 q1 r4 K2 |

developmentでは全302点のprimary点最小がB2だが、厳しい精度ほどqが増える。
M2固定5構成ではε=0.005に適格なのはB3だけであり、移したB2 q1構成はその要求精度に適格ではない。
B2 r4の適格境界はε_min=0.00525637654710で等号を除外する。
適格化直後もshot負担が大きく、B3→B2の点順位変化は上表の別の表示点間で起きる。
これは元のε=0.05でのM2 `TRANSFER_SUPPORTED` を取り消すものでも、held-out上のmethod optimumでもない。

developmentのB0 rank5 q1はε>0.0186305147953で適格になる。
同rankのq2/4/8はそれぞれε>0.0264386154414/0.0282755427931/0.0287279977061。
qを増やせばdiscardの総biasが必ず小さくなる、とは保存値から言えない。
pure discard/PF biasは欠測のまま、誤差相殺の機構を確認したとはしない。

## shot負担と回路費用の違い

元ε=0.05、developmentの各methodのprimary点最小代表は次である。
Nはcorrected Hoeffding十分shot式による解析値で、実行した量子shot数ではない。
この代表4件はcosine/sineの保存平均RZが同じなので、表の1-shot値は軸の共通値である。
一般の解析では軸別費用を使い、元C_effを固定流用していない。

| methodと構成 | N_total | 保存1-shot RZ | G_RZ(P=0) |
|---|---:|---:|---:|
| B0 rank5 q1 | 25,910 | 8,866 | 229,718,060 |
| B1 rank12 q1 | 18,471 | 20,168 | 372,523,128 |
| B2 rank3 q1 r4 K2 | 20,563 | 6,359.71875 | 130,774,896.65625 |
| B3 rank0 q8 r32 K4 | 36,032 | 31,993.53125 | 1,152,790,918 |

B2はB1よりshotを要する一方で1-shot回路が安い。B0 rank5はaccuracy適格でも、
残るheadroomと1-shot費用を掛けた結果がB2より大きい。
したがってaccuracy適格性、少ないshots、安い回路、低い測定込み資源は別の判定量である。
これは保存値の費用項による説明であり、Hamiltonian誤差の新しい機構分解ではない。

## 共通状態準備費用と不確かさ

Pは共通の仮想RZ-equivalent準備費用/shot。実準備回路の評価ではない。
元ε=0.05のdevelopmentではP=0のB2 r4から、P≈1,796.42でB2 r8へ移る点envelopeを得た。
全218候補のこのεでのenvelopeはB2構成だけである。
M2元5構成ではP≈385.08でB2 r4→r8、P≈208,735.13以降はB1になる。
候補集合が違うため、この差をgeometry効果だけへ帰属しない。
精度を変えればP envelope自体も変わり、例えばdevelopment ε=0.005の極大P領域にはB0 rank9 q8が入る。
巨大Pは仮想感度であり、整数shot丸めにも依存するため、実用準備費用の予測として扱わない。

元32 paired cost標本のcovarianceを保持した点±2SEはengineering intervalで、formal CIではない。
developmentでは全302点、M2固定5構成では272点で、primary点最小と複数候補の区間が重なる。
上表のq/K/rの厳密winnerを認定せず、追加trajectoryで自動救済しない。

独立照合の最初のcheckerはshot式の乗除順を入れ替え、巨大な浮動小数点boundのceilで24軸行が不一致になった。
固定sourceの演算順では全行一致することを確認し、checkerだけを訂正した。source・式・threshold・結果は変更せず本解析も再実行していない。
巨大なshot整数を任意精度の厳密最少値として主張しない。詳細はvalidation auditに保存する。

## 実行と成果物の照合

一回のrunnerは2026-10-05 07:58:59〜07:59:02 UTC、wall約2.92秒、peak RSS210,872 KiB、CPU1、BLAS各1。
pre/post各62 synthetic tests passed、fail/skip0。source6 blobs、入力4 blobs、
runner manifestの7 fileをhash/bytes照合し、全ledgerのshots・6費用・paired SE・point Pareto、
method代表tiesと604 domain-ε集合のP interval minimality・欠けないcoverageを検査した。
新signal、trajectory、build/compile、ground-state、分子データ/runtime/registry、量子shots、GPUは0。
Python診断guardはOS sandboxではない。local実行でありimmutable CI・外部再現ではない。

[成果物directory](../artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/)に
summary、4 CSV、claim audit、全表示点report、runner manifestを保存した。
[execution audit](../artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/execution_audit_v1.json)と
[validation audit](../artifacts/resource_applicability/pr2_pm2_precision_analysis/2026-10-05/validation_audit_v1.json)は別監査として追加し、
runner manifestとその7対象fileを変更していない。
summary SHA-256は `7b026d4cc657cf43ad23fd7d6e10d5aa31cd58af555649b7a8e858ea12155845`、
runner manifest SHA-256は `546cdfaf8c77f349f6f55b346e93749ce5956c9a605281d169887257377c843f`。

次は、このprecision/resource mapが限定resource/applicability studyの完成に十分かをreviewする。
PM-3、追加trajectory、別geometry/分子、strong synthesis/higher-order PF、energy/RPE、Track B統合は未認可。
