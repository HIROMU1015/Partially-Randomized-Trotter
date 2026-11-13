# H4 DF10：DF head＋Pauli tailの全13round signal

**承認された未評価22 meanを一回実行し、全件の局所数値gateが通過した。P1のm0/m6×2prefixの4件を再計算せず統合し、LD1/LD2それぞれ全13roundのsignal packetを完成した。** 新規66complex/normalizer22組、統合後78complex。新reference・P1 mean再計算・full256 mean・N/MSE・compile・価格/Gは0。

[全26row packet](MEAN_HANDOFF.json)／[全signal・正規化・残差CSV](ROUND_SUMMARY.csv)／[HANDOFF](HANDOFF.md)／[SOURCE](SOURCE_INDEX.json)／[新22件のraw結果](run01/result.json)／[固定実行計画](EXECUTION_PLAN.md)。全signal取得は全RPEのRMSE・十分Nの資格付けではない。

## 固定条件とSOURCE

保存Linear H4/隣接1.00 Å/STO-3G、tol-only1e−8 DF10、同H/state/snapshot、weighted-Hermitian completion・fragment順、sector(2,2)/36、target=保存DF-Hの同sector基底状態、保存数値stateによる条件付き評価。S2、ΔPF=.1、K2、h=.1、M12、P=0、元prior/decoder/shift0/誤差契約、scalar一回会計を維持した。

| 役割 | 固定SOURCE |
|---|---|
| 新22件の科学SOURCE | `7084fbc9a43cf7b7e24681b4005d37cb59c1e465` |
| 新22件のraw carrier | `fce4e4f7eae620e4440ac4103312d7afed9e9608` |
| P1結果・再利用する旧4件 | `e8641b84fd8a3e98a49b65ecaa8fbd371a9d8ab7` |
| 旧4件の数値SOURCE | `05447d1d4f1e3c0e9e335a7085aa58a877c139f2` |
| P0辞書/operator | `4220f3006d0d8d413e195ed948e250578513b557` |
| B canonical validator | P1と同じ`ce0517de6e7b58cf4bf4c69f99f998d86099c406` |

元exactκは
`13112614441153360648422410673503886836581176429267589536059228160/539496659426909122096437389293855970805348475396548904610992201`。
P1で生成済みの共通κ scheduleをそのまま使用し、κ/λ/Rの再探索はない。

- LD1 canonical：`d905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27`
- LD2 canonical：`d7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3`

Pauli辞書は各184非identity項、λ_PはLD1=3.3066052942233157、LD2=2.0550998769244275。保存s_R=.4177175120626939/.2223759094107018、c=2.29310124732を保持し、P0のs_Pとの差を補償identityへ変換していない。旧DF ID・価格は流用していない。

## 再利用・計算方法

数値provider/P1 adapterはe8641b84時点からbyte不変。新規実装は未評価roundだけを選ぶbatchとSOURCE付きpacket統合だけである。P0保存operatorと保存stateを既存loaderで読み、sector36でideal finite meanを計算した。full256のoperator検証はP0、mean一対照はP1の証拠を保持し、今回の新full256伝播は0。

既存のdirect-expm head、paired finite Taylor、独立raw/corrected、exact-tail、scalar全Tで一回、matrix power/cache、R-aware empirical allowance、normalizer丸め診断を使用した。最大44,584,960 logical event位置でもevent列を明示生成していない。最初の失敗で停止する契約に従いm昇順・LD1/LD2交互に処理した。CPU/RAM/workerの独自capや共有設定変更はない。

P1のm0/m6は内容・SOURCE・identityの照合のみで、signal/B/logB/allowanceは再計算していない。旧4点のCSV診断もP1保存CSVから転記した。P1のfixed-R/full256対照は元SOURCEへの参照として保全し、本番26rowへ混ぜていない。

## 全13roundの結果

全rowでT=.1×2^m、qPF=2^m、actualΔ=.1、signed weights=[1]、allocation=[R_outer]。qRPEは別fieldである。biasは無次元のabs(corrected−保存state full-H reference)であり、energy誤差/RMSEではない。m0/m6はP1再利用値。

| m | T | LD1 R | LD2 R | LD1 attenuation | LD2 attenuation | LD1 signal bias | LD2 signal bias |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .1 | 3 | 2 | .9644631477 | .9792334922 | 1.340504e−5 | 1.337405e−5 |
| 1 | .2 | 6 | 3 | .9642742619 | .9723128830 | 2.680980e−5 | 2.674722e−5 |
| 2 | .4 | 11 | 5 | .9610419238 | .9668098498 | 5.361772e−5 | 5.349230e−5 |
| 3 | .8 | 22 | 9 | .9610264733 | .9631655575 | 1.072221e−4 | 1.069711e−4 |
| 4 | 1.6 | 43 | 17 | .9601345293 | .9610330183 | 2.143749e−4 | 2.138731e−4 |
| 5 | 3.2 | 86 | 33 | .9601334948 | .9598736886 | 4.286479e−4 | 4.276442e−4 |
| 6 | 6.4 | 171 | 66 | .9599048356 | .9598730057 | 8.572683e−4 | 8.552606e−4 |
| 7 | 12.8 | 341 | 132 | .9597895849 | .9598728350 | 1.714595e−3 | 1.710579e−3 |
| 8 | 25.6 | 681 | 263 | .9597317273 | .9597233321 | 3.429215e−3 | 3.421184e−3 |
| 9 | 51.2 | 1361 | 526 | .9597027401 | .9597233213 | 6.858366e−3 | 6.842304e−3 |
| 10 | 102.4 | 2722 | 1052 | .9597027390 | .9597233185 | .01371669093 | .01368456708 |
| 11 | 204.8 | 5443 | 2103 | .9596954888 | .9597045580 | .02743280859 | .02736856588 |
| 12 | 409.6 | 10885 | 4205 | .9596918617 | .9596951671 | .05486043983 | .05473199018 |

全raw/corrected/exact-tail/reference、B/logB、残差、allowance、SOURCEの正本はCSV/packetと各JSON。m12の主要値は以下のとおり。

| 値 | LD1 | LD2 |
|---|---|---|
| raw | .19300554765656575 + .9400835738805661i | .19288541138989315 + .940111603443775i |
| corrected | .20111199791275977 + .9795681421792797i | .20098612350720568 + .9795939748819018i |
| exact-tail | .2011119974925422 + .9795681401326963i | .20098612336716287 + .9795939742013502i |
| logB | .04114302348091545 | .04113957916704647 |
| B | 1.0420011255166373 | 1.0419975365438898 |
| event位置数 | 44,584,960 | 17,223,680 |
| empirical u | 2.536089367456673e−6 | 9.800455268122632e−7 |
| empirical logB margin | 1.3196194538894483e−6 | 5.10089570425538e−7 |

共通の保存referenceは.1470900528467786+.989123104742252iで、厳密ground固有stateのexp(−iE0T)に置き換えていない。corrected−exact-tailの小差も浮動小数点誤差を含む観測であり、empirical allowanceより小さい成分を純粋な物理finite-K biasとして分離・保証しない。

## 数値検査と限界

新22件すべてで既存のexact-tail norm/unitarity、head unitarity、独立raw/corrected closure、raw outcome range、reference pair、normalizer精度gateが通過した。

| 新22件の診断 | 最大保存値 | 既定gate/扱い |
|---|---:|---|
| raw/corrected vector closure | 4.3728735746241794e−10（LD1/m11） | 1e−9 |
| power unitarity残差 | 8.81893828278443e−12（LD2/m12） | 1e−9 |
| head unitarity残差 | 1.0839888571524473e−15 | 1e−9 |
| reference pair差 | 9.279812219411241e−12（m12） | 1e−10 |
| normalizer 1ULP log感度 | 1.979969963915674e−8（LD1/m12） | 従来empirical logB marginへ反映 |

normalizer22組はすべて有限/b>1、既存判定で精度未解決/underflow/overflowは0。長時間のclosureや丸め感度はそのまま記録し、閾値緩和・unitary化・再正規化は0。allowanceはempiricalでrigorous上界ではなく、norm観測domainもmacro/power境界に限る。

新人工6＋P1回帰6の12件PASS。[TESTS](TESTS.md)。科学実行4.0085秒、Python3.11.0rc1/NumPy1.26.4/SciPy1.14.1。science起動一回、retry/fallback/追加call0。実行後のsnapshot/receipt/context hash一致。旧P0/P1 raw・結果・SOURCE・KEEP/STOP、他担当worktree/processは変更していない。

## 引渡しとSTOP

packetはP1と同じcanonical configuration/各round identity/lawを保持する。旧4rowはP1 commit/数値SOURCE、新22rowはraw carrier fce4e4f7e/科学SOURCE7084fbc9aへ各mean_sourceで接続した。旧mean_recordは元の値とSOURCEのまま。

Bは全round raw/law/allowanceを既存RPE評価器へ受け入れられる。C/Dには同ID・辞書/λ/scalar・全schedule/R/q/K/profileを渡す。今回N/MSE・価格/Gは未実行である。厳密ground同定・actual sampler/control law・finite-Kから全RMSEへの移送は未資格、十分N/global RMSE/C/G/winnerはnullを維持する。

次：Codex Bがpacketの受入確認を行い、ユーザーがN/MSE/価格等の次の科学scopeを判断・承認する。Codex Aは今回のsignal取得を完了し、commit/push・remote SHA/tree/主要成果物照合後mandatory STOP。追加LD/K/κ/PF幅や対象外計算へ進まない。
