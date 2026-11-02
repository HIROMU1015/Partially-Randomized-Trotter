# C1：H6全基底合成誤差上界・限定実行結果（2026-10-26）

**全3,200 unique入力列が完了し、8経路×13 roundの104行すべてが登録位相十分条件を満たした。** 全経路の状態は`POTENTIALLY_PHASE_ELIGIBLE_NUMERICAL`。科学callは3,335／上限6,535、correctness FAIL 0、retry／resume 0。実行SOURCEと契約は変更していない。

取得したのは**ancilla×sector400の800入力部分空間に対する資格Bの数値的operator上界**である。full8192成分の出力を保ったが、full8192入力全体の上界ではない。丸めを囲った資格C、最終CA、shots、総費用の結論は取得していない。

## 固定条件・SOURCE

直鎖H6・1.00 Å・STO-3G、Nα=Nβ=3／sector400、修正codecを用いた保存fit最終CDF rank14／保存selected XDF rank14。通常number表現、元g補正κ・scalar・元frame順を保持。全15因子のordered symmetric S2、s=1／4、literal／sharedの8経路、δ=0.1、m=0..12、n=s·2^m、最大16,384。PR splitは使わない。

共通CA=0.00159 Ha、DF reserve=CA/8=0.00019875 Ha、own-DF ground oracle、状態準備費用除外を維持した。両armの物理DF誤差は異なり、今回再計算しない。固定P0はQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation degree1／qubits initially zero=false。保存QPYを読むだけで、新build／compileは行っていない。

- 科学SOURCE：`e36e0560624c66df0aedf619fc5c8c33ecb4f837`、tree `1c8d0b5186c82b1f236575bc1cf66d7cfd70da6b`。
- [固定契約](research_c_c1_h6_synthesis_bound_contract_unapproved_20261025.json)、scope `C1_H6_COMPLETE_SECTOR_SYNTHESIS_BOUND_V1`。
- 契約SHA256：`39a2fe9e6f5fa60a05c7b0c965127f87a53b58d9d831eb7c1b95d263cf35e734`。
- SOURCE manifest SHA256：`61eaad15866a643a9fca34d686e4cf4c9357d81ba1ba1294625516b9f20a36d9`。
- Branch：`codex/research-c-c1-h6-synthesis-bound-execution-20261026`。
- 実行日付は10月26日JST。outputは契約で固定された歴史的な`artifacts/research_c_c1_h6_synthesis_bound/2026-10-25/run01`を使用。

ユーザーのチャット上の開始許可で実行した。契約の歴史的な`unapproved`名／metadataは変更していない。承認証明・承認原文保存・新grantは作成していない。

## 事前・事後gateと完全coverage

| 対象 | 結果 |
|---|---|
| SOURCE51件・保存input38件・契約／manifest・依存lock／環境 | preflight／postflight PASS |
| 旧protected727件＋直前native raw122件 | 849件すべてbyte不変 |
| 保存PF reference4件 | complex128／shape400×400／finite／unitarity defect gate PASS |
| compiled QPY8件 | 13 qubits、official有限実角P0 gate、classical操作なし、layout／phase等の構造gate PASS |
| 完全基底 | 4 unique groups×800列、各0..799を1回、欠落／重複0 |
| full-output state gate | 全3,200列・各8192成分を比較。component／norm-drift各1e-10以内 |
| 8 kernel／104 round／8 summary | 全件完了、phase eligibility104／104 PASS |

literal/sharedの各対は、同arm・同s・同保存PF targetに加え、ordered gate class／qubit／parameter float値／declared phaseのIR signatureが完全一致した。[route identity](../../artifacts/research_c_c1_h6_synthesis_bound/2026-10-25/run01/route_identity.json)に全signatureとaliasを保存した。RZ数や有限probeの近似一致を理由に共用していない。G=4が確定したため、同じ800列をshared側で重複実行せず、8経路へ4 groupの結果を対応付けた。

## 8経路のkernel数値上界

Vを保存compiled回路、Sをancilla×物理sector、U|S=diag(I400,W)とする。全列・full出力からεF=||(V−U)P_S||Fを集計し、exact arithmeticで成立するspectral≤Frobeniusの関係を使う。以下の上界値とmax-entryはdimensionless。

| 経路 | coverage | εFによる数値operator上界 | 最大成分差 | 最大norm drift |
|---|---:|---:|---:|---:|
| CDF s1 literal | 800/800 | 7.9697438830e-10 | 2.8377317493e-11 | 7.3019368330e-13 |
| CDF s1 shared | 800/800（IR共用） | 7.9697438830e-10 | 2.8377317493e-11 | 7.3019368330e-13 |
| CDF s4 literal | 800/800 | 8.1857544161e-10 | 2.9575643602e-11 | 7.1243011490e-13 |
| CDF s4 shared | 800/800（IR共用） | 8.1857544161e-10 | 2.9575643602e-11 | 7.1243011490e-13 |
| XDF s1 literal | 800/800 | 1.5602042434e-9 | 5.5313285329e-11 | 7.3974160131e-13 |
| XDF s1 shared | 800/800（IR共用） | 1.5602042434e-9 | 5.5313285329e-11 | 7.3974160131e-13 |
| XDF s4 literal | 800/800 | 1.5455465726e-9 | 5.4842650961e-11 | 7.3752115526e-13 |
| XDF s4 shared | 800/800（IR共用） | 1.5455465726e-9 | 5.4842650961e-11 | 7.3752115526e-13 |

CDFの数値上界値は同sのXDFより小さいが、これは実operator誤差そのものの厳密な順位・DF近似精度の優位・費用優位を示さない。

leakageを落とすsector射影は行っていない。参照出力はS内なので、S外へのactual leakageはfull残差L2に含まれる。そのため各入力列のleakage L2≤その列の全残差L2、部分空間上のleakage operator norm≤上表εFという**資格Bの上界**を利用できる。leakageだけを分離した観測値は保存しておらず、ゼロとは報告しない。

記録粒度も区別する。各列L2は独立保存される一方、最大成分差とnorm driftは固定runnerの**8列batch最大値**である。[列CSV](../../artifacts/research_c_c1_h6_synthesis_bound/2026-10-25/column_residuals.csv)は3,200行に列L2と、その列を含むbatchのcomponent／norm上界を格納した。後者を個々の列の点測定値とは扱わない。これらの記録を増やすための再state-actionは行っていない。

## reference欠損・不変性と反復guard

| 保存W | max-entry of W*W−I | Δ=||W*W−I||F | α guard |
|---|---:|---:|---:|
| CDF s1 | 8.6597395921e-15 | 3.3304429895e-14 | 1.0000000000000169 |
| CDF s4 | 4.7739590059e-15 | 3.0018930687e-14 | 1.0000000000000153 |
| XDF s1 | 5.7731597281e-15 | 3.3632648971e-14 | 1.0000000000000170 |
| XDF s4 | 4.6629367034e-15 | 3.1059548387e-14 | 1.0000000000000158 |

いずれも固定reference gate1e-10以内。Uはsquare WをS内へ埋め込む定義によってSを不変にする。own-DF groundをPF固有状態と仮定しない。Vのfull-space unitary性はofficial gateのideal semanticsで構造確認し、actual出力のS外成分を残した。

数値Wの欠損をゼロとせず、K_n≤min(1+α^n, εF Σ_{k=0}^{n−1}α^k)を使った。m12でreference norm-growth上界は1.000000000069～1.000000000259、telescoping factorもnより僅かに大きい。unitaryを無条件に仮定してnεだけを採用していない。

## 104 roundと位相十分条件

保存head／X／Y shellの全ancilla2×2 operator評価を再利用した。kernel反復とhead／tailのstate差からcomplex synthesis boundを作り、同じtargetのcomplex signalへ変換した保存PF operator誤差を1回だけ加算した。DF誤差（Ha）は加えていない。導出と資格の詳細は[準備報告](research_c_c1_h6_synthesis_bound_preparation_20261025.md)の§4–5を参照。

以下は各13 roundで最大となったm12の値。literal/sharedは同じIRなので共通値である。全104行は[round CSV](../../artifacts/research_c_c1_h6_synthesis_bound/2026-10-25/round_bounds.csv)と[元result](../../artifacts/research_c_c1_h6_synthesis_bound/2026-10-25/run01/result.json)に保存した。

| arm・s | n | 反復kernel上界K_n | 合成complex上界 | 保存PF誤差 | PF＋合成complex上界 | 位相上界rad |
|---|---:|---:|---:|---:|---:|---:|
| CDF s1 | 4096 | 3.2644070946e-6 | 9.2331375739e-6 | 0.122226755849 | 0.122235988986 | 0.122542455051 |
| XDF s1 | 4096 | 6.3905965812e-6 | 1.8075336715e-5 | 0.122231116310 | 0.122249191647 | 0.122555757477 |
| CDF s4 | 16384 | 1.3411540037e-5 | 3.7933563631e-5 | 0.007643081720 | 0.007681015284 | 0.007681090813 |
| XDF s4 | 16384 | 2.5322235049e-5 | 7.1622096481e-5 | 0.007643354759 | 0.007714976856 | 0.007715053392 |

登録phase roomはπ/12=0.2617993877991494 rad。8経路すべてのm0..12がこれを下回り、分類は8／8で`POTENTIALLY_PHASE_ELIGIBLE_NUMERICAL`。`INFEASIBLE_REGISTERED_BOUND`／`UNRESOLVED`は0。

s4では反復が増えて合成上界はs1より大きいが、保存PF誤差が小さいためcombined位相上界は小さい。今回の全列上界は登録位相予算に対して有用だった。ただし十分条件の数値PASSであり、実位相誤差の測定や厳密CA保証ではない。

## call・資源実績・保全

| 処理 | COMPLETE logical units |
|---|---:|
| saved bundle | 1 |
| sector basis index | 1 |
| reference checks | 4 |
| QPY structural checks | 8 |
| exact-IR grouping | 1 |
| full-output input-column action | 3200 |
| kernel bound reports | 8 |
| round bound reports | 104 |
| route summaries | 8 |
| **合計** | **3335** |

全attemptにCOMPLETE terminalが対応し、FAIL／inflight 0。1列=1 logical unitで、8列batchを1 callに値切っていない。G=4なので135+800G=3335となる。上限の未使用分を追加計算へ流用しない。新fit／optimizer／rank探索／DF・PF再計算／新build・compile／CA・shots・総費用計算はすべて0、retry／resumeも0。

資源planは実際のaffinity32 CPU、利用可能RAM約56.16 GBを確認し、4 group workers・Aer各8 threads・batch8、現job内BLAS1に調整した。worker／CPU／RAM／wallの恣意的上限は追加していない。固定SOURCEのcached conversionとbatchをそのまま利用した。

runner wall **5173.3409秒（約86.2分）**、process CPU10058.0603秒、peak RSS5,048,440 KiB（約4.81 GiB）。whole-job `/usr/bin/time`は1:26:15、平均CPU194%、swaps0、exit0。設定上32 CPUを利用可能としたことと、実効平均CPU使用量は区別する。Python／Qiskit側の組立・serializationが律速した可能性はあるが、今回profiler診断・SOURCE変更は行っておらず原因を断定しない。高速化倍率も主張しない。

資源観測を採取した時点ではCPU／memory pressureは0だった。全時間連続観測とは主張しない。共有venv／cache／設定／他ユーザーjobの変更は0。task-local cacheのみを使用した。[資源記録・実行identity・postflight](../../artifacts/research_c_c1_h6_synthesis_bound/2026-10-25/)に保存した。

postflightはSOURCE51件、入力38件、契約・manifest、旧証拠849件の全byte一致を確認。raw run01は23ファイルを保存し、hash一覧を[postflight.json](../../artifacts/research_c_c1_h6_synthesis_bound/2026-10-25/postflight.json)に記録した。旧nativeの104未資格profile、旧FAIL、627-call履歴、grant／ledger／marker／STOPを変更していない。新規データのCSV／要約化は保存JSONの整形であり、新しいstate-action／operator norm／PF計算を呼ばない。

実行コマンド：

```sh
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v \
  /tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python \
  scripts/run_research_c_c1_h6_synthesis_bound.py \
  --contract docs/research/research_c_c1_h6_synthesis_bound_contract_unapproved_20261025.json \
  --execute --output artifacts/research_c_c1_h6_synthesis_bound/2026-10-25/run01
```

このコマンドは消費済みrun01の実行履歴であり、再起動しない。

## 残る資格と次の判断

今回、旧有限probe（資格A）から新しい完全基底の部分空間数値上界（資格B）へ進み、登録13 roundの位相条件を評価できた。Aer／小gate変換／保存CAR-PF reference／norm／伝播／asin等の丸めenclosureはなく、資格Cへ昇格していない。`rigorous_enclosure`、full-Fock全入力operator bound、CA、shots、total costはnullのまま。

次の最小作業は、この**資格Bの合成誤差＋保存DF／PF**を既存RPE統計・energy・native費用へ接続する限定数値pilotの条件整理である。共通CAを満たすshotsと条件付き総RZは今回計算していない。H6では前段の1 h-step kernel RZがCDF側で約2%多かった事実も維持し、今回の小さな合成上界から総資源優位を推定しない。

**ユーザー／研究C**が、その次段階を依頼・許可するか判断する。**Codex**は別途の具体的依頼と必要な明示科学許可後にのみ実施する。詳細科学レビューも今回の実行許可に含まれない。今回の結果・SOURCE・保存input・成果物をcommit／pushしてremote照合後、**mandatory STOP**。CA／shots／総費用や厳密enclosureの追加計算へ自動進行しない。
