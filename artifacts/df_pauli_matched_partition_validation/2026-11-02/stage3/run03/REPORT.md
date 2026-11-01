# P3 Stage 3 run03：未取得13件のprocess分離回復・36価格完成

2026-11-02 JST。**13件の限定回復が成功し、Stage 3固定cohortは36/36件完成。**
[STOP](STOP.json)：`STOP_STAGE3_FIXED_COHORT_COST_COMPLETE_NO_AUTOMATIC_FOLLOWUP`。
取得済み23価格はrun02からそのまま継承し、新13件だけを異なる13 processで各1回compileした。
新sampling・native回路再生成・native preflight再実行・既存23件の再compile・automatic retryはすべて0。
科学SOURCEは`c0d9d3bbe66750d0b20acd9e71d55525f429b229`、基点結果は`2da49fa4e8578227c130fc6061a0b3061276f893`、run02 native SOURCEは`9c4006bff3e76ea2ba7fbe4a10e679a8693c3a5c`。
[SOURCEと入力](source_and_inputs.json)／[限定対象13件](recovery_targets.json)／[実件数](execution_status.json)。

## 条件と証拠の継承

linear H4、隣接1Å、STO-3G、中性singlet、system8/full256、保存DF tol-only1e-8/rank10。
案A、共通Gaussian T_eff保持、M_B=Pauli184残差、M_C=DF rotated Z/ZZ360残差・10 basis。
T=exact binary64 .8（3602879701896397/4503599627370496）、S2/q8/r4/K4、P=0。
ancilla8、cosine/+Im sine、measure込みのfull wrapper。各arm4 trajectory、各32 event、計8本/256位置をrun02から再利用した。
P seed310201–310204、DF seed310301–310304は方式別独立標本で、同じindexをpaired科学標本として扱わない。
order4が出なかったrun02のevent列を選び直していない。coverage・native位相/actual-adjointの根拠は
[run02報告](../run02/REPORT.md)、[native preflight](../run02/preflight.json)、
[共通保持/basis](../run02/native_bases.json)を参照する。
81項目のnative PASSは再利用証拠であり、本runで再計算した値ではない。

run02の36件thread並列compileはQiskit共有内部状態の`Already mutably borrowed`で停止し、23件が正常返却・保存された。
本runは公開commitのpending listとarchive原bytesを照合し、13件のinput QPYだけを読み込んだ。
旧23件はその共有processで正常返却された来歴ごと継承し、今回の13件と同じ実行履歴だったとは扱わない。
[価格別の由来](price_provenance.json)はinput/outputのnamespaceを明記する。

## Process修正・compiler・検証

既存`matched_partition_stage3.price`、`transpile_and_measure_cost`、QPY保存関数を再利用した。
`spawn`＋`max_tasks_per_child=1`で1 target/processとし、同じprocessで同時にtranspileしない。
13 targetに異なる13 PID、各compile call1回、全workerをjoinし、例外0でprocess exit0。
wall 20.909099秒。SOURCEや共有package/venv/OS設定は変更していない。

Qiskit1.3.0、qiskit-nature0.7.2、numpy1.26.4、scipy1.14.1、OpenFermion1.6.1、Python3.11.0rc1。
compilerはbasis rz/sx/x/cx、optimization_level1、seed17、backend/coupling/layout/routing指定なし。
compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b` は旧23件・新13件で一致した。
人工4 tests PASS：saved bytes破損検知、worker数超過taskでもprocess非再利用、既存compilerの価格/fingerprint一致、
非上書きmergeと同軸非加法性。保存JSONのlistとlive dataclassのtupleはJSON表現へ正規化して照合した。
人工PASSはH4全compiled operator作用保証を意味しない。

## 観測価格範囲と解釈

以下は固定4 trajectory/arm×2軸の観測min/maxのみ。母平均・標準誤差・tail・方式順位・同精度N/Gは算出していない。

| arm/program | RZ min–max | CX min–max | depth min–max |
|---|---:|---:|---:|
| P/F | 2916–3070 | 1098–1186 | 1507–1579 |
| P/E | 98–155 | 118–178 | 155–229 |
| DF/F | 8092–9978 | 2628–3170 | 3749–4502 |
| DF/E | 3655–6210 | 1178–1914 | 1728–2748 |

同じ共通保持programを使うこのcohortでは、Pauli残差のEとFullの価格がDFより小さい。
Stage 2保存BはPauli39.125909965750068、DF34.527999187486905であり、これらのBは本runで再評価していない。
この観測は、共通分割でも単回native価格差が生じることを示す。元の純RTE対DF B3と同一比較ではなく、
辞書・sampling law・native残差実装の効果を単独に識別した結果ではない。
4 trajectory/armの有限cohortだけから一般的優位・期待費用・元研究の低費用原因の一意性を認定しない。

## 全36価格

F16/E16/R2/Empty2、両測定軸を含む。[CSV](compiled_prices.csv)／[全metrics](prices.json)／[新13 metrics](recovered_prices.json)。

| target | 取得run | RZ | CX | depth | size |
|---|---|---:|---:|---:|---:|
| DF_0/E/cosine | run02 | 4832 | 1551 | 2197 | 9479 |
| DF_0/E/sine | run03 | 4831 | 1551 | 2197 | 9478 |
| DF_0/F/cosine | run02 | 9231 | 2991 | 4215 | 17985 |
| DF_0/F/sine | run02 | 9231 | 2991 | 4215 | 17985 |
| DF_1/E/cosine | run03 | 4618 | 1445 | 2129 | 9028 |
| DF_1/E/sine | run02 | 4617 | 1445 | 2129 | 9027 |
| DF_1/F/cosine | run03 | 8862 | 2839 | 4118 | 17228 |
| DF_1/F/sine | run03 | 8862 | 2839 | 4118 | 17228 |
| DF_2/E/cosine | run02 | 6210 | 1914 | 2748 | 12111 |
| DF_2/E/sine | run02 | 6209 | 1914 | 2748 | 12110 |
| DF_2/F/cosine | run02 | 9978 | 3170 | 4502 | 19381 |
| DF_2/F/sine | run02 | 9978 | 3170 | 4502 | 19381 |
| DF_3/E/cosine | run03 | 3656 | 1178 | 1728 | 7192 |
| DF_3/E/sine | run02 | 3655 | 1178 | 1728 | 7191 |
| DF_3/F/cosine | run03 | 8092 | 2628 | 3749 | 15766 |
| DF_3/F/sine | run02 | 8092 | 2628 | 3749 | 15766 |
| Empty/cosine | run02 | 0 | 0 | 1 | 1 |
| Empty/sine | run02 | 2 | 0 | 4 | 4 |
| P_0/E/cosine | run03 | 155 | 155 | 224 | 362 |
| P_0/E/sine | run02 | 155 | 155 | 224 | 362 |
| P_0/F/cosine | run03 | 3070 | 1163 | 1579 | 5988 |
| P_0/F/sine | run02 | 3070 | 1163 | 1579 | 5988 |
| P_1/E/cosine | run02 | 131 | 178 | 229 | 358 |
| P_1/E/sine | run02 | 130 | 178 | 228 | 357 |
| P_1/F/cosine | run03 | 3044 | 1186 | 1571 | 5978 |
| P_1/F/sine | run02 | 3044 | 1186 | 1571 | 5978 |
| P_2/E/cosine | run02 | 99 | 121 | 155 | 241 |
| P_2/E/sine | run03 | 98 | 121 | 155 | 240 |
| P_2/F/cosine | run02 | 2992 | 1121 | 1507 | 5806 |
| P_2/F/sine | run03 | 2992 | 1121 | 1507 | 5806 |
| P_3/E/cosine | run02 | 112 | 118 | 169 | 263 |
| P_3/E/sine | run02 | 111 | 118 | 168 | 262 |
| P_3/F/cosine | run03 | 2916 | 1098 | 1516 | 5669 |
| P_3/F/sine | run03 | 2916 | 1098 | 1516 | 5669 |
| R/cosine | run02 | 579 | 340 | 511 | 1114 |
| R/sine | run02 | 579 | 340 | 511 | 1114 |

## 予定済み16件の非加法性

[interaction.json](interaction.json)の式は同じ軸のΔ=F−R−E+Empty。
RZ/CX/sizeの固定program削除介入に対するcompiler非加法性で、一意な因果配賦ではない。
depthは各行で保存し、部品depthの加算や同じ差分診断には使っていない。
R単独は残差削除でT-halfが隣接しcompiler簡約が効くため、R579 RZ/CX340をFullへ単純加算できない。

| trajectory/axis | ΔRZ | ΔCX | Δsize |
|---|---:|---:|---:|
| P_0/cosine | 2336 | 668 | 4513 |
| P_0/sine | 2338 | 668 | 4516 |
| P_1/cosine | 2334 | 668 | 4507 |
| P_1/sine | 2337 | 668 | 4511 |
| P_2/cosine | 2314 | 660 | 4452 |
| P_2/sine | 2317 | 660 | 4456 |
| P_3/cosine | 2225 | 640 | 4293 |
| P_3/sine | 2228 | 640 | 4297 |
| DF_0/cosine | 3820 | 1100 | 7393 |
| DF_0/sine | 3823 | 1100 | 7397 |
| DF_1/cosine | 3665 | 1054 | 7087 |
| DF_1/sine | 3668 | 1054 | 7091 |
| DF_2/cosine | 3189 | 916 | 6157 |
| DF_2/sine | 3192 | 916 | 6161 |
| DF_3/cosine | 3857 | 1110 | 7461 |
| DF_3/sine | 3860 | 1110 | 7465 |

## 原値の保全・未解決事項・STOP

[recovery_evidence.zip](recovery_evidence.zip)は新output QPY13・各metrics13・worker記録13の計39 memberを可逆保存する。
[archive_members.json](archive_members.json)で各原bytesのSHA256を示す。旧input36・旧output23・native/trajectoryは
[run02 archive](../run02/native_evidence.zip)に保全し、今回のarchiveで置き換えない。
元のlocal rawファイルも保持した。[verification.json](verification.json)にhash/可逆性/旧証拠の一致を記す。
検証時にQPYを再build/再compileしていない。compiled global phaseは元metrics/QPYのまま保存する。

Stage 2のfinite bias両軸UNRESOLVED、finite-bit実効law平均、全lawρ、全compiled operator作用精度、
母平均費用/tail、shot/RPE同精度総資源は未解決。今回の36価格完成はこれらを解決したことを意味しない。
旧Stage 2 `STOP_STAGE2_UNRESOLVED_COMPONENTS`、run01 FAIL、run02 partial failureと全STOPは変更しない。

次の担当はユーザー：この固定cohortの価格差・非加法性を踏まえ、追加平均費用評価やStage 4準備の必要性を判断する。
Codexは別の明示指示まで追加標本・新条件・Stage 4・主要claim採択・論文改訂を開始しない。
**今回の13件限定回復は完了。公開・remote一致確認後mandatory STOP。**
