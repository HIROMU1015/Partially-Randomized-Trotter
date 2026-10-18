# H4固定1 Å：PR境界・希少層・時間・DF政策の追加検証

利用者の「この未実施についてH4でも行ってほしい」を4系統の実行指示として受領した。旧H4のD/A/N、H6/H8の結果・SOURCE・grant・ledgerは変更しない。geometryの6距離系列は対象外。

## 固定した範囲

| Stage | 追加範囲 | 再利用と限定 |
|---|---|---|
| R | p3、q1/2/4/8のr1、q16のr2/4、K2/4。12 signal | rank12/T=.8の保存入力・full reference。sampling/compile 0。費用優位は評価しない |
| B | p3、q4/8、r4、K4。s2_multi/s4各8軌道、合計32軌道64軸 | 元のmultinomialを条件付け、component・量子重み・normalizationを保持。母平均保証・無条件費用再会計なし |
| T | δ=.2、T=.4/.8/1.6、q2/4/8。固定5構成 | p9 B0/B0I S2、full12 S2/S4、p3/r4/K2 PR。T=.8のsignal・実測を全て再利用。新PR各32、合計144軸 |
| F | H4/1 Å/STO-3G、SCF/integrals各1、DF分解1、tol=1e-8、固定rank指定なし | generation順・signed係数・cutoff0。weighted Hermitian projection、N=8 full-Fockのengineering budget 9.9e-11 Hartree。独立した入力と状態を保存 |
| FP | F正常終了後、その保存入力を別grantへ結合して固定5構成を比較 | q4/T=.8。p=ceil(3rank/4) B0I S2/S4、full S2/S4、p=ceil(rank/4)/r4/K4 PR各32。72軸。旧rank12の対応5構成は保存結果を再利用 |

4つの問いをR/B/T/F+FPで扱い、各実行単位は別grant・one-shot lock・output・native wait4を持つ。Fの終了でSTOPを記録した後、今回の明示指示に含まれるFPだけを別に起動する。次の研究課題への自動進行はしない。

通常signed biasと経験的非相殺bias、5精度(.05/.01/.005/.001/.0005)、allowance倍率1/10を区別する。厳密bias保証はnull。paired trajectoryの実測整数費用を用い、母平均や全候補最良の保証は付けない。未取得費用を0補完しない。

## 実装・高速化

`h4_followup_contract.py`、`h4_followup_science.py`、`h4_tol_only_input.py`、`run_h4_followup.py`、`h4_followup_witness.py`、`prepare_h4_followup.py`を追加した。既存のH4/native forward、条件付きmultinomial、shot会計、fingerprint再利用、所有workerのfork/wait4を再利用する。既存の封印済みmoduleは変更しない。

R/B/T/Fを同時起動し、F終了後は空いたCPUをFPに割り当てる。freshな物理core・RAMに応じてworkerを選び、計算workerは1 thread。独立候補・軌道を並列に処理し、同じ軌道のcosine/sineではevolutionとevent列を共有する。各群の先頭軌道は独立event oracle・control/branch/X/Yを通してから残りを処理する。preparation・spectraは親で一度構築してCOW継承し、Numba/OpenMPのproduction operatorはfork後にのみ初期化する。Fは小さい36次元sectorを直接対角化する。

資源使用の数値は起動時receiptに保存する。過去のworker数・memory ceiling・PSI閾値は移送しない。SOURCE/input/compiler/runtime変更、nonfinite、独立oracle/controlの不一致、実native exit異常ではSTOPする。PSI上昇・無関係なworktree更新だけでは停止しない。共有環境と他利用者のprocessを変更しない。

## 人工検証

新13件と既存H4の17件、計30件がPASS。新規の人工native接続試験は8qubitの合成DFを使い、T=.4/1.6と条件付き2層についてpaired実compile・control・独立event oracleを検査した。実H4を使った試験ではない。独立レビューとは呼ばない。

## 解釈の範囲

時間比較は固定δの累積時間依存であり、固定qでのT効果ではない。希少層のn8は条件付きpilotの点推定である。F/FPではSCF軌道と保存状態も新規生成されるため、旧系列との差をDF rankだけの因果効果と断定しない。共通のDF生成政策による代表panelを追加する検証であり、legacy系列の置換や全格子再最適化ではない。

実行状態・結果・SOURCEは後続のreceiptに追記する。この準備本文だけでは正常完了を主張しない。


## 実行追記：R/B/F完了

実行SOURCEは `dfef38fc8107dd76f5d71dcdeca31f8d3dc39f64`。R/B/T/Fを別grantで並行起動した。fresh時点の64物理coreからR12、B26、T25、F1を選択し、各workerは1 thread。R/F完了後、B/Tと重ならない13 coreへFPを結合した（FPのSOURCEは `64e0fc91`、科学moduleは元SOURCEと同一、追加は資源割当preparerだけ）。

Rは12/12 signalでnative exit0・final gate PASS・mandatory STOP。5精度の経験的適格件数は通常/非相殺とも12/10/10/8/6。これはsignal適格性で、費用winnerではない。wallはdriver 11.79秒。

Bは32/32軌道・64/64軸、全worker exit0/reap・final gate PASS・mandatory STOP。driverを含むnative wallは133.01秒。以下のcosine/sine単価は同値で、各n=8の条件付き平均とsample SE。

| q | 層 | 平均RZ/軸 | sample SE |
|---|---|---:|---:|
|4|s2_multi|13,004.75|131.58|
|4|s4|13,144.25|200.34|
|8|s2_multi|25,306.125|157.07|
|8|s4|25,731.625|263.65|

Fもnative exit0・final gate PASS・mandatory STOP。tol-onlyの実rankは10、sector状態残差は1.96e-15未満、projectionのN=8 engineering評価は約8.28e-14 Hartree。旧rank12とその状態・SOURCEは変更していない。SCF/DF/state生成はFだけで各1回、新費用比較はFPの別grantに限定した。

FPの追加preparerは `prepare_h4_followup_fp.py`。起動時に稼働中の兄弟stageが使うCPUを除き、終了したstageのCPUを再利用する。既存stageのSOURCEやgrantは変更しない。


FPも72/72軸、native exit0・最終gate PASS・全回収で終了（native wall289.09秒）。終了時metadataを確認する3人工試験もPASSし、今回の人工検証は計33件。残るTはT=1.6 full S4を処理中。`collect_h4_followup_results.py` は全stageの終了後に予定軸数・168 workerのnative回収・結果SHA・SOURCE/gateを照合して資料化する。科学実行・retry・追加候補は行わない。
