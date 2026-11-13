# 固定H4：最大20 complex meanの有限screen案

2026-11-13 JST。**準備完了・科学未実行・実行承認は別途必要。** 正本は[PLAN.json](PLAN.json)。B51500b7の11候補を暫定screen集合として保持し、最適候補やRMSE適格候補として採択しない。

## 共通条件

Linear H4/1.00 Å/STO-3G、保存tol-only DF10/weighted-Hermitian completion、sector(2,2)/36、同じH/state/snapshot。T=.1×2^m、M12、元prior/decoder/誤差budget・P0を変更しない。Hamiltonian=`c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0`、state=`9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b`、snapshot=`12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96`。

Full PFはone-body＋全10fragment。標準Y4は既存3 signed piece、M5 bestは既存11 signed pieceを順序・binary64値のまま使う。Δはmacro-step幅。T4はB0I/prefix7/Y4/Δ.4でtail identity scalarだけを追加し、B0と同一視しない。

Full-H RTEはone-body込み全非scalarをランダム化、決定論block0、K2又はK4、scalar抽出後total Tで一回。κなし、q=max(1,ceil(T/.25))、R=max(1,ceil(448T/2))、単一正時間piece allocation=[R]。旧light q16/R16やT2/q8/R448は新候補と異なる。

## 新規20セル（X/Yでmeanを二重計数しない）

| label | m | T | q | R | physical candidate ID |
|---|---:|---:|---:|---:|---|
| F2-FINE | 6 | 6.4 | 64 | — | de9bc15cbc089f14b47f |
| F2-FINE | 12 | 409.6 | 4096 | — | 4841a53bcaf02927b34a |
| F2-COARSE | 6 | 6.4 | 16 | — | ad136c58d4184a236b9c |
| F2-COARSE | 12 | 409.6 | 1024 | — | 4f0aca817edccc7fbe17 |
| F4-FINE | 6 | 6.4 | 64 | — | b7008f551fc7645bbb62 |
| F4-FINE | 12 | 409.6 | 4096 | — | 91c97ad812a8c2f2606e |
| F4-COARSE | 6 | 6.4 | 16 | — | 90e364e6a0f5a079514b |
| F4-COARSE | 12 | 409.6 | 1024 | — | 7b27153f537fc83d4994 |
| F5-FINE | 6 | 6.4 | 64 | — | 2dae32828bbe34571df3 |
| F5-FINE | 12 | 409.6 | 4096 | — | c3d00525f7540edaf3c0 |
| F5-COARSE | 6 | 6.4 | 16 | — | d2201e76da058de4381b |
| F5-COARSE | 12 | 409.6 | 1024 | — | bee121c3e964c6b46e6a |
| T4-SCALAR | 6 | 6.4 | 16 | — | cb106dda2345d806dc52 |
| T4-SCALAR | 12 | 409.6 | 1024 | — | 6127f59aed85e2343729 |
| R2-ADAPT | 3 | .8 | 4 | 180 | FULL_H_11ee99e775aaf0e9b1d2 |
| R2-ADAPT | 6 | 6.4 | 26 | 1434 | FULL_H_afc046d3dd644fa7e50d |
| R2-ADAPT | 12 | 409.6 | 1639 | 91751 | FULL_H_71274f99be4b14071df3 |
| R4-ADAPT | 3 | .8 | 4 | 180 | FULL_H_12fef747dbd0e525271e |
| R4-ADAPT | 6 | 6.4 | 26 | 1434 | FULL_H_ec1405577ba19c2a83b3 |
| R4-ADAPT | 12 | 409.6 | 1639 | 91751 | FULL_H_3c0064b76925e75132d7 |

F2/F4/F5はS2/Y4/M5、FINE=.1・COARSE=.4。Full-Hのevent数はm3/m6/m12で720/37,284/150,379,889。これは**構造上の位置数**で、event列・trajectoryを生成しない。q_RPE=2^mは上記qと別定義。

## 保存再利用とreference

| 対象 | 再利用セルとSOURCE |
|---|---|
| Full PF6案 | m3/m4の12セル。155529eの長時間signal carrier、B51500 metadataのlocatorとphysical IDを固定 |
| B0I/Y4/.4 | m3 PF common 0b1886f、m4長時間155529eの2セル |
| B0/p7/S2/.1 | 旧89c91c6等の全13セル。m3は元PF common SOURCE。旧HP型Nは利用しない |
| PR-KEEP | A169d/bbaの全13セル。新mean取得・旧nominal再計算なし |
| 共通reference | m3/m6/m12ともbbaの保存独立pairを参照。m3はS1の旧中心＋新2経路、m6は元S0/P1、m12は旧89c91c6のpair |

合計40保存セルのlocatorをPLANへ固定し、新20と交差しないことを検証。保存fieldの存在・metadata一致は今回確認済みだが、数値の新しい比較判定はしていない。保存PR/B0のbias等を実行後のscreen表示へ使う場合は元の既存診断値をcopyし、再解析・新スコアを暗黙に追加しない。

**新referenceは0 action。** 保存pairのbinding/T/SOURCE/有限性・差の既存gateを実行時に確認する。必須referenceが不一致・欠測ならSTOPし、勝手にreference actionを追加しない。代替の新referenceが本質的に必要になった場合だけ、該当するTの2経路1組（最大2 action/時点）を別scopeで提示する。今回その追加は計画に含めない。

## 取得項目・精度・call上限

| 項目 | 固定scope |
|---|---|
| complex mean | Full PF12＋B0I2＋Full-H6＝**最大20**。raw/correctedの独立列は同じ候補の1mean評価に含む |
| finite-K normalizer | Full-H6候補のevent normalizer **6回**。logB/B/attenuation、one-ULP log感度、raw/corrected closure |
| 保存入力support | 同H/stateを既存loaderで1回復元、36次元の11既存block spectra＋全H非scalar spectrum1。PFのmacro-stepは7種類を共有。これはsignal計算の内部処理で、SCF/DF/state/ground-state生成ではない |
| 数値実装 | complex128、NumPy1.26.4。既存spectrumのHermiticity/再構成1e−9、保存pair1e−10、raw/corrected closure1e−9、PF power norm/unitarity1e−9。投影・正規化で救済しない |
| allowance | 旧R-aware算術式を再利用し、logB marginはnormalizer感度を下回らない。経験的・未認証。power境界normを全event観測最大値と偽らない |
| 局所出力 | corrected-reference complex bias、PF/discard・finite-K成分の適用可能な分解、corrected/raw phase・radius、数値検査・SOURCE。zero時のphaseはnull |
| しないもの | 新reference、trajectory、回路build/compile、N算出、decoder simulation、RMSE/G・候補最適化、8次PF、全13roundへの追加は0 |

Full-Hではevent時間`(T/q)/R`の演算順を固定。group numerator P(dt)^Rとnormalizer b^Rから、corrected/rawを各々q乗して伝播する。rawは最終corrected/Bの代入ではない。保存full-H referenceをexact-tail対照として使い、新norm観測はnull。PFは元signed順でmacro-stepを形成してq乗し、Fullはconstant、B0Iはconstantとtail identityを元規則で各一回だけ適用する。

## 実行条件とSTOP

本準備の公開commitを新科学SOURCEとして指定し、`h4_four_method_screen.execute(root, fresh_output, source)`を**ユーザーの明示承認後だけ**呼ぶ。sourceは科学コード同定で、承認証明書ではない。準備CLIはmetadata-onlyのままにする。固定20セルを一括実行でき、段階ごとの新grantは不要。

出力はPLAN、20個の個別record、全体resultとSOURCE/保存再利用locator。N・global RMSE・科学的PASS/FAILはnull、資格はLOCAL_SCREEN_NOT_GLOBAL_RMSE。保存科学結果やBのcanonical IDを上書きしない。

SOURCE/input/binding不一致、非有限値、normalizer b≤1等の精度未解決、closure・spectral・power gate失敗で部分保存し**STOP**。候補差替え・R/K/δ/M変更・閾値緩和・retryをしない。並列中の開始済みworkerは結果を保存し、未開始cellは新たに始めない。科学的にbiasが大きいだけなら観測として残し、それだけで全RMSE FAILとしない。

局所screenは長時間のbias/減衰/数値安定性を次の候補選択の材料にするもの。新順位・自動filterの閾値は設けていない。**Bとユーザー**が必要な次の有限候補案を決め、全13round評価・全risk・価格へ進むscopeは別途承認する。今回の準備完了後はmandatory STOP。
