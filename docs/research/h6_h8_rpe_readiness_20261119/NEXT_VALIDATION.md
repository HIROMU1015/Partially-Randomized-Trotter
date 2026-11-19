# 初回H₆の通常探索に接続する手順

## 準備

- [COVERAGE.json](COVERAGE.json)のFull19/M5 1構成を使う。h=0.1、要求幅1.6、M=12、保存全DF19 solver基準、目標エネルギー誤差の十進値 `0.00159360010199040 Ha` を厳密に二乗する。旧priorは入力診断として保全し、Q3には通常の終端decoder・law・lossで正しく受理された入力を渡す。
- H₆ loaderと既存Full PF mapperを使い、固定H₄ spec・p10を参照する候補生成を物理系・rankを引数に取る形へ分ける。初回の探索範囲はFull19/M5/要求幅1.6の1構成。全13個の (T,q_{PF},T/q_{PF}) とfragment/scalarを照合してH₆のcanonical IDを発行する。旧ラウンドIDとの対応を保持する。
- 保存raw 13個を新configurationに対応するpacketへ接続する。古い候補のX/Y符号、identityの位相会計、raw/correctedを混同しない。保存されたQ3用law・closureとの適合はadapterと実入力dry-checkで検査する。M=12は既存13行の受入口を使える。M=11は同じ信号・価格のprefixを使う場合も、独立した評価IDとMから再構成した整数配分で評価する。
- 価格と信号をregistryに別々に登録する。価格の単位は状態準備を除いた測定付きfull controlled Hadamard wrapperの1 shot当たりcompiled RZ。Qiskit 1.3.0、rz/sx/x/cx、opt1、seed17、元のcontrol・scalar、軸別証拠を保持する。受付を満たさない軸は不足のまま残す。H₄係数・構造価格・旧暫定予測は代入しない。
- 保存カタログと探索マップを物理系・snapshot・評価契約で分け、旧H₄のrunは固定版で再開する。PRを追加するときは同じ (L_D)、tail dictionary、K、κに由来する全roundのscheduleを使う。既存H₆のPR資料には途中でprefixとRを切り替えた系列があるため、単一のPR候補として結合する前に範囲ごとのbindingを確認する。

## 最初の有限科学案

1. **初手はm0、m1、m2のX/Y。** §10.5の最新結果・旧ID・元receipt・進行中jobの差分を確認し、同等証拠を再利用する。残る不足に限り最大6 wrapper compile。対象は保存DF19 Full/M5、(q_{PF}=1) で (T=delta_{m actual}=0.1,0.2,0.4)。m3の幅0.8の価格を代用する根拠がなく、短時間の3幅を独立に埋められる。compile後は作用・compiler・軸別receiptが通った価格だけを保存する。異常・出所不明は対象軸を保留し、6軸の結果保存で停止する。この段階の新しい信号・Q3は0件。
2. **m7〜m12は長時間価格を別に検証。** 保存q=1/2/4のQPYとH₄の閉包検査の入口を照合し、H₆固定幅1.6で同じ検査が立つかを調べる。QPYや同等証拠が不足すればm7のq=8・X/Yを最大2 compileする案を別に固定する。q=8での一致が確認できてもq=256は未資格。固定幅の反復・終端処理を保証できる範囲と、独立長点が必要な範囲を分け、採用範囲まで覆えない場合は総額を未確定とする。 [H₄で使用した閉包採用条件](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c3f761f0/docs/research/pf_closure_price_recommendation_20261115/REPORT.md)
3. **全入力が揃ったら通常探索を1構成で検収。** 最初はM=12・格子点 (s=1) の最大1 Q3。入力dry-check、worker、結果保存、探索器の採用、同じ予算での再開0件を確認する。worker制約なら技術保留、MSE上界で適合、下界で不適合を区別する。M=11はその検収後に同じ構成で独立した有限案として追加する。M=13はm13入力と固定Q3の検証範囲が別に必要。
4. **H₈を同じ入口に接続する。** 保存DF28 solver・位相枝の条件を確認してh/M/精度を固定し、[H₈の照合表](COVERAGE.json)のm4と同じ条件を使う場合だけ信号と価格を再利用する。spanの広い全round取得に入る前に、採択された初回Full PFの不足だけを見積もる。H₈旧Stage B/Cの単時刻適格性はRPE終端MSEへ移さない。

今回の公開対象は保存証拠の照合表とこの計画。新規科学実行のSOURCE、入出力先、有限範囲は実行前に固定する。
