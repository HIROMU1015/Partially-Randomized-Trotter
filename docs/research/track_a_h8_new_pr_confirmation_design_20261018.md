# Track A H8：新PR2候補の独立確認設計（実行未承認）

利用者の「その方針で進めて大丈夫」に基づく**対象・比較・必要標本数の技術設計**。
新sampling/native build/compile/signal/fit/shot再計算は0。新seedの割当も0。
H6高次PFの13/45/45本の独立確認とはbranch・SOURCE・grant・output・標本を分離する。

現結果の正本は[64c85f7a833b5950ef2ccae623708b73081edcd6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64c85f7a833b5950ef2ccae623708b73081edcd6/docs/research/track_a_pr_new_a_total_cost_results_20261018.md)。
既存の科学計算SOURCEは変更しない。本案のproduction adapter・fresh seed台帳・最終input/runtime sealは未作成で、
`future_SOURCE=null`、`technical_sealed=false`、`approved=false`、`runtime_authorization=false`。
[設計bundle](../../artifacts/resource_applicability/track_a_h8_new_pr_confirmation_design/2026-10-18/README.md)。

## 1. 対象と固定比較

共通条件：linear H8、1.0Å、STO-3G、tol-only1e-8 DF28、16 spin modes/sector4900、T=.8、P=0。
既存S2 PR・有限K4 RTE・元component law・ordinary ancilla・symmetric-directional control、元DF順序を保持。
新しい候補探索ではなく、既に選んだ2候補を**新標本を一つも見る前に固定する**。

| 独立scope | 固定新PR | ε | δ=T/q | R=q*r | 固定PF対照 | PF RZ/軸 |
|---|---|---:|---:|---:|---|---:|
| H8_CONFIRM_P8_Q2 | B2_p8_q2_2nd_r4_K4 | .05 | .4 | 8 | B0I_p11_q1_2nd | 42,268 |
| H8_CONFIRM_P7_Q16 | B2_p7_q16_2nd_r4_K4 | .0005 | .05 | 64 | B0I_p15_q4_4th | 668,636 |

主評価は通常signed bias・allowance倍率1。元signal/reference/normalization、保存された軸別shot、
α配分、numerical allowanceを固定し、再最適化しない。対照単価は既存の直接実測で、新PF compileは0。
対応する元preparation、signal record、primary/secondary accountingのhashをplanへ結合した。
H8 profileは元識別子の継承で、原profile fileのbyte SHA照合済みとは主張しない。
実行時compiler/runtimeの固定・fresh byte照合を別途完成する。

主estimandは元無条件lawの `μ_j=E[N_real*C_cosine+N_imag*C_sine]`。
直接得るのは新しい固定標本の平均/SD/SEであり、今回の設計だけでμ_jの上界は認定できない。
対照のmeasurement noiseを測る実験ではなく、固定compilerの決定論的単価を比較する。
184集合の他候補にはmodel推定が残るので、対照固定2件への比較と全184候補実測最良を混同しない。

副評価は、同じ新標本を同じε・倍率1の保存済み経験的非相殺shotで重み付けする記述的比較とする。
通常/非相殺に別々のtrajectoryを生成しない。主検証は通常biasの2条件で、4条件の同時保証は主張しない。
旧PR `B2_p7_q2_2nd_r4_K2` と `B2_p7_q8_2nd_r4_K4` のn32平均/SEは歴史的な比較として保持する。
旧平均を既知母平均として検定しない。旧PR側の追加fresh samplingは本案に含めない。

## 2. 保存分散を用いた標本数案

公開済みの新A平均・paired SEだけを読み、原trajectoryの費用平均や候補順位は再計算していない。

| 候補 | 探索n8のG平均 | 保存paired SE | 固定PF G | 探索点の対PF削減率 |
|---|---:|---:|---:|---:|
| p8/q2/r4/K4 | 1,227,928,105.5 | 8,346,671.197 | 1,267,321,444 | 3.1084% |
| p7/q16/r4/K4 | 91,256,285,948,891 | 224,590,173,149.943 | 94,193,016,652,860 | 3.1178% |

計画上のnormal plug-in式は `n=ceil[(z_(1−α_C/2)+z_.8)^2*(f*s_8/Δ)^2]`。
`s_8=SE_8*sqrt(8)`、Δは保存PF費用−探索平均、fはSD倍率1/2/4。
2件を事前固定したfamilyとして、計画上α_C=.05、各片側α=.025、仮定power=.8を使う。
このα_Cは**古典回路費用標本の計画値**であり、量子shotのαやβ/RPE誤差予算を変更しない。
[NISTの標本数式](https://itl.nist.gov/div898/handbook/prc/section2/prc222.htm)は既知σ/normal条件の式で、
ここでは未知のσを小さい選抜n8から置き換えた参考シナリオにすぎない。
[Bonferroni](https://itl.nist.gov/div898/handbook/prc/section4/prc473.htm)も、各marginal inferenceが妥当であって初めて同時保証へ使える。

| 候補 | f=1 | f=2 | f=4 | 本案 |
|---|---:|---:|---:|---:|
| p8/q2 | 3 | 12 | 46 | 新48軌道 |
| p7/q16 | 1 | 2 | 6 | 新48軌道 |

**推奨案は両候補各48本**。最大46を8-task単位で切り上げ、事前固定の同数holdoutとした。
小さいplug-in値1/3を「十分な標本数」として採用せず、従来n32も無条件に踏襲しない。
SD4倍は保証された安全係数でなく、感度シナリオ。n48の実power、CLTの精度、tail分散は未保証。
この提案は点費用の独立再現に限定した実用的な選択で、母平均保証に必要なnとは呼ばない。

## 3. 希少eventの可視性：nを増やすだけでは解消しない

2候補の保存8軌道ずつを、event digest・seed・元law・native終了receiptへ照合した。
p8/q2は8×8=64 event、p7/q16は8×64=512 eventが**全てTaylor次数0**だった。
次数0でもcomponentは乱択され、回路費用は0ではない。既存SEはその範囲のばらつきを含む。
次数2/4の費用が0と確認されたのではなく、これらは観測されていない。

元lawのIID occurrenceという条件で、保存p2/p4から
`π_nonzero=1−(1−p2−p4)^R`、`π4=1−(1−p4)^R`、
`P(一度以上観測)=1−(1−π)^n`を計画算術で評価した。新event抽選は0。
保存floatの丸めを元lawの再fit/再正規化で補正していない。

| 候補 | n48で次数2又は4を一度以上 | n48で次数4を一度以上 | 97.5%で非zeroを一度以上見るn | 同じく次数4のn |
|---|---:|---:|---:|---:|
| p8/q2 | 29.17% | 0.00517% | 514 | 3,427,720 |
| p7/q16 | 34.60% | 0.000978% | 417 | 18,095,918 |

最後の2列は**可視性だけ**のnであり、費用平均の精度や優位性の必要標本数ではない。
514/417や数百万本を推奨するものではなく、今回のcall予算にも含めない。
n48で未知層を見ない結果は十分予想され、その場合も費用0や厳密上界を代入しない。
仮に48軌道中の該当eventが0なら、IID指標の片側97.5%二項上限は約7.40%。
これはevent確率の上限で、費用寄与の上限ではない。[NIST exact binomial](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbino.htm)。

元lawはp/q/rで変わる。旧p7/q4・q8の条件付きs2/multi/s4費用を、p8/q2又はp7/q16へ移送しない。
無条件標本へ条件付きpilotを足して平均することもしない。
母平均優位を主張するには、候補固有の完全な層質量と費用推定/有効上界を結ぶ別設計が必要。
その条件付きsampling・層別会計は今回採択・実行せず、別scopeとして残す。

## 4. 将来の判定：実行完了と科学的な結論を分ける

各候補の48本全てが正常に両軸compile/native回収でき、事前固定byte/normalization gateに合格した場合だけ、
固定shotで `G_i=N_real*C_i,cosine+N_imag*C_i,sine` を作り、平均・SD・paired SE・次数別観測を保存する。
両軸を独立な費用標本として数えず、同じevent列からの共分散を保つ。

| 出力 | 条件・意味 |
|---|---|
| POINT_REPRODUCED | 新48本の点平均が固定PF費用未満 |
| POINT_NOT_REPRODUCED | 点平均が固定PF費用を上回る |
| EXACT_POINT_TIE | exact算術で同費用 |
| POPULATION_CONCLUSION_UNRESOLVED | 全supportに妥当な推定・上界が未検証。上のpoint結果と別軸 |
| TECHNICAL_STOP | SOURCE/input/law/seed/イベント対/normalization/native/IO/実OOM等の不合格 |

探索n8をn48へpoolingしない。費用が悪いtrajectory、希少event、外れ値を削除しない。
確認結果で別候補へ変更せず、途中平均を見た成功終了・標本追加・サンプル数再推定をしない。
SEだけの95%保証、valid p-value、母平均最良は出さない。点再現PASSと母平均未確定は両立する。

## 5. seed・source・入力・準備を固定する手順

1. 本設計/対象2件のselection receiptを、新抽選より前に固定する。
2. production adapterのSOURCEと9系統のbinding（科学条件・元source・入力・preparation・law・compiler/runtime・shot・seed・output/grant）を固定する。
3. 新phaseを`INDEPENDENT_UNCONDITIONAL_CONFIRMATION_N48`として実装する。旧A backendの`EXPLORATION_N8`出力を新しい確認済み結果と読み替えて流用しない。
4. seed namespaceを`H8_NEW_PR_HOLDOUT_V1|future_SOURCE|sealed_plan_digest|cell|replica`に分ける。
   元master→step seed APIと元event分布は保持。旧探索/確認/pilot/STOPのmaster・派生seed、今回A128 masterと派生seedとの衝突を拒否する。
   実装が変わってもAの元seedを導出し直して確認標本と呼ばない。今は新seed0、台帳は未seal。
5. seedの非重複は独立stream設計の証拠で、数学的独立性そのものの証明とはしない。
6. 元p7/p8 preparation archiveと保存referenceを再利用する。SCF/DF/state/full-H reference/A_SIGNAL生成なし。
7. 元native quantum kernel、scalar一回適用、reference不回転、DF順序、symmetric control、compiler opt1/seed17/rz,sx,x,cxを維持する。
8. 実入力manifest/venv/runtimeをbyte照合。未使用output/lock、fresh CPU/memory/OOM/filesystem block/inode/quotaを確認する。

既存の採択済み2セルのnative control証拠は照合済み。本案の予算は保守的に**新代表1本/候補のcontrolも含む**。
新adapterのphase/law/normalizationを人工・metadata検査してから、各候補の新代表control合格後に残47本を投入する。
これはpipelineの再準備がまだ必要であることを意味し、現在のplan/grant案だけでは起動できない。

## 6. 並列化・call予算・停止

| prospective call | 2候補合計 |
|---|---:|
| 無条件trajectory | 96 |
| full Hadamard wrapper compile | 192 |
| native evolution | 96 |
| occurrence | 864 |
| RTE event | 3,456 |
| control probe上限 | 8 |
| independent event oracle上限 | 2 |
| 新PF compile/SCF/DF/state/signal/fit/量子shot実行 | 0 |

2候補のgrant/ledger/output/STOPを独立にする。各candidateの最初のcontrolを並行し、合格したcandidateの残りを順次並列投入する。
q16の予測負荷を考慮して長いtaskを先に処理し、同じevolution/preparationを両軸で再利用する。
元のhash/fingerprint再利用を保ち、候補ごと/trajectoryごとの重複状態準備を避ける。内部thread1でnested oversubscriptionを避ける。
worker数はfreshなCPUと実効memoryから決め、旧15worker等を恒常capにしない。最大96独立taskという有限仕事量は資源制限ではない。
phase/total wallやAS/RSSの新しい強制上限は設けず、共有環境・他job・既存venvを変更しない。

保存native CPUからの単純な参考値は、fresh代表controlを含め16 workerで約2.12時間、32 workerで約1.57時間。
実測speedupや完了保証ではなく、最初のq16 controlがcritical pathになる。希少event・混雑・IO等で延び得る。
admissionはcontext＋worker見積り＋headroomを使い、過去の最大RSSを未観測eventの上界にしない。
一時的PSI・CPU競合・無関係worktreeの変更で強制終了せず、実OOM/継続不能・自分のnative異常・科学不一致はown-scope STOP/reap。
before-call予約と過去STOPを保全し、返却/resetなし。失敗したscopeのretry・自動拡張・次stageは0。

## 7. 今回作成したものと次の承認

[計画算術tool](../../scripts/h8_confirmation_sample_design.py)と[人工11検査](../../tests/test_h8_confirmation_sample_design.py)を追加。
NIST例の式、SD/多重性、exact margin、極小確率の桁落ち、検出nの最小性、zero-count上限、call budget、異常入力を検査した。
実H8 metadata16件の読取と、公開済みscalarに基づく計画算術のみ。新科学sampling/compile/native buildは0。
作者の技術設計/人工検査であり、独立したGPT/Claude科学reviewではない。
大きな既存索引、現結果64c85f7a…、H6高次PFや他jobを変更しない。

今後の承認対象案は、**上記2候補各48本、192 wrapper compileと必要なcontrol、固定4 endpoint（通常2・記述的非相殺2）への限定会計**。
承認を受けた場合、Codexがproduction adapter・人工検証・SOURCE/seed/input/runtime sealを完成し、
fresh gateの後に独立one-shot scopeを実行、結果を保存してmandatory STOPする。
本資料自体は実行承認ではない。母平均保証を必須とする場合は、本案を十分とはせず別の層別/上界設計を選ぶ必要がある。

**次の担当：ユーザーが検証目的と実行範囲を承認。Codexは承認された範囲の実装・seal・実行を担当する。現在はSTOP。**
