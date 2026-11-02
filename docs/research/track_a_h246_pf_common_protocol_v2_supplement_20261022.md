# 共通実験プロトコルv2補足：H2/H4/H6共通決定論PFの正式採用

2026-10-22 JST。**ADOPTED_FIXED_DOMAIN_SUPPLEMENT_DOCUMENT_ONLY**。
ユーザーが結果commit `90f1cd55a0d963953696bd0a27d9d041ff939bb7` の588候補・未解決0件を正式な研究成果として採用した。
これは既存結果の実験後の整理であり、過去への事前登録、v2本則の条件変更、AD03の正式採択、最終evidence freezeではない。
[v2本則](track_a_pr_resource_evaluation_protocol_v2.md)の既存本文・過去の固定科学契約は保全し、末尾に本補足への参照だけを追加する。

## 比較単位と固定共通格子

linear H2/H4/H6、隣接1 Å、STO-3G、tol-only DF threshold 1e-8、保存rank3/10/19、T=.8、P=0。
各系の保存Hamiltonian/state/snapshot・preparation・referenceのidentityを比較単位とする。
S2、Yoshida4、m5_bestの固定binary64係数・signed schedule、q=1/2/4/8、δ=.8/.4/.2/.1。
full cosine/sine Hadamard wrapper、symmetric-directional control、元scalar一回、Qiskit1.3.0・rz/sx/x/cx・opt1・seed17、backend/coupling/layout未指定。
H2既存scalar guardを他系へ移さない。

| 系 | DF rank | B0/B0IのL_D | Full | 候補数 |
|---|---:|---|---:|---:|
| H2 | 3 | 0..2 | 3 | 84 |
| H4 | 10 | 1..9 | 10 | 228 |
| H6 | 19 | 5..15 | 19 | 276 |

各方法・prefixに3公式×4q。全588候補・2940 paired精度endpoint。
H2 prefix0とFullは別登録のまま保持する。q16、H4 DF12、過去の広域候補、別公式を含めない。
今回の最良はq≤8の有限共通格子内の最良であり、歴史的全域最良・連続域最適ではない。

H2/H4は既に科学配列hash一致を確認した再生成容器、H6は原snapshotのbyte復元。
同じrankやcompiler optionsだけで同一入力とはしない。[SOURCE・入力identity](../../artifacts/resource_applicability/track_a_h246_pf_common_adopted/2026-10-22/source_catalog.json)を正とする。
本資料統合ではsnapshot・state・reference等を開かず、保存済みidentity証拠を参照する。

## 今回の共通評価政策

通常signed bias、ε=.05/.01/.005/.001/.0005、allowance倍率1、α_axis=.025。
元m5系列の経験的allowance、floor1e-9、安全倍率10、roundoff倍率64、保存uを下回らない規約、log_B margin floor1e-12かつ保存margin以上、u/headroom≤.01を継承する。
凍結binary64 shot算術と保存済み別経路照合を使用し、費用は `N_cos × RZ_cos + N_sin × RZ_sin`、状態準備費用P=0。
適格性・shot・費用・順位は結果commitで評価済みの値を転記する。本統合で再評価しない。
経験的allowanceは厳密な誤差certificateではなく、analytic shotは量子測定shotの実行実績ではない。
過去のu・shot・非相殺政策・v2正式条件を遡及変更しない。

## PR cohortと二つの比較

PRは保存EXPLORATION outer32/K4、各系prefix1/3/5、3公式×4qの元探索集合を参照する。
outer32は内部RTE規約であり、費用標本数n=32を意味しない。各比較行の保存n・平均・不偏分散・paired SE・SOURCEをそのまま保持する。
pilot、旧cohort、PR対PRの118独立確認標本、v2本則の旧独立確認Pは混合しない。

比較Aは同一公式内の45行、比較Bは保存PR探索集合最良対全3公式のPF共通格子最良の15行。
例えばm5内のPR低費用を比較Bの結論へ移さない。
[採用データセット](../../artifacts/resource_applicability/track_a_h246_pf_common_adopted/2026-10-22/README.md)と
[固定結果報告](track_a_h246_pf_common_final_20261022.md)に両者を分けて保存する。

## 完了状態と取得履歴

全候補の適格性が確定し、必要な候補の費用が揃って未解決0。全精度不適格候補をcompile済みと解釈しない。
新規signal、旧signal再利用、metadata別観測、保存同native価格、未使用保存価格、不適格を区別する。
旧legacy217価格はidentity未証明のため使用0。同native保存価格の使用はH4 22/H6 30 pair。
共通stageの新signal186、固有新価格378候補756軸、非返却予約を含むcompile実呼出776軸は保存値であり、今回の新規call数は0。

H6 metadata63は別契約で観測済み。旧52signalを保持し11候補のrawは新規取得と区別した。
中断COSTの187完成pairは保存byte/vector/gate受入監査による利用、新22pairは新grantの正常終了。
元parentの終了時刻・原因・exit/reapは不明のまま。旧STOP・394 compile予約を保持し、中断stageの成功receiptを補作しない。
新22workerの正常終了と旧parentの終了状態を混同しない。

## 採用範囲と次の判断

比較Bの保存点ではH2は5精度すべてPF低費用。H4/H6はε=.001/.0005でPR低費用、他の3精度はPF低費用。
H6の保存分母分解ではm5 prefix拡張比は全5精度で1であり、S2/Yoshida4を含める公式拡張の影響が主だった。
入れ子集合の保存点比較であり、探索範囲だけの単独因果効果ではない。

有限候補・保存探索標本の選択効果は未補正。保存SEは母平均順位の保証や選択後CIではない。
S4未観測、rare eventの費用・分散・tail上界はnullで0ではない。
経験的allowance、指定compiler・P=0・保存DF・有限Tに限り、energy/QPE/RPE全round・実backend/noise・FT資源・一般的PR優位を主張しない。
既存主表・Figure2・独立確認はそのまま保全する。
AD03、掲載先の配置変更、最終evidence freeze、詳細科学レビューは別判断として未開始。
**文書・構造検証・許可範囲の公開完了後mandatory STOP。**
