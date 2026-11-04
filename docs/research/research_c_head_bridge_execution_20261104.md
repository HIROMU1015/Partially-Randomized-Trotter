# 研究C：Head bridge・26-call限定科学実行結果

2026-11-04 JST。**固定SOURCEを変更せずHead bridgeだけを一度実行し、26/26 COMPLETE、correctness FAIL・retry・resumeは0。** Full-CAR headと保存sector headの差をArb384で包囲し、kernel head差上界約3.059775044622e-16、m12のhead transport上界約1.253283858278e-12を取得した。B2二項との接続後はm12で約2.670395084860e-11。資格は保存数学target間のsector36 reference接続であり、Stage P／全law synthesis／校正／CAは未評価。

## 固定SOURCE・対象・事前照合

- 科学SOURCE：`f658f72d5f309538cba23b0745c7e9389411758f`、tree `74488b1f734233e2611f80cace4308ba79fcb9d8`。
- [固定H契約](research_c_head_bridge_contract_unapproved_20261104.json)：`C_PR_R32_SAVED_HEAD_BRIDGE_V1`、最大26 logical units、Arb384。
- 契約SHA256：`306003b926bddb35b2011ad2dbb00d96801885df4937a0e540b87ba660fb232d`。
- [固定SOURCE manifest](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/technical/SOURCE.json) SHA256：`61684d17fa849d67f6717b019f3098edaef0c32daa06c85af8ccf07c7ca7775a`、162 file、保存入力48件。
- [技術準備・新43／関連355 PASS](research_c_head_bridge_stage_p_readiness_20261104.md)。ユーザーの実行許可は直前に推奨したH26-callに適用し、P41-callは実行していない。歴史的unapprovedファイル名・metadataを保持し、承認証明・原文保存・新grantは作成しない。

対象は保存H4・直鎖1.00 Å/STO-3G、number CDF8、元順LD2、partial ordered symmetric S2/s1/r32/K0、h=δ=.1、m0..12。Nα=Nβ=2、physical sector36。Endpoint SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。元g補正κ、核scalar、tail scalar、frame順、own-DF ground oracle／状態準備除外、既存P0・IID-grid・CA/RPE政策を変更していない。

独立branch `codex/research-c-head-bridge-execution-20261104`を固定SOURCEから作成し、fresh `artifacts/research_c_head_bridge_stage_p/2026-11-04/H-run01`へ一度実行した。[事前照合](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/H_execution_preflight.json)で公開SHA/tree・clean HEAD・契約/SOURCE/input hash・package・FE_TONEAREST・fresh不在を確認した。Python3.11.1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、python-flint0.8.0／FLINT3.3.1。SOURCE、runner、数値条件は無変更である。

## 実行・call実績

[Journal](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/H-run01/science_calls.jsonl)は26 attempt／26 COMPLETE、FAIL0。[STOP](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/H-run01/STOP.json)はretry/resume0とmandatory STOPを記録する。

| 区分 | COMPLETE |
|---|---:|
| Stage R/B2 metadata | 2 |
| 保存配列group | 1 |
| candidate・shape identity | 1 |
| 共通tail位相のmodel binding | 1 |
| sector不変性 | 1 |
| 3 head対の指数化・差包囲 | 3 |
| forward/reverse products | 1 |
| kernel head bridge | 1 |
| 13round transport | 13 |
| B2三項接続 | 1 |
| summary | 1 |
| **合計** | **26** |

保存J/Uのcandidate一致、full heads3×256×256／sector generators9×36×36、basis indices36を確認した。`bit_reverse(saved_interleaved_sector_indices)`で揃え、sector↔complementの両cross blockが全3 headでexact zeroであることを検査した。したがってcomplete head作用のsector制限が可能であり、Gaussian途中gateを射影してはいない。共通tail phase hex `0x1.51a72cd5c1278p-5`も保持した。

Stage Rのfull headsを再生成せず保存bitsから読込み、B2が使った保存sector generatorも変更しない。核scalarはG0内で一度、tail identityは共通の位相一度。κの対称化やUのpolar化、phase fitは行っていない。

## Head差の包囲

[head0](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/H-run01/head0.json)／[head1](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/H-run01/head1.json)／[head2](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/H-run01/head2.json)に、両側exp(−iG/20)のArb interval、generator差、半時間作用差を保存した。

| head | generator差のoperator2上界（表示近似） | 半時間作用差のoperator2上界（表示近似） |
|---:|---:|---:|
| G0 | 2.175583928817e-15 | 1.087771799577e-16 |
| G1 | 2.591433097601e-15 | 1.295686304370e-16 |
| G2 | 4.124631109945e-16 | 2.062260017411e-17 |

全3件でgenerator差と半時間作用差の保存lower endpointも正だった。従ってbit不一致は、捕捉した二つの数学的head targetの非零差として確認できた。Exact入力を使用し、Arb指数・積・差・normのroundingを包含している。表は読みやすい近似であり、保証に使う正本は原本JSONのexact rational endpointである。

[Head products](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/H-run01/head_products.json)のforward／reverse差上界は、いずれも表示近似約1.529887522311e-16。[Kernel bridge](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/H-run01/kernel_bridge.json)は、同じliteral tailを共有する二つのhead target間について

`||D_full−D_sector|| ≤ ||E|| (||R_full−R_sector|| ||L_full|| + ||R_sector|| ||L_full−L_sector||)`

を適用し、上界約3.059775044622e-16を得た。Stage Rのexact a/Mと保存event上界を使用した。Normを1、rangeを2へ丸めて代入していない。非unitary mapにcap2はない。

静的調査で確認したdense／sparse contraction・積和の丸め順序差と整合する大きさである。今回、正しいcandidate/basis/scalar/phaseを固定して差そのものを包囲した。一方、各entryの差を特定の積和演算へ割り当てる追加診断は実施していない。

## 13roundとB2への接続

[結果原本](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/H-run01/result.json)には、n=2^mのtelescoping上界 `H_m ≤ n e max(a,b)^(n−1)`を保存した。全13行でfinite exact上界を得ており、clipや未知値の0補完はない。

| m | head transport上界（表示近似） |
|---:|---:|
| 0 | 3.059775044622e-16 |
| 1 | 6.119550089244e-16 |
| 2 | 1.223910017849e-15 |
| 3 | 2.447820035697e-15 |
| 4 | 4.895640071395e-15 |
| 5 | 9.791280142790e-15 |
| 6 | 1.958256028558e-14 |
| 7 | 3.916512057116e-14 |
| 8 | 7.833024114232e-14 |
| 9 | 1.566604822847e-13 |
| 10 | 3.133209645693e-13 |
| 11 | 6.266419291388e-13 |
| 12 | 1.253283858278e-12 |

[B2 connection](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/H-run01/B2_connection.json)は、新Head差H_m、保存B2のatom→CAR上界B_m、CAR→binary64 referenceのU_mを**一回ずつ**加算した。既存totalを第四項にしていない。既包含のrounding、PF/RTE、normalization、categorical、shellの追加加算もない。

| m | H_m（表示近似） | 旧B_m（表示近似） | 旧U_m（表示近似） | H_m+B_m+U_m（表示近似） |
|---:|---:|---:|---:|---:|
| 0 | 3.059775044622e-16 | 2.805897600126e-17 | 1.004349214834e-14 | 1.037752862880e-14 |
| 6 | 1.958256028558e-14 | 1.474161094368e-15 | 5.666886902456e-13 | 5.877454116255e-13 |
| 12 | 1.253283858278e-12 | 6.840274819091e-14 | 2.538226424213e-11 | 2.670395084860e-11 |

これは**physical sector36のreference target間のraw complex signal差**の接続である。同じtail mapのheadだけを替えた差で、normalized sector状態すべてに作用するoperator上界から換算できる。Ground近似誤差を新たに保証したものではない。表示近似ではなく、保存exact rational三項の一致を検査した。

旧Stage Rの`B2_head_identity_qualified=false`は原本のままである。今回得たのは同一性ではなく、**異なるtargetの間を明示的な誤差上界で移送する証拠**。片側headを他方へ置き換えていない。

## 残件・資源・保全・STOP

Stage Pのcompiled→raw→literal full512 per-word上界は未取得。全word compiler成功、全law synthesis、期待bias校正、shots/CA/総費用は未資格のまま。今回のHead接続をnative synthesisやfull512保証に昇格させない。P41-callは独立した未承認契約で、fresh P出力を作っていない。Stage R、PF/RTE、fit/optimizer、ground solve、sampling、native build/compile、Stage C、重要科学レビューの新規実行は0。

実行前CPU affinity32、空きRAM約52 GiB、空きdisk約80 GiB、CPU/memory pressure avg10=0を確認。独立3 head対は3 process・process-local Arb384で並列化した。保存wallは2.1394417732954025秒、親process peak RSS392028 KiB。子processを含む合計RSS／CPU時間は未計測で、speedupは主張しない。恣意的なCPU/RAM/worker/wall capや共有venv/OS/cache/scheduler／他jobの変更はない。

[保存照合](../../artifacts/research_c_head_bridge_stage_p/2026-11-04/H_execution_verification.json)は、原本25 file／8,272,544 bytes、26 COMPLETE、SOURCE162／入力48、固定210 fileの不変更、13行・三項会計・P未実行を確認した。旧SOURCE/raw/契約/grant/ledger/marker/STOP、およびStage R成功・旧失敗の両treeを保全した。

公開時にSOURCE/raw/journal/STOPを保存し、remote SHA/tree・主要本文・large head blobを照合する。**公開照合後mandatory STOP。** 次の担当はユーザー／研究C：独立したStage Pの固定41-callを実行するか判断する。Codexはその別途明示承認後だけ実行し、今回のH枠やP/Cへの自動進行を行わない。
