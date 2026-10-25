# C1：保存H4 XDF prefix1〜9・DF予算の限定実行結果

2026-10-26 JST。**保存順の最初のDF予算適格prefixはrank8だった。rank1〜7はNOT_MET、rank8／9はPASS。** 全9prefixを完了し、356/356 logical units、correctness FAIL／inflight0、retry／resume0。保存rank10は新しいnormを計算せず、既存PASSを再利用した。

これは直鎖H4・1.00 Å・STO-3G、Nα=Nβ2／sector36、保存OpenFermion1.6.1 eigh由来の元順prefixと、既存engineering allowanceの下での**数値DF予算適格性**である。rank8を新しい比較baselineとして正式採用したこと、PF／native／最終CA／費用の資格が得られたこと、全XDF方式中での最小rankは意味しない。

## 固定SOURCE・契約・入力

- 科学SOURCE：`f7bc730d12ad704cdb72cc8a82d2bd973d01b2e5`、tree `1f70265e13045492554591f55df5a71d0d14fd1c`。SOURCE／runnerを変更せず実行した。
- Branch：`codex/research-c-c1-h4-prefix-execution-20261026`。
- [固定356-call契約](research_c_c1_h4_xdf_prefix_budget_contract_unapproved_20261026.json)、scope `C1_H4_SAVED_XDF_PREFIX_DF_BUDGET_V1`。
- 契約SHA256：`56ad290738e1faec290bd66e896fc7da8370e4f31760ddd3429d5fea1c6787d0`。
- SOURCE manifest SHA256：`b63ef45279a27c09fcb0626628b025e4891c6ea58f602ffddce9489cf022a516`。
- 元P-H4 SHA256：`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。公開packet `86b8b002704aced8ca239414080f53793de81658`、生成SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12`を保持。
- fresh output：`artifacts/research_c_c1_h4_xdf_prefix_budget/2026-10-26/run01`。このrunは消費済みで再起動しない。

前回の固定契約による限定科学実行へユーザーが明示許可したため、独立worktreeから一度実行した。契約の歴史的`UNAPPROVED` metadata／ファイル名は変更せず、承認原文保存・証明書・新grantは追加していない。

## 全9prefixの結果

元g補正済みの保存κ・核scalar・spin／bit／frame順・保存hermitizationを維持し、λ[:r]／G[:r]だけを使った。κの確認は一度、prefixごとの再補正・reordering・再分解は0。DF normは保存physical sector行列と各prefixの差のspectral norm。CA=.00159 Ha、DF予算CA/8=.00019875 Ha、guardは`binary64_operation_scale_guard8_v1`をrank別に算定した。

| rank | DF sector norm（Ha） | allowance（Ha） | norm＋allowance（Ha） | CA/8 |
|---:|---:|---:|---:|---|
| 1 | 1.525369368179258 | 1.462256852586962e-9 | 1.525369369641515 | NOT_MET |
| 2 | 1.170623034105351 | 1.470516726819096e-9 | 1.170623035575868 | NOT_MET |
| 3 | .6045212350459327 | 1.475567103626314e-9 | .6045212365214998 | NOT_MET |
| 4 | .05165191678498344 | 1.478318645469369e-9 | .05165191826330209 | NOT_MET |
| 5 | .03975348633849828 | 1.479002715376580e-9 | .03975348781750099 | NOT_MET |
| 6 | .02594692067539543 | 1.479554403829278e-9 | .02594692215494984 | NOT_MET |
| 7 | .0002633436643642781 | 1.480028834985097e-9 | .0002633451443931131 | NOT_MET |
| **8** | **.00009533753411107529** | **1.480439843662034e-9** | **.00009533901455091894** | **PASS** |
| 9 | .0000002079904965185448 | 1.480850073743132e-9 | .0000002094713465922879 | PASS |

[全精度CSV](../../artifacts/research_c_c1_h4_xdf_prefix_budget/2026-10-26/prefix_budget.csv)／[9prefix・保存anchor・選定原本](../../artifacts/research_c_c1_h4_xdf_prefix_budget/2026-10-26/run01/result.json)。最初のPASS後もrank9を評価し、単調性を仮定した省略は行っていない。

保存anchorはXDF10 norm `3.3751558354542566e-15`／allowance `1.4812598092146554e-9`／保存合計 `1.4812631843704909e-9 Ha`、CDF8 norm `7.666027761392501e-5`／allowance `1.881549094133745e-9`／保存合計 `7.666215916301914e-5 Ha`。両方の既存PASSを引用しただけで、rank10／CDF8のnormやguardを新規評価していない。

## gate・call・資源

SOURCE15、保存input7、過去SOURCE3点、元packet・契約・環境のpreflightはPASS。κ max-entry差 `2.7755575615628914e-17`、scalarの一度使用、sector indices／OF-Qiskit permutationがPASS。保存physicalのHermiticity max-entryは`1.457167719820518e-16`、prefix各行も固定1e-10／rtol0内で、finiteを含むcorrectness FAILは0。元physical行列の再生成はしていない。

| logical unit | COMPLETE |
|---|---:|
| saved bundle load | 1 |
| sector context | 1 |
| κ／scalar／basis identity | 1 |
| 保存physical attachment／formation scale | 1 |
| prefix model | 9 |
| 実sector matvec | 324 |
| sector error norm | 9 |
| allowance／Hermiticity／budget | 9 |
| prefix selection report | 1 |
| **合計** | **356** |

journalは356 attempt／356 COMPLETEで一対一、FAIL／inflight0。新fit／optimizer／分解、physical生成、rank10 norm、PF、native回路／compile、合成誤差、shots／費用はすべて0。retry／resume0、科学条件・閾値・SOURCE変更0。

affinity32 CPU・空きRAM約52.6 GiB・低pressureを確認した。既存planに従い独立9prefix、worker内BLAS1、read-only packet／physical／contextの共有、Python sector backendを用いた。CPU／RAM／worker／wallの人工上限や共有環境／他ユーザーjobの変更はない。runner wall1.8672秒、process CPU10.3343秒、peak RSS277488 KiB。whole-process `/usr/bin/time`は2.08秒、peak RSS277744 KiB、swaps0、exit0。速度比較・speedupの主張はしていない。

[実行identity・資源・要約](../../artifacts/research_c_c1_h4_xdf_prefix_budget/2026-10-26/)／[raw19ファイル・SOURCE・契約・call journal・STOP](../../artifacts/research_c_c1_h4_xdf_prefix_budget/2026-10-26/run01/)／[postflight](../../artifacts/research_c_c1_h4_xdf_prefix_budget/2026-10-26/postflight.json)。SOURCE15／保存input7を含む旧protected44ファイルと元packetのhashは実行後も一致した。全9行のCSV作成と保存JSON／journalの照合だけを追加し、科学evaluatorを再実行していない。

直前[技術準備](research_c_c1_h4_xdf_prefix_budget_preparation_20261026.md)の人工29＋関連回帰28 PASSを再利用した。旧full256／32-thread SVD回帰1件の異常は未解決のまま保全し、今回のsector36正常完了で解消したとは扱わない。そのfull256経路の新規実行も0。

## 現在言えることと次の判断

保存XDFの同じ生成順でrank8／9がDF予算を満たすことを確認したため、**rank10は当該DF予算のfirst-passingではなかった**。従来未解決だった保存順のfirst-passingは今回rank8と判定された。ただし別順序・別分解法の最小rankは未評価である。

旧H4 CDF8／XDF10の約27%条件付き総RZ差は、その登録候補の保存結果として維持する。今回のXDF8のDF PASSは、その値を修正したり、同rankの費用・shots・最終CA・資源順位を認定したりするものではない。XDF8／9のPF、ground／alias、native correctness、合成誤差、RPE、費用の新資格は今回取得していない。数値guardは厳密enclosureではなく、`CA_achieved`と`rigorous_enclosure`はnullを維持する。

**ユーザー／研究C**が、XDF8をより小さい保存baselineとして次の比較対象に採用するか判断する。採用する場合、**Codex**が保存CDF8と新XDF8のPF／native／合成／RPE／費用へ進むための最小技術設計を、具体的な依頼後に準備する。科学実行・重要な詳細レビューにはそれぞれ必要な別明示許可が要る。今回から自動進行しない。

結果・SOURCEをcommit／pushし、remote SHA／tree／input／成果物を照合した後、**mandatory STOP**。run01の再起動・追加rank・H6 rank13・回路／費用計算は行わない。
