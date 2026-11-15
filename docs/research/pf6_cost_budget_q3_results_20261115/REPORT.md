# PF6・費用予算3/4：固定12配分のQ3結果

2026-11-15 JST、Codex B。**新規Q3は許可された12件を各1回実行し、適合10件・モデル不適合2件。UNKNOWN／INCONCLUSIVE／retryは0件。全件の原API return・partials・資源記録をworker自身で保存し、callerも終了code0で完了した。**

計画7f27e18f06dde12127c85be898777b685d29baccのREPORT §3をユーザーが採択した固定実行。今回は現6幅だけであり、別幅・新signal・MC・sampling・build/compile・新価格は実行していない。結果を見て配分を変更していない。

## 1. 有限集合で確認できた最小適合費用

元6基準点N0と、今回の各2配分を合わせた**方式ごと9点、計18点**の範囲。価格はB2の固定156軸を用いた条件付き総compiled RZ数であり、十分N・物理的global RMSE・G_qualifiedの認定ではない。

| 方式 | 公式・要求幅 | 配分 | 確認された最小適合費用 T（RZ） | MSE上界 Ha²（表示） |
|---|---|---|---:|---:|
| Full PF | S2・16h/7 | B_PRICE | 615,304,502 | 0.000001495783385949356167254624 |
| B0I p7 | S2・16h/7 | B_PRICE | 435,481,090 | 0.000001506660616251764083585362 |

両最小点の整数配分は、今回の生成結果として一致した。方式間で同じNを課した結果ではない。m0〜m12、**X/Y各軸**：

```text
N_X = N_Y = [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 4]
```

元configurationはFull PF S2 11cf8ca6、p7 S2 a6e55777。元physical schedule・各roundの実際のq/T/幅・scalar・targetは不変。16h/7は要求幅であり、early roundも一律にこの幅へ変更したものではない。allocation IDと全桁ID、入力SOURCEはINPUTSとC_RESULT_JOINを参照する。

## 2. 全18点：元基準点6件と新配分12件

元N0は両軸 [59,55,51,47,43,39,35,31,27,23,19,15,11]。Aは形保持w=N0、Bは単価を考慮したw=N0/C。具体的な全X/Y列・実費用・未使用予算は実行前固定の[ALLOCATIONS](ALLOCATIONS.md)、正本[INPUTS](INPUTS.json)。費用を使い切れない整数端点はそのままとし、MSEを見てλや列を変更していない。

| 物理候補 | 配分 | 実費用 T（RZ） | MSE下界 Ha²（表示） | MSE上界 Ha²（表示） | 判定 |
|---|---|---:|---:|---:|---|
| Full PF S2・16h/7 | N0（保存再利用） | 838,693,580 | 8.255048976770948750408424E-7 | 8.255048976770948750408802E-7 | 適合 |
| Full PF Yoshida4・16h/3 | N0（保存再利用） | 1,089,674,842 | 7.969265386751240788457598E-7 | 7.969265386751240788458870E-7 | 適合 |
| Full PF M5・16h | N0（保存再利用） | 1,343,396,772 | 4.625148099114778355837148E-7 | 4.625148099114778355838201E-7 | 適合 |
| B0I p7 S2・16h/7 | N0（保存再利用） | 593,555,352 | 8.356033913794202152062946E-7 | 8.356033913794202152063318E-7 | 適合 |
| B0I p7 Yoshida4・16h/3 | N0（保存再利用） | 765,902,758 | 7.882628817223086740102446E-7 | 7.882628817223086740103736E-7 | 適合 |
| B0I p7 M5・16h | N0（保存再利用） | 951,282,896 | 4.616158368748513116040767E-7 | 4.616158368748513116041819E-7 | 適合 |
| Full PF S2・16h/7 | A_SHAPE | 622,182,168 | 9.698881900054125353117738E-7 | 9.698881900054126236340622E-7 | 適合 |
| Full PF S2・16h/7 | B_PRICE | 615,304,502 | 0.000001495783385949356167254620 | 0.000001495783385949356167254624 | 適合 |
| Full PF Yoshida4・16h/3 | A_SHAPE | 628,216,816 | 0.000001188885887595055326434029 | 0.000001188885887595371771803337 | 適合 |
| Full PF Yoshida4・16h/3 | B_PRICE | 619,717,524 | 0.000002426124932361972328572360 | 0.000002426124932361972328572366 | 適合 |
| Full PF M5・16h | A_SHAPE | 628,785,102 | 0.000001474459957703556200561039 | 0.000001474459957709756777790896 | 適合 |
| Full PF M5・16h | B_PRICE | 619,784,264 | 0.000004895357753720505023070850 | 0.000004895357753720505023070851 | モデル不適合 |
| B0I p7 S2・16h/7 | A_SHAPE | 440,329,944 | 9.797745654510525900321019E-7 | 9.797745654510526762188602E-7 | 適合 |
| B0I p7 S2・16h/7 | B_PRICE | 435,481,090 | 0.000001506660616251764083585358 | 0.000001506660616251764083585362 | 適合 |
| B0I p7 Yoshida4・16h/3 | A_SHAPE | 444,807,010 | 0.000001174900865076655516197599 | 0.000001174900865076752235939216 | 適合 |
| B0I p7 Yoshida4・16h/3 | B_PRICE | 435,641,568 | 0.000002418371981954278179229700 | 0.000002418371981954278179229706 | 適合 |
| B0I p7 M5・16h | A_SHAPE | 445,015,812 | 0.000001528257151522752342191913 | 0.000001528257151524630648960704 | 適合 |
| B0I p7 M5・16h | B_PRICE | 438,954,868 | 0.000004899004506054312599493318 | 0.000004899004506054312599493319 | モデル不適合 |

端点の表示は25桁の近似。判定は十進εE=0.00159360010199040 Haのexact二乗τと、原APIのexact有理数で行った。モデル不適合2件はFull/p7それぞれのM5・B_PRICEで、**下界>τ**であり、単に保守的上界が大きいだけの判定ではない。

旧6基準点は8dba096dcの保存component判定と1002f1dfeaの費用を引用した。原API return・CPU/peak RAM欠測の部分保存・運用UNKNOWNを保ち、今回の完全保存12件と混同しない。基準点の補完目的で再評価していない。

今回の有限集合ではS2・B_PRICEが各方式内の最小適合点になった。高次PFにも新しい費用目標で適合する点があり、M5は形保持では適合した一方、安いB_PRICEでは不適合だった。配分の単価最小化だけで精度は保証されず、特定roundを不適合の原因と断定する比較実験は行っていない。幅全体・公式全体を棄却する結果ではない。

## 3. SOURCE・契約・価格の位置付け

- 科学実行SOURCE：**06f4210e1bdfd5b07f42b941fa22d17e228533fc**、caller scripts/run_pf6_cost_budget_q3_fixed.py、実行前固定INPUTS。結果公開commitとは区別する。
- 保存Linear H4/1.00 Å/STO-3G/tol-only1e−8 DF10・sector(2,2)・元H/state/snapshot、h=0.1/M12/P0。A正本1adb7e65a9、科学fb9a1878、carrier06a15ee4を再取得せず使用。
- 採択Z1＋T1 total decoder、保存full DF10 solver energy proxy −2.1663874486347634 Ha、全事象平方loss、保存deterministic raw/B=1・shots/X/Y/round間IID二項、EXACT_AFFINE_OF_BINARY64_RAWを維持。PR finite-K lawをPFへ追加していない。
- ground/state/targetの厳密包含、実samplerへの移送、数値certificateは未認証。transition allowanceは元のengineering allowanceであり、狭いMSE区間を認証済み物理保証とはしない。AδE/δE²を追加せず、元Minkowskiは別参照として保全。
- 価格は**B2 1002f1dfeaの96直接＋60条件付き閉包**。同じ元configuration・physical ID・round/axis・元/total candidate IDで照合し、元receipt/SOURCEは変更しない。新Nとの312軸別積はINPUTSに保存。normalizationや価格標本数を二重乗算していない。
- PF_FIXED_COMPILER_CLOSURE_CONDITIONAL_60_AXES_V1の固定compiler・障壁/反復/終端構造等の前提を引き継ぐ。60軸を直接観測や無条件QUALIFIEDへ昇格せず、価格誤差上界/CIはnull、2%保証なし。条件が崩れれば今回のT/最小点も条件付きのままである。

元6結果・新12結果は[SUMMARY](SUMMARY.json)、短表[RESULTS.csv](RESULTS.csv)、C向け[候補・配分・価格join](C_RESULT_JOIN.json)から追跡できる。新12件は各run01/<label>/API_RETURN.jsonが原返却値、RESULT/3種PARTIAL/RESOURCE_FINAL/PERSISTEDがworker保存原記録。

## 4. 数値gate・保存整合

12件とも新整数count格子に対するρ=1の有限binary64閉包を確認。各列のcount-onlyと、outcomes4225、verification55043173、非終端frontier8192、transitions141559036／callの範囲を保持した。旧配分の閉包PASSを流用していない。

kernel内の質量検査に加え、保存結果のfirst-unsafe13分割＋all-good=1、各round good＋unsafe=1、保持MSE寄与＋残余loss上界=DP上界、moments区間・MSE上下界・exact閾値判定の整合、原APIとRESULT.recordの一致を確認した。これは保存出力の検算であり、Q3・DPを再実行したものではない。zero HOLD/tie argmin、principal guard、範囲外質量の残余会計を保持し、total decoderの全事象lossを捨てていない。

全runでSCIENTIFIC_CALL_STARTは1回、最終stageはEVALUATION_COMPLETE。原API/partials/RESOURCE_FINALを12件とも確認し、UNKNOWN・監視エラー・保存エラーは0。caller終了code0、記録した子process1,189件も全て終了code0。子process数は科学call数ではない。新規callは12件のみ、retry・結果依存の配分変更なし。

## 5. 実行負荷・テスト

親7 job並列、各8 DP worker・batch8という既存高速経路を使用。全体caller wallは259.126秒（約4分19秒）。個別記録はSUMMARY.rows.resources、全体の監視・終了はrun01/COMPLETION.json・PROCESS_RETURN.json。

資源fieldの範囲：worker/child CPUは各job区間の差分。worker peak RSSと最大単独child RSSはOSのprocess-lifetime high-waterで、親poolを再利用したjobでは以前のjobの最大も含み得る。親monitorのsampled process-tree RSSは観測値であり、共有pageの重複やsample間の未観測を含む。個別peakを足して同時RAMとしない。

実行前は新caller6＋旧PF6＋保存worker5＝17 test method PASS。actual installed worker/supervisor入口で親monitor例外を注入し、開始済みworkerの原API/partials/資源保存を確認した。実行後の集約器は人工2件PASS（TEST_SUMMARY.txt）：shot総数ではなく費用を選び、不適合を除外し、UNKNOWN/INCONCLUSIVEと同額候補を保持する。新科学モデル・decoder・DP kernelの実装変更は0。

## 6. 範囲・引渡し・STOP

結論は現6幅×N0/A/Bの有限menu内で、固定価格前提とQ3モデル閾値に条件付く最小適合費用である。別幅、N0超の振替、別β、他の配分族、PR/Full-Hとの新費用比は評価していない。最小N、幅を含む大域最適、十分N・物理的global RMSE・G_qualified・方式全体winnerを認定しない。

[HANDOFF](HANDOFF.md)にC/A/Dとの結合とロードマップへの差分を記載する。ロードマップ本体・旧SOURCE/raw/結果/grant/ledger/marker/STOP・他担当worktreeを変更しない。公開とremote照合後 **mandatory STOP**。追加の科学call・別幅・価格・G比較へ自動進行しない。
