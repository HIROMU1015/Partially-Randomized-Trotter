# 2026-11-14 D：Full-H pure RTEのκ型schedule移行調査

C `367dc98d` のFull-H R448（q=2^m、R=448/group、K2）を確認した。R448はT2.0最良q8/R448/K2（κ_eff 2.03）から
R/groupだけを流用した接続seedで、group時間が0.25から0.1に変わっている。s∝TなのでlogBがroundごとに倍になり、
κ_effはm0の101からm12の0.025へ下がる（m5とm6の間で2を横切る）。13 meanはR448構成の平均として有効だが、
Full-H pure RTEの最適性能ではなく、κ候補へ流用しない。

原論文E21のr=κλ²T²（A `510fbacd` の転記）とB `paper_event_mapper` の総数を再利用した。
総数をgrouping q=2^mとR=ceil(r/q)へ分ける最小helper `full_h_kappa_layout.py` を追加した。
D `full_h_rte_spec(kappa)`、B generate/_validate、C configure/prepare、数値evaluate、C guardは無変更で通る。
保存R448 configurationと13 physical IDを同じbuilderで再現し、κ2/1/4のconfigurationとphysical IDをmetadata生成した。
人工τ gridでb≤exp(τ²)にbinary64の1 ULP超過があり、実行gateには既存のone-ULP感度を余裕として加える。

計画はS-1（κ2 signal 13、数値のみ）、S-2（κ1/4、任意）、P-1（κ2 m3、新ZZ、normal1 path、compile2）、
P-2（group分解の価格model用cell、任意・Cのscope）。κ型ではm10〜12がGを支配し、直接compileできない。
長時間価格はUNQUALIFIED model scenarioのままとする。κ・K・groupingは採択していない。

人工・metadata test 10/10 PASS。新科学計算・回路build/compile・保存値の数値再解析・N・価格・Gは0。
[REPORT](../d_full_h_kappa_schedule_20261114/REPORT.md)／[計画](../d_full_h_kappa_schedule_20261114/FINITE_SCIENCE_PLAN.md)。公開照合後mandatory STOP。
