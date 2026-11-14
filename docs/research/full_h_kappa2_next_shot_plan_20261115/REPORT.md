# Full-H pure RTE κ2：次のwhole-shot配分案

2026-11-15 JST。既存B branch `codex/rpe-total-dp-acceleration-20261114`、結果commit **a82b0f0ec7f0f11d9981ddb2343fcd5378a87503** を継続使用。今回は保存資料の閲覧・重複照合・計画公開のみ。新Q3・DP・二項確率・MC・sampling・価格・G計算は**0件**。保存MSEの再評価、比率・倍率の推定、round別lossの再集計はしていない。

## 推奨する最初の一手

**同じFull-H κ2の既評価controlを、全13round・X/Y各軸で一律2倍にした1本**を提案する。

```text
m   = [0,1,2,3,4,5,6,7,8,9,10,11,12]
N_X = [128,120,112,104,94,86,76,68,60,50,42,34,24]
N_Y = [128,120,112,104,94,86,76,68,60,50,42,34,24]
allocation_id = FULL_H_KAPPA2_TOTAL_Q3_CONTROL_SCALE2_PROPOSAL_01
```

全roundを変更し、physical/canonical ID、raw、κ、grouping、decoder、target、law、lossは変更しない。[PROPOSAL.json](PROPOSAL.json)を入力対応・整数列・提案scopeの正本とする。**未採択・未評価の提案**で、必要N、nominal N、最適Nではない。同等結果がなければ次回の新規科学callは最大**1件**。

2倍は、既評価の配分形状を保つ少数候補の最初のchallengeとして選んだ。MSEが1/Nに比例するという仮定、保存MSE/τ比、適合倍率の予測から決めていない。round別の損失寄与を特定する証拠が弱いので、特定roundだけを増やすより変更を解釈しやすい。shot増加でMSEが必ず単調減少するとも仮定しない。旧vectorの奇数round（m4/5/9/10/11）も偶数になるため、zero/tieのcount格子も変わる。その効果を情報量増加だけと同一視せず、次回は全事象を同じdecoderで評価する。

## 保存診断から言えること

出所はa82b0f0の[前回REPORT](../pr_df_fullh_q3_fixed_results_20261115/REPORT.md)、[SUMMARY](../pr_df_fullh_q3_fixed_results_20261115/SUMMARY.json)、`run01/FULL_H_KAPPA2/{API_RETURN,PARTIAL_COARSE,PARTIAL_LEMMA,PARTIAL_TOTAL_DP,RESOURCE_FINAL}.json`。以下の数値は保存表示値の転記で、新しい数値再解析ではない。exact端点は原API returnが正本。

| 保存診断 | 値・意味 |
|---|---|
| 既評価X/Y control | `[64,60,56,52,47,43,38,34,30,25,21,17,12]`、USER_SUPPLIED |
| MSE区間（Ha²） | `[0.0001624431036689670410203549, 0.0001633618706479668178967357]` |
| 閾値判定 | `MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD`。下界>τであり、上界の保守性だけが原因のINCONCLUSIVEではない |
| bias²区間（Ha²） | `[2.868161477561078564583462E-12, 2.872144182161031235037416E-12]` |
| variance区間（Ha²） | `[0.0001624431007968228588593236, 0.0001633618677798053403356571]` |
| DP残余mass / 残余loss上界 | `2.588278666480358978222801E-8` / `9.187669789997768763807948E-7 Ha²` |
| any-zero確率 | `0.003391162117029792340028986` |
| m12 zero HOLD / tie argmin追跡mass | `0.003335822449490626852658056` / `0.001075484489097027156998885` |
| 旧partial規則の未解決率区間 | `[0.004478852723212924306149894, 0.004478870323678232566843041]`。total decoderの未定義率ではない |
| ρ閉包 / 質量 | `VERIFIED_FINITE_BINARY64_SUPPORT`（2,318,425組）。保存round・DPの質量gateはPASS |

この固定controlではvariance診断が主成分で、bias²の小ささだけでは適合しない。残余lossは保存区間へ既に含まれており、追加加算も除外もしない。zero/tieはZ1＋T1に従い出力を定義済み。追跡massは残余込みの厳密な全履歴率とは限らず、zero/tieの大きさだけからMSE原因や改善効果を断定できない。

### round別の読み取り

13rawはce6aacfaの`B_SIGNAL_INPUT.json`、first-unsafeはa82b0f0の`SUMMARY.rows[label=FULL_H_KAPPA2].first_unsafe`。下表は保存値をそのまま転記した。first-unsafeはρ=1の安全集合を初めて外れる確率で、**当該roundのMSE寄与、実際の誤branch率、zero/tie率のいずれとも同一ではない**。後続roundの修正・alias・最終energy変換を経るため、この表だけからround優先順位を定量決定しない。

| m | raw X | raw Y | attenuation | first-unsafe（保存表示） |
|---:|---:|---:|---:|---:|
| 0 | 0.6067576022863022 | 0.13354293306420212 | 0.6212797702440143 | 0.000006262939451038551649264661 |
| 1 | 0.5574945527839351 | 0.2578935595101613 | 0.6142550582969805 | 0.00002008016127787398167229052 |
| 2 | 0.3965269287910298 | 0.4667412469510861 | 0.6124385682659564 | 0.00005004295063195018254668834 |
| 3 | -0.09821625614483642 | 0.5997784061916837 | 0.6077668716579394 | 0.00005723152589173460153608146 |
| 4 | -0.5759103692766756 | -0.19381270283091534 | 0.6076480209907765 | 0.0001557609085769302718069372 |
| 5 | 0.48313783506712704 | 0.36671552036303695 | 0.6065496192380675 | 0.0004271359878968599147087688 |
| 6 | 0.16312049070105 | 0.5841960705999619 | 0.6065421200566398 | 0.0006045200137657153670218556 |
| 7 | -0.5188028519817799 | 0.31422073419148716 | 0.6065402451847742 | 0.002096407803644498780323984 |
| 8 | 0.2809732727284654 | -0.5375355992218813 | 0.6065397764669999 | 0.003044016956139759773386451 |
| 9 | -0.3462237894286338 | -0.49801550758792773 | 0.6065396591604958 | 0.007490517263700896209860356 |
| 10 | -0.21127806800224744 | 0.5685524613264552 | 0.6065396302631736 | 0.01135669693779073500000082 |
| 11 | -0.4593491691446569 | -0.3960917237676677 | 0.6065396216338035 | 0.02819720141543896325522094 |
| 12 | 0.08921471521257078 | 0.5999340838614607 | 0.6065312568694629 | 0.03885266825613058694843529 |

保存減衰は全roundにあり、特定roundだけの問題とする根拠はない。末尾のfirst-unsafe頻度が大きいことも、そのroundだけを増やせば全MSEが適合するという証拠にはならない。m0/6/12を含むzero/tie・残余の全13行は元SUMMARYに保全されており、今回はloss別因果分解や新しい寄与率を作っていない。

## §10.5：既存のFull-H配分案・結果・job

| 固定資料・観測 | 今回への利用可否 |
|---|---|
| 26d4f23e7 `d_full_h_kappa_schedule_20261114/KAPPA_CONNECTION.md` §1、`KAPPA_CANDIDATES.json` κ2 | Full-H自身のBを使う原論文型 `ceil(B_m²[11+4(M−m)])` という固有の名目配分式がある。ただし候補の全13round `N=null`。整数化済みvectorや同Q3結果はこの資料にはない |
| C c628e9e8 `paper_g_price_routes_20261114/REPORT.md` §3、`paper_nominal_cost` | 同式を使う既存producer/APIはある。nominalは十分Nでなく、Q3閾値適合証拠でもない。今回は実Bを式へ代入せず、原論文型N/Gを新規生成しない |
| B 999304918 `rpe_required_n_cross_method_preparation_20261114/FINITE_N_PLAN.md` | Full-H初回の「外部指定whole-vector各1本」案で、Full-H固有の整数列は未指定。Pauliの3/4縮小案をFull-Hの既評価と数えない |
| B 0629a8cf3 → 27e70efe5 → a82b0f0ec | 同じκ2・現契約のcontrol 1本を評価済み。原API return・partials・CPU/RSSを再利用して解釈する。既評価controlは再実行しない |
| PR-DF LD1/LD2、Pauli control/3/4、PF6、旧partial DP | physical候補／vector／decoderが異なる。今回の新vectorに同等な結果として使わず、再実行対象にも加えない |
| Full-H R448、T=2旧HP型配分・R2-ADAPT screen | κ2の13round raw/normalizer/law/decoder契約と異なる。配分の出発点やQ3結果として移植しない |

2026-11-15 JST（公開ref観測・job観測は同日、記録時刻2026-11-14T19:29:56Z）にB、Full-H S1、C paper-G、統合roadmap、D P2の関連公開refとlocal履歴を確認。Bはa82b0f0、S1はce6aacfa、Cはc628e9e8、roadmapは73b23e01c。D P2は887c3b9d0。確認範囲では、提案2倍vectorについて保存Q3結果・計画中の同一Q3 jobは同定されなかった。全サーバー・非公開記録の不存在を証明したものではなく、実行前に変化分を再照合する。

自ユーザーの関連Python script名とcwdのみを読取観測した。稼働中だった`continue_full_h_p2_checkpoint.py`／`run_full_h_kappa2_p2_accelerated.py`は固定SOURCEの説明でもP2価格処理と確認でき、Q3重複ではない。該当job、他担当worktree、設定へ介入していない。D1の完了はこのrisk計画の依存条件ではない。

## 実行可能性と次回の判断

新vectorの整数だけからcount組数を確認した。**旧scopeを自動継承すると不十分**であり、[FINITE_SCIENCE_PLAN](FINITE_SCIENCE_PLAN.md)の新しい有限範囲を次回指示で明示採択する必要がある。確率、phase、閉包、DP状態分布、MSEは新規計算していない。ρ閉包、数値allowance、mass・残余loss・MSE区間は新vectorで未確認。

前回の保存負荷はwall349.2605965558905 s、process-tree CPU560.02039335 s、worker peak RSS1,265,364 KiB、sampled tree peak8,006,676 KiB。新vectorは支持・整数分母・状態集約・残余が変わるため、これを実行時間やメモリの保証へ外挿しない。既存高速backendとworker自身の保存処理を再利用し、実行前に同時稼働jobを読取確認して資源利用を判断する。CPU/RAM/workerの新たな恒常上限は設けない。

## 検証・ロードマップへの影響・STOP

計画内のX/Y13整数、旧配分の2倍、候補ID・13physical行の保全、SOURCE参照、純整数count式を照合した。既存kernel・caller・入力・結果に変更はない。人工Q3や既存回帰testの再実行も行わず、計画metadataの整合確認のみ。

統合担当への変更点：Full-H κ2は**既評価controlでモデル閾値不適合／次の未評価2倍vector 1本を提案／新scope採択と科学実行許可待ち**。PR-DF2件の適合状態は不変。適合vector取得済み、必要N確定、方式間winnerとは表示しない。ロードマップ本体は変更していない。

`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`。原Minkowski、ground未認証、実sampler移送未認証、数値certificateなしを保持し、AのδEを二重加算しない。**計画公開後mandatory STOP。科学実行は別途明示許可後。**
