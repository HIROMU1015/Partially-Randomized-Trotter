# B0I p7/S2・幅h：粗幅達成費用を予算とする固定2配分Q3結果

2026-11-16 JST。**今回のA_SHAPE・B_PRICEの2配分では、幅h=0.1で「435,481,090 RZ以下かつQ3モデル適合」の配分は見つからなかった。** 両列とも予算内だが、有効MSE下界が閾値を超えるモデル不適合である。UNKNOWN・INCONCLUSIVEは0件。これは評価した2列の結論であり、幅hの全shot配分に適合解がないことや、粗幅の大域最適性を示すものではない。

## 固定した配分と判定

H4 linear / 1.00 Å / STO-3G / 保存tol-only DF10、sector(2,2)、B0I p7/S2、一体項＋保存fragments0..6。元configuration **e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc**、幅h固有の13round schedule qPF=2^m・実幅0.1を維持した。RPE h=0.1/M12、Z1＋T1 total decoder、保存full DF10 solver proxy −2.1663874486347634 Ha、全事象平方loss、保存deterministic rawのideal IID二項モデルを使用。exact τ = **0.0000025395612850638132820416921600 Ha²**。

| 規則 | N_X=N_Y（m0〜m12） | モデル費用T RZ | 粗幅達成額との差 T−435,481,090 | 判定 |
|---|---|---:|---:|---|
| A_SHAPE | [21,19,18,17,15,14,12,11,9,8,6,5,3] | **421,323,290** | **−14,157,800** | モデル不適合 |
| B_PRICE | [59,55,51,47,43,39,35,31,27,16,7,2,1] | **434,203,864** | **−1,277,226** | モデル不適合 |

| 規則 | 直接モデルMSE下界 Ha²（表示） | 上界 Ha²（表示） |
|---|---:|---:|
| A_SHAPE | 2.643169809552747106267978E−5 | 2.643169812287933598398398E−5 |
| B_PRICE | 1.005010279348543459706972798193894E−5 | 1.005010279348543459706972804518317E−5 |

両者とも**下界>τ**であり、単に保守的上界が超えただけの不確定判定ではない。原[API_RETURN](run01/B0I_P7_S2_H__A_SHAPE/API_RETURN.json)／[B_PRICE原API](run01/B0I_P7_S2_H__B_PRICE/API_RETURN.json)と[EXACT_RESULTS](EXACT_RESULTS.json)のexact有理数で判定した。表の小数は表示用で、[SUMMARY](SUMMARY.json)の25桁表示ではB_PRICEの両端が同じに丸まるが、exact区間幅は0ではない。

粗幅16h/7・B_PRICEはb4296b110b129d226dbef2743c14f725a27c3bf6の保存結果を再利用し、費用435,481,090 RZ、MSE約1.506660616E−6 Ha²で適合。元配分[59,55,51,47,43,39,35,31,27,23,19,13,4]を再評価していない。今回両列の費用差は負だが、適合を伴わないため**達成費用の厳密改善や同額達成とは扱わない**。今回menu内の確認済み最小適合費用はnull。

## 直接価格・外挿価格の寄与

固定価格は **ecf45508a63ddf75a4806b47179f5f5102fa90e0** の `docs/research/d_pf_p7_s2_h_comparison_prices_20261115/run01/COMPARISON_PRICES.json`。m0〜m4の10直接軸と、m5〜m12の16外挿軸を区別する。

| 規則 | 直接部分 m0〜m4 RZ | 外挿部分 m5〜m12 RZ | 合計 RZ | この配分の外挿割合 | 未使用予算 RZ |
|---|---:|---:|---:|---:|---:|
| A_SHAPE | 5,645,442 | 415,677,848 | 421,323,290 | **98.6600688512%** | 14,157,800 |
| B_PRICE | 16,000,932 | 418,202,932 | 434,203,864 | **96.3148803300%** | 1,277,226 |

割合は各列のexact整数積算から算出し、直前配分の98.388806%を流用していない。単価・全26軸のN×C・直接/外挿区分・元receipt・physical ID・固定SOURCEは実行前固定[INPUTS](INPUTS.json)と[ALLOCATIONS](ALLOCATIONS.md)に保存。外挿価格は採用済み比較規則による値で、未観測長qの実測値や厳密価格上界ではない。モデル誤差上界・CIはnullのまま。実装guardを外挿価格のPASSへ付け替えていない。

## 生成・重複照合・実行SOURCE

- 上限は**元N0=[59,55,51,47,43,39,35,31,27,23,19,15,11]**、各軸1以上。直前の適合列の末尾13,4は探索上限へ使っていない。
- 粗幅の既存 `run_pf6_cost_budget_q3_fixed.generate` をbyte不変で再利用。Aはw=N0、Bはw=N0/C_h、既存の整数breakpoint chainで予算内最大λを選んだ。次breakpointで予算超過することを保存。新しい配分族や結果に応じた調整は追加していない。
- 同configuration・vector・decoder・target・law・lossでの保存/進行中Q3を§10.5に従い限定照合。既存幅hの末尾13,4、粗幅PF6、Pauli再配分は今回2列の代替ではない。2規則の列は異なり、**新規2件・保存Q3再利用0件・retry0件**。調査範囲は[SCOPE_AND_PREFLIGHT](SCOPE_AND_PREFLIGHT.md)。信号・価格・旧比較結果は保存再利用した。
- caller・具体列・価格参照・実費用/残額・テストを **8f9bf401009329683d1244675d6fdfd663689159** にcommit/pushしてから実行した。callerは `scripts/run_p7_s2_h_budget_q3_fixed.py`、既存PF intake/total binding/Q3 kernel/高速DP/worker保存/親supervisorを再利用した。科学実行SOURCEと今回の結果公開SHAは異なる。
- 新3件の人工/metadataテストがPASS。実signalのriskを呼ばず、具体列・上限・単価積・SOURCE/ID/区分不一致拒否、同列2規則の1call化を確認。科学実行前に検出したT/δの有理数とbinary64表現の照合不具合は既存exact_scalarで修正し、科学条件を変更していない。[TESTS](TESTS.txt)。

## gate・残余・保存状態

| 診断 | A_SHAPE | B_PRICE |
|---|---:|---:|
| 新count格子で確認した閉包隣接組 | 37,628 | 2,517,970 |
| 実transition組数 | 5,798,898 | 80,429,718 |
| 追跡外残余mass | 2.728328396069545E−16 | 1.449705098251911E−33 |
| 残余loss上界 Ha² | 2.735186492130420E−14 | 6.324423132567378E−32 |
| any-zero確率 | 2.793440200263359E−4 | 5.989010347556895E−2 |
| bias² Ha²（概数） | 6.519000524924E−9 | 4.828554815207E−8 |
| variance Ha²（概数） | 2.64251791E−5 | 1.000181724533E−5 |

両件で新しいbinary64 good閉包、各roundとfirst-unsafe分割の質量、残余loss、MSE上下界・moments、exact閾値、原API/RESULT整合を検査した。非終端frontier上限8,192、ρ=1、outcomes4,225、verification55,043,173、transitions141,559,036は不変。終端状態はA10,354/B3,127で、終端には非終端frontier制限を課していない。zero/tieをtotal decoderの保持/argmin規則で全事象lossへ含め、未解決として捨てていない。first-unsafeの確率をMSE寄与と同定していない。

科学判定は両方 `MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD`、保存状態は両方 `WORKER_ARTIFACTS_SAVED`。全原API・coarse/lemma/total-DP partials・RESULT・RESOURCE_FINAL・PERSISTEDをworker自身で保存し、親caller/記録済み子181件は終了code0、monitor errorなし。[COMPLETION](run01/COMPLETION.json)。

| 資源記録 | A_SHAPE | B_PRICE |
|---|---:|---:|
| worker wall秒 | 36.752 | 114.263 |
| worker CPU秒 | 41.880 | 101.222 |
| child CPU秒 | 11.481 | 137.401 |
| worker peak RSS KiB | 202,260 | 600,292 |
| 完了child単体peak最大 KiB | 167,460 | 439,216 |
| 親monitorのsampled subtree peak KiB | 568,816 | 3,977,872 |

親2件並列・各最大8 DP worker/batch8の既存経路、caller wall114.311秒。RSSは記載範囲の観測値であり、個別peakを加算した物理RAM peakではない。新しい高速化倍率を測定したものではない。

## 解釈と終了

今回、価格を考慮した配分でもこの固定予算ではQ3適合を確認できなかった。別のshot振替や予算、幅全体を否定せず、この2点の結果として引き渡す。価格の大半は未検証長q外挿であり、正式E[C]/Gの資格も付与しない。

元solver proxyのground資格、state/target包含、実sampler lawへの移送、binary64 allowanceの厳密certificateは未認証。AδEをMSEへ二重加算せず、元Minkowskiは別参照。`sufficient_N`、global RMSE、G、G_qualified、方式winnerはnullを維持する。

新compile・fit・signal・MC・結果に応じた配分変更・追加Q3・retryなし。旧結果・SOURCE・STOPと他担当環境を保全し、commit/push・remote照合後 **mandatory STOP**。
