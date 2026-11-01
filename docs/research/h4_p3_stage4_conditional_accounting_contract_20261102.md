# P3 Stage 4：保存結果だけによる条件付きN/G会計の実行案

2026-11-02 JST。基点はStage 3 run04結果`e44e87a306d05c8549f1a044b7fd056c2e1055a3`。
ユーザーはStage 3限定価格評価を完了と判断し、下記Stage 4を推奨したが、
新規科学的数値解析には別途明示承認を要すると指定した。**本資料は非実行の案で、N/G等の新計算は0。**
今回は保存schema・SOURCE・入力bytesの静的確認とMarkdown作成だけを行い、旧mandatory STOPを維持する。

## 1. 限定対象・仮定・出力件数

| 項目 | 今回提案するStage 4の範囲 |
|---|---|
| 系・入力 | linear H4、隣接1Å、STO-3G、中性singlet、system8/full256、保存DF tol-only1e-8/rank10 |
| 分割・方式 | 案A、共通T_eff保持/V_square残差、M_B=Pauli184、M_C=DF rotated Z/ZZ360・10 basis |
| tuple・測定 | T=exact binary64 .8、S2/q8/r4/K4、P=0、cosine/+Im sine full wrapper |
| 精度 | ε_sig=.05/.01/.005/.001/.0005、ε_a=ε_sig/√2、α_a=.025 |
| bias/reference | Stage 2の補正済みnom_P/nom_DFと保存physical_reference区間・transfer。不確かさを含める |
| normalization | 方式別の保存BとB_uncertainty、decimal50検算値。上界候補の仮定・丸めを下記で明記 |
| 価格 | Stage 3 run04の独立32 trajectory/方式・各軸32価格と共分散。旧pilotをpoolしない |
| 実装仮定 | finite-bit law/angle gap=0、compiled ρ=0。実測済みの事実ではなく、nominalモデルの条件 |
| 主出力 | 2方式×5精度=10会計行、計20軸の適格性・headroom・N。RZ総費用Gと価格標本SE |
| 補助出力 | CX総数、shot加重depth指標、各指標の価格標本SE、方式間差・比と比較対象の適格性 |
| 禁止 | 新sampling/compile/Gaussian合成・対角化、reference/state/Hamiltonian生成、finite bias高精度化、新tuple/探索 |

単一Tにおける同じsignal精度の条件付き会計であり、energy精度やRPE全round総費用ではない。
実装誤差仮定・engineering数値不確かさ・価格標本の不確かさ・未取得の保証を別欄で報告する。
新しい母平均tail上界・全law保証・重要claim採択を完成条件に追加しない。

## 2. 保存入力・SOURCEの接続

Stage 2は結果`8bcf5851f99f980dc08c5885ace54874b554c815`、科学SOURCE`4bfbbaad1f8178077489bf06b085377256d6fb80`。
Stage 3 run04は上記基点、科学SOURCE`1cfce5e961e3d063741a1e0656052b10c7b27392`。
両者の保存共通H4/state/ν・T/scalar・law・compiler/測定境界の対応を既存来歴から確認する。
Stage 4では原分子入力やQPYを読み直して回路・referenceを再構築する必要はない。

入力は以下の保存JSON8件。今回ローカルbytesと固定commitの一致を確認した。

| 固定ref | 保存ファイル | SHA256 |
|---|---|---|
| `8bcf5851f99f980dc08c5885ace54874b554c815` | `artifacts/df_pauli_matched_partition_validation/2026-11-02/stage2/run01/results.json` | `398e43cff0c1a314994c337a08601aa80753482acd72b30541d4bc6a413d2a92` |
| `8bcf5851f99f980dc08c5885ace54874b554c815` | `artifacts/df_pauli_matched_partition_validation/2026-11-02/stage2/run01/source_and_inputs.json` | `9a0acbab136fb8102e6cf44ee1211590ac7d4dacc38e5bef8ece107c540ed4fc` |
| `8bcf5851f99f980dc08c5885ace54874b554c815` | `artifacts/df_pauli_matched_partition_validation/2026-11-02/stage2/run01/STOP.json` | `ad3c238820cc2bcb5d620e223e5b273e37acd82236fc449aea95672cfc9267a2` |
| `e44e87a306d05c8549f1a044b7fd056c2e1055a3` | `artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run04/prices.json` | `f06494c8fc03af157e750399913956d2f34ab3c2d1cdbfdc62104bc8d76d83d3` |
| `e44e87a306d05c8549f1a044b7fd056c2e1055a3` | `artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run04/price_statistics.json` | `29676d603d7d086a80345b316e71e1f9ff785592788fad549b072569c61c3085` |
| `e44e87a306d05c8549f1a044b7fd056c2e1055a3` | `artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run04/source_and_inputs.json` | `d1e307104f7ac1d3a1038ac1dad14f2227a287b986602c11ceb8f735d87776b8` |
| `e44e87a306d05c8549f1a044b7fd056c2e1055a3` | `artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run04/event_coverage.json` | `d567c1bdfe34ac9afef28a28480a8e14a6637dc4c67c0c20430c92fd993b9b9a` |
| `e44e87a306d05c8549f1a044b7fd056c2e1055a3` | `artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run04/STOP.json` | `0940d37a406c583d43a29caf765ad0edfa6246ab91831e08cf0f3aa7841e513c` |

会計で読む主fieldは以下であり、今回それらへ新しい数値関数を適用していない。

| 入力 | 使用する保存field |
|---|---|
| Stage 2 results | signals.nom_P/nom_DF.real/imag/u_num/u_bind/u_ref_transfer、reference.physical_interval/transfer_uncertainty、normalization.P/DF.B/B_uncertainty/decimal50.B/log_B |
| Stage 3 statistics | arms.*.n=32、axes.*.*.mean_exact/unbiased_sample_variance/standard_error、cosine_sine_sample_covariance_exact |
| Stage 3 prices | 方式・index・軸別のRZ/CX/depth。共分散・重み付き標本SEのscalar照合に使う |
| 各source/STOP/coverage | 科学条件・由来・完了状態、旧STOP、order4未観測・未保証事項 |

再利用するのはfixed_signal_accounting.fixed_signal_shots、必要なscalar ln端点には
h4_conditional_cng.trans_bounds、平均/軸共分散には既存の価格統計規約。
h4_conditional_cngの旧1%guard・旧H4候補・rho探索・科学runnerを本会計へ持ち込まない。
小さい純scalar adapter/runnerと人工testsで接続し、不要な管理frameworkや承認証明を作らない。

## 3. total biasの保守側評価：split誤差を二重計上しない

各arm X、軸a（real=cosine、imag=+Im sine）について、保存nominal点をz_Xa、
保存physical reference区間を[L_a,U_a]とする。将来の評価定義は

```
d_Xa = z_Xa - (L_a+U_a)/2                 # signed表示用
u_signal_X = nom_X.u_num + nom_X.u_bind + nom_X.u_ref_transfer
b_Xa,upper = max(|z_Xa-L_a|, |z_Xa-U_a|)
             + u_signal_X + reference.transfer_uncertainty
u_a,remaining = 0                        # 下記の実装gap仮定のみ
h_Xa = epsilon_a - b_Xa,upper - u_a,remaining
```

referenceの幅は端点距離に含むため、reference.display.axis_resolutionを重ねて足さない。
nominalの数値/binding不確かさとreference transferを各1回だけ含める。
outerとfiniteを合成する経路を使わず、共通splitのu_numを二度足すことを避ける。
u_remaining=0は残るfinite-bit gapとcompiled ρをゼロと置く仮定であり、数値不確かさを捨てる意味ではない。
Stage 2 finite bias成分をゼロ認定せず、既存のUNRESOLVED/STOPも変更しない。

すべての保存binary64値とreferenceのrational端点をexact Fractionへ束縛してscalar演算する。
上界をbinary64の既存関数へ渡す場合は上向き、許容精度を渡す場合は下向きの丸めを用い、実operandを保存する。
この端点演算の正確さを、元のengineering u_num/u_bindの厳密保証へ格上げしない。
本資料ではd、b_upper、hの数値は作成していない。

## 4. B・shot式・ceilの扱い

保存BそのものB_0と、Hoeffding式に使う上界候補B_eng,upを区別する。
今回提案する保守側候補は、保存値をexact rationalへ束縛して

```
B_eng,up = max(B_0 + saved.B_uncertainty, saved.decimal50.B)
```

と定義する。B_uncertaintyはStage 2のengineering目安なので、B_eng,upが名目振幅を覆うことも
本条件付き会計の仮定として記す。decimal50の文字列を厳密なnormalization certificateとは呼ばない。
新しい係数和・β・Bの再生成をせず、保存値に対するscalar上界候補と丸めだけを評価する。
補正済みnominal信号へBをもう一度掛けない。B_eng,upはshot式の振幅上限に使うもので、
信号を事後に再スケールしてbiasを変える処理ではない。
名目平均と実binary64 law/角度の対応は、実装gap=0という仮定の範囲で扱う。

各軸は既存fixed_signal_shotsへbias=b_upper、allowance=0、対応するlog_B_upperを渡す。
evidence_kindはEMPIRICAL、certificate_verifiedはfalse。既存関数の入力条件と2^52の安全整数domainを維持する。
必要な新しいlog_B_upperはB_eng,upのscalar logから保守側に丸め、expとの対応を確認・記録する。

```
N_a = ceil(2 B_eng,up^2 ln(2/alpha_a) / h_a^2),  h_a > 0
```

関数返却のbinary64 ceilだけに依存せず、既存trans_boundsのln端点と有理数演算で
同じ不等式のscalar上側を照合する。過小なら保守側ceilへ上げ、元返却値・修正量・理由を保存する。
これは保存結果のscalar会計であり、finite bias・信号・referenceの高精度再計算ではない。
h>0を確認できない行、必要field欠落、B上界条件不足、整数domain外はN/Gをnull/UNRESOLVEDとし、0で補完しない。
正のheadroomを示せないことだけから、真のbiasの物理的不適格を断定しない。

## 5. 価格とGの標本SE

各armの新32 trajectoryだけについて、保存mean_exactを使い

```
G_m = N_cos * Cbar_cos,m + N_sin * Cbar_sin,m
SE(G_m)^2 = (N_cos^2*s_cos,m^2 + N_sin^2*s_sin,m^2
             + 2*N_cos*N_sin*s_cos,sin,m) / 32
```

とする。mは主RZ、補助CX/depth。covarianceは同trajectoryの不偏標本共分散で、平均の共分散へは32で割る。
同じ保存32本のg_i=N_cos*C_i,cos+N_sin*C_i,sinからの標本分散ともscalar照合する。
旧pilotのpooling、cos/sinを64独立標本へ数え替える処理、方式間paired SEは行わない。
方式間G差のSEは独立標本の二乗和平方根。比G_P/G_DFは両方式のGが定義できる場合だけ表示する。
SEはNとbias/B政策を固定した価格標本の不確かさで、N自体の誤差や数値保証を確率化した値ではない。
depthはshot加重depth指標であり、装置上の並列wall timeとは呼ばない。

stage3のorder4未観測をそのまま引き継ぐ。標本価格SE・観測非重複から母平均の大小を保証せず、
population_mean_upper・population_tail_bound・実compiled/law保証はnull/未取得のまま残す。

## 6. 実装・実行・報告と今回のSTOP

承認後は次を一つの限定作業として行い、途中の通常技術判断ごとに再承認を求めない。

1. 保存JSONだけを読む最小接続を実装し、人工scalarデータで区間bias、誤差の一回計上、B丸め、ceil、
   headroom境界/null、軸共分散・独立方式差、既存関数との整合を検査する。
2. 新SOURCEをcommit固定・公開し、上記8入力と条件を照合して10行/20軸の科学的scalar会計を1 batch実行する。
3. biasと不確かさ・B上界の仮定・適格性・N・RZ/CX/depth費用とSE・方式間比較・未解決事項をJSON/CSV/REPORTへ保存する。
   新namespace案は`artifacts/df_pauli_matched_partition_validation/2026-11-02/stage4/run01`。
4. 原入力・旧STOPを保全し、公開・remote一致確認後mandatory STOPする。
   失敗時も途中出力を残し、自動retry・条件変更・追加科学計算を行わない。

小さいscalar会計なので巨大な並列処理を強制しない。恣意的なCPU/RAM上限を設けず、必要に応じcache等を使う。
共有環境や他利用者・他worktreeのSOURCE/入力/出力は変更しない。
科学結果報告は、nominal条件付き会計／価格標本SE／未取得の実装保証の3層を区別する。

**今回は本案の作成・入力bytes確認・公開だけで完了し、会計は未実行。**
次の担当はユーザー：本案のH4・5精度・M_B/M_C、finite-bit gap=0/compiled ρ=0、
保存engineering不確かさに基づくB上界候補を前提とする条件付き会計を明示許可するか判断する。
Codexはその許可後だけ上記の実装・人工検査・SOURCE固定・限定実行・公開へ進む。
別承認の理由はユーザーの今回指示、および共通AGENTS.md/統一指示§8.1が保存値の新科学数値解析を別scopeとするため。
承認証明書・原文保存・grant発行は不要で、チャットの具体的許可で足りる。
