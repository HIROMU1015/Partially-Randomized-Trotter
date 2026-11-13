# A → C/B：Full PF追加幅のmetadataと科学範囲

[REPORT](REPORT.md)／[未実行SCIENCE_PLAN](SCIENCE_PLAN.md)／[PLAN.json](PLAN.json)／[ROUND_SCHEDULE.csv](ROUND_SCHEDULE.csv)。新H4数値/保存数値再解析/N/C/Gは0。最初の5配置と分枝未解決のM5/32hを区別する。

## Cへ渡す情報

`configurations[].constructor`は公開C86f37d25/D e7dca5f7/B437ba4ac APIでcanonical V2を再生成する。独自IDや旧IDの再ラベルはない。6配置・78行それぞれにcanonical candidate ID、完全physical candidate、exactT、qRPE、qPF、actual幅、対応step cellとSOURCE、旧catalog XY status/locatorを保持する。

| label | configuration ID | status |
|---|---|---|
| S2_16h_over_7 | `11cf8ca6677a7ce651a00d827d0948e758510f95876acd24bf4d826c2bb6ca5d` | 初回承認候補 |
| S2_8h_over_3 | `7754671bba8a61cf3054f4e43f9672517f3f786f81c21d0bdea02cfdef30b575` | 初回承認候補 |
| Y4_16h_over_3 | `a799697fc8ca36b024acec26af43313777948ead432e66804c670649a18afe76` | 初回承認候補 |
| Y4_32h_over_5 | `9e930a476209c65a0d9a17b8a2bf4a8605fa43cb70018d1e37c5d960a3f58c27` | 初回承認候補 |
| M5_16h | `70f7b31744ecd87fe277511bad7d1fc7dace9b1a097c315729c566f706c75edd` | 窓規約を含む承認候補 |
| M5_32h | `a8853e14bd7cfe698e3de3d5e6d4fb99a780e5bbf647eb3290988343c02cfd64` | 分枝未解決・初回範囲外 |

exact h-relative分数を表示floatに置換しない。全round qPF=ceil(T/requestedΔ)、actualΔ=T/qPFを参照する。signed piece順・重み・負時間・backend signed-time hexはstep_cellsに保持し、同step共有を検査済み。S2/Y4/M5は1/3/11 pieceである。これはfull measured wrapperのRZ数ではない。

初回5配置のm12は旧正本catalog照合域外。全13roundでのEXACT_METADATA_MATCHのXセル数は順に1/0/1/1/2、Yも同じ。残りはOUTSIDE_CHECKED_CATALOGで、新しい価格受入ではない。compiler/control/native/lowering/profile、scalar、axis、physical recipe、P0を一致させ、Cが既存coverage/追加価格の必要性を判断する。旧PR C1やE[S]係数を移植せず、q比例・piece比例のcompiled価格を仮定しない。

最新C公開context `367dc98d832ecb4d769f2a3da3d174bc697331a5`は確認したが、主にB0I/Full-H signalで、ここで必要な新Full PF長q価格を提供する成果ではない。B/C/Dのbranch/worktreeは変更していない。

## Aの次回callerとBへの意味

元scanner `run_h4_full_pf_heff_scan.py`は旧15cell/PRINCIPAL固定。新規約を暗黙適用せず、scope5cellと各branchを渡す最小caller変更を科学SOURCEとして固定する。元`pf_effective_hamiltonian.evaluate`のENERGY_WINDOWは再利用可能で、評価器やPF式の再実装は不要。

M5/16hの[-2.5,1.0] Haはlogの数値分枝候補で、物理energy shiftやdecoder prior変更ではない。元H全固有系/旧M5/.8との対応を検査するが、solver energyで位相を合わせない。保留M5/32hは現global window APIの制約を越えるため、mode/group分枝規約が別途必要。旧target中心Schur helperの無断流用はしない。

次回出力には全spectral成分・保存state weights・元Hとの対応・δE・数値残差・指定二時点meanを含める予定。Bはearlyを含む同じ全scheduleへ接続し、各δEをそのままRPE bias上界やRMSEへ置換しない。raw signalを用いるriskがPF効果を既に含む場合はδEを再加算しない。十分N/global RMSE/G/winnerはnullを維持する。

## 次の役割とSTOP

- ユーザー：初回5cell、M5 ENERGY_WINDOW、保存M5/.8 spectrum検査1件と指定mean20出力の有限範囲を承認するか決定する。
- Codex A：承認後だけ最小callerを固定し科学実行。SOURCE/数値/分枝不一致なら部分結果でSTOP、自動retryなし。
- Codex C：別途承認された価格範囲で、全round exact recipeとdomainを比較する。
- Codex B：物理結果・shot/statistics・同candidate価格が揃った範囲で全RPE risk/G比較を行う。今回は依頼・実行しない。

今回の技術成果公開・remote照合後mandatory STOP。保留候補、追加科学、価格、N/Gへ自動進行しない。
