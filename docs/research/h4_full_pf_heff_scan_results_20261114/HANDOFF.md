# A → C/B：Full PF有限格子の局所候補

[REPORT](REPORT.md)／[機械可読C packet](C_HANDOFF.json)／[15 cell一覧](CELL_SUMMARY.csv)／[raw run](run01/result.json)。科学SOURCE=`b64f22b97c641fc97d13e9872b723f789e4fa454`、固定計画=`878000d8d5d956b1d2f5513986ae40d0612e0bdd`。H4 DF10/sector(2,2)/Full PF/prefix10、同H/state/snapshot/full DF-H target・h.1/M12/P0。

## Cへ渡す3候補

| ラベル | canonical configuration ID | 局所δE（Ha） | m12 qPF / signed S2 piece数 |
|---|---|---:|---|
| S2/.2 | `9e96ed75f7f10b5f993ba9e2841684e52ad97b7d4e9df07b7b0070f02c852981` | +5.35412051879e−4 | 2048 / 2048 |
| Y4/.4 | `99e539037a07452f2f5a4029549980c82ec72d1b832cfcbcff59313ee3b0ae0e` | −1.90607716779e−4 | 1024 / 3072 |
| M5/.8 | `24286ac55e9b96a58fa4f11ae4fe5e6c4f9f9ca0c6a9cf24f501dc146db810e2` | −7.28958845286e−7 | 512 / 5632 |

元εtrot=.00079680005099520 Haとの局所spectral screenに基づく、格子内で最も粗い幅である。S2/.2はこの3配置の全13roundで構造上のpiece数が最少で、保存locatorもあるためcompiled比較の最初の優先候補にするが、compiled価格・G順位を認定したものではない。

packetの`shortlist[].constructor`から既存B/D APIで完全V2 configurationを再生成できる。roundごとのq_RPE/qPF/actual幅、canonical candidate ID、完全physical candidate、step-cellと固有系SOURCE、正本catalog rowへのlocatorを保持する。独自ID・旧IDの再ラベルはない。

## early roundと価格domain

- S2/.2：m0は.1の既存69ee cell、m≥1は今回の.2。
- Y4/.4：m0=.1、m1=.2、m≥2=.4の今回のY4 cell。
- M5/.8：m0=.1、m1=.2、m2=.4、m≥3=.8の今回のM5 cell。

要求幅ではなくactual幅の固有系を参照する。同一物理stepの共有と、異なるcanonical configurationの受入を区別する。S2/.1の既存mean/stepは今回再計算・再解析していない。

S2/.2とY4/.4は保存m3/m4のXY価格が旧D catalogでEXACT。M5/.8はm3 EXACT/m4 MISSING。3候補ともm12は照合域外。これは既存statusの転記で、新しい価格受入ではない。full controlled measured X/Y wrapperのcompiled RZ、P0、元profile/native/scalar/seed/domainを一致させる。旧nativeと新ZZ loweringをcompiler hashだけで混ぜない。価格値の新解析・予測・fit・compile・Gは今回0。

## 物理結果の意味

新14 H_eff、直接PFと全spectral mean各28、元H新eigh/reference0。全数値gate通過。principal分枝の最低groupは全点で元最低groupとの最大trace overlapを持ったが、ground/branchの科学的認定やRPE誤差上界ではない。全energy/vector/state weights・overlap・残差は各`cell_XX/EIGENSYSTEM.json`に保持した。

M5は.8でも小さいshiftを示すが、今回格子の上端であり最適幅ではない。S2/Y4の予算境界補完と併せ、必要な追加科学範囲はユーザーが別に決定する。M5の拡張ではlog branch/aliasとearly actual幅を改めて固定し、無断unwrapしない。

**B**は全spectral成分、状態仮定、prior/decoder/shot/lossを含む既存risk契約へ接続する。raw直接riskがPF効果を含むならδEを再加算しない。今回全13roundの新mean、N、MC、RMSE、Gを生成していない。十分N/global RMSE/Gはnull、資格falseを維持する。

**ユーザー**がCへの価格取得又は追加物理探索の有限範囲を承認した場合に担当Codexが実行する。Aは今回の公開・remote照合後mandatory STOP。
