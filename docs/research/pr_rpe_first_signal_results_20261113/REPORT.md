# PR-LD1/LD2：全13round signal科学実行結果

2026-11-13 JST。**固定計画a9abf8deのLD1/LD2を一回実行し、全26候補の数値gateが通過した。raw/corrected/exact-tailを78 complex出力、finite normalizer26件を保存した。新reference・旧保存mean再解析・retry・N/MC/C/Gは0。十分N・global RMSE・winnerは未資格/nullのまま。**

[SUMMARY](SUMMARY.json)／[全26round表](ROUND_SUMMARY.csv)／[元run](run01/result.json)／[実行log](SCIENCE_RUN.txt)／[HANDOFF](HANDOFF.md)。結果はRPE全体の精度・総コスト最適化への物理入力であり、最適PRの確定ではない。公開照合後mandatory STOP。

## 1. 固定条件とSOURCE

| 項目 | 今回実行した条件 |
|---|---|
| 系 | Linear H4、隣接1.00 Å、STO-3G、保存tol-only1e−8 DF10、sector(2,2)、dimension36 |
| 比較 | PR-LD1とPR-LD2。one-bodyはdeterministic、元順のDF prefixとtailを分割。全保存DF-Hがtarget、保存数値state条件付き、P0 |
| PF/RTE | S2、ΔPF=.1、K2、既存paired Taylor/normalizer、同exactκ、元scalar/fragment順/lawを保持 |
| RPE | h=.1、M12、全m0…12、元prior/decoder、shift0、εE=.00159360010199040 Ha、εqpe=εtrot=.00079680005099520 Ha、minkowski/finite_B_diagnostic |
| 固定計画 | **a9abf8de75bf3432fcd714e868e5c14c8884b8f8**、`docs/research/pr_rpe_first_signal_menu_20261113/`。exactκはMENUと両configurationの保存有理数をそのまま使用 |
| 数値SOURCE | **fefd548ecd56f60d72031b3d385f6f5e6ee95d06**、`pr_rpe_signal_provider`。変更なし |
| caller SOURCE | **041ac435a38afec3cf2a426661351871a1bac8ef**、`scripts/run_pr_rpe_first_signal.py`。既存batchを順に呼ぶ最小caller。新numerical algorithmなし |
| raw結果carrier | **16210932e5dabe7d12d41f35fea0e0a0d7a9b271**。各m record・configuration結果・menu結果を同内容で保存し、HANDOFFはこのcommitへ参照 |
| B | **437ba4acdb82708b6fc98e3cd119a2a734f7cf64**のV2 fixed-A identity/metadata受入。Bの統計計算は未実行 |
| 保存input/tail | INDEX **7a8bc7f132e166390c943e6b32c92cc6abe1baf4**、同H/state/snapshot、prefix1/2の既存λ/scalar。既存loaderのhash/layout検査を保持 |
| 保存reference | **bba150834ca30bdb186fbfd58c55a8d51177b855**の13full-H独立pairを共用。新action0 |

canonical IDはLD1 `a4dda8f7e3fc07666ca2aabf634917e8bced14eb62151eb629ef23c0cce21d3d`、LD2 `1fa33556c17303f51ba03c4eae11189d35e59b3f57faad0e69d9fd08bf9245ae`。固定configurationのfield/identityを変更せず、旧S0結果を新IDへ再ラベルしていない。

実行日時は2026-11-13 **19:15:09–19:15:19 JST**（run内はUTC記録）。Python3.11.0rc1/NumPy1.26.4/SciPy1.14.1。36次元のmatrix power/cacheを使い、全event列は生成せず、CPU/RAM/workerの独自capを追加していない。実行後もINDEXのH4入力/context計3ファイルが元hashと一致した。LD1成功後にLD2を開始する順序で、menu全体のSTOP条件を保持した。

## 2. 全13roundの局所結果

全roundでS2、qPF=2^m、actualδ=.1、piece allocation=[R_outer]。qRPEは別fieldとして保持する。以下のbiasは**|corrected−保存state full-H reference|という無次元signal空間の差**であり、energy誤差/RMSEではない。全complex値・SOURCE・各数値gateは[ROUND_SUMMARY.csv](ROUND_SUMMARY.csv)と各mのraw recordにある。

| m | T | LD1 R | LD2 R | LD1 attenuation | LD2 attenuation | LD1 signal bias | LD2 signal bias |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .1 | 2 | 1 | .96031935 | .97500480 | 1.340513e−5 | 1.338028e−5 |
| 1 | .2 | 4 | 2 | .95996567 | .97472386 | 2.680981e−5 | 2.674786e−5 |
| 2 | .4 | 8 | 3 | .95987587 | .96637121 | 5.361772e−5 | 5.349257e−5 |
| 3 | .8 | 16 | 5 | .95985334 | .95973946 | 1.072221e−4 | 1.069712e−4 |
| 4 | 1.6 | 32 | 10 | .95984770 | .95972131 | 2.143749e−4 | 2.138731e−4 |
| 5 | 3.2 | 64 | 20 | .95984629 | .95971677 | 4.286479e−4 | 4.276442e−4 |
| 6 | 6.4 | 128 | 40 | .95984593 | .95971563 | 8.572683e−4 | 8.552606e−4 |
| 7 | 12.8 | 255 | 80 | .95969159 | .95971535 | 1.714595e−3 | 1.710579e−3 |
| 8 | 25.6 | 510 | 160 | .95969157 | .95971528 | 3.429215e−3 | 3.421184e−3 |
| 9 | 51.2 | 1020 | 320 | .95969157 | .95971526 | 6.858366e−3 | 6.842304e−3 |
| 10 | 102.4 | 2040 | 640 | .95969157 | .95971525 | 1.371669e−2 | 1.368457e−2 |
| 11 | 204.8 | 4080 | 1280 | .95969157 | .95971525 | 2.743281e−2 | 2.736857e−2 |
| 12 | 409.6 | 8160 | 2559 | .95969157 | .95969983 | 5.486044e−2 | 5.473199e−2 |

integer Rの差は共通κとprefix別保存λから派生したscheduleで、独立R探索の結果ではない。m0でLD2 R=1も元最小event条件のまま実行した。LD2の少ないevent数や局所bias値だけでN/C/Gの優位を認定しない。

## 3. m12の代表complex signal

共通referenceは `0.1470900528467786 + 0.989123104742252 i`。同じ保存H/state/Tのfull-H pairで、厳密groundの単一位相信号とは自動同一視しない。

| 値 | LD1 | LD2 |
|---|---|---|
| exact-tail | .2011119974925329 + .9795681401326937 i | .2009861233671559 + .9795939742013534 i |
| corrected | .20111199756547962 + .9795681404874257 i | .20098612342494984 + .9795939744815362 i |
| raw | .19300548788357885 + .9400832827419839 i | .19288634937111757 + .9401161751062688 i |
| logB | .0411433313021524 | .041134716030912746 |
| B | 1.042001446266762 | 1.0419924691803404 |
| abs(corrected−exact-tail) | 3.6215e−10 | 2.8608e−10 |
| empirical u | 1.9030279617441277e−6 | 5.979488514004631e−7 |
| logB margin | 9.893361209945881e−7 | 3.1058534943974276e−7 |

finite-RTE差が小さいという出力と、その差を厳密に解像できたという主張は別である。m12の差はempirical allowanceより小さく、これを有限K誤差の厳密上界や候補間の確定的な性能差へ昇格しない。三つのcomplex差分は元recordに保存し、局所位相/Tからenergy biasを生成していない。

## 4. 数値検査・人工test

| 検査（26候補） | LD1最大 | LD2最大 | 元gate |
|---|---:|---:|---:|
| exact-tail power unitarity | 5.829194913871128e−12 | 8.550215301967306e−12 | 1e−9 |
| raw/corrected独立vector closure | 2.538524340812354e−10 | 1.311018718304664e−10 | 1e−9 |
| 保存reference pair差 | 9.279812219411241e−12 | 9.279812219411241e−12 | 1e−10 |

input norm、head unitarity、exact-tail norm、raw座標範囲、finite normalizer丸め/overflow/underflowも全件通過。threshold変更・再正規化・fallback・retryなし。rawはcorrected/Bの事後代入ではなく、各eventのP/bから独立power伝播した。scalarは全Tで一回。corrected/raw operatorにunitarityを要求していない。

allowanceは既存R-aware経験式とnormalizer ULP感度を保持したもので、logical event数に基づく。最大normはmacro/power出力境界での観測であり、全event・行列累乗中間値のrigorous enclosureではない。長時間のclosure増加とu/logB marginをBへそのまま渡す。旧m3/m4の観測norm欠測は旧SOURCEのまま保持し、今回の新観測で過去recordを埋めていない。

新callerは[mock5件](TESTS_CALLER.txt)PASS。LD1失敗時のLD2非開始、既存output拒否、途中例外時の先行結果保全を確認した。providerの人工10＋metadata7は固定fefdの既存検証を再利用し、今回実H4で追加のexpm正当性比較や旧mean照合を行わなかった。作者の技術確認であり独立科学レビューではない。

## 5. B・Dへの受渡しと未資格

[B_MEANS_LD1](B_MEANS_LD1.json)／[B_MEANS_LD2](B_MEANS_LD2.json)にcanonical identity/law/targetと今回raw carrierへのcommit/pathを保存した。[B_READINESS](B_READINESS.json)はB437の既存`prepare`だけを使うmetadata確認で、両構成のmean fieldを受入れた。残るblockerは`MISSING_N_X/Y`と`PHYSICAL_MEAN_UNQUALIFIED`。shotを指定/生成せず、統計実行0、十分N/global bound/Gはnull。

[D_PRICE_REQUESTS](D_PRICE_REQUESTS.json)は26candidate×X/Y=52軸の価格metadata。全q/R/piece/K/fragment順・profile/P0・SOURCEを揃え、価格取得/compile/fitは0。matrix powerによる古典高速化を回路event削減に読み替えない。

保存stateの厳密ground同定、DF前の元分子Hamiltonianへの誤差、finite-K lawとactual samplerの同一性、全decoder/branch/alias/loss、有限shot統計とRMSE移送は未資格。局所signalが揃ったことだけでこの資格を変更しない。

## 6. 次の設計自由度と判断に必要な証拠

まず**B**がこの二構成の全rawから同target/decoder/claimでwhole-shot配分と全riskを評価し、**D**が同scheduleの平均compiled価格・domain/不確かさを取得する。NとCの証拠が揃った範囲で全ΣN E[C]を比較する。今回はN最小・m12最安・event最小・PR winnerを認定しない。

その後の有限menuは、(a) head/tail価格の交換が支配的なら保留prefix3/5/7/9、(b) normalization由来の測定負担とevent価格の交換が支配的なら共通κ、(c) decoder精度が物理近似に敏感ならPF幅/既存対応formula、(d) finite-K効果が数値allowanceを超えて識別され費用上重要ならK、の順に**必要な証拠に応じて**提案する。今回これらの値を選択・探索していない。実Gが未資格なら順位は保留し、初回二候補で探索を終了しない。

**ユーザー**が次のB統計/N・D価格・追加物理menuの有限範囲を判断/承認する。**Codex A**は今回の結果を公開照合した後mandatory STOPし、追加signal・旧mean再解析・MC・価格・Gへ進まない。
