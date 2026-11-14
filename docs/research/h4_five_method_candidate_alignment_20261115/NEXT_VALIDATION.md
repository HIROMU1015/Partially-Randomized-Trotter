# 最小追加検証と科学実行の前提（未承認・未実行）

最終G比較への不足を7種に分ける。全13round価格を直接compileする前提を置かず、既存receipt・保存mean・E[S]を先に利用する。新計算の必要性は指定claim水準・価格精度・対象候補をユーザーが決めた後に確定する。

| 種別 | 解決手段 | 実行条件 |
|---|---|---|
| 1 SAVED_RESULT_ADMISSION | 保存mean/receiptの受入、元cohort・role保持 | 算術再解析が必要なら別科学承認。今回の4保存receiptは再受入不要 |
| 2 METADATA_OR_DECODER_BRIDGE | 元signal→B packet、PR-DF fixed-grid→total、scalar/source/価格sidecarの接続 | 科学law・decoder意味・固定契約を無断変更しない |
| 3 NEW_SIGNAL_SCIENCE | 未取得round mean | candidate/round/backend/gate/SOURCEを固定した明示承認 |
| 4 DIRECT_COMPILED_PRICE_SCIENCE | 同物理条件の新direct/paired S,C | lowering/scalar/compiler/axis/cohort/law/seed/roleを事前固定して明示承認 |
| 5 INDEPENDENT_PRICE_MODEL_VALIDATION | cal→係数凍結→独立holdout、長q/R移送 | 指標・許容差・停止・未知CI/law/rareの扱いを結果を見る前に採択 |
| 6 MSE_OR_SUFFICIENT_N_SCIENCE | 同signal/decoder/lossでwhole-vector risk・十分N | Bの有限scopeと統計・数値契約の明示承認 |
| 7 USER_COMPARISON_OR_SCIENTIFIC_CONTRACT_DECISION | 正式候補、claim水準、許容価格誤差・risk/decoder/loss方針 | Cは採択しない。G差やwinnerを先に作らない |

## 候補ごとの最小案

| 候補 | 先に再利用・技術接続 | 承認が必要な不足と判断可能になること |
|---|---|---|
| Full PF / p7 shortlist | A c212のS2保存meanをpacket化。S2L旧値は再資格化/再計算の選択。M5/m4保存4receiptを再利用 | A §7のS3（各方式S2 8、Y4 7、M5 7 mean、旧m3を使わなければS2/Y4各1追加）だけを候補として提示。既存step活用。全signalが揃ったらBがrisk/N。価格は既存CELL_PLANで新cal20・holdout12の32wrapper案。12軸別groupのfit/凍結/holdoutは別科学範囲 |
| p7 S2/.1を比較側に選ぶ場合 | 全13signal/B metadata既存。m3/m4旧価格locatorの受入可能性を先に確認 | shortlistへのsignal移送ではなく、このschedule専用の不足価格・検証を立てる。旧原本の量/measurement/実装証拠が不足なら自動補完しない |
| PR-DF LD1 | 全13signal、旧N/partial MSE、E[S]26軸、m0/m6のP1/P2を再利用。B9c0でversioned decoder bridge完成。残る技術不足はtotal scalar等の価格受入 | total decoderでの新MSE/Nは別承認。同domain residual modelの識別・cal/holdout不足をDへ照会。必要なq/Rセルを既存P1/P2との重複を避けて事前指定。ideal E[S]＋sampler residualのlaw差、有限cohort/rare/共有誤差/数値資格を明示。LD1全lawE[C]が何精度で利用可能か判断するための検証 |
| PR-DF LD2 | 全13signalとB9c0の版付きdecoder橋渡しを保持。LD1価格を移植しない | total MSE/Nと同LD2 preparation/law/new-ZZの最小paired S/C、モデル識別が必要。LD1kernelの流用は自動承認されない。cohort数・精度は未決で、巨大m12 T2は前提でない |
| PR-Pauli LD1/2 | 全13signal、B total橋渡し、保存model MSE/N。LD1 B0はwiring参照 | Aの未承認B1-W（LD2 m0 2wrapper）、B1-V（提案6）、B1-CAL（辞書ごと24）の段階案を再利用し、目的に必要な段階だけを選ぶ。B0のcal再利用は取得前policyがない限り昇格させない。辞書別・q/R別holdoutとfinite-cohort/law/rareの資格が必要 |
| Full-H κ2 | 全13signal、B total橋渡し、P-1 m3 wiringと別revalidation evidenceを再利用 | A P-2旧q1-only較正はrank2で3係数を識別できない。D7等のrank3案（24又は26wrapper）、q方向比較・独立holdoutを既存設計から選ぶ。P-1をcal点としない。非configuration calセルをD callerへつなぐ技術差分は科学取得前に必要 |
| R448 / Full PF S2/.1 | 補助candidateとして独立保存 | 正式比較に採るかを先に判断。κ2/shortlistのGへ混ぜない。採択前に新科学callを増やさない |

上のwrapper数・mean数は既存A/D計画の提案範囲であり、本作業の実行承認ではない。新しいcohort平均・係数・残差・Gを計算していない。旧STOP/失敗runを変更しない。

## 価格精度と停止

決定論PFの観測値基準2%（`abs(Cobs−Cpred) <= .02*Cobs`）はCの**未採択案**。各route/formula/X/Yごとにm4,m5校正、係数・指標を凍結してからm6独立holdoutとする。許容差を結果に合わせて変えない。不一致・入力SOURCE不明・scalar/実装差は保存してSTOPし、閾値緩和・自動retry/別law/再compileへ移らない。m7〜m12はm6合格だけで正式に資格されない。

ランダム方式の価格精度は、有限cohort分散、モデル系統差、ideal/sampler差、rare未観測、共有covarianceを含むclaimに依存する。必要な数値閾値・cohort数を決めるだけの適格N・比較margin・policyは現時点で不足している。従って精度を0や保証済みとしない。

一般の感度関係 `|δG| <= Σ N·|δC|` は計画上の関係式に留め、実Nとの積・総和を実行していない。Bが同候補の適格Nと比較目的を提示した後、ユーザーが必要価格精度を事前固定する。小価格差を根拠に事後許容差を選ばない。

**mandatory STOP**。SOURCE確認・metadata接続が完了したことを、上記科学処理への許可と解釈しない。
