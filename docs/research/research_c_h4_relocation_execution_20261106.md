# H4線形Z再配置：保存Jの固定mass診断結果（2026-11-06）

**36 checkpoint＋別枠中立C0参照1件の診断を完了。主比較の再配置後ZZ-only massはENTRYが小さく、元のmapped tail massの順位と逆転した。**

H4直鎖・隣接1.00 Å/STO-3G、number-CDF8、Nα=Nβ=2、元frame順LD2。C0/P1/P2×strength .03/.1/.3×PR/ENTRYの保存済み16/64更新checkpointを固定し、Jを再最適化せずに同じ線形Z除去規則を適用した。PFのδ/r/Kやcompilerはこのmass診断の引数ではなく、今回のPF/RTE・回路・費用再評価は0である。

## 主比較：C0・64更新

単位はHa。D_diagは対角係数massで、強決定論baseline Dとは別の記号。Mは元の非identity mapped mass、M_ZZ=O+D_diag/4、M=M_ZZ+Z。

| arm / strength | 元M | 線形Z | ZZ-only M_ZZ | Z/M |
|---|---:|---:|---:|---:|
| PR / 0.1 | 0.498794143081 | 0.026562486113 | 0.472231656968 | 5.32534042% |
| ENTRY / 0.1 | 0.945068647756 | 0.564237101034 | 0.380831546723 | 59.70329271% |
| ENTRY / 0.03 | 0.944266678742 | 0.563217940683 | 0.381048738058 | 59.64606751% |

事前固定したcommon政策候補（PR .1 / ENTRY .1）のM_ZZ比PR/ENTRYは **1.240001415407**、差は **+0.0914001102453 Ha**。fair-retuned政策候補（PR .1 / ENTRY .03、異なるstrengthを明示）の比は **1.239294635574**、差は **+0.0911829189096 Ha**。いずれもENTRY_LOWER。結果を見て候補・strength・checkpointは変更していない。

PR .1では元massの約5.33%が線形Zだが、ENTRYの主候補では約59.6〜59.7%を線形Zが占める。この固定J集合では、元のtail mass差が線形Zをrandomized tailに残す辞書の選択に強く依存することを観測した。これは現在の固定endpointの比較であり、再配置後の目的で最適化したときの順位や、目的関数だけの因果効果を認定する結果ではない。

## 全18同条件組

36 checkpointを全数処理し、同一start/strength/予算の18組ではENTRY_LOWER 17、PR_LOWER 1、数値tie/判断不能0。唯一PR_LOWERなのはP1/.1/64で、比.995319373041。他の候補へ主比較を変更していない。

| start | strength | budget | M_ZZ PR/ENTRY | 判定 |
|---|---:|---:|---:|---|
| C0 | 0.03 | 16 | 1.347224895375 | ENTRY_LOWER |
| C0 | 0.03 | 64 | 1.326403941293 | ENTRY_LOWER |
| C0 | 0.1 | 16 | 1.241208849410 | ENTRY_LOWER |
| C0 | 0.1 | 64 | 1.240001415407 | ENTRY_LOWER |
| C0 | 0.3 | 16 | 1.241174224119 | ENTRY_LOWER |
| C0 | 0.3 | 64 | 1.241067984266 | ENTRY_LOWER |
| P1 | 0.03 | 16 | 1.065050408016 | ENTRY_LOWER |
| P1 | 0.03 | 64 | 1.028372621577 | ENTRY_LOWER |
| P1 | 0.1 | 16 | 1.048851882388 | ENTRY_LOWER |
| P1 | 0.1 | 64 | 0.995319373041 | PR_LOWER |
| P1 | 0.3 | 16 | 1.066322750085 | ENTRY_LOWER |
| P1 | 0.3 | 64 | 1.045667251917 | ENTRY_LOWER |
| P2 | 0.03 | 16 | 1.146863915427 | ENTRY_LOWER |
| P2 | 0.03 | 64 | 1.105410325082 | ENTRY_LOWER |
| P2 | 0.1 | 16 | 1.145126630748 | ENTRY_LOWER |
| P2 | 0.1 | 64 | 1.117004253236 | ENTRY_LOWER |
| P2 | 0.3 | 16 | 1.146274235462 | ENTRY_LOWER |
| P2 | 0.3 | 64 | 1.129072305489 | ENTRY_LOWER |

[36件のO/D_diag/Z/M/E/gap/M_ZZ全表](../../artifacts/research_c_h4_relocation/2026-11-06/science/run01/checkpoints.tsv)、[18比較の値・差・比](../../artifacts/research_c_h4_relocation/2026-11-06/science/run01/matched_pairs.tsv)、[全結果JSON](../../artifacts/research_c_h4_relocation/2026-11-06/science/run01/result.json)に固定表と候補identityを保存した。中立C0参照は別枠で、M=2.209371937066、M_ZZ=1.124347459961 Ha。gridや主比較に混ぜていない。

## 数値資格・来歴・停止

保存Jのmember hash、shape(8,4,4)、binary64、有限性、厳密な対称性は37入力すべてPASS。36 checkpointの元M/E/gapとC0参照の元Mを照合した。最大保存値残差は3.89e-16 Ha、M=M_ZZ+Z・ENTRY/gap式・E/5≤M_ZZ≤E/3等の最大恒等式残差は1.23e-15 Ha以内。登録allowance atol1e-12 Ha/rtol1e-12を変更していない。C0参照の元E/gapは保存metadataにないため照合値を捏造していない。

実行SOURCEは公開commit 052941ab7ce680f57b0b5ba9ef27ef8cdd4937c1、SOURCE SHA256 947f693b8092c47abdd0a35af45d0a268263e47653284f7d5f2cc0b0eda4ef83、契約SHA256 1d532e446ee13c4d01b83524d03aa91998de3452b43ffdccfbaa3cf1c32d8b3a。会話の明示実行指示に基づき、準備時の未承認filename/statusと契約本体はそのまま保持してfresh run01を一度実行した。旧runの再開やgrant再発行はない。既存sha/array_record/whitelist loader、coefficients/mass_diagnosticsを再利用し、U/h/gはdecodeしていない。

numpy1.26.4の既存専用fit環境を変更せず利用。独立workerを増やす必要のない128係数×37入力のvector化された軽量縮約で、SOURCE/inputのpre/post byte照合とimportを含む実測は約0.875秒。共有package・OS・他process・旧raw/J/U・旧SOURCE/契約/STOP・H6は変更していない。

[STOP.json](../../artifacts/research_c_h4_relocation/2026-11-06/science/run01/STOP.json)はfailed=false、checkpoint_rows=36、reference_rows=1、retry/resume=0、mandatory_stop=true、downstream_started=false。[結果hash・実測・集計](../../artifacts/research_c_h4_relocation/2026-11-06/science/summary_run01.json)を別ファイルに保存し、停止済みrun01の出力を後から書き換えていない。

## この結果から認定しない事項

再配置後のPF/RTE誤差、shots、W、native kernel/shell費用、総RZは未評価・nullである。Hamiltonian総和の代数的等価性だけでは積公式・誤差・価格は等価にならない。したがって今回のM_ZZの逆転から、再配置後の総RZの順位、旧辞書における条件付き45.47%総RZ結果の数値的否定、PR-aware目的の新規性・一般性のGO/NO-GOを直接導かない。

次はユーザー／GPTが、この固定J診断を踏まえて再配置後の公平なPF/RTE・native比較へ進むか、その科学条件と許可範囲を判断する。重要科学レビュー、再最適化、H2/H6、H4・1.15 Åは開始していない。**登録済み診断の完了によりMANDATORY STOP。**
