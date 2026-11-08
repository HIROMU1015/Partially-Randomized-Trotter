# N5-B2終了後：最小の科学実行候補と判断

2026-11-08 JST。[REPORT](REPORT.md)。**提案のみ・未実行**。新しい科学条件・precision・policy・保証方式を採択した文書ではない。承認証明書／grantは不要だが、次の数値実行にはユーザーの明示承認が必要である。

## 1. 優先順位

N5-B1の160 fixtureは再実行しない。同じconsumerの任意有限列に対する因子分離はSOURCEで条件を示せる。m17への主要な障害は、現policyの域外rejectとlogical precision資格である。巨大wholeを作ってもこの障害は解消しない。

最初の判断担当は**GPT／ユーザー**。Gを将来のraw信号保証へ使う条件と、logical precision／domain方針の採否を判断する。新規合成の必要性を決める前に、保存wordと現b36がどこまで使えるか知りたい場合には、下記の**r32中央4 positive literalだけの保存入力数値診断**が最小候補となる。GをRPE会計へ正式採用する前でも、このlogical成分の診断自体は区別して実施できる。

## 2. 固定入力とcall範囲

[NEXT_NUMERIC_INPUTS.json](../../../artifacts/pr_reference_n5b2_static/2026-11-08/NEXT_NUMERIC_INPUTS.json)が対象sliceの正本。すべて既存SOURCEの保存literalで、新しい到達集合ではない。

| 項目 | 固定内容 |
|---|---|
| 条件 | m17、B′／Reference×X/Y、H4／1Å／STO-3G／DF12、Partial S2 LD3 h=.02 K2。88 bits・保存626／630・alpha1/720を維持 |
| B′ positive | 0x1.7e4088e024c5fp-12、0x1.fdab627b8d37ap-14 |
| Reference positive | 0x1.7e4088e024c5fp-11、0x1.fdab627b8d37ap-13 |
| 符号 | 上記各literalと保存negative literal。別角／mod／snapを加えない |
| Policy参照 | **現行のまま**b36、j=-36…6、nearest ties-to-even、ascending written order、保存43 basic。新b／precision候補は0 |
| Guard入力 | 保存BASIC_WORDS_MANIFESTとbasic_records.tar.xz、record／sequence hash、actual outward guard。Guardを再計算しない |
| 量子化入力 | N3-B1b-1 INPUTSのerror_budgetにある保存sqrt2上端、actual binary64 theta |
| 比較条件 | N5-B0 M17_INPUTSの当該arm X/Y uniform_gate_error_strict_upperをそのまま使う。必要precisionへの採択ではない |

**Codex**が明示実行承認後に実施する候補callは次だけ。

1. 固定入力と43 recordの保存対応を照合し、既存`round_index`を変更せず再利用する。4 positiveのindex評価と4 negativeの符号対称性確認、合計8 index評価まで。
2. 4 positiveについて、同じ固定basic選択規則からselected jを求める。Actual binary64とdyadic indexの差delta、保存sqrt2上端によるQ(theta)、選択済み保存guardの和G(theta)、Q+Gを評価する。
3. 論理的に同じboundを持つnegativeを確認し、各armの保存X/Y十分capと照合する。4 positive×2 axisの8比較を超えてroleを追加しない。

`DictionaryProvider.provide`全体はT集計やword連結も行うため呼ばない。再利用するのはround_indexと、provideのindices／selected／delta／q_upper／g_upperに対応する必要な純粋処理だけ。小さいwrapperの作成・数値実行をこの範囲で別途承認する案である。**Word文字列の新生成、strict_guard、acquire、emitter／consumer、native build/compile、Operator、sampling、RPE、T価格callは0**とする。

これは現policyが定義するlogical productの**成分上界**を調べる処理で、実emitted circuitの新strict certificateを作る処理ではない。新しいprecision policyやC0優先の変更もない。

## 3. 判定とSTOP

| 結果 | 研究判断へ返す内容 |
|---|---|
| 保存Q+Gがcap内 | 当該中央sliceでは保存word構成を再利用できる十分証拠候補。全role・全support・m17 RPEには移さない |
| 保存Q+Gがcap外 | 現b36／保存guardのこの十分証明では足りない。新b／別word／再合成が必須という下界にはしない |
| Quantization又は選択guardが主要な項 | どの成分の仕様を次に判断するかの材料にする。新precision探索を追加しない |
| 入力・record・domain・SOURCE不一致 | 欠損を保存してSTOP。0／推定値で補完しない |

4 positiveの評価と指定検算の完了時、又は上記不一致時にmandatory STOP。成功／不十分のいずれでも、別literal、追加role、新guard、b探索、合成へ自動拡張しない。

保存capは十分割当であって必要条件ではない。Scopeの出力は成分・符号対称性・cap照合・SOURCE・入力・STOPだけとし、m17成功／失敗、T優越、最適precisionを認定しない。

## 4. それ以外の残課題

| 作業単位 | 現証拠と不足 | 再利用／実装・計算／承認 |
|---|---|---|
| 全pathへのG適用 | ConsumerのSOURCE代数とN5-B1 runtimeあり。Actual m17 terminalと完成policyは未資格 | 既存consumerを再利用。実本番bindingが必要ならその最小差分だけ別scope化。新経路の局所実行には承認 |
| Logical word strict資格 | C0／43／85の保存guardあり。全必要位置の目標保証は未完 | 上記中央sliceを優先候補。同じwordの新guardが必要なら対象recordを別途固定して承認 |
| Quantization precision | b36の既存証明あり。Actual m17 targetへの適合未完 | 上記診断の後、必要なら新b等の選択規則をGPT／ユーザーが判断。今回未選択 |
| Vacuum／scalar／event域 | 保存包絡、実域外scalar、event supersetあり | 域保証／policy方針を先に選ぶ。新包絡数値評価や新key生成が必要なら有限scopeと承認。中央診断では解決しない |
| 共通policyの完成 | C0優先、有限停止、actual adjointを再利用可能。全m17入力へ返せない | 選んだprecision／域／reuse規則を固定してから必要最小限のprovider差分を設計・承認。取得済み43成功を新keyの取得保証にしない |
| m17 RPE | 旧counts／d_old／N／geometryとN4-CのX/Y max資産あり | 完成policy・pathが資格された後に、logical全位置和→raw disk→半径／位相／保存Nの4条件だけを別途承認。旧runner.mainを起動しない |
| T-count・shot総費用 | M7 anchorと有限価格は保存。M17の共通policy価格なし | 実行可能policy確定後の別scope。626／630のため絶対frame価格も必要。新価格は明示承認、全18round総Tは自動追加しない |

後続のRPEでは旧diskを正しい単位で一度だけ加え、pure scalarの扱いを明文化し、Gで消せない残項を保持する。T比較では同一law・同一情報・同一policyと各axis価格を使う。共通frameの1-shot取消しはshot重み付き総費用差にはそのまま使えない。

**現在は静的N5-B2完了によるmandatory STOP。次担当はGPT／ユーザーで、保証・policy方針と上記限定数値診断の採否／承認を判断する。Codexは次段階を開始していない。**
