# 最小D1の承認候補と、D2/D3への境界

**未採択・未承認・未実行。** [REPORTのPhase A](REPORT.md)の選択をユーザーが決めた後でだけ科学評価に入る。現指示は実装・人工テスト・保存値照合・計画と公開まで。

## ユーザーがD1前に決めること

1. **target/stateの主張水準**：最小案は保存DF10-Hと同じ保存数値stateを用いる条件付き診断。保存stateを厳密groundと証明したとは扱わない。元Hの厳密ground energy保証を要求する案は、DF/state/reference誤差の追加資格が必要。D1で未知reference条件を0又は単位半径へ埋めない。
2. **固定する時間とPF/PR**：推奨案はT=.8の`49709717a6c3d405c60d`（B0/p7/S2/q_PF8）対`14bd81a375400c9b0305`（PR/p1/S2/q_PF8/R4/K2）。B0切捨てPFを含む旧候補集合を認めるか、Full PFだけにするかを明示する。後者では同じ保存signal表・価格表から該当Full行を別途固定し、新たな順位最適化はしない。T2のm5対は代替案で、同時追加しない。
3. **energy/phase/failure条件**：共通energy精度εE、各algorithmのαtotal、D1のαX/Y、βtotal/βstat、外部energy priorと位相符号・decoder条件。既存signal ε=.0005をεEと呼ばない。例えばD1だけの二軸α=.025ずつは全RPEの配分とは別で、正式採用ではない。
4. **時間scheduleの位置付け**：推奨案はbase h=.8、D1=m0。次のM/TmaxはεE等から後に決める。旧DF12 18round・CDF8 13roundは採用しない。base h=.1等を希望する場合もD1の保存Tとq_RPEを明示し、q_PFを同一視しない。
5. **費用の扱い**：D1は保存直接価格のidentity/coverage照合まで。主指標RZ・CX等の副指標、D3で探索と独立確認のどちらを凍結価格に使うか、許容する条件付き価格モデル／不確かさは別に選ぶ。D1でGや新勝者は作らない。

選択肢を示しただけで採択・実行を推定しない。別のレビュー開始手続きやapproval証明の作成は求めない。

## D1でCodexがまとめて実行できる範囲（明示許可後）

- **入力/SOURCE**：今回公開する`h4_df10_rpe_connection.py`、既存geometry/Hoeffdingは基点fd76a3ed（元実装1f5ea971f）。主入力は155529e2の保存signal/endpoint/科学binding/price。既存signal SOURCE `e417ffc9f52b13c18f5493193ce510d858664259`と、元コード・dirty/継続履歴はscientific_provenanceから保持する。公開carrierを元実行SOURCEへ付け替えない。
- **処理**：選択済み一時刻・二candidateの保存reference/corrected/raw、total bias、u、log B/marginを読み、同じtargetへのreference phase/radius差、誤差disk、raw余裕、β条件、X/Y別十分Nを評価する。保存E0を使う場合は診断referenceとしてだけ用い、decoderのbranch選択には渡さない。旧finite-time Nとの区別、α配分、保存価格IDの対応まで一回のcampaignで出力する。
- **資格**：経験的u・数値固有state・ideal/actual law・reference条件を明記し、数値pilot結果をrigorous boundへ格上げしない。保存log_B_marginを扱うraw誤差が不明、reference資格が不足、又はrho/β条件を満たさない場合は該当結果を未資格／N=nullとし、追加のH/state/signal計算へ自動進行しない。
- **出力**：新しい独立namespaceの一時刻×二candidate表。phase/radius/誤差/α/Nの由来、欠測、価格coverage、条件付きPASS又は未資格を記録。単一時刻から全RPE energy成功・費用優位を認定しない。
- **除外**：SCF/DF/state再生成、固有値再計算、新finite-mean signal、trajectory、build/compile、価格再fit/再標本化、PR/PF再選択、G集計・順位・成功率simulationは0。必要になれば、その具体的な追加範囲をユーザーへ戻す。
- **終了**：保存原本・STOPを保持して結果公開・照合後mandatory STOP。原科学runやconsumed grantを再開しない。

D1で数値化するerror diskの仕様は、ユーザーが選んだstate/energy referenceと主張水準に従う。式や引数が揃っていることは、現在未確定のreference条件を満たす証拠ではない。

## D2：全roundへ進む条件

D1が接続可能と判断され、ユーザーが**共通task・有限候補menu・h/M・T_m=2^m h・α/β配分・state/reference/law誤差**を採択し、その数値実行範囲を明示許可した場合のみ進む。

- h=.8案なら保存T=.8/1.6の同一構成行を利用できる可能性がある。T1.2/2.0はそのdyadic scheduleへ自動挿入しない。T≥3.2等の未取得時間は新しいsignal範囲である。これは時間coverageの整理であり、今回その時間のsignalを評価していない。
- 時間・q_PF・δ・formula・prefix・R_outer/K・state・scalar/lawが完全一致する保存点は再利用する。別時間のsignalを補間・冪乗しない。
- 未取得点は既存`h46_long_time_signal.evolve_batch`等のfinite mean/bias計算を優先。保存input loader・spectral batch・不変sectorの加速を適合範囲で使い、SCF/DF/stateの再生成は不要なら行わない。負時間pieceやR配分を変えない。
- reference/state条件が閉じ、十分N・全軸α union・全round phaseとbranch/prior条件が揃うかを判定する。decoderは既存関数を使い、known E0にbranchを合わせる処理は行わない。

## D3：同じenergy taskの総資源

D2の十分Nと全round条件が揃った後、別途許可した価格・総会計の範囲で実施する。

1. 保存直接価格を優先し、time/δ/q_PF/formula/prefix/R/K/H/state/law/compiler/control/scopeが一致する行だけを結合する。量子Nと価格nを分ける。
2. モデルが必要ならqの定義、条件、fit/holdout/外挿domain、共有校正・covariance・残差を明示する。D C6〜C8のCDF8/Strong-Reference係数、DF12 opt2の係数、Track C/newAの改善率はDF10へ移植しない。コードと検証方法だけを再利用する。
3. 新adapterの軸別ΣNC、又は等Nなら既存aggregate_totalsを使う。RZ/CX/各depth/sizeとpeak幅を別表示し、normalizationや価格nを再乗算しない。
4. 直接/model/外挿の寄与と、順位を変え得るmodel不確かさを報告する。未価格化点・未取得covariance・未観測tailを0補完せず、全体CI未取得はnullのまま。

DF10で方法と条件を確認した後のH6 DF19は、別の実行範囲としてユーザーへ戻す。D1完了をD2/D3/H6の自動実行許可にしない。
