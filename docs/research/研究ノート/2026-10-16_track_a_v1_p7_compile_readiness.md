# 2026-10-16 V1：B0+I p7/q1 S2実測の準備

利用者の指定科学レビュー第8節の第1優先として、[p7 S2の2軸実測契約](../track_a_v1_p7_compile_readiness_20261016.md)を準備した。新compileの実行は明示的に未承認。
結果commit25bbdd33…のremote一致後、専用branch/worktreeを分離し、元V1結果branchと科学SOURCEは不変。固定予測72621/5 RZ/軸、各軸2%以上で即STOP、合計2compile、再fit/retry/拡張禁止を採用した。
scalar C・full-H reference・経験的allowanceと保存shotを再利用し、signal/shotを再計算しない。native control検査は将来の回路同一性検査として上限4回を明記する。
実測後のweighted RZとPR保存値403423200の比較は、さらに別の数値再解析承認後に行い、compile phaseから自動接続しない。
p7専用19を含むpinned90人工tests PASS、metadataでSOURCE/provider/runtime/inputと既存660ファイル不変。新科学計算・有効grant発行0。commit/push/remote確認後mandatory STOPし、実行承認を待つ。


## 後続：利用者実行承認・2軸実測完了

利用者の「進めてください」を公開SOURCE84b9ea22…の最大2軸one-shot compile承認として記録し、fresh SOURCE/input/profile/資源/保全gate後に起動。cosine/sineとも14,492 RZ、固定予測14,524.2との差は各161/72460（0.222191554%）でPASS。cosineの個別合格を確認後にsineへ進み、再fit・追加compile・retryは0。

wall73.241100秒、native exit0、全回収、最大monitor gap1.003944秒。SOURCE21/provider230/runtime34273/input15/旧grant等660ファイル不変。旧draftを保全したまま新grantのみ発行し、one-shot lock・予約・結果を保存。新signal/reference/scalar/SCF/DF/state/primitive/sampling/V2/総費用再会計なし。

[実測報告](../track_a_v1_p7_compile_results_20261016.md)。この候補の費用を実測したが、ε=.05の総費用・削減率・優劣は未更新。別承認待ちのmandatory STOPを維持する。厳密bias保証・全grid費用誤差保証・PR母平均保証は未認定のまま。


## 後続：別承認による実測費用のε=.05限定比較

利用者の「次の作業に進んで大丈夫」を受け、入力公開結果a73ce484…とremote一致、実行code e078a7e6…を固定し、新grantで1候補・2倍率行だけone-shot再会計した。保存shot cosine10216/sine14356と各実測14492 RZにより、決定論費用356097424 RZ、保存PR確認点403423200 RZ。PR削減率は−13.29012029%、倍率1/10で同じ。旧modelの−13.03895727%から逆転の符号を維持した。

[結果・到達範囲](../track_a_v1_p7_measured_reaccount_results_20261016.md)。全136候補の再ランキング・他精度更新・保守的bias ruleの最良選択は未実行。通常signed bias・経験的allowanceの保存点比較であり、母平均・厳密bias・全候補実測最良は認定しない。

wall0.010498秒/exit0、Decimal/cross-productと旧model再現PASS。SOURCE22/provider230/runtime34273/旧入力15/新入力8、旧compile raw22/旧grant等663不変。新compile/signal/reference/scalar/primitive/SCF/DF/state/sampling/shot/fit/候補再選択/V2なし。旧grant/lock/原planを保全し、新grant/lockだけを追加。mandatory STOP、成果物はlocal commit、pushは利用者が行う。
