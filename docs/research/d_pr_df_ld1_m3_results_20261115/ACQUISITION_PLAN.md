# PR-DF LD1 m3：取得前の固定条件と後続m4規則

今回の採択・実行範囲はm3 (T=.8/q8/R16、physical 045843b2429bece6f29b) の元normal sampler fresh index0..3、同path X/YのS/C、最大8実compileだけ。作業branchは `codex/d2-pr-df-ld1-m3-acquisition-20261115`、元P2 `39ddd59edf54a3a901e5302207b2765ddbd03fbb` から独立worktreeを作成した。Pauli LD2 holdout/STOPとDのm9 branch/worktree/jobは変更しない。

[INPUTS](INPUTS.json)はC計画 `8eef0ad7a9b04e04d766f5e05b3b32eee50a794e`、C受入索引 `c99d511e2422b8f250490b11054f754a81e536c5`、元P1/P2のSOURCEを参照する。元exactκ・H/state/snapshot・fragment/basis順、H4/linear1Å/STO-3G/tol-only DF10/LD1/S2/Δ.1/K2、scalar c+s_Rを総Tで一回、CONTROLLED_ZZ_CX_CRZ_CX_V1、元control/P0/測定付きfull Hadamard wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17を保持する。

元runner全体はreplay/m0/m6/残差summaryを含むので実行しない。新しい薄い入口 `scripts/run_pr_df_ld1_m3.py` は元の `P.load_c1_backend` → `F.make_backend_class` → `Backend.context` / 元 `sample_requests` / `price_requests` を使用する。P1/P2 backend・新ZZ・高速S/compiled価格本体は変更しない。両軸を別々にcompileし、同じrequests/eventsからSとCを得る。

取得前の§10.5差分は[PREFLIGHT_METADATA](PREFLIGHT_METADATA.json)。元C索引でm3有限pathは0、構造E_I[S]をcompiled価格へ繰り入れず、旧C1の異R/旧ZZを代替にしない。確認した範囲に受入可能なm3 receipt/実行jobなし、再利用0、新規上限8。外部非公開jobまで未実施と断定しない。出所不明のcache hitは未解決として停止し、compileで上書きしない。

## 固定した較正・holdoutの区別

- m0/m6：保存P2 fresh各4 path、元cohort `D_P2_DIRECT_PILOT_V1` と元seed・SOURCEを保つ。INPUTSにC索引と元receiptのlocatorを保存した。P1/replayは参照のままでpoolしない。今回再compile・新平均・残差集計を行わない。
- m3：`D_PR_DF_LD1_M3_CAL_NORMAL_20261115_V1`、index0..3。元carrier `trial_seed` で各8 step seedを固定した。
- 後続m4：q16/R32/T1.6、physical `3df0cfd3356c358aeda8`、`D_PR_DF_LD1_M4_HOLDOUT_NORMAL_20261115_V1`、index0..3、各16 step seedを固定した。m0/m3/m6の較正seedと重複しない。**今回はm4のsampling/取得/開封0、実行jobへ含めない。**

後続Cの固定案は軸別 `saved E_I[S] + theta0 + thetaq*q + thetaL*qR`。今回の受渡しはm3の個別S/Cだけで、残差の集計、fit、係数・m4予測の凍結はCへの別指示。ideal law Iとsampler Qの差、有限標本変動、未観測rareは未評価。全law E[C]/CIを4 pathから認定しない。

## m3取得前に記録するm4の記述的比較規則

後続のC凍結公開後、さらに別指示で取得するm4 fresh4 pathを対象に、軸ごとに凍結予測Pと観測Oを比較する。`O=(Z0+Z1+Z2+Z3)/4`、`d=P-O`、`abs(d)`、`d/O`、`abs(d)/O`をexact整数/有理数で保存する。正は過大、負は過小予測。X/Yをpoolせず、欠測時にpath数4を変更しない。

O=0ならd/abs(d)を保存し相対はnull、理由ZERO_OBSERVED_PRICE。receipt欠落・gate/identity異常では該当軸のO/誤差をnullにし理由を残す。モデル数値許容差を設けず、PASS/FAILを付けず、`model_verdict=NOT_ASSESSED_NO_ADOPTED_TOLERANCE`。観測後の閾値設定、refitや追加pathへ進まない。この規則の記録はm4実行の許可ではない。

## 取得gate・資源・STOP

元P2の入力file/preparation/hash・fragment順、normal seed、signed event allocation/time、newZZ有効化、同event digest、軸別compiler/fingerprint・wrapper mutation・整数S/Cを確認する。新たなfull-wrapper statevector gateを追加したとはしない（元P2 freshはfalse）。既存P1/P2の人工・replay証拠は元SOURCEの記録として再利用し、今回m0/m6 replayは実行しない。

実contextは一回復元し4独立pathを並行処理する。元C1のcompiler内部pool1を保持し、path単位で並列化する。context・sampler・S・compose・軸別compile/saveの開始/終了・時間とpath別完了を保存し、必要なら60秒の定期stackを専用private fileへ記録する。共有設定や他jobに変更を加えない。実行SOURCEはこのcaller/INPUTSを含むcommitを `--source` で明示し、取得前に固定する。

重大異常は部分rawと失敗軸を保存し新規投入を停止する。無断retry/seed変更/別軸補完0。今回の終了点は8個別S/Cの公開・CへのHANDOFF、commit/push/remote照合後のmandatory STOP。残差集計/fit、m4、E[C]/CI/N/G/winnerへ進まない。
