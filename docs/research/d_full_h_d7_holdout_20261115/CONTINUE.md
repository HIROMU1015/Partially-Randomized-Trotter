# D7独立holdout：固定job実行中（再投入禁止）

利用者の実行指示・続行指示に基づき、観測前結合/重複照合、人工32件、import-only23 moduleのPASS後に起動した。
SOURCE/input `9cbabc617e95a81810924413c1b2b06ced329514` は指定branchへpushしremote一致確認済み。
予測b5165f11bb63f9eb2a6bfe385ba75d8003b7e724、規則f0765067a5a8b39888e36f3a8b80f1bdbda69223を変更しない。

## 既存jobを観測する

- 実行checkout：`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d-full-h-d7-holdout-20261115`、detached HEAD固定。
- 元D1公開worktree：`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d-full-h-p2-caller-prep-20261114`。
- M5_ROUND session31027：fresh path0/1、各X/Yの4 compile。
- M6_ROUND session70309：fresh path0、各X/Yの2 compile。
- private：`/home/AbeHiromu/projects/d-full-h-d7-holdout-20261115-private/run01/<cell>`。
- public：実行checkoutの `docs/research/d_full_h_d7_holdout_20261115/run01/<cell>`。
- console/time/JOB：run01直下。PROGRESS/OBSERVATIONS/CHILD_EXIT、PATH_<index>_PROGRESSはprivate。

**同じjobを再投入しない。正常workerのSOURCEを書き換えず、停止・再起動しない。**
実行済み/途中のnative record、個別compile、作用・実装gate、子process終了codeを保持する。
異常時は新規投入を止め、部分receiptと理由を保全してSTOP。無断retry・seed変更なし。

## 完了後の残作業（追加科学条件なし）

1. 既存RESULT・CHILD_EXIT・PATH log・timeを確認する。新しいpath/compileを起動しない。
2. 実行checkoutのrun01を元D1 worktreeへcopyし、`RESULT.json`をlocal保全、ignoreする。
   元D1側で `publish_full_h_d7_holdout.py --prepare <cell> --execution-source 9cbabc617e95a81810924413c1b2b06ced329514`。
   これは保存元receiptと6個別軸の価格/元compiler bytes/実装gateを検査し、既存READ_RECEIPTでlossless圧縮する。
   prepare済みのcellを繰り返さない。大きいgzipは64MiB以下のbyte partsを公開する。
3. 結果fileを指定branchへcommit/pushして結果SHAを固定する。
4. 同scriptの `--bind <結果SHA> --execution-source 9cbabc617e95a81810924413c1b2b06ced329514`。
   公開SHAからraw/元caller record/6軸compiler bytesを再照合し、COMPARISON.json/csvとPUBLICATION_CHECKを生成。
   固定予測は分子/分母を直接読み、m5同軸2 path平均とm6同軸1 path値の4比較だけをexactに計算する。
5. REPORT/HANDOFF_Cを作り、全比較行からC予測とD1個別receiptへ固定SHA/path/selectorを記す。
   予定/再利用/新規compile/有効/欠測・資源・ロードマップへの差分を報告する。ロードマップ本体は編集しない。
6. 関連索引/現状/manifest/dated noteを適切に更新し、commit/push・remote/公開成果物を照合してmandatory STOP。

4件有効/全O>0はDESCRIPTIVE_COMPLETE、O=0を含むならDESCRIPTIVE_COMPLETE_RELATIVE_UNDEFINED。
欠測/不一致/gate不成立はINCOMPLETE_OR_INVALID。m5片方だけを平均の代用にしない。
tolerance=null、model_verdict=NOT_ASSESSED_NO_ADOPTED_TOLERANCE。モデルPASS/FAILなし。
m6 n2、追加round、再fit、事後閾値、全law E[C]/CI/N/G/winnerは範囲外。
