# LD3 VP total/Q3 HANDOFF

LD3は指定VPで条件付きモデル閾値に適合。新Q3は1件、LD2は50be68b80の保存VP引用。新signal/sampling/MC/compile/価格/shot探索/retryは0件。原API、partials、CPU/RSS、親process終了codeを保存済み。mandatory STOP。

| 構成・同一VP | MSE下界 Ha² | MSE上界 Ha² | 判定 |
|---|---:|---:|---|
| LD2 保存 | 6.207498269405979019554497794E−7 | 6.207498269406694956690683824E−7 | 条件付きモデル適合 |
| LD3 新1件 | 6.206210455307360129958626682E−7 | 6.206210455307820120226411551E−7 | 条件付きモデル適合 |

VPはX/Y各`[48,45,42,39,36,33,29,26,23,19,16,13,9]`、USER_SUPPLIED。τは保存十進εEのexact二乗。decoder/target/law/lossはLD2 VPと同一。LD3自身の辞書・13信号・κ/scheduleを使用し、normalization/scalarは再適用していない。

## 受入・SOURCE

- 実行SOURCE：7733089b85d013bad314e108d539b3afe0271635、`scripts/run_pauli_ld3_vp_q3_fixed.py`。
- 原API snapshot：c7961238679f7dc7dd358baa924ec1e3579a7121、`run01/PAULI_LD3_VP/API_RETURN.json`。RESULT_SOURCE.jsonはこの解決可能なGit参照を持つ。
- 元ID dd5b881d308f6fc48d3c91d08cec22c5fbbac296f60242e9bfb433c3e0321847 → total ID 6757998a50bec243e1cd14d1b968457c673d383c47cb95c6b52e8b0ed7e1a0eb。
- `TOTAL_BINDING.json`：元configurationと新decoder契約の完全対応。
- `TOTAL_SIGNAL_VIEW.json`：元packetを保全したSOURCE-bound view。
- `C_SIGNAL_INPUT.json`：Cの既存signal_input契約への受入view。
- `C_RESULT_JOIN.json`：元/total configuration、candidate、law、各round/axis、allocation、元physical context、辞書、signal/numerical/result SOURCEの26軸索引。価格欄とGはnull。元価格生成IDの付替えには使わない。
- Aの8c8ce542eのpure helperを`bind_packet.helpers.dictionary_identity`に使用。P1 jobは変更なし。元A carrier・target・qualificationは変更していない。

## 数値状態・保存状態

rho閉包・質量保存・残余会計・MSE/moments・scopeと保存後gateはPASS。残余mass約2.39284E−21、loss上界約4.59990E−20 Ha²を残した。zero/tieは全関数規則で保持。これはUNKNOWNではない。

原API/RESULT/RESOURCEはworker自身で親return前に保存。104 DP子processのOS終了codeは全0、最上位workerの個別OS終了codeはrun01で未記録。成功futureとcaller終了code 0は保存済み。観測hookのmodule参照のみ完了後に人工修正し、実H4は再実行していない。詳細と測定範囲はREPORT/TESTSを参照。

## ロードマップへの差分・残る資格

PR-Pauli LD3は、13信号coverage/total metadata接続に加え、LD2共通VPの限定Q3モデル適合証拠が取得済みとなる。LD2を再取得する必要はない。今回の2行のMSEは条件付きモデルであり、LD3の費用優位や方式最適性は未評価。価格担当へは同じ元物理候補・辞書・lawを束縛する価格証拠が必要で、今回のC索引は価格資格を与えない。

ground energy認証、実samplerへのlaw移送、厳密数値certificate、十分N/global RMSE/G_qualifiedは未取得。追加配分、費用比較、科学計算は別指示が必要。ロードマップ本体は統合担当が更新する。旧STOPを保全し、今回もmandatory STOP。
