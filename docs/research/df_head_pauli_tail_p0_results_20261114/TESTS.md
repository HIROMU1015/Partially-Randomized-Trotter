# P0の技術テストと実科学数値gate

プロジェクト専用Python `/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B` を使用。システム環境やCPU/RAM/workerの上限は変更していない。

| 実行 | 結果 | 範囲 |
|---|---|---|
| `tests/check_df_head_pauli_tail_p0.py` | 5 PASS | 人工3mode/非可換complex Hermitian factor・signed weight、LD1/2、CAR↔JW bit順、identity/scalar、sector label分類、scalar不一致・非Hermitian・state lift不一致時STOP |
| `tests/check_df_head_pauli_tail_contract.py` | 4 PASS | 既存人工N1 sector漏れ、射影非involution、finite RTE signed eventと辞書依存normalizer、full/reduced macro mean、global collectとprefix差 |
| caller `--preflight` | PASS | 固定configuration/context/input byte bindingのみ。保存NPZ array load・科学leafなし |
| 承認済みrun01 | 81/81数値gate PASS | 保存H4 DF10/LD1/LD2のP0だけ。全operator/sector/closure gate1e−9、state1e−12 |

人工テストでは`np.load`とPauli moduleの`QuantumCircuit`を禁止し、保存分子入力を使っていない。最初の人工fixtureが一粒子二modeだけでは個別label漏れを示さないことをテスト作成中に確認し、spectatorを含む人工三mode一粒子sectorへ修正した。科学run開始前のmetadata-only照合では、historical contextに対する後年のcomposition追加を別記して全元binding項目を照合するよう修正した。これらはP0科学runのretryではない。

その後SOURCEをfd111f989に固定し、fresh run01を一回実行した。実科学エラー・retry・閾値緩和・fallback・再正規化は0。旧4件の実行で既存ファイルのDeprecationWarningが出たが、testは通過し、当該ファイルは変更していない。

これは著者による技術・数値検査であり、独立レビューや外部再現ではない。人工PASS・P0 gate PASSは、P1 mean/actual sampler/十分N/全RPE RMSE/価格優位の資格ではない。
