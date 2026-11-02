# 旧H6環境からの準備値export（新しい信号計算・transpileなし）

SOURCE: `b53271f59609f9b633f1771a85c7ca7181fdba8a`。このpacketにはSOURCEのexporterとcodecだけを含む。
旧科学SOURCE `5a903aea2492c2d211c1e15e7737df0d50694c12` の224 pathsを維持する。

旧H6を計算した `.venv311` を変更せず使用する。既存研究worktree/dirty/untracked/結果を保持し、下の独立worktreeを作る。
保存Hamiltonian/stateの再生成、SCF、DF分解、state solver、signal、transpile、grant発行は行わない。
読み取るのは既存snapshot。小さなDF基底・係数準備7種類を保存し、旧prepared hash7件・旧450 eventsを厳密照合する。
exporterはprocess単位でthread1、nice10、AS2GiB、wall120秒、file16MiBを設定する。共用設定・venv・他jobは変更しない。

## 旧環境側Codexへの依頼

1. packetを旧利用者home配下の新exclusive directoryへ展開し、同梱FILE_HASHES.jsonと各file SHAを確認する。
2. 旧repositoryとPythonの実在を確認する。パスが違う場合だけ、実在する既存pathを使う。
3. 公開commit `054bb7b2b2ca788b47ebdc5c14c115f24c8b6379` の独立worktreeをhome配下に作る。branch切替/reset/clean/venv変更を行わない。
4. exporterを一度実行する。既存source224/input/environment/全7準備hashが一致しない場合、入力再生成や検査緩和をせずexport_terminal.jsonを報告する。
5. 成功時は `h6-frozen-preparations-v1.json` と `SHA256SUMS` のbytes/SHAを報告し、利用者へ引き渡す。export_terminal.jsonも保全する。

## コマンド例

以下ではpacket展開先を `$HOME/h6-basis-exporter-20261012` とする。既存directoryがあれば新しい連番を使い、上書きしない。

```bash
git -C '/home/abe/Project/Partially Randomized Trotter' \
  -c maintenance.auto=false -c gc.auto=0 \
  worktree add --detach "$HOME/h6-frozen-basis-source-20261012" \
  054bb7b2b2ca788b47ebdc5c14c115f24c8b6379

'/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python' -I -P -B \
  "$HOME/h6-basis-exporter-20261012/export_track_a_h6_frozen_preparations_v1.py" \
  --repository "$HOME/h6-frozen-basis-source-20261012" \
  --codec "$HOME/h6-basis-exporter-20261012/h6_frozen_preparation_v1.py" \
  --output "$HOME/h6-frozen-basis-export-20261012" \
  --export-old-preparations
```

新サーバーの受領先:
`/home/AbeHiromu/projects/h6-handoff-evidence/20261012/native-reproduction-fix-v1/incoming/`

転送後にserver側のbytes/SHA、全7準備hash、450 seed/events、既存447 full-wrapper fingerprintを照合する。
旧450の一致gate、seed、compiler、full-wrapper、科学条件は緩和しない。
既存447のfingerprintが異なれば、その2本を元費用系列へ混ぜず、再測定が必要と報告する。
既存H4は停止・変更しない。今回のexportで本計算runnerは起動しない。
