# SERVER_B研究単位別一括整理候補・保全計画

**計画のみ。実削除・登録解除・branch削除は未承認であり実行しない。83件を研究単位ごとの一括判断に提示する。**

B3保護3件、保持59件、追加保全171件、他環境条件63件は今回の対象ID集合に含めない。主registryの344件と他3管理領域の追加35件を合わせた379件のうち、下記83 IDだけが候補である。IDと非公開実pathのbindingは本人専用原本に保持する。

| 作業単位 | 件数 | 正確なID集合 |
| --- | ---: | --- |
| BASELINE_OR_OTHER_UNRESOLVED | 1 | SERVER_B-WT-0344 |
| DF_H2 | 1 | SERVER_B-WT-0054 |
| DF_H46_PF_PR | 5 | SERVER_B-WT-0133, SERVER_B-WT-0134, SERVER_B-WT-0135, SERVER_B-WT-0343, SERVER_B-WT-0345 |
| DF_H5_H7_OPEN_SHELL | 7 | SERVER_B-WT-0143, SERVER_B-WT-0144, SERVER_B-WT-0145, SERVER_B-WT-0146, SERVER_B-WT-0191, SERVER_B-WT-0192, SERVER_B-WT-0194 |
| DF_H6 | 10 | SERVER_B-WT-0155, SERVER_B-WT-0164, SERVER_B-WT-0168, SERVER_B-WT-0173, SERVER_B-WT-0174, SERVER_B-WT-0175, SERVER_B-WT-0178, SERVER_B-WT-0180, SERVER_B-WT-0181, SERVER_B-WT-0182 |
| DF_H8 | 28 | SERVER_B-WT-0195, SERVER_B-WT-0196, SERVER_B-WT-0198, SERVER_B-WT-0199, SERVER_B-WT-0200, SERVER_B-WT-0201, SERVER_B-WT-0202, SERVER_B-WT-0203, SERVER_B-WT-0204, SERVER_B-WT-0205, SERVER_B-WT-0206, SERVER_B-WT-0207, SERVER_B-WT-0208, SERVER_B-WT-0209, SERVER_B-WT-0212, SERVER_B-WT-0218, SERVER_B-WT-0219, SERVER_B-WT-0222, SERVER_B-WT-0226, SERVER_B-WT-0227, SERVER_B-WT-0229, SERVER_B-WT-0230, SERVER_B-WT-0231, SERVER_B-WT-0232, SERVER_B-WT-0246, SERVER_B-WT-0247, SERVER_B-WT-0248, SERVER_B-WT-0259 |
| DF_NATIVE_TRACK_A | 18 | SERVER_B-WT-0004, SERVER_B-WT-0007, SERVER_B-WT-0250, SERVER_B-WT-0252, SERVER_B-WT-0254, SERVER_B-WT-0261, SERVER_B-WT-0324, SERVER_B-WT-0329, SERVER_B-WT-0331, SERVER_B-WT-0332, SERVER_B-WT-0335, SERVER_B-WT-0336, SERVER_B-WT-0348, SERVER_B-WT-0349, SERVER_B-WT-0354, SERVER_B-WT-0358, SERVER_B-WT-0359, SERVER_B-WT-0360 |
| PAPER_EVIDENCE_PRESENTATION | 4 | SERVER_B-WT-0139, SERVER_B-WT-0172, SERVER_B-WT-0210, SERVER_B-WT-0330 |
| TRACK_B_G6_HISTORY | 5 | SERVER_B-WT-0023, SERVER_B-WT-0025, SERVER_B-WT-0027, SERVER_B-WT-0028, SERVER_B-WT-0030 |
| TRACK_B_OTHER_HISTORY | 4 | SERVER_B-WT-0361, SERVER_B-WT-0372, SERVER_B-WT-0374, SERVER_B-WT-0375 |

## 保持資産とB3非干渉の根拠

候補全件は前回の限定依存監査で候補だった同HEADのcheckoutで、現在のtracked変更0・untracked0・ignored0を確認し、少なくとも1つのlive GitHub tipとlocal保持refで同commitを保持している。共通Git owner、B3 SOURCE/baseline checkout、可視利用/lock、旧metadata/code bindingがあるものは除外した。これは完全な全環境依存閉包ではなく、SERVER_B所有者の将来利用終了は未確定である。

branch/tag・共通Git・固定SOURCE/input/result/STOP・grant/ledger/marker・private原本と既存cohortを保持し、物理展開copyとGit履歴を分離する。候補外の未公開HEAD/変更/ignored資産は保全計画CSVの個別IDへ返す。追加小型copy47件は完全custody認定ではない。候補で外部private原本・契約が見つかった場合もそのIDを除外する。

## ユーザー判断と直前preflight

1. ユーザー/研究担当がSERVER_Bの上記作業単位で将来利用・再開予定・他Codex利用終了を判断する。LOCAL131件の利用終了判断は代用しない。
2. Codexが対象IDを固定し、他環境・SOURCE/provider/inputの具体的物理path条件、外部private custody、保存参照の到達性を閉じる。未解決IDはバッチから外す。
3. 実整理の具体的許可後、HEAD/branch/status/untracked/ignored・lock/使用・共通Git/backlink・保持GitHub ref・archive可読性・SOURCE/path差分をfreshに確認する。B3状態が不明なら関連資産は保持する。
4. 合格集合へ通常git worktree removeだけを適用する計画とする。branch削除、force、prune、clean、reset --hard、gc、手動rmで拒否を回避しない。今回はremoveを実行しない。

## STOP条件と復元計画

HEAD/statusの変化、未保全原本、入力/固定SOURCE不一致、想定外job/lock、B3依存、保持ref欠落、共通Git/backlinkの異常、Gitの通常remove拒否では該当IDを保留する。共通異常なら全実整理を止め、他研究processを操作しない。復元は保持branch/commitから別の新規checkoutを作る計画とし、固定path契約に関わる復元はその契約を先に確認する。

現在の占有量・解放量は再計測していない。前回容量を今の削減効果として使用しない。保全/owner/他環境条件が閉じた研究単位なら、数件ずつに恣意的分割する必要はない。
