# 実H4 PBS一件の技術的選定

Codex sub-agent `pbs_readiness` が保存記録だけを調べ、`FULL_PF` の元job
`98693b829295fc884330` を選択した。新しい数値評価・adapter実行・dry-check・PBS投入は行っていない。

対象configurationは `70f7b31744ecd87fe277511bad7d1fc7dace9b1a097c315729c566f706c75edd`、
`N_X=N_Y=[30,28,26,24,22,20,17,15,13,11,9,5,3]`。
アクセスできた5,372件のREQUESTと5,332件の正常終了PROCESS_RETURNを技術的に確認し、
保存されたprocess wall timeが最短の58.225秒、監視したprocess treeの最大RSSが1,764,976 KiBの記録を採用した。
これは科学的な方式順位や新しい性能測定ではない。他方式の保存時間と資源は[CANDIDATE_INVENTORY](CANDIDATE_INVENTORY.json)。

元の条件はlinear H4 1 Å、STO-3G、tol-only DF 1e-8・観測rank10、`L_D=10`のFull H、
`4th(m5_best)`、保存binary64の`h≈0.1`・要求PF幅`16h≈1.6`、round `m=0..12`。
各roundの実PF幅、law、原SOURCEは[SCIENTIFIC_CONDITIONS](SCIENTIFIC_CONDITIONS.json)へ保存値のまま転記した。
保存した判定は`MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD`。
元のraw model診断を再現する参照であり、物理的総誤差、十分N、ground-energy認証、方式の優位性を新たに主張しない。

元の[REQUEST](REQUEST.json)・[OUTCOME](OUTCOME.json)・[ROUND](ROUND.json)・[PROBLEM](PROBLEM.json)はGitのraw byteを保持している。
固定locatorとhashは[ORIGINAL_FILE_LOCATORS](ORIGINAL_FILE_LOCATORS.json)、入力配置に必要なlocatorは
[INPUT_LOCATORS](INPUT_LOCATORS.json)。M5のPLANと保存signalのGit objectは、選定時点ではクラスタ側に未配置だった。

元jobのmetadata identityを同じhash規則で照合し、contract digest `2e3dac22b1065151`、job IDの一致を確認した。
[JOB_IDENTITY_CONTRACT](JOB_IDENTITY_CONTRACT.json)はmetadata hashの確認であり、数値kernelを呼んでいない。
[EXACT_REFERENCE](EXACT_REFERENCE.json)の上下界と判定は元OUTCOMEからそのままコピーした。
decoder、独立IID law、loss、threshold、既存価格・signal出所は[EVALUATION_CONTRACT](EVALUATION_CONTRACT.json)。

元ROUNDのUNLIMITED等は履歴であり、今回の実行条件ではない。親担当が新しい一件のscopeへ固定して実行を所有する。
元計算のoptimizer SOURCEは53bb、今回のcontrollerは925、科学Q3 checkoutは28d、固定routeはcbcを使用する予定である。
その対応とprofileの未確認差分は[PROFILE_MATCH](PROFILE_MATCH.json)に記録した。
既存のPBS FullPF dry-checkは別のS2候補を検査しており、このM5 REQUESTのPASSへ転用していない。

[DUPLICATE_CHECK](DUPLICATE_CHECK.json)では、元の科学的結果はすでに存在する一方、
利用者専用クラスタ領域にある8件のassignmentは全て人工・TRANSFERREDで、実H4のPBS完了・進行記録がなかった。
今回の目的はその元結果を参照したPBS移植性の一件確認であり、元runのSTOPや結果は変更しない。
参照できた2026-10-23・10-25の統一指示は10.3までで、番号10.5は取得できなかった。
利用者の重複照合指示と、参照可能な再利用・SOURCE/raw/STOP保全規則を適用した。

次の実行責任者は親Codex。必要入力の配置、完全REQUESTのmetadata検査、固定scope・資源・重複の最終照合後に、
許可された実H4一件だけを開始する。
