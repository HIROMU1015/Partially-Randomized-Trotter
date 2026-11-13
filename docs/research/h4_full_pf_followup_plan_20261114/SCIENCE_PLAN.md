# 未実行：Full PF追加5 actual-stepの科学計画

今回は計画のみ。以下をユーザーが明示承認した場合に限り実行する。保留M5/32h、N/MC/compiled価格/Gは含まない。[PLAN.json](PLAN.json)がexact幅・canonical/physical ID・SOURCE・全roundの正本。

## 固定条件と再利用

Linear H4/1.00Å/STO-3G/tol-only1e−8 DF10、weighted-Hermitian completion、sector(2,2)/36次元、保存H/state/snapshot、Full PF/prefix10、元fragment順・signed weights、scalar一回会計を保持する。h/M12/P0、target、energy_shift=0、prior、decoder、元εE=.00159360010199040、εqpe=εtrot=.00079680005099520 Ha/Minkowskiを変更しない。元H/state/snapshot bindingはPLAN.sharedに保存。

既存direct-expm PF step builder、matrix power、`pf_effective_hamiltonian.evaluate`、保存input loaderと元固有系cacheを再利用する。元固有系はcommit `69ee0ff8898960d5f9959af8d34d89f223caec87`の`docs/research/h4_full_pf_heff_results_20261113/run01/EIGENSYSTEM.json:reference_comparison`。cacheのSOURCE/H/全basis/residual gateを保持し、不一致なら新eighへfallbackしない。

旧actual幅9cellはPLAN内saved_sourceで参照し再計算しない。ただし下記M5/.8のwindow整合検査1件だけは新しい保存スペクトル検査として承認範囲に含める。元run/STOPは上書きしない。

## 新5cellと指定二時点

| 新actual幅 | 分枝 | 最初の使用round・q | m12 q | 新step/Schur | direct/spectral mean |
|---|---|---|---:|---:|---:|
| S2・16h/7 | PRINCIPAL | m4・7 | 1792 | 1/1 | 各2 |
| S2・8h/3 | PRINCIPAL | m3・3 | 1536 | 1/1 | 各2 |
| Y4・16h/3 | PRINCIPAL | m4・3 | 768 | 1/1 | 各2 |
| Y4・32h/5 | PRINCIPAL | m5・5 | 640 | 1/1 | 各2 |
| M5・16h | ENERGY_WINDOW [-2.5,1.0] Ha | m4・1 | 256 | 1/1 | 各2 |

新PF step最大5、Schur最大5、直接PF complex mean最大10、同全spectral再構成最大10。新full-H reference0、元H新eigh0、旧保存signal再解析0、normalizer/N/MC/C/G0。保存M5/.8の全スペクトルwindow整合検査は1件。別の元H対角化や新referenceで未解決を埋めない。

ENERGY_WINDOWの窓SOURCEはこの計画、purpose=EFFECTIVE_SPECTRUM_WINDOW。幅×Δ<2π、boundary/cut、再構成を既存規約で検査する。旧M5/.8全スペクトルのエネルギー・重み・group対応がこの窓と矛盾しないことを先に検査し、失敗時は自動unwrapや窓移動をせずSTOP。保存solver energyは分枝選択に使用しない。

## 手順・出力・数値条件

1. metadata生成の確定commit、実際のcaller修正の確定科学SOURCE、入力とcacheのbindingを記録する。旧scannerを無変更で再実行せず、新5cell/branchを渡す最小callerを固定する。新承認証明書・grantは不要。
2. 既存contextを1回取得し、同H/state/scalarと元固有系cacheを照合する。上記window整合検査を行う。
3. 各新actual幅で直接expmによりscalar込みstepを構成する。同一stepはcache共有、巨大event列生成なし。既存Schur評価器で全energy/vector/state weights/group、元Hとのbasis/projector overlapを取得する。
4. 対応groupを確認しδEを保存する。最低値同士の差と元最低groupに対応するshiftを区別し、対応不明はUNKNOWN。δEだけからgroundやRPE bias上界を認定しない。
5. 表の初使用時点とm12で、q powerの直接PF meanと全spectral meanを照合する。新full-H signal・旧保存signal診断を追加しない。earlyは保存actual cellのSOURCE対応を保持する。

既存数値policyを継承する。state norm1e−12、unitarity/Schur Gram・residual/reconstruction/power・mean照合1e−9、Hermitian固有系1e−9 Ha、energy group1e−8 Ha、phase cut guard1e−8 rad。元PLANの規約を緩和・再正規化・射影補正しない。値・失敗診断とpartial outputをfresh runに保存する。

## STOPと解釈

SOURCE/H/state/snapshot/recipe不一致、cache/basis不一致、非有限、branch/window不成立、既存数値gate失敗、direct/spectral不一致なら部分結果を保存しmandatory STOP。自動retry・新幅・別分枝・閾値緩和は禁止。単にδEが局所予算を超えることは有効な観測結果で、科学的FAILを隠して候補を変えない。別の承認済み点は数値/分枝gateが成立する範囲で評価できる。

この有限案は局所δEと時間発展再構成を検証する。全13round mean、全RPE RMSE/十分N、compiled価格、Gや最適候補を認定しない。raw由来riskへδEを二重加算しない。

## 保留候補

M5/32hのconfigurationと全13round metadataはPLANにあるが、上の科学承認範囲外。全スペクトルの分枝規約と必要continuation callが未固定であり、additional operator/Schur各1という最終点だけでは科学範囲を定義できない。必要な技術準備と有限callを別途提示した後で承認を得る。今回の結果次第で自動実行しない。

承認後の実行成果はREPORT/HANDOFF・rawとともにcommit/push、remote SHA/tree/blob照合後mandatory STOP。新幅探索・C/N/Gは別途承認対象。
