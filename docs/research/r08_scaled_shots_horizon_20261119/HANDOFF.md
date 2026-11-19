# R08の共通ショット倍率と最大ラウンド番号

## RomuTail通常運用への採用（2026-11-20 JST）

Windowsの通常Previewが解決していた `romutail-mixed-methods-0c67daacd` を基点に、
検収済み `romutail-scaled-shots-9ab86dadc` を新規runの `current` に採用した。
controller・Q3 adapter・worker輸送は `9ab86dadc274a501c9c535cc1c00b92197d992bf`。
科学Q3 SOURCEは `28d976b99cdf8329adcc1b79327e329e947ef774`。
RomuTail、server2、PBSで配置を照合し、各ホストで保存PR-Pauli要求のM=11、12のdry-checkが通過した。
M=13はm13信号・価格待ちであり、固定Q3の検証域外も検査した。既存の科学SOURCE・route view・
DP cache・受付と容量設定を引き継いだ。PBS所要時間規則は新adapterのSHAへ結び直し、
各工程の見積りと資源上限を継承した。

Windows通常入口から隔離した人工run `search-ecee387725493ee8588f` を開始した。
初回応答は試験中の同期実行で時間切れになったが、PBS `267499.CPUcluster-PRISM` は
2 CPUで正常終了し、中央で12ラウンド10件、13ラウンド10件、14ラウンド8件を採用した。
Full PFの別候補がm13入力待ちでも残りの候補を評価できた。同じrunをWindowsからResumeし、
追加job・二重採用・PBS再投入が各0件だった。旧runのResumeは人工検収で旧固定版を解決し、
実運用の旧runはStatusで旧releaseを確認した。切替前後で旧runのROUND、設定、起動記録は一致した。
倍率・マップ・worker・固定版・準備環境の回帰31件と、固定route viewでのDP回帰19件が通過した。
別途、旧ショット選択のAST固定1件と旧owner照合2経路の失敗は、切替前の採用版でも再現した。

Windowsのrelease明示PreviewでH₄、H₆、H₈を確認し、切替後のrelease指定なしPreviewでも
新currentを確認した。H₆は `{"scope":{"system":"H6"}}` で固定Full PF／M5の保存メタデータを表示する。
保存信号13ラウンド、直接価格14/26軸、m7〜m12の価格12軸とH₆専用Q3入力の不足を返す。
H₈は同じ形式で `"H8"` を指定し、準備中と表示する。H₆とH₈はPreview専用。
新規の実信号・実compile・実Q3・本番探索は各0件。

Cの通常同期には倍率・Mの読込と従来の方向表示を反映した。公開版は
`5cba7efd4307ca0edaf576998b6f7d22c566f1f6`。停止・保留・費用省略の理由、M=12と全Mの最安、
同じ回路でMを変えた費用差と未確定の扱いは、隔離した公開fixtureで通常同期を検収した。
Romulaptopの通常閲覧用 `H4-search-map/index.html` は公開版とSHA-256が一致し、倍率欄と
従来の方向表示の実装が届いたことを確認した。人工結果は検証用の保存先で扱った。

WSLは既存の通常配備基点がなく、補完用bundleを用意した時点で停止中だった。
次回の起動後は `.rpe-central/deployment/romutail-scaled-shots-9ab86dadc/WSL-CATCHUP.bundle` を
既存 `3cea00961` 系列から取り込み、新releaseの `HOST.json` と `RELEASE_COMPONENTS.json` で配置照合する。
server1は復帰後に同じ構成で配置し、対応routeの保存入力dry-checkを通してから参加する。
各ホストに必要な配置・証拠はRomuTailの `.rpe-central/deployment/romutail-scaled-shots-9ab86dadc`
にある。[共通運用ルール](../rpe_distributed_operations.md)

Full PF、B0I、PR-Pauli、PR-DFの通常探索に、`RPE_SCALED_SHOTS_V1` を追加した。各回路と `M` について、X/Y共通の倍率から全shot列を作る。各軸の整数配分は `ceil(s*(11+4*(M-m)))`。配分の軸別修正と旧上限は適用しない。Q3の採用条件は保存H₄全DF10のsolver値に対するMSE上界が `Fraction("0.00159360010199040")**2` 以下であること。固定routeからtarget、law、lossを継承する。

## 固定入口と配備

公開された配置記録では、RomuTailの新規run用releaseは `romutail-fleet-efficiency-ffd406f4e`。controllerとworker輸送は `ffd406f4ed966f8ec712cba65a4d378cec43cac8`、科学Q3は `28d976b99cdf8329adcc1b79327e329e947ef774`。このreleaseを実装の基点にした。保存継続run `search-04c58930814ee4ae1d25` は `4d20f4b7a009c42d9a4918f8db17dc87503d060e` とQ3 adapter `349558b748a82f384a4589434a63ef9b5572032c` を固定している。この環境でのRomuTailへの直接照合は名前解決で停止した。
[配置記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ee7c7d0b2/docs/research/rpe_work_conserving_search_20261118/HANDOFF.md) · [保存runのrelease](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dfeb6b3b/docs/research/romutail_controller_runs/search-04c58930814ee4ae1d25/RELEASE.json)

配備担当は新しいreleaseを既存の `prepare_rpe_auto_profiles.py` と `prepare_romutail_controller_deployment.py` で別ディレクトリに用意する。controller、Q3 adapterの新SOURCEをホストprofilesとfleet manifestに結び付け、元の科学root、各route view、runtimeを使う。controllerの配備用生成は、新ルールを持つSOURCEに対して、全Q3 route profileのadapterが同じSOURCEであることを検査し、新しいSERVICEで倍率ルールを既定にする。既存の固定dry-checkは新adapterの配備時に実施する。旧runのResumeは保存済みrelease、SERVICE、ROUNDの旧配分規則を解決する。本作業で稼働中の `current` の切り替えは行っていない。

## local0からの指定

`REQUEST.json` の例を示す。省略した場合のM・倍率格子はこの例と同じ。

```json
{
  "scope": {
    "methods": ["FULL_PF", "TRUNCATED_PF", "PR_PAULI", "PR_DF"],
    "scaled_shots": {
      "M": [11, 12, 13],
      "grid": {"min_k": -12, "max_k": 16, "divisor": 4},
      "compare_paper": true
    }
  }
}
```

```powershell
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Preview -RequestFile .\REQUEST.json
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Start -RequestFile .\REQUEST.json -Instruction "<今回承認した対象と実行範囲>"
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Resume -RunId "<既存run ID>"
```

新releaseのPreviewでは `applied.search.scaled_shots` を確認できる。格子の指数は `k/divisor`。原論文の比較を有効にするとPRでは `exp(2/κ)` を近傍格子に丸めず直接整数化する。単位倍率 `s=1` は常に記録する。同一回路、同一Mで整数配分が一致する点はQ3を共有し、格子・単位・原論文の由来が残る。格子に対応配分がない比較点は別集計に載る。

## 結果の読み方

通常の `ROUND_RESULT.json` の `methods.<method>.scaled_shots` にM=12・指定M全体の評価済み最安、比較専用の最安と差額、旧自由配分の参照値を保存する。`by_configuration` には標準化した回路条件、κ、M、倍率の由来、X/Y整数配分、条件付き単価と費用、MSE上下界、保留理由、価格・信号の参照を保存する。探索マップ同期はこの部分を読み込み、同じ公式・幅・pまたはL_D・Mでκ別の倍率と費用を表示する。Cの正式価格採用は既存catalogの受入規則に従う。

費用による省略は同じ回路とMの格子内で、確認済み適合点の費用と倍率を更新できない配分に限定する。同額で小さい倍率は残す。別のκ・Mの最安でその組を打ち切らない。技術保留・未評価に更新の余地があるときは評価済み最安。価格の条件付き採用と、探索器に未登録の回路を含む物理設計空間での最適性は別に扱う。

通常同期の倍率・M欄では、指定範囲、停止・継続理由、省略・保留理由を表示する。同じ元configurationに結び付いたMを順に並べ、両端にQ3適合と条件付き費用がある場合にだけ差額（後のMの費用−前のMの費用）を記録する。両端の状態と元結果locatorを保持し、片端が入力待ちなどの場合は差額未確定とする。Mを変えると配分も生成し直すため、各Mの格子内の評価済み最安を比較する。別runや別回路を一つの曲線に接続しない。

従来の回路変数ごとの方向・探索境界の表示は、別のマップ実装 `cd52c799e` に収録済み。今回の倍率・M欄を定期表示に出す際は、Cの通常同期に該当reader・HTMLを取り込む。
[従来の方向記録と導入箇所](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cd52c799e/docs/research/h4_search_map_20261117/HANDOFF.md)

## 検収範囲と残る条件

- 通常Gatewayから通常schedule CLIとconcurrent executorに人工要求を渡し、4方式のM=11/12/13で12/13/14個のshotを使った。M=12の最安回路と、M=11を含めた最安回路が異なる例を確認した。M=13待ちの別候補があっても評価が進み、同じrunの再開で追加起動は0件。別releaseの旧runはResumeで旧版を選んだ。
- 固定Q3と同じ専用Python・route viewを使い、M=12の人工入力から検証済みprefixを取ったQ3と、同じM=11の独立人工入力の診断が厳密一致した。固定Z1＋T1のDPは12、13、14の人工count tableで参照実装と厳密一致した。通常同期の人工Git検収では、倍率結果の固定locatorがマップへ入り、既存catalogの点数を増やさなかった。
- M<12は元の固定configuration、信号packet、価格の対応済みprefixを使い、Mに固有の評価ID、配分、復号の終端と費用で評価する。元のM=12のMSEを短いMへ流用しない。保存H₄を使った新しいM=11の実Q3は未実施。
- M>12では不足ラウンドとX/Y価格を `HORIZON_INPUTS_MISSING` として列挙する。保存H₄のM=13にはm13の信号・価格を用意する既存13-round providerの拡張が必要。正式な固定Q3の `OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN` が残るため、M=13の正式評価をscience queueへ自動接続するところまでは未対応。固定ガードを維持し、実Q3へ進める条件は別途検証する。
  [固定Q3の検証域](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/710aa69b/src/trottertracks/rpe_deterministic_risk.py#L286)
- 実H₄での新規信号、compile、実Q3、実探索投入は各0件。次の実探索の前に、指定範囲と新規不足を§10.5で照合する。
