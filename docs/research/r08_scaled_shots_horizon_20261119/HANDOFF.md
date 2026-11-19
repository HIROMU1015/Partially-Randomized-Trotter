# R08の共通ショット倍率と最大ラウンド番号

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

## 検収範囲と残る条件

- 通常Gatewayから通常schedule CLIとconcurrent executorに人工要求を渡し、4方式のM=11/12/13で12/13/14個のshotを使った。M=12の最安回路と、M=11を含めた最安回路が異なる例を確認した。M=13待ちの別候補があっても評価が進み、同じrunの再開で追加起動は0件。別releaseの旧runはResumeで旧版を選んだ。
- 固定Q3と同じ専用Python・route viewを使い、M=12の人工入力から検証済みprefixを取ったQ3と、同じM=11の独立人工入力の診断が厳密一致した。固定Z1＋T1のDPは12、13、14の人工count tableで参照実装と厳密一致した。通常同期の人工Git検収では、倍率結果の固定locatorがマップへ入り、既存catalogの点数を増やさなかった。
- M<12は元の固定configuration、信号packet、価格の対応済みprefixを使い、Mに固有の評価ID、配分、復号の終端と費用で評価する。元のM=12のMSEを短いMへ流用しない。保存H₄を使った新しいM=11の実Q3は未実施。
- M>12では不足ラウンドとX/Y価格を `HORIZON_INPUTS_MISSING` として列挙する。保存H₄のM=13にはm13の信号・価格を用意する既存13-round providerの拡張が必要。正式な固定Q3の `OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN` が残るため、M=13の正式評価をscience queueへ自動接続するところまでは未対応。固定ガードを維持し、実Q3へ進める条件は別途検証する。
  [固定Q3の検証域](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/710aa69b/src/trottertracks/rpe_deterministic_risk.py#L286)
- 固定SHAから人工profileを組み立て、通常Previewの既定値と旧adapter profileの拒否を確認した。旧テスト `test_status_reports_completed_exit_and_preserves_scientific_stop_reason` は基点と今回の両方で `COMPLETE` の代わりに `RECONCILIATION_REQUIRED` を返した。
- 実H₄での新規信号、compile、実Q3、実探索投入は各0件。次の実探索の前に、指定範囲と新規不足を§10.5で照合する。
