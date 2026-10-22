# GAP-001 旧BLISSとUWCの探索結果

2026年10月23日 JST。公開用に内部パス・host/user識別情報・private cache情報を除いた報告。環境IDは `ENV-LOCAL-01`。

**判定 `NOT_FOUND_IN_SEARCH_SCOPE`。** 指定の原JSON、比較raw、theta sweep、run log/manifestは今回の探索範囲で見つからなかった。関連SOURCEとproseは存在するが、原データの代用にしていない。

| 対象 | 判定 |
|---|---|
| `H2_H6_2nd_grouped_uwc_alpha_bliss_quadratic_theta_0p01_gpu.json` | `NOT_FOUND_IN_SEARCH_SCOPE` |
| `H3_bliss_sector_scaling_diagnostics.json` | `NOT_FOUND_IN_SEARCH_SCOPE` |
| `H2_H6_2nd_grouped_uwc_alpha_simple_shift_gpu.json` | `NOT_FOUND_IN_SEARCH_SCOPE` |
| `grouped_uwc_pf_qpe`配下のtheta sweep JSON/CSV、比較結果、元run log/manifest | `NOT_FOUND_IN_SEARCH_SCOPE` |
| 対応Hamiltonian/sector/grouping入力を元runに結び付ける記録 | `NOT_FOUND_IN_SEARCH_SCOPE` |

## 同一性と実行条件

[固定UWC資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/231d17e393f573fd29c659eaf1fd9bae2a68d8f9/notes/uwc_current_implementation_and_results.md)はraw欠落を明記している。確認した固定資料には、上記原JSONの独立した期待SHA256は見つからなかった。原JSONがないためbytes/SHA256を計測できず、原本identityは未確認。

資料に報告された条件は、H2–H6、STO-3G、距離1.0 Å、`bliss`、quadratic、`theta=0.01`、`2nd`、`target_error=1e-2`、`rz_layers`、GPU IDs 0,1、`baseline_alpha_source=artifact`。H3については旧`N_target=2`と修正版`N_target=3`を混同しない。診断time gridは0.750/0.752/0.754/0.756、theta表は0/0.001/0.005/0.01/0.05/0.1。これらはprose由来の探索条件であり、実runの一次記録による検証ではない。

実runの生成commit、正確なcommand、backend/依存版、実行日時、変換前後H hash、入力state・grouping・fitの原記録は未発見。保存済みの別PF係数や現行SOURCEを原runの確定入力とみなしていない。

## 発見した関連SOURCE

既存manifestがUWC実装追加とtheta sweep追加に対応付ける歴史的commitから、次のSOURCE snapshotを読み取った。Git blob IDとの一致とbytes/SHA256は確認済みだが、実runがこのsnapshotで走ったことは未確認。`requirements`も宣言上の依存であり、当時のruntime inventoryではない。

| SOURCE path | 保存commit | bytes | SHA256 |
|---|---|---:|---|
| `src/trotterlib/uwc.py` | `0ce0389c5d907995558f4a3b1b8c593fd6494d77` | 24903 | `24107639fd4768eae39e1717b637c9c321874bd6685e7be5904cd7ec01cdf880` |
| `src/trotterlib/grouped_uwc_comparison.py` | `0ce0389c5d907995558f4a3b1b8c593fd6494d77` | 42669 | `42d363dcf78c56113f37496ddc8bc37912d04f89b2c73c75f689b5edf208062f` |
| `scripts/run_grouped_uwc_pf_qpe.py` | `0ce0389c5d907995558f4a3b1b8c593fd6494d77` | 9346 | `697bb4054b3433744a0d97c253507b7a607d3fa11ee604560359ecc01d411868` |
| `scripts/run_uwc_grouped_alpha_h2_h6.py` | `0ce0389c5d907995558f4a3b1b8c593fd6494d77` | 14992 | `edfc3e2d580bcd526106ea99b6f7096988e997a77d8f10e55c4280fc89a3b37b` |
| `src/trotterlib/config.py` | `0ce0389c5d907995558f4a3b1b8c593fd6494d77` | 12783 | `15e12517f82ed3e9d56eac8a272693ced87226a8ae5e97351e1c2c9b4e3415ca` |
| `src/trotterlib/io_cache.py` | `0ce0389c5d907995558f4a3b1b8c593fd6494d77` | 1580 | `e7504b620eabf71af5f88992113a7306ed0506d5e7b99ca705aa43afb1d00ca1` |
| `requirements.txt` | `0ce0389c5d907995558f4a3b1b8c593fd6494d77` | 297 | `d9f2e0af70bf58be27c6e46b3c28bfece031591de9988420f73d08cecfa2c111` |
| `requirements-gpu.txt` | `0ce0389c5d907995558f4a3b1b8c593fd6494d77` | 43 | `cbc27b46c5766d24e10de2936a27932bd222a1c64acca52b7967c5ff35ae676a` |
| `src/trotterlib/grouped_uwc_theta_sweep.py` | `c774c671c0a4eaf91045b7844bdcc9be41d2f453` | 17178 | `7acbb11c5343f119582cf88b302f7bd12d7a6040e3823aa77f12c24a892cbf64` |
| `scripts/run_grouped_uwc_theta_sweep.py` | `c774c671c0a4eaf91045b7844bdcc9be41d2f453` | 8091 | `42318d6b505df258186ea272cfd70aa23cc8d6495cf9249b480a1f7d578e1af2` |

## 探索範囲と引継ぎ

範囲・除外・未調査archiveは[探索総報告](SEARCH_REPORT.md)と[SEARCH_SCOPE.json](SEARCH_SCOPE.json)に記載。登録worktree255件の既知相対保存先probeで`artifacts/grouped_uwc_pf_qpe`は0件。追加のworktree領域全体のファイル名検索でも指定rawは未発見。対象pathの既存ローカルGit履歴にもcommitted rawは見つからなかった。これは別環境やarchive内部の不存在を意味しない。

引継ぎGAP-IDは **GAP-001**。次の担当はユーザーが別途選んだ環境のCodex。今回のCodexは報告後mandatory STOP。原データの回収・公開・科学実行は開始しない。
