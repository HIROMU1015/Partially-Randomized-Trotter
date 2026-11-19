# H₆・H₈のRPE最適化に向けた保存入力の照合

H₆の初回候補は、**Full PF／M5／DF19全fragment／要求幅1.6／h=0.1／M=12**。m0〜m12の保存raw信号13ラウンドを元記録まで照合した。compiled RZの直接価格の証拠はm3〜m6のX/Y計8軸。m0〜m2の6軸とm7〜m12の12軸は、調査した同一候補の価格表で欠測している。通常探索への登録には、H₆固有のcanonical configuration、信号packet、Q3 route、26軸価格が残る。

H₈はFull PF／M5／DF28の (T=1.6, q_{PF}=1) で、保存信号とX/Y直接価格が一致する候補を確定した。新しい (h)・(M)・精度閾値の採択は保留した。

[機械可読coverageと固定locator](COVERAGE.json) · [次の有限実行と接続手順](NEXT_VALIDATION.md)

## H₆

直線1.00 Å・STO-3G・tol-only DF19、12 system qubit、粒子数固定sector 400。保存H・state・snapshot・rank policyは各元結果で一致した。基準は保存全DF19 solver値 `-3.2360662799087647 Ha`。候補のh、要求幅、各ラウンドの実幅、元候補IDを区別した。旧保存資料のラウンド別条件付きshot数は、今回のRPE終端MSE判定に転用しない。 [保存coverage](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f4c0bdabdf2afdc5bc4f20573bb10677917f4c1a/docs/research/h6_df19_rpe_m8_12_results_20261110/run01/COVERAGE.json)

| ラウンド | (q_{PF}) | 実幅 | 信号 | compiled RZ価格の証拠 |
| --- | --- | --- | --- | --- |
| m0〜m2 | 1, 1, 1 | 0.1, 0.2, 0.4 | 保存3 | X/Yの6軸が欠測。どれもq=1だが実幅が異なる |
| m3 | 1 | 0.8 | 保存1 | X/Yの2軸。保存Full PFの固有receiptに対応 |
| m4〜m6 | 1, 2, 4 | 1.6 | 保存3 | X/Yの6軸。q=1/2は凍結モデルの較正、q=4は独立検証 |
| m7〜m12 | 8, 16, 32, 64, 128, 256 | 1.6 | 保存6 | X/Yの12軸が欠測。保存モデルの検証範囲外 |

旧coverage、後続のFull PF短時間棚卸し、H₆価格providerを同じ物理候補IDで照合した。m3の価格は再利用元の候補receipt、m4は探索carrier、m5とm6は保存試行の軸別フィールドまで特定した。新しいregistryで使うときは元のcompiler・control・lowering・測定・P0と照合し、同じ契約の価格だけを受け入れる。ここでいう8軸は保存証拠の所在であり、今回のQ3登録済み軸数は0。 [全13ラウンド価格coverage](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cb61ef4633467446122954ed6cc43462e0306a9/docs/research/h6_provider_integration_nc_coverage_20261111/REPORT.md) · [Full PF棚卸し](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/92ff17dededea6e1ee273b2eee6a08e292ee9e5b/docs/research/d_full_pf_saved_evidence_coverage_20261113/REPORT.md)

固定幅1.6のH₆ affineはq=4で一致したが、長い反復のcompiler境界を確認していない。q=1が共通のm0〜m3へ適用する根拠もない。保存q=4のFull PF取得は約44分だった。長時間の直接取得は、保存QPYの再利用可否と費用への寄与を確認してから決める。 [独立q検証と伸長の限界](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73dcbe23678bfc50d5654f09e600d4cfbe75e697/docs/research/h6_df19_price_v2_20261110/REPORT.md)

## H₈

直線1.00 Å・STO-3G・tol-only DF28、16 system qubit、sector 4900。仮にH₆と同じ `h=0.1、要求幅1.6、M=12` を照合座標として使うと、m4が保存Full28/M5の `c46590722df98f6ebffd` に一致する。Stage Aのraw信号とStage Bの決定論1回路のX/Y価格が同じcandidate IDで、保存bindingも一致した。checked carrierの残り12ラウンドは未結合。この仮scheduleを採用する前に、保存全DF28 solver基準、初期の位相枝、精度閾値を別に確認する。 [Stage A/Bと保存原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7fb1d125e05238a7d932fc89b8dab6ac6a7f8c84/docs/research/h8_long_time_execution_20261103/EXPLORATION_RESULTS_20261104.md)

Stage Cまで確認したところ、H₈には同じ (T=1.6) のPR-DF M5／prefix8／(R=32,K=4) の独立有限cohortもある。後続PR-DFの候補に使えるが、全roundの価格・law期待値や新Q3の適合は未取得。旧 `ε=.0005` は単時刻で使った条件なので、エネルギーMSE閾値は別に固定する。 [H₈ Stage C](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eb07156a37b1938ed3aed95c13abbb545466c573/docs/research/h8_stage_c_execution_20261105/REPORT.md)

## 照合範囲

保存rawを新しく数値評価せず、SOURCE・候補・schedule・axis・signal locatorを照合した。H₆価格は既存の固定価格表と後続棚卸し、H₈のm4は長時間探索の結合済みcarrierを対象とした。現在のH₄探索branchと、公開されている関連H₆・H₈のbranchを確認した。ローカルプロセス照合で今回の構成のQ3は見つからず、H₈旧tmuxの表示は完了済みcarrierの存在を覆す証拠と扱っていない。非公開の全ホスト台帳との照合は実行前の差分確認に残す。

固定locatorを別の読取で開き、13連続ラウンド、T=q×実幅、軸別locator、H₆の13信号・8価格軸、H₈仮scheduleの1信号・2価格軸を再確認した。既存の倍率・M探索の人工回帰は6件PASS。H₆/H₈の通常CLI受入は今後の接続検収に含める。

新規信号、reference、build、compile、価格の予測・再計数、Q3、総費用計算は各0件。
