# Validation status

## 結論

**監査基準 commit [`cf285c0`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/cf285c0ac1e3d587df4a8eb6bee2279a12ced462) の内容だけでは、現在の DF screening / UWC 検証結果を外部から再現・追跡できません。**

ここで「外部から再現可能」とは、clean checkout から、commit 済みの入力と手順を使って結果を再生成し、その結果が公開済みの数値と一致することを確認できる状態を指します。本書は既存 artifact の棚卸しであり、新たな科学計算を実行した結果ではありません。

> **DO NOT USE:** 現在 commit されている DF screening JSON を、修正済みの結果または最終結果として引用しないでください。ファイル内の算術は整合していますが、その Cgs 入力は後の commit で基底状態の不整合を理由に削除され、screening は再生成されていません。

## ステータス一覧

| 対象 | commit `cf285c0` にある証拠 | 判定 | 読み方 |
|---|---|---|---|
| 旧来の高次 Trotter 評価 | 出力付き `abe_trotter_project.ipynb` と、`artifacts/trotter_expo_coeff_gr{,_original}/` 内の係数 pickle 計 540 個 | **historical** | README が説明する旧来の高次 Trotter 解析の成果。現在の DF screening / UWC の検証証拠ではない |
| DF reduced screening | `epsilon_total=1e-4` の JSON 1件。635候補、12分子の best を収録 | **DO NOT USE / stale** | 保存値の加算と best 選択は内部整合するが、元の Cgs 表が削除済みで再生成不能。protocol 上も shortlist 前の近似 screening |
| DF 最終評価 | protocol と実装 | **incomplete** | shortlist の explicit-`L_D` Cgs 再 fit、H14 `8th(Morales)`、`4th(new_2)` が未完了 |
| UWC | 実装説明と H2--H6 等の数値表を含む Markdown | **reported only** | 表が参照する machine-readable JSON は commit されておらず、表から元 run を追跡できない |
| テスト | 4ファイルに `test_*` 関数定義が28件。UWC note に過去の `26 passed` の記録 | **current result unknown** | `cf285c0` に対するテスト実行結果ではない。この変更で追加する manifest 構造検査も科学計算・全 test suite は実行しない |

## 証拠と監査結果

### 1. DF screening

対象 artifact:

- [`artifacts/partial_randomized_pf/screening_results/df_screening_cost_minimization_eps_1.000e-04.json`](artifacts/partial_randomized_pf/screening_results/df_screening_cost_minimization_eps_1.000e-04.json)
- [`Partial Randomized Study Protocol.md`](Partial%20Randomized%20Study%20Protocol.md)
- [`artifacts/partial_randomized_pf/README.md`](artifacts/partial_randomized_pf/README.md)

JSON 自体について確認できる範囲は次のとおりです。

- `candidates` は635件で、1件は `(molecule, PF, L_D)` の組です。
- `best_by_molecule` は H3--H14 の12件です。
- 全635候補で、保存値の `g_total` は `g_det + g_rand` と一致します（最大絶対差 0）。
- 12件の `best_by_molecule` は、それぞれ同じ molecule の候補中で最小の `g_total` と一致します。

これは **JSON 内部の算術と選択処理だけ** の確認です。入力データ、Cgs fit、物理モデル、または結果の科学的妥当性を検証したことにはなりません。

再現性を失っている直接の理由は次のとおりです。

1. JSON の `cgs_table` は `/home/AbeHiromu/Project/.../df_cgs_cost_table.json` という生成環境の絶対パスを指します。
2. commit [`98f960c` (`基底状態ずれてたので削除`)](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/98f960c2dd09fc1ae6b8b5c802dc5ce84fc61604) は、集約 Cgs 表、split 表、index の計37ファイルを削除しています。
3. その後も上記 screening JSON は残っていますが、削除理由を反映した正しい Cgs 入力から再生成された artifact はありません。

また protocol は、この計算を候補を絞るための近似と定義しています。screening では anchor の `C_gs,D(p,L_anchor)` を各 `L_D` に使い回し、**最終評価では shortlist の各 `(p, L_D)` で Cgs を再 fit して `G_total` を再計算する必要があります**。同じ protocol には、次も未完了と記録されています。

- H14 `8th(Morales)` の anchor Cgs
- H3--H14 `4th(new_2)` の anchor Cgs 計算、cost table への merge、再 screening
- shortlist に対する explicit-`L_D` Cgs の再 fit

したがって、入力問題がなかったとしても現在の JSON は最終結果ではありません。

### 2. UWC

[`notes/uwc_current_implementation_and_results.md`](notes/uwc_current_implementation_and_results.md) には、H2--H6 grouped UWC、H3 time-grid 診断、theta sweep、simple shift の条件と数値表があります。一方、同文書が参照する次の出力を含む `artifacts/grouped_uwc_pf_qpe/` は commit `cf285c0` に存在せず、`.gitignore` でディレクトリ全体が除外されています。

- `H2_H6_2nd_grouped_uwc_alpha_bliss_quadratic_theta_0p01_gpu.json`
- `H3_bliss_sector_scaling_diagnostics.json`
- `H2_H6_2nd_grouped_uwc_alpha_simple_shift_gpu.json`
- theta sweep 表の元になった run 出力

したがって Markdown の表は「報告された数値」として読めますが、repository 内の canonical raw/summary artifact と照合することはできません。なお文書自身の結論も、現在の simple BLISS quadratic shift では grouped PF+QPE cost がほぼ低下していない、という限定的なものです。

### 3. テストと CI

commit `cf285c0` の `tests/` には、静的に数えた `test_*` 関数定義が28件あります。

- `tests/test_df_hamiltonian.py`: 5件
- `tests/test_df_partial_randomized_pf.py`: 9件
- `tests/test_grouped_uwc_comparison.py`: 7件
- `tests/test_uwc_preprocessor.py`: 7件

UWC note が保存している実行記録は `.venv/bin/python -m pytest -q` の `26 passed` です。これは後から追加されたテストを含む現在の suite に対する結果ではなく、実行 commit、依存環境、完全なログも記録されていません。監査基準 commit `cf285c0` には `.github/workflows/` もありませんでした。この変更では manifest と記載パスの構造検査だけを追加しており、科学計算または全 test suite の CI ではありません。このため、`cf285c0` の28定義が pass するとは本監査から主張できません。

## 再現を妨げているもの

- 修正済みの DF Cgs 集約表・split 表・index がない。
- stale screening JSON に入力 hash、生成元 commit、実行環境、実行 command がない。
- DF screening の修正後再実行と shortlist の explicit-`L_D` 再 fit がない。
- protocol に記載された H14 `8th(Morales)` と `4th(new_2)` が未完了。
- UWC の Markdown 表に対応する machine-readable run artifact がない。
- UWC artifact の保存先が `.gitignore` され、レビュー可能な canonical summary の例外設定がない。
- 現在の HEAD を対象とする自動テスト結果がない。

## 「検証完了」とするための条件

以下をすべて満たした時点で、DF / UWC の結果を repository から外部検証可能と扱います。

1. **DF 入力を修正して固定する。** ground-state のずれを修正した Cgs を再計算し、集約表、全 split 表、index を同時生成する。各表に molecule、PF、`L_D`、入力 Hamiltonian hash、生成元 commit、生成 command を記録し、相互の件数と hash を検査する。
2. **未完了の DF ケースを埋める。** H3--H14 `4th(new_2)` と H14 `8th(Morales)` の必要な anchor Cgs を生成し、同じ canonical table に merge する。失敗または除外する場合は、対象、理由、結果への影響を明記する。
3. **screening を再生成する。** 修正後の canonical table だけを入力として `epsilon_total=1e-4` screening を実行する。結果には相対的な入力 path、全入力の content hash、生成元 commit、command、依存環境、candidate 件数を保存する。`g_total = g_det + g_rand` と molecule ごとの best 選択を自動検査し、旧 JSON を stale として置換または明確に隔離する。
4. **最終 DF 評価を実行する。** screening の shortlist と選定規則を保存し、各 `(PF, L_D)` で anchor ではない explicit-`L_D` Cgs を再 fit して `G_total` を再計算する。最終表から各 fit の machine-readable artifact と入力 hash へ追跡できるようにする。
5. **UWC の根拠データを公開する。** Markdown に載せる全表について canonical JSON/CSV を commit し、条件、baseline、seed（使用時）、backend、入力 hash、生成元 commit、command を保存する。Markdown の値が artifact から自動生成または自動照合されるようにし、必要な summary だけを `.gitignore` の例外にする。
6. **clean checkout で検証する。** 固定した依存環境と文書化した command で、小規模な end-to-end 再生成および全 test suite を CI から実行する。結果 artifact を作った commit に対する成功 check を GitHub 上に残し、比較 tolerance と期待値を test または検証 script に固定する。

上記が完了するまでは、旧来の高次 Trotter artifact、DF screening、UWC 表を互いに独立した進捗資料として扱い、現在の partial-randomized DF/UWC の完成済み検証結果として一括して引用しないでください。
