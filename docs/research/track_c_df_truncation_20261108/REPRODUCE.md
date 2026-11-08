# SOURCE・rawの復元と再現条件

科学計算・再解析は`STUDY_STOP.json`でmandatory STOP済み。以下は再現方法の説明であり、自動再開する指示ではない。原runやそのSTOPは編集せず、再実行には別の出力・cacheを使用する。hash確認・archive復元は数値再計算を必要としない。

## 公開ファイル

- `artifacts/track_c_df_truncation/2026-11-08/ordering/{RESULTS.json,COMPARISON.csv,ROUNDS.csv,CANDIDATES.csv,ERROR_BUDGET.csv,TRUNCATION_MECHANISMS.csv,VALIDATION.json}`：最終比較。ルートの同名summaryは順序audit前の保存履歴。
- `ordering/{POLICY.json,FINAL.json,QUALIFIED_COMPLETE.json,SHORTLIST_COMPLETENESS_CHECK.json}`：凍結policy、独立結果、全48点、policy不変の確認。SOURCEの差と公開snapshot SHAを混同しない。
- `H{2,4,6}/INPUT.json`とraw内`INPUT_ARRAYS.npz`：public replay用の保存H/state/B/sp・exact係数・gate IR。private原本pickleは不要。
- H2/H6の`ORIGINAL_INPUT.npz`、`SAVED_FULL_DF.npz`、`ORIGINAL_INPUT_IDENTITY.json`：5385c6の公開保存入力のbyte-identical copy。元記録の旧run制限は保持。
- `SOURCE.json`、`ENVIRONMENT.json`、`SOURCE_HISTORY.bundle`：loaded科学SOURCE、依存基点、環境。bundle tipはa4e32484254a090eab6d221776a50a1cfd9f0f17、必要な既存基点はc5a3c102199fe97f4d5c21837c059ae2684f5ef2。
- `RAW_INDEX.json`と`ARCHIVE_PARTS.json`：全memberと分割archiveのbytes/SHA256。rawには全候補、scan/価格、interval証拠、旧・修正後の結果、独立検査、失敗logを保全。

## rawのlossless復元

同じdirectoryの`raw.tar.gz.part000`から番号順に連結すると標準tar.gzになる。partごとのhashと全体hashをARCHIVE_PARTS.jsonに照合してから、空のscratch directoryへ展開する。公開checkoutや旧rawを展開先に指定しない。

```bash
RAW_BASE=artifacts/track_c_df_truncation/2026-11-08
RESTORE_DIR=$(mktemp -d /tmp/track-c-df-raw.XXXXXX)
cat "$RAW_BASE"/raw.tar.gz.part??? | tar -xz -C "$RESTORE_DIR"
```

tar memberはこの研究rootからの相対path。RAW_INDEX.jsonの各entryについてbytes/SHA256を照合すれば、数値再計算なしに全rawの内容一致を確認できる。梱包はfile bytesを保ち、tar headerのmtime/uid/gidを標準化している。元の科学fileは変更しない。

## 歴代SOURCE

公開commitのtreeと、実行時の科学SOURCE commitは別の識別子である。bundleは科学実行時の各commitをそのまま保全する。基点c5a3を持つrepositoryで確認できる。

```bash
git bundle verify artifacts/track_c_df_truncation/2026-11-08/SOURCE_HISTORY.bundle
git bundle list-heads artifacts/track_c_df_truncation/2026-11-08/SOURCE_HISTORY.bundle
```

別の再現checkoutを作る場合、bundleからHEADを新しいlocal refへfetchし、SOURCE.jsonのstage_loaded_sourcesに対応するcommitを使う。過去rawの初期source欄はprocess中に観測したHEADであり、実load時のcommitはSOURCE表で明示した。初期記録を後から改竄していない。

## 環境と入力replay

主な版はPython3.12.3、numpy1.26.4、scipy1.14.1、mpmath1.3.0、Qiskit1.3.0、OpenFermion1.6.1、openfermionpyscf0.5、PySCF2.7.0、pytest8.4.2。ENVIRONMENT.jsonを参照する。環境はproject専用venvに構成する。共有library・OS設定の変更は不要。`PYTHONPATH=src`、各BLAS1threadでworkerをaffinity/task数へ展開した。

H2/H6には保存h/g/constantとfull J/Uを使用し、新規SCF/DF fitなし。H4にはpublic replay。公開INPUT_ARRAYSには元のHとstateがあり、Hamiltonian/groundを再生成する必要はない。`replay_model`がH/state/B/sp/matrix/diagonal/coefficients/gate parametersについて元cacheと完全一致したことを`INPUT_REPLAY_CHECK.json`へ保存した。

再実行する独立scopeでは、元公開rootのH2/H4/H6へ**欠けているINPUT_ARRAYS.npzだけ**を復元し、fresh cache/fresh outputを設定する。保存metadata/原本を上書きしない。

```bash
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TRACK_C_DF_CACHE=/tmp/track-c-df-new-cache
export TRACK_C_DF_TRUNCATION_OUTPUT=/tmp/track-c-df-new-output
# 両pathが未使用であることを確認してから、別の許可scopeでのみ実行する。
python -m trottertracks.track_c_df_truncation inputs-replay
```

`inputs-replay`は新SCF/fit/ground solveを行わず、公開数値/IRからcacheを作る。既存cacheがある場合は拒否する。元targetはraw HのHermitian部分であり、元配列自体は保持する。

## 実行経路の対応

新しい科学runを開始しない読取確認では、REPORT、SOURCE、RAW_INDEX、FINAL_CHECKと保存policy/資格/MC trajectoryを用いる。人工テストは`tests/test_track_c_df_truncation.py`（8 PASS）。人工PASSと分子の資源性能を区別する。

再実行のstageは次の順で接続される。これは必要な依存順の説明であり、STOP済みnamespaceを再開しない。

1. 入力replay → `pilot` → `certificates` → `scan` → `reprice` → `qualify-repriced` → `freeze` → `independent`。
2. PRの`pr-isometries`、`pr-bch` → `pr-probes` → `pr-qualify` → `pr-training` → `pr-freeze` → `pr-holdout` → `pr-finish`。
3. 有限の順序audit：`python -m trottertracks.track_c_df_order_audit`の`pilot` → `qualify` → `train` → `freeze` → `holdout` → `finish`。
4. 保存結果の読取export：`python -m trottertracks.track_c_df_truncation_report --ordering`。

当時の途中修正はSOURCE_HISTORYとREPORTの技術修正節で辿る。最終コードのfresh runは過去のpreliminary記録の作成を再現する必要はない。保存policyを検証する再現と、候補探索を新たにやり直す実験を区別する。platform/BLAS差による候補の微差をbitwise一致の前提にせず、科学的なinterval/shot/decoder条件は再検査する。

H4の再利用Full/PRは基点c5a3の`artifacts/track_c_closed_p3_joint_opt/2026-11-08/extended/FINAL.json`と同研究REPORT/rawから辿る。H2/H6の旧13round・別compiler価格、Pauli-JW B0/B0I価格はこの再現比較に入れない。

## 誤差と価格の検査単位

選択結果は36軸αのexact和≤.05、全18roundのrho/β>0、十分N、alias、mean branch error<1.5936e-4 Haを持つ。価格には同じRZ優先native実装のCX/size/depth、scalar/X/Yを含む。PRのYはweightedであり、range normalizationを無重みHadamard平均へ置換しない。価格MCはunweighted期待回路価格で、shared-cell covarianceを残す。

初期training/holdoutは追加auditのselection dataへ再分類した。最終の`ordering_holdout`は29cell・3,712 trajectoryで別seed namespace、policy凍結後のデータ。旧holdoutを最終独立評価として重複利用していない。H4は旧c5a3の凍結policyとその独立holdoutを再利用する。

科学的な追加実行・研究拡張は行わず、mandatory STOPを保持する。
