# 448-shot科学SOURCE候補と実行前preflight

2026-11-07 / SERVER1 GPUA100-gleap-01。**実装仕様は採択済み。実H4一campaignの実行許可はまだない。** 本文を含む公開commitが新SOURCE候補であり、正確なSHAはGitHubのcommit metadataと公開照合後の最終報告に記す。commit内へ自身のSHAを書き込む循環参照は設けない。将来のSPEC/EXECUTION SOURCEは実際の`git rev-parse HEAD`を保存し、remote branch SHAと一致させる。

## 固定仕様

保存linear H4／1.00 Å／STO-3G／legacy DF12／8 system qubits／全4電子sector、L_D3、m0、q1、δ=0.02、r1、K2、coefficient_atol=0。N1 actual finite-grid Q/selected full、N2保証ground、認証済みexact Fraction engine、理想gate、X/Y。対象target・qubit ordering・phase・X/Y bit規約を変更しない。

| 項目 | 固定値 |
|---|---|
| Campaign | `SCV-SIGNED-H4-H1-PAIRED-20261107-v1` |
| Purpose | `SCV_SIGNED_SUPPORT_H1_FRESH_PAIRED_DIAGNOSTIC` |
| Train | X h1=64、Y h1=64、両取得後X/Y各一fit |
| Evaluation | 各軸(8,128,8,8,8)、全5正層、同bitのB1-OL/SCV paired |
| 合計 | 448 physical shots、1 campaign、retry0、追加0 |
| Seed/stream | 20261110、PCG64/SeedSequence、12 domain/24 children、token辞書順、path/bit分離 |
| Range | exact D1=2+2abs(λ1)≤6、inactive4層λ0 |
| 集計 | 各軸Full＋trial ID固定half A/B、計6、追加bit/価格なし |
| Price | Qiskit1.3.0、basis=(rz,sx,x,cx)、opt2、seed_transpiler17、backend/coupling/layout/routing=None、selected full no-prep |
| 上限 | engine/Born/threshold/price各448、path448、build896、compile448、word29120、apply427840（各955以下）、OLS2、paired6、restore/decode/stream各1 |

完全設定とspawn keyは[SETTINGS.json](SETTINGS.json)、新SOURCE/inputの結合は[SOURCE.json](SOURCE.json)、[INPUT_BINDING.json](INPUT_BINDING.json)、[PROFILE_LOCK.json](PROFILE_LOCK.json)を参照する。

## Input binding

- N1 `744e1a09661ebba6841fe5faf456f88c41c47780`
- N2 `5e9bd08068738cdf1070bdc43bee8194b113ae33`
- Engine certification `fdf00cee3c87d804845497afc89ac6a71f3b3f59`
- Metadata SOURCE `71778dae6954ed9a7a6821062a2a6e6be17672c4`、結果 `b4512675ae1d650858256547d359210361d77f68`
- Signed kernel `b23b3c059dd1fdcc86178659311983048dea2257`、採択仕様 `138d071805430ef1e556655392e4d9f572feba40`
- LAW SHA256 `b77980d198cc9edc620daec9bf87df02b66e96575fd85327aed0793f3e86ecef`
- Catalog/PREPARATION SHA256 `ca897db7be52ad30dcfab0512d7530188fc13a9d8ca0c78430809492d01f738b`
- Saved result SHA256 `c5895f7696711c92bf87e591a3c2264cd7672d9af70d13027db790950e305ba0`
- Metadata identity `c5e0c5c26b926eb4a73d1808a1c61222dc45934248c30d8aea4f8fc32c44b7e2`
- Control identity `5e252ff9fa5682850683ced97d007b59fecbf6d7617c561adb4a34d4cc7c7f89`

今回は既存inspect_certified_bundleを使うbyte/identity照合のみPASS。H4 SignedControl通常構築・saved復元・mean有限和・state decode・認証再実行は0。保存Σ1>0/ELIGIBLEは既存結果の継承であり、相関/改善の新評価ではない。

## 環境とSOURCE

Python3.12.3、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、mpmath1.3.0を含むnative/dependency/thread profileを固定。詳細はPROFILE_LOCK。共有venvへinstall/変更をしない。private cache pathだけを用い、既存verify_runtimeは旧認証native profileと一致PASS。SOURCE hash979ファイルで推移的library Pythonも静的に固定する。新inputや実componentをprofile収集へ渡さない。

旧SOURCE・旧PROFILE_LOCKは上書きしない。認証済み数値kernelのfloat/GPU/O1/O2版への置換経路はない。Saved control snapshotのscience_ready=falseは旧metadata段階の表示として保存し、今回の技術準備完了と区別する。

## 別途実行許可後に限るpreflight

1. この公開SOURCE commitとbranch SHA、clean tracked files、想定以外のuntrackedなし、SOURCE/dependency/native/compiler/settings hash一致。
2. N1/N2/engine証明とmetadata9ファイルの固定commit/全bytesを再照合。不一致時STOP、別SOURCEへの自動追随なし。
3. output `artifacts/scv_signed_support_fresh_pilot/2026-11-07/campaign_run01/`、同親 `manager_run01/`、cacheをfreshとする。旧data/coef/word/cacheの持込み、run02/resumeなし。
4. 既存own crontabを保護。qualified manager route、kind/purpose/seed/Source/profile/input、consume-once、Restart=no、cron cgroupを確認。持続条件不成立なら実science前STOP。
5. CPU二窓idle/affinityとmemory/cgroup headroom・pressureを確認。初期Wは実資源と有用task数に基づく。parent一回decodeをfork継承、worker再decode/再生成なし。path開始前にもmemory圧力・headroomを検査する。Fraction成長の将来peakや無期限の資源可用性は未保証。
6. saved controlを一回typed復元し、保存値・scope・identityを検査。新moment有限和をしない。bundle/proof一回decode/attach後、saved samplerとの完全一致を確認。
7. 新stream book一回、mappingを先行保存。実測wordは許可されたphaseでのみ消費。両train→fit/guard→凍結→evalの順序、有限capを維持する。

現在、上記のstatic bytes/native項は確認済み。実H4 control復元・state/engine接続・資源admission・新cache/outputの消費・manager cgroup・実測学習rank/guardは**科学実行時の未実施preflight**。今回の人工PASSで代用しない。

## STOPと次回承認

identity/SOURCE/profile/law/control/scope/stream/trial/word/価格/保存/finite capの不整合、rank/nonfinite/幅6超過、worker異常、資源・非干渉不成立でSTOP。STOP後に新path/callを開始せず、in-flightとbit/価格/UNKNOWN tail・全学習費を保全する。clip/refit/seed変更・科学retryなし。完了でもmandatory STOP、保証付き本検証や追加campaignへ進まない。

次回必要なのは、公開照合済みのこのSOURCEで**同じ固定仕様の448-shot一campaign（保存control一回復元、bundle decode一回、新stream、fresh train/paired評価、必要full build/compile/価格、全保存・公開・STOP）を実行する明示許可**。新grant/approval certificateは要求しない。今回はその許可を推定せず、実submit/cron trigger/本番managerを有効化していない。
