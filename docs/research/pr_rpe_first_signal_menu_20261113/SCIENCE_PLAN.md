# 承認待ち：PR-LD1/LD2・全13round signal有限実行

2026-11-13 JST。**未実行**。[MENU.json](MENU.json)と[二構成・根拠](REPORT.md)を正本の承認対象とする。今回の依頼はmenu具体化までであり、新科学実行の許可と解釈しない。

## 固定入力・計算SOURCE

- 数値provider：`fefd548ecd56f60d72031b3d385f6f5e6ee95d06:src/trottertracks/pr_rpe_signal_provider.py`。
- B identity/fixed-A metadata adapter：`437ba4acdb82708b6fc98e3cd119a2a734f7cf64`。A fixed-grid/commonκは元510fbacdの方式を保つ。
- 保存input/tail INDEX：`7a8bc7f132e166390c943e6b32c92cc6abe1baf4`、H4 context SHA `8396605fa0c5ee34f29b436ee16e0f731cfaa6cd94f3304033106a85e9f5d52f`、prefixes1/2。
- 保存13reference pair：`bba150834ca30bdb186fbfd58c55a8d51177b855:docs/research/h4_pr_physical_error_contract_20261113/PHYSICAL_INPUTS.json`。
- 全spec：[CONFIGURATION_LD1.json](CONFIGURATION_LD1.json)／[CONFIGURATION_LD2.json](CONFIGURATION_LD2.json)。旧S0 exactκ/h/M/予算・H/state/snapshot・profile・prior/decoderを保持。新構成SOURCEを入力SOURCEと分け、旧IDを再ラベルしない。

## 提案する対象・上限

Linear H4/1.00 Å/STO-3G/保存tol-only DF10/sector(2,2)、S2、Δ=.1、K2、h=.1/M12、P0。変更する外側変数はL_D=1→2のみ。Rは同じ元exactκと各prefix保存λから既存規則で生成済みで、自由に選び直さない。

| 対象 | round | 新candidate評価 | complex出力 | normalizer | 新reference |
|---|---|---:|---:|---:|---:|
| PR-LD1 | 0…12 | ≤13 | ≤39（三mean） | ≤13 | 0 |
| PR-LD2 | 0…12 | ≤13 | ≤39（三mean） | ≤13 | 0 |
| **全体** | 26round-candidate | **≤26** | **≤78** | **≤26** | **0** |

保存referenceのpair差・SOURCE・時間gateの確認、数値allowance算術と取得した新meanのsignal-space bias/normalization診断を、この科学実行の範囲に含める。旧保存mean・旧Nの新受入診断・再集計は含めない。

## 承認後の実行順

1. 今回の提案config/全scheduleと固定B validatorを照合する。数値SOURCE fefdに必要な既存関数の差分がない状態で実行する。無関係なworktreeを凍結しない。
2. 元H/state/snapshot・selected tail descriptorを既存loaderから復元する。保存λ/scalarとB split SOURCEが一致することを確認。新SCF/DF/state/ground生成なし。同じ保存inputとreferenceはcacheして使う。
3. 新規outputへPR-LD1の`SignalBatch.run`、次にPR-LD2を実行する。context/cacheは構成ごとに所有し、保存reference cacheは共有できる。独立数値処理の並列化は許可範囲内で実施可能だが、失敗後に追加候補を起動しない。CPU/RAM/workerの恣意的cap・共有設定変更はしない。
4. 各pieceのdirect expm head、既存paired Taylor/normalizer、raw/corrected独立powerとexact-tail対照を評価。fragment順・適用順・signed時間・scalar `(constant+s_tail)T` の一回会計を保持する。巨大event列・trajectory・新reference fallbackなし。
5. 各m recordを保存し、全13が揃う場合だけその構成の`FINITE_MEAN_SCHEDULE_COMPLETE_NOT_RMSE_QUALIFIED`を記録。両構成についてB正本identity/law/targetへSOURCE付きでHANDOFFする。単一構成のlocal成功でもglobal RMSE/N/C/Gはnull。
6. REPORT/HANDOFF・結果をcommit/pushしremote SHA/tree/主要成果物を照合してmandatory STOP。

実計算を開始するcallerは上記既存APIを呼ぶ最小のものとし、必要なら実行前にそのcall範囲とSOURCEを明記する。caller追加を理由に新しいgrantや細分化承認を求めず、同じ有限scopeの許可内で作業する。numerical recipeや科学条件の変更が必要なら停止して報告する。

## 数値条件・STOP

元input norm差1e−12、独立reference pair差1e−10、head/exact-tail norm/unitarity・raw/corrected closureの1e−9 gateを維持する。finite-K raw/correctedの非unitarityを誤ってexact-tail gateへ適用しない。allowanceは元R-aware経験式・logical event数とnormalizer ULP感度を保存し、rigorousな境界と主張しない。

SOURCE/hash/binding/λ/scalar/candidate/law/profile不一致、field欠測、保存pair FAIL、非有限、normalizer丸め未解決、B overflow/attenuation underflow、norm/unitarity/closure gate失敗で取得済み値・残差・例外型と未開始roundを保存して**全menuをSTOP**。LD1の失敗時はLD2を開始せず、並列実行済み処理がある場合は結果を部分取得/未確定として扱い科学的なretryをしない。

閾値緩和・再正規化・別prefix/R/κ/幅への差替え、追加参考点、旧meanの再解析、無断retryは禁止。失敗は数値/入力の理由と局所科学所見を区別し、候補の全RMSE不可能性へ読み替えない。

## この承認に含めないもの

新N/nominal/十分N、decoder/MC・shot familyの実評価、compiled取得/fit/C/G、winner、他prefix・他formula/K/κ、旧screen再開、8次PF、詳細科学レビュー。これらは後続の明示scopeで扱う。

次の担当：**ユーザー**がこの二構成と有限範囲を承認。**Codex A**が承認範囲のsignalだけを実行。**Codex B**が供給全meanから同一claim/targetのN・全RPE統計を別承認で評価し、**価格担当**が同identityのCを評価する。
