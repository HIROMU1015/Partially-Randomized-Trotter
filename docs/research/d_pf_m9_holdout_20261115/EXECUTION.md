# m9限定の実行SOURCE固定と有限scope

利用者の明示許可：Full PF/p7 × S2/Y4/M5の指定6物理点、m9/T51.2、X/Yの不足最大12 compile、固定予測の受入・取得・個別判定・公開まで。追加承認証明は作らない。Cの正本とpolicyはcaller内の固定commit/path/SHAに束縛する。

- C HANDOFF：04a68f5932f53377595b2d69bbf8f46e02e82a01。
- PREDICTIONS：b1a7cd358daeab51ce538c747d0cdb8ad86c97d4、SHA-256 `e5e00558669fd9568d40fcc2deba5e157a2c21c5505c5bb3268ea1a185701fc5`。
- M9_POLICY：375f026a30e3c1ed3179a73cbbe333578abff00f。observed分母、exact `<=1/50`、全12個別PASSでcampaign成功。
- [観測前metadata](PREOPEN.json)、[§10.5差分・入力bytes・資源観測](DUPLICATE_AND_INPUT_PREFLIGHT.json)、[人工/回帰テスト](TESTS.txt)。保存受入候補0、不明重複0、12軸不足（確認scope内）。未列挙archiveは網羅していない。

caller：`scripts/run_pf_m9_prices.py`。既存F/Tの`plans`が持つ全schedule metadataから固定m9だけを選択する。`pf_price_stages.STAGES`、m6専用reader、既存F/T callerは無変更。固定予測値を計算し直さず、元係数とSOURCEのmetadata一致だけを検査する。

| route | 公式 / q / physical ID（各XY） | 新規compile上限 | 出力 / private namespace |
|---|---|---:|---|
| Full | S2/224/a4ec53ff4e60d60a2930、Y4/96/b2c479e7805c4d1f0b30、M5/32/a53547a5fb707b9550d5 | 6 | `d_full_pf_shortlist_prices_20261114/m9_holdout/run01` / 既存Full private rootの`m9_holdout/run01` |
| p7 B0I | S2/224/1bd0a5dd4bbf876da1d6、Y4/96/9924c226ee246283ec54、M5/32/60111109e0024d1a1064 | 6 | `d_truncated_p7_t3_prices_20261114/m9_holdout/run01` / 既存p7 private rootの`m9_holdout/run01` |

上記public pathの接頭辞は`docs/research/`。private rootはF/Tの元定数を利用し、旧calibration/holdoutやone-shot枠を再利用しない。deterministicなのでsampling/path/event=0。cohortは各routeの既存決定論系列名、index0、実際のm9 physical ID・SOURCE・新namespaceで区別し、独立な確率標本の主張はしない。

childは既存`signal.install`を先に実施し、provider c31ef424 bytesを既存検査で確認してから`native_context`を各route一度作る。Cのtotal scalarとのhex完全一致をbuild前に要求する。Fullはc_H、p7はc_H＋tail identity、fragment0..prefix−1を生成順に保持する。

実回路は元`full_trial/checked_trial`→`cost_trial/compose`、native v5、測定付きHadamard XY、Accounting/compile/cache/fingerprintを再利用する。Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、symmetric_directional、P0除外、`H46_LONG_TIME_NATIVE_COMPOSE / NOT_APPLICABLE_NO_RTE`。native/測定gate `<1e-9`、compiled-vector `<1e-7`、元index0の7 gateをすべて保持。片軸の価格流用やcompiled-vector省略はしない。

実行はFull route、その有効取得・公開・guard bindingを確認後にp7 route。route内のworker数は既存`run_cells`の実資源観測による（3 cellを超える科学対象はない）。入力固定時はaffinity64、idle約61、利用可能RAM約516GB。恒常的なCPU/RAM/worker capは設けず、共有設定・他processは変更しない。

```bash
python -B scripts/run_pf_m9_prices.py --execute-approved-route FULL_PF
python -B scripts/run_pf_m9_prices.py --bind-route FULL_PF --publication <Full raw SHA>
python -B scripts/run_pf_m9_prices.py --execute-approved-route TRUNCATED_PF
python -B scripts/run_pf_m9_prices.py --bind-route TRUNCATED_PF --publication <p7 raw SHA>
python -B scripts/run_pf_m9_prices.py --evaluate-publications <route-to-bound-commit.json> --output docs/research/d_pf_m9_holdout_20261115/run01
```

実環境のPythonはStage 2と同じ専用venvを用いる。各routeは実compile前に当該caller/input/preopenをcommitで固定し、結果SOURCEはRESULT/sourceへ記録する。raw公開とbound公開を分け、C guardはモデル適合判定と区別する。予測との2%超過FAILでも有効な直接価格を破棄しない。

重大実装失敗/取得不能/保存不能/OOMでは新規投入停止、部分trial/cache/STOPを保全する。自動retry、refit、閾値変更、別軸/roundへの置換をしない。全判定または重大異常でcommit/push・remote照合してmandatory STOP。結論は指定m9点のみ。m7/m8/m10～12の資格、N/G/winnerへ進まない。
