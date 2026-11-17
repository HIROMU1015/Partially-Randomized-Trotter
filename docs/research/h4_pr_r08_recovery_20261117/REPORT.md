# R08への失敗継続修正の適用

R08の科学条件を変えず、修正SOURCE `fdb4453a0fae36c8b8734a225a19316446dee62e` で通常探索を再開する入口を固定した。旧controllerは例外後の準備回収中で、最後のLD4価格較正が残っている。走行中SOURCEの変更・signal送信・科学stageの強制終了は行わない。所有lockの自然解放を待つ引継ぎprocessを別tmuxで起動する。

**修正版の科学controllerが稼働したかは `HANDOVER_STATUS.json` と `run02/CONTROLLER_PID.json` で区別する。`WAITING_FOR_OLD_OWNER` は切替待機であり、修正版での探索再開完了を意味しない。** 旧runと旧STOPは保全する。

## 適用内容

- SOURCE314524a91の即時例外記録、候補単位の限定UNKNOWN保留、未開始stageでの協調yield、request key保全を採用。
- 既存の `scripts/run_rpe_schedule_search.py`、Completion、concurrent queue、DP workerを再利用。新しい科学worker・候補選択器は作らない。
- PR-Pauli要求幅 `.1/.2`、PR-DF `.1/.05`、S2、L_D1〜9、元mapperのκ探索、shot探索、target/law/decoder/loss/exact τを維持。binary64の `.2` を有理数 `1/5` へ変換しない。
- 元のcycle上限200を維持。保存stateのcycle3なら残り197。queue20、DP worker3、checkpoint m5/m11と既存cacheを維持する。技術的移行で予算をリセット・拡張しない。

## §10.5差分と保存状態

専用の修正worktreeを使い、旧R08は観測のみ。別runや本番queueへの重複投入は0。旧controllerの所有lockが残る間、新しいcontrollerを起動しない。

保存Q3は316件（適合218、不適合98）。全件に成功したPROCESS_RETURNとOUTCOMEがあり、この対象内で未回収0、UNKNOWN0を確認した。MSE再計算・価格再計数・fitは行っていない。再利用時には元Executorのjob key・route・SOURCE・binding guardが改めて適用されるため、316件全てが無条件に再利用可能とは認定しない。

観測時の準備はREADY26、既存HELD9、WAITING1、SELECTED1。SELECTEDは進行中LD4の較正であり、実runを保留へ変更していない。人工的な移行検査では死んだownerのSELECTEDを保留する分岐も試した。実移行は旧owner終了後の最終stateを読み直す。

旧R08は16:00 JSTの `WIDTH_READY_DURING_PREPARATION` ERROR後、Q3投入が止まり、準備の有限waveが続いている。旧controllerの例外tracebackはまだ回収待ちで、具体的原因は未確定。新SOURCEがこの実例の原因を解消済みとは主張しない。

実metadata検査では、DFのLD4/.05・LD8/.1・LD8/.05で `DIFFERENT_SPLIT_LAMBDA_SOURCE` を検出した。元WidthDriverと同じ規則でそのpointのみHELDにし、guardを緩めない。他の有効候補の探索を続ける。元record/keyは付け替えない。所在・理由は `checks/METADATA_SMOKE_SUMMARY.json`。

## 検証

- 引継ぎ専用の人工4件PASS。成功・失敗・曖昧stageの保持、過去key、signed zero、元configuration guardによる個別保留、Q3の欠測・異常終了による保護を確認。
- native Python3.11.0rc1でも同じ4件PASS。
- 既存の失敗継続回帰10件PASS。以前公開した92件の修正検証を再利用し、全82既存回帰を重複実行していない。
- 固定SOURCEで保存H4 metadataだけを通常WidthDriverへ通し、36 family・271 point・316 receiptを確認。科学call0。候補のSOURCE保留3件を維持。
- 実H4の新規科学計算はこの技術準備・検証では0。旧R08の既承認stageは旧SOURCEで継続している。修正版による科学的成果・速度倍率・最安更新はまだ未取得。

## 引継ぎ動作と保護

`resume_pr_r08_after_drain.py` は旧wrapperのflockを待つ。解放後、CONTROLLER_RETURNと元SOURCE/入力を確認し、元branchへ終端metadataだけを追記commit/pushする。raw・STOP・SOURCEの内容は変えない。新しいrun02へ元keyのterminal record・価格registry・stateを引き継ぐ。死んだownerの曖昧SELECTED、失敗stage、STOPは再試行しない。過去のcontroller停止理由は `resumed_from.previous_stop` に残す。

元family/physical/binding guardを通して新入力をcommit/pushし、remote一致を確認した後だけ、通常CLIを一度起動する。未回収Q3・数値UNKNOWN・SOURCE不整合・保存/公開不能では `HANDOVER_STOP` になり、新規科学投入しない。自動retry loopはない。稼働後の例外は新しい `RUNTIME_ERRORS.jsonl` に回収待ち前から残る。

ロードマップ差分はR08の安全なSOURCE引継ぎ入口・検証・切替待機の公開。探索完了、例外原因解消、価格資格拡張、最安更新の認定ではない。ロードマップ本体は編集しない。
