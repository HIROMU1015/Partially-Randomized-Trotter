# A1-0：技術接続・最小科学実行scope

2026-11-06 JST。SERVER1 `GPUA100-gleap-01`。
専用branch：`codex/o2-a1-0-connections-20261106`。基点：[前回公開a2fcbe2][BASE]。

**opt2 threshold0のfinite targetをSOURCEで一意に指定し、実CDF/固定controlとselected-only full価格への薄い接続を追加した。人工24件PASS。実科学payload・bias上界・state/価格は未取得のため、科学測定は開始していない。**
同じ静的preflightを繰り返す代わりに、次に許可が必要な数値処理をQ1–Q5として具体化した。

## 成果物と差分

| 成果物 | 内容 |
|---|---|
| [A1_0_CONNECTION_STATUS.md](A1_0_CONNECTION_STATUS.md) | 今回閉じた技術接続、保存catalog/CDFの限界、threshold/target/state/bias/価格の残課題 |
| [A1_0_SCIENCE_RUN_SCOPE.md](A1_0_SCIENCE_RUN_SCOPE.md) | 固定SOURCE/input、Q1–Q5の関数・出力・目的・数値条件・STOP。未承認の実行仕様 |
| [a1_connections.py](../../../src/trottertracks/algorithm_codesign/a1_connections.py) | caller提供sampler入力→実CDF→既存law/controlと、指定軸だけのselected full価格接続 |
| [test_a1_connections.py](../../../tests/tracks/algorithm_codesign/test_a1_connections.py) | 人工CDF、既存純粋validatorの限定利用、mock builder/compiler/cache |

既存A1の5ファイルは変更しない。PROJECT_MAPと研究別SOURCE索引に案内を追加する。
科学runner・新artifact・承認管理機構は作らず、研究概要・科学manifest・正式契約・元の判断履歴は変更しない。科学的な新結果/採択/validation pathの追加ではない。

## 今回新たに閉じた点

1. **finite target**：opt2のcutoff0/preparation/tail、元selected回路、m0のK2 paired Taylor degree3とattenuation/phaseを固定した。実53-bit法の平均とalgebraic finite targetは別記する。
2. **threshold差の範囲**：cutoffはidentity抽出前に適用される。従ってtail係数だけでなくidentity phase、λ/B、有限operatorの差を扱う必要がある。保存dropped L1を小さいという理由で0扱いしない。
3. **catalogの不足**：WP06bに保存された324件のsupport基底定義を確認したが、係数・確率・抽出順付きcatalogではない。canonical保存JSONと元WP11 opt2 rawを探した範囲では完全CDF/catalogは未発見。固定NPZ→同preparationで取得する科学処理まで接続を限定した。
4. **実law接口**：元validatorと同じ正規化順、NumPy1.26.4 choice CDF、既存53-bit adapter、固定3control/order/5層bankへ接続した。positive入力でも格子supportが0になる人工例を拒否し、無断label置換を行わない。
5. **価格接口**：旧paired関数の不要なfull-basis比較を呼ばず、元selected evolutionと必要な軸のfull wrapperだけを既存compile/cacheへ渡す。cache hitでも量子trial価格recordは返し、shot費を消さない。
6. **取得scopeの具体化**：B1のattempt capはstageごとにquotaを凍結して適用する。production quotaは独立calibration後・production前に固定するため、前回草案の時間順の曖昧さを解消した。

## 人工テスト

既存opt2 venvを変更せず使用。新9件＋既存A1の15件、**24 tests / OK**（最終fresh process約0.470秒）。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src <existing-opt2-python> -B -m unittest discover \
  -s tests/tracks/algorithm_codesign -p 'test_a1*.py' -v
```

確認範囲は、CDF境界/格子質量、fsum/除算順、ν/joint/5層の人工恒等式、support消失、入力/profile拒否、selected-onlyの両軸/片軸call、cache hitの課金recordである。
旧人工15件も含め、実科学CDF・保存測定値・NPZは入力にしていない。改善率・真variance・十分N・費用順位は評価していない。

初回testのvalidator importがpackage全体を初期化し、Qiskit内部warningを出したため、そのimport経路は修正した。
最終testは既存の純粋validator関数一つをAST抽出して人工入力へ適用し、科学module/Qiskitが未importであることも検査した。最終runのwarningはなかった。
初回library初期化内部の全call数は記録していないので、その内部処理まで一律0とは認定しない。A1/H4の科学入力再構成・測定・full回路/実価格取得は実施していない。
固定SOURCEやpackageを修正して回避したものではない。

## 未閉鎖の条件と最小の次段階

| 項目 | 今後必要なもの |
|---|---|
| Q1 実law | 同じthreshold0の完全payload、実CDF/support、ν/joint moment。新規科学数値取得 |
| Q2-L finite-law/numerical bias | μ_qと固定finite targetの差の正当な上界、centering条件と正の座標残余幅 |
| Q2-T threshold橋渡し | WP04のsignal/PF/RTE/radius保証を使う場合だけ必要。旧保証を移送しないfinite-mean診断なら初回から外せる |
| Q3 state | 保存vectorを実回収するか、同じDF 4電子ground一つを限定再構成。fingerprint欄だけではsimulationできない |
| Q4 full価格 | fresh path/axisの不足selected build/compile。元8 trajectoryの平均は全層へ無条件移送不可。期待価格confidenceは追加条件 |
| Q5 測定・統計・費用 | 既存提案の採否後、独立train/cal/prod。実現campaign費用と期待費用を分離する |

最小の次の科学実行候補は**Q1＋Q2-Lだけの非sampling取得→STOP**。これで実law/既知平均とfinite-target誤差予算を数値で閉じられるか判断できる。
初期をfinite meanの座標診断に限る場合、新PF/radius/18round計算を要求しない。旧εは事前比較幅の候補であり、旧RPE保証を引き継いだとの意味ではない。
Q3–Q5まで実行するには、それらも含む明示的な科学実行許可が必要。Q1/Q2成功だけでは測定へ進まない。

control a/b・3features、B0/B1-O/B1-OL/A1、train1024/axis、B1 pilot64/層/axis、η=1/8、cal式、production8192上限、range/attempt guard、ε/α配分は**前回の未承認提案のまま**。
次のユーザー判断は、その採否、finite-mean診断かWP04保証の移送も要求するか、state再構成・不足compile・期待価格confidenceをどこまで許可するかである。
承認証明書・新grant発行を要求しない。重要科学レビューは開始していない。

## 公開・保全・mandatory STOP

通常sandboxには既知のbwrap起動障害があり、実行環境の承認機構を通した限定操作で作業した。
元main/既存dirty-untracked、前回worktree、旧SOURCE/input/raw/result/grant/ledger/marker/STOP、one-shot/retryを保全する。
C1凍結、S2のT主指標追加不改善、R1候補なしを維持する。共有Git認証、venv/package/cache設定、OS/quota、他ユーザーjob/filesは変更しない。

新規実科学CDF/support/moment/normalizer、threshold/bias/radius/残余幅、RTE sampling、Hadamard simulation、train/cal/prod、variance/N/費用評価は未実施。
science worker/GPU jobも起動していない。今回は技術接続の人工検証と実行仕様までである。

専用branchへcommit/pushし、公開remote SHA・3文書本文・基点からの限定差分と原本保全を照合する。
確定SHAと取得実績は最終チャットで報告し、自己SHAを書き込む追加commitは不要とする。
**公開・本文照合後mandatory STOP。Q1–Q5、追加実装、別Trackへ自動進行しない。**

[BASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a2fcbe212ebd66b563dd1413768e5569bf5fdfed
