# Track A Codex 2：H4/H6高次PF・統合比較 run01

2026-10-18。**18適格群の費用比較と6層の点q選択を完了。独立確認は未実行。mandatory STOP。**

## 1. 固定条件と実行範囲

利用者の「検証を進めて」を、直前に提示した統合比較一回と、独立確認の候補・標本計画案の作成への承認として受領した。Cの新sampling/compileへ承認を広げていない。

linear H4/H6・隣接1Å・STO-3G・保存DF rank12/19・prefix3/5・T=.8・K4・外側1step当たり32 RTE short steps（全体32q）・q1/2/4/8。保存state、fragment ordering、signed絶対値largest remainder配分、元8桁new_2を維持した。ε_sig=.001、各軸ε/√2、α_axis=.025、経験的allowance1e-9、logB margin1e-12、P0。

ordinary control・explicit XXPlusYY/number Phase・Qiskit1.3.0・rz/sx/x/cx・optimization1・seed17・backend/couplingなし、profile1239304c…。旧6群と追加12群の各n8、合計144 paired trajectory/288保存compiled軸を再利用した。新SCF/DF/state、signal、sampling、build/compile、係数変更、fitは0。

## 2. 統合比較の結果

24条件の保存shot数を再現し、18適格・6不適格を確認した。既存6群はpaired費用列・平均・分散・SEを含む保存cost構造が完全一致した。全18適格群に実測費用が揃い、6層とも未解決候補なし。各層の適格qについて正確な有理数平均を比較し、同値なら小さいq→candidate IDの固定規則を適用した。

| 系 | 方式 | 点費用最小q | shot込みRZ標本平均 | 標本SE |
|---|---|---|---|---|
| H4 | 2次PR | 8 | 3.207092876e+12 | 3.040084e+10 |
| H4 | Yoshida4 PR | 2 | 2.262585049e+12 | 3.451087e+10 |
| H4 | new_2・元8桁版 | 1 | 1.321206962e+12 | 1.724511e+10 |
| H6 | 2次PR | 8 | 1.409622002e+13 | 1.473278e+11 |
| H6 | Yoshida4 PR | 2 | 1.363182105e+13 | 7.676737e+10 |
| H6 | new_2・元8桁版 | 1 | 5.993261164e+12 | 2.256837e+10 |

S2は両系とも旧代表点q4からq8へ変わった。Yoshida4 q2、new_2 q1は維持された。これは登録q1/2/4/8と経験的適格性の下での探索標本の点選択であり、母平均最適qやq>8を含む最適性の保証ではない。

選択点同士では、Yoshida4のS2に対する点費用削減率はH4 29.4506%、H6 3.29449%。new_2のS2に対する点削減率はH4 58.8036%、H6 57.4832%、Yoshida4に対して41.6063%/56.0348%。候補間独立性を比較で認定しておらず、差や比の厳密区間・母平均の優位性を示していない。

**H6 Yoshida4 q4はq2より点費用が0.974353%高いだけ**である。q2の母平均最適性まで確認するなら、選択6候補だけの独立確認では足りず、q4を対抗点として事前登録する必要がある。またH6のS2対Yoshida4の差は約3.29%であり、10%差用の標本計画を、その差を検出する設計とは扱わない。

`comparison_table.csv` は24条件についてsigned軸bias、logB/B、allowance、軸別shot数、軸別RZ平均、shot込みRZ/CX平均・SE、適格性、選択点を対応させる。不適格条件の費用は空欄で、0補完しない。通常signed biasの既存会計であり、非相殺bias・母平均保証・厳密bias保証を追加した結果ではない。B0/B0+IやRPE全round費用とも統合していない。

## 3. 独立確認の候補と未採用の標本数案

基本候補はH4/H6各S2 q8、Yoshida4 q2、new_2 q1の6点。結果SHAと候補集合をselection digest `667d15d18f02414c530d4f1c0d94355f6b58d5ccf53833a8fce3b2375ba93e68` に固定した。C起動前には新しいseed namespace、PRNG/event law、旧・追加探索seedの全除外集合を再照合し、別SOURCE/入力seal/grantと固定標本数が必要である。namespaceの相違だけで確率変数の独立性は証明されない。旧n8も今回の探索n8も確認標本へ混ぜない。

下表は既存の`planning_n`による保存n8分散のプラグイン試算。目標相対MCSEは**案**であり、採用値・必要十分な標本数・precision保証ではない。

| 候補 | 相対MCSE 1%案 | 2%案 | 5%案 |
|---|---|---|---|
| H4_S2_q8 | 8 | 2 | 2 |
| H4_Y4_q2 | 19 | 5 | 2 |
| H4_N2_q1 | 14 | 4 | 2 |
| H6_S2_q8 | 9 | 3 | 2 |
| H6_Y4_q2 | 3 | 2 | 2 |
| H6_N2_q1 | 2 | 2 | 2 |

1%案の試算を単純合計すると55 trajectory/110 cost-wrapper軸だが、**adopted_n=null**を維持する。n=2等の小さい試算値をそのまま実行推奨にしない。pilot n8では未観測希少層・分散の過小推定を除けず、厳密なMCSE・coverage保証はない。未観測層の費用を0や任意値で補完していない。

実行前に登録した別の設計例は、同一系の3方式間全pair・両系でfamily6、名目α=.05、normal近似power=.8、検出差=大きい方のpilot平均の10%。候補独立を**計画上だけ**仮定したBonferroni式であり、実際のpowerを保証しない。実データ比較の独立性認定やconfidence budgetの変更ではない。Cで約3.29%の方式差や約0.97%のq差を調べるなら、その目的に合わせて検出差・比較family・標本数を改めて事前固定する。

## 4. SOURCE・実行・技術監査

- 比較SOURCE：`a65e4f6e7bcc83f90a054f6aafc4ee2e0ca5d80a`。取得SOURCE：`d77e439a16d6953cb17e6d1ef91b0d9430144848`。既存のkernel/adapter/runnerを変更せず再利用した。
- 登録後処理補助：`postprocess_registered.py`、SHA256 `d943ba9e5043de6635cdd3884472664a687c1d89935989d6e8a858667d08ba43`。Git SOURCEに元々含まれない補助scriptであり、起動前に独立したSHAでsealし公開bundleへ同一bytesを保存した。
- 実行private plan digest：`57f238326f8b3fea5d89fe18646b9c61c326bca84df530926694a30d7174030b`。grant `H46_HIGHER_PF_INTEGRATED_20261018_RUN01`、exclusive消費marker1件。公開するのは消費receiptだけ。
- native exit0、wait回収済み、wall 0.580246秒。登録後処理もexit0で回収。139fileを前後照合した。42人工/mock PASSを専用cloneでも確認した。作者自身の確認であり独立reviewではない。
- 24行・144 cost標本の軽量処理なので1workerを使用した。native compileを伴わず、process起動が処理時間を支配するため余分なprocess分割はしない。将来Cのtrajectory並列度には移送しない。
- 起動前に組込みmathをfileとして扱うmetadata処理を修正した。この時点はgrant作成・科学dispatchとも0。Git object参照の階層警告も自分のclone内のalternatesだけで解消した。科学retryは0、旧run/grant/ledger/STOP・他worktree・共有環境は変更していない。

[公開bundle](../../artifacts/resource_applicability/track_a_h46_higher_pf_integrated_run01/2026-10-18/README.md)。raw入力・NPZ/QPY・有効grantは非公開のまま。次Phaseを起動せずmandatory STOPした。

## 5. 次の作業

**ユーザー**：独立確認の目的を選ぶ。選択6点の費用再現と方式比較を中心にするか、H6 Yoshida4 q4を追加してq2との僅差も検討するか。その上で目標MCSE・検出差・比較familyと具体的n_Cを含むC契約を承認するか判断する。

**Codex**：採用された確認目的に合わせて固定標本計画・seed除外監査・実行契約を仕上げる。Cの新sampling/compileは別の明示承認後に限る。new_2の厳密4次係数版、H8、geometry、別ε、決定論対照・RPE主表へ自動拡張しない。
