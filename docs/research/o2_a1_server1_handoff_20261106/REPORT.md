# A1 SERVER1：研究引き継ぎ・科学実行契約草案

2026-11-06 JST。branch `codex/o2-a1-server1-handoff-20261106`。
基点・引き継ぎ元：[`6e8d099a5252d3b424045fb22aeb112d5ebad5c9`][A1]。

**SERVER2の固定5ファイルをSERVER1の専用worktreeへ変更せず継承し、最初のm0/X/Y検証の契約草案を作成した。科学計算は未実施。**
保存H4入力・opt2環境・元価格rawを確認でき、人工テスト15件PASS。
ただしWP04信号とopt2回路のtail threshold/preparationに版差があり、実CDF・bias・state/価格の接続を閉じずに科学測定を開始できる状態ではない。
草案の完成と科学実行の開始可能性を区別する。

## 成果物

| 文書 | 内容 |
|---|---|
| [A1_SERVER1_HANDOFF.md](A1_SERVER1_HANDOFF.md) | SERVER1/remote/branch、固定5ファイルの取得・hash、依存環境、保存入力・SOURCE・rawの所在、版差、再利用、保全 |
| [A1_SCIENCE_EXECUTION_CONTRACT_DRAFT.md](A1_SCIENCE_EXECUTION_CONTRACT_DRAFT.md) | model/state/finite target、実law/ν、control/強い層別baseline、有限測定・confidence・価格、未承認call、STOP |
| 本REPORT | 実績、完成範囲、不足、次の判断、公開・停止条件 |

## 1. 実施したこと

ユーザー指定のSERVER1を実測hostname `GPUA100-gleap-01` と保存環境記録・SERVER1 inventory・既存WP11資産で照合した。
主checkoutはmain `0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf`、既存dirty/untrackedあり。原状を記録し、mainを更新せず、固定commitから自分専用branch/worktreeを作成した。
SERVER2 worktree・他sessionのbranch/lockには操作していない。

固定commitの3文書・A1 module・人工testを実読した。GitHubから5本文を実取得し、local Git blob OIDおよび全文一致を確認した。
正式共通指示2026-10-25を実読し、公開保存コピーとのbyte一致も確認した。適用AGENTS/PROJECT_MAP、研究概要全文、関連規範・validation status/manifest、opt2、V1/V2を参照した。
保存欄の引用・SOURCE照合だけで、新しい科学的再集計・順位付けはしていない。

既存WP11 opt2環境のpackage metadataはPython3.12.3、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、PySCF2.7.0、OpenFermion1.6.1。
環境を変更せず、次の**人工テストだけ**を再実行し、15 tests / OK（約0.227秒）だった。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src <existing-opt2-python> -B -m unittest discover \
  -s tests/tracks/algorithm_codesign -p test_a1_control_variates.py -v
```

架空label/CDF、手書きbit、mock metadata、readoutのAST検査に限定した。科学入力・保存測定値は使わず、既存wrapper SOURCEはreadout規約の静的AST照合だけに用いた。RNG・量子生成はない。
これは主エージェントの技術確認で、独立科学レビュー・CI・実PRでの改善実証ではない。sub-agentも使用していない。

## 2. 入力・roundの適格性と未解決接続

| 項目 | 判定 |
|---|---|
| 保存H4 NPZ | Git objectとSERVER1に存在。byte SHA256は既存記録と一致、decodeなし |
| 候補 | linear H4/1.00 Å/STO-3G/legacy DF12/4電子/8qubit/L_D3、最終opt2 m0/q1/δ.02/r1/K2、X/Yを提案 |
| state | 保存DF全体のphysical-sector groundを対象とする。H_D ground/PF controlと区別。state fingerprintは記録として対応するが、完全state payloadの利用経路は未閉鎖 |
| 元物理H | 保存DF modelとのoperator/state binding未閉鎖。未近似Hへの保証・厳密半径1を主張しない |
| opt2 raw価格 | SERVER1のWP11にm0の既存8 trajectory checkpointあり。記録10 SOURCE-file hash一致。全event/CDF/fresh bitは含まれず、そのまま学習用入力にはできない |
| 重要な版差 | WP04 signalはcoefficient_atol=1e-12/preparation `89145d…`、opt2回路側は既定0.0/preparation `50c77f…`。同じH hashから同じlaw/finite targetとは認定しない |
| 実law/ν | generator・53-bit CDF adapterの対応を静読。完全catalog/actual CDF・momentsは未取得・未計算。Parent dyadic lawは代用しない |
| bias/radius | V1/V2の条件付きstatusを維持。threshold差、finite-grid law、centering、native/数値実装、正残余幅は実行前条件 |

必要な元NPZ・元SOURCE・元価格rawが全て欠落した状況ではない。一方、完全catalog、対応CDF、fresh bit、状態payload/限定再構成、価格confidence接続は未取得/未実装として明示した。
未発見資産を存在しないと断定せず、別入力による無断代替はしていない。

## 3. 草案で具体化した提案

1. **control**：既存WP06aのpositive非identity代表IDを文字列昇順に選び、a=`df-fragment-3:Z0Z1`、b=`df-fragment-6:Z0`。n0/local、n2/local、n2/pairの3 indicator。選択に測定outcome・未知期待値・compiled costは使わない。実qでpositive supportを確認できなければSTOP。
2. **baseline**：B0 rawにもvariance-sensitive confidenceを認め、B1-Oのorder2層、B1-OLのorder/label共通細分5層、A1を比較。全正層のconditional取得・有料pilot/cost-aware配分を含める。5層に吸収される利益をCV独自利益と呼ばない。
3. **有限測定**：A1 train各axis1024、B0 train不要、B1各正層/axis64 pilot。calibrationはη=1/8を用いる事前固定式、productionは独立calibrationから十分Nを決定して各方式/axis合計8192 shot上限とする候補。これらは未承認のscope案で、新しいN計算結果ではない。
4. **confidence**：凍結係数の有界range、独立較正によるvariance上界、fresh productionのBernstein/Hoeffding十分条件を全方式に共通化。保存α/axisを4方式、cal/prod/price、必要なら層に配分。旧N=510やbinary V2 PMFをCVへ移さない。
5. **価格**：元Qiskit1.3.0/opt2/seed17/selected policyのfull no-prep価格。全D halves・中央RTE・control・scalar phase・ancilla・X/Y readoutを含み、train/cal/prod、古典棄却/fit/setupも計上。T/native未取得はUNKNOWN。
6. **STOP**：target/law/centering/bias/価格不一致、独立性破損、range/shot/attempt/call scope超過、実際の資源不足等を具体化。全18roundや別parameterへ自動拡張しない。

候補予算は有限性と統計保証の接続方法を示すための提案であり、性能最適値ではない。rare featureがtrainに現れない可能性、層別取得・較正費が大きくなる可能性を残す。
分散削減、十分Nの値、総費用比、新規性、科学的GOを発行していない。

## 4. 次のユーザー／GPT判断

まず、保存DFモデル限定のm0両軸診断を採るか、opt2のthreshold0 lawとWP04保存signalとの差をどの範囲で接続するかを判断する。
その上でcontrol/比較4方式、精度・α、train/cal規則・測定scope・価格報告範囲を固定する。
完全catalog/CDF・state payloadを回収できない場合の限定再構成、bias評価、fresh Hadamard simulation、train/cal/prod、variance/十分N/費用評価、不足full build/compileには**新しい明示的な科学実行許可**が必要。
今回はそれらを開始していない。承認証明書・新grant発行・段階ごとの再承認を標準要件にしない。

## 5. 公開・保全・mandatory STOP

今回の差分はこのdirectoryの3文書だけ。固定A1の5ファイル、科学SOURCE・input・raw・result・grant・ledger・marker・旧STOP、元mainのdirty/untrackedを保全する。
正式研究契約やvalidation manifestを変更していない。C1凍結、S2のT主指標追加不改善、R1候補なし、既存one-shot/retry条件を維持する。
新しいHamiltonian/state生成、NPZ配列decode、科学CDF/moment/normalizer評価、RNG sampling、Hadamard simulation、科学train/cal/prod、variance/N/費用再解析、回路build/compile、native synthesisは全て0件。
新実装・新汎用pipeline・重要科学レビューは行っていない。

通常sandboxのbwrap起動障害に対し、実行環境の承認機構を通した限定操作で対応した。共有設定、Git認証設定、package/cache、他ユーザーprocess/filesを変更していない。
3文書を専用branchへcommit/pushし、remote SHA、3本文の実取得一致、基点からの3追加以外の差分なしを確認する。
確定SHA・取得照合実績と元main保全の最終確認は最終チャットで報告する。自己SHAを文書に埋めるための追加commitは行わない。

**公開・取得照合でmandatory STOP。ユーザーの新しい明示許可前に科学測定・数値計算・実装・別Trackへ進まない。**

[A1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/6e8d099a5252d3b424045fb22aeb112d5ebad5c9
