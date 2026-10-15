# H8 native B0+I：実装・実行準備

H8 B0+I prefix7/14/21×q1/2/4/8の12構成/最大24軸を実装。SOURCE65a9d801…、人工38件と未承認CLI拒否PASS、provider265/input7 byte照合PASS。16 modes/DF28/sector4900、保存referenceと経験的u、caseごとの2軸並列・CPU carryを固定。実新scalar/signal/native/compile0、CPU[]/false/未seal/commandnull、資源案は承認待ち。subagents0、準備後STOP。

[準備入口](../track_a_h8_native_b0i_readiness_20261015.md)。利用者がH6全登録結果を受け、追加H6なしでH8対照coverageへ進む方針を示した。既存10月15日科学reviewに従う実装であり、新しい詳細科学reviewや本計算は行っていない。H6の許可・旧grantをH8へ移送せず、実入力会計/native correctness/compileと資源は次の明示認可へ戻す。

## 追記：4 worker対応

SOURCE54ed98c5…、2構成×cosine/sineの最大4 workerへ変更。人工45件PASS、回路保持最大2構成・奇数残り2軸・4 identity監視・途中起動失敗cleanupを検査。科学/compiler/最大24compileは不変。AS64GiB/RSS32GiBは各process上限案で実測消費量ではない。実科学/worker起動0、CPU[]/false/未sealを維持。

利用者の指示を受け、先頭から最大2構成を検証して4軸を並列compileする方式へ変更。2番目の検証失敗時はそのまとまりの全compileを開始せず、既完了分と最初のSTOP理由を保持する。実16-mode回路の速度・最大RSSは未測定。336.5GiBは旧AS合算方式を4 workerへ換算した未承認起動判定案であり、必要物理メモリを測定した値ではない。新grant発行・本計算起動・共有環境変更・subagent利用は0。

## 追記：4 worker one-shot実行承認

利用者の起動指示を受領。SOURCE54ed98c5…、H8全12構成screen/適格のみ最大24軸・4 worker one-shot。source13/provider265/profile34,273/input7 byte照合PASS、旧H8消費/保留/不明と停止証拠を保持。CPU[63, 62, 61, 59, 57]をown-run用に固定、最終auth/sealとfresh gate合格後だけ起動。この記録は起動前、科学結果未取得。自動retry/次stageなし、subagents0。
[実行認可の固定記録](../track_a_h8_native_b0i_authorized_launch_20261015.md)。

## 追記：4 worker one-shot実行承認

利用者の起動指示を受領。SOURCE54ed98c5…、H8全12構成screen/適格のみ最大24軸・4 worker one-shot。source13/provider265/profile34,273/input7 byte照合PASS、旧H8消費/保留/不明と停止証拠を保持。CPU[63, 61, 60, 56, 54]をown-run用に固定、最終auth/sealとfresh gate合格後だけ起動。この記録は起動前、科学結果未取得。自動retry/次stageなし、subagents0。
[実行認可の固定記録](../track_a_h8_native_b0i_authorized_launch_20261015.md)。

## 追記：4 worker one-shot実行承認

利用者の起動指示を受領。SOURCE54ed98c5…、H8全12構成screen/適格のみ最大24軸・4 worker one-shot。source13/provider265/profile34,273/input7 byte照合PASS、旧H8消費/保留/不明と停止証拠を保持。CPU[3, 4, 5, 6, 8]をown-run用に固定、最終auth/sealとfresh gate合格後だけ起動。この記録は起動前、科学結果未取得。自動retry/次stageなし、subagents0。
[実行認可の固定記録](../track_a_h8_native_b0i_authorized_launch_20261015.md)。
