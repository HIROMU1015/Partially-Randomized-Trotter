# H4凍結受領・独立最終review（2026-10-08）

**独立reviewはTECHNICAL FAIL、受領はNOT_EVALUABLE。本計算・再sealは行わない。**
H4_RECEIPT_PENDING_INDEPENDENT_TECHNICAL_REVIEW_FAILED_STOP。SOURCE `ad57d1639133f7158cce58d767b8e0aa179bf044`、準備REVIEW `798270161d85af6361a4b08cdee9e4c879e6ffd7`、origin起点branchのSHA一致を確認。
資料branch `track-a-h4-receipt-final-review-20261008`。source32のactual blob/bytesは全一致し、source/tests/schemaの変更0。
この資料を含むREVIEW commitのactual SHA/remote SHAは固定後の外部receiptと最終報告で確認する。

一つの入口は本書。[独立review](../../artifacts/resource_applicability/track_a_h4_receipt_final_review/2026-10-08/independent_final_review_v1.json)、[技術binding](../../artifacts/resource_applicability/track_a_h4_receipt_final_review/2026-10-08/technical_binding_assessment_v1.json)、
[受領監査](../../artifacts/resource_applicability/track_a_h4_receipt_final_review/2026-10-08/receipt_audit_v1.json)、[最終承認状態](../../artifacts/resource_applicability/track_a_h4_receipt_final_review/2026-10-08/final_approval_status_v3.json)、[commit対象](../../artifacts/resource_applicability/track_a_h4_receipt_final_review/2026-10-08/commit_inventory_v1.json)を結合する。
[従来の準備・57人工結果](track_a_h4_prelaunch_preparation_20261008.md)は保存する。57PASSを最終技術review合格/production成功へ読み替えない。

## 受領と利用者が行う最小転送

既知受領directoryは空。NPZ6、generation-freeze、run05 byte journal/log/native controlは未受領。
streaming byte SHA照合0、科学array読込0、入力再生成0。old handoffには6 NPZ/3補助fileのSHAはあるが個別byte数がない。
転送元でmanifestへbyte数/SHA/file一覧を捕捉し、受領packet/manifest/known SHAの三段階で照合する。
[file要件](../../artifacts/resource_applicability/track_a_h4_receipt_final_review/2026-10-08/TRANSFER_REQUIREMENTS_v1.md)に期待値を固定した。

実行可能な具体手順は[local-only転送手順](/home/AbeHiromu/projects/h4-handoff-evidence/20261008/receipt-final-review/USER_TRANSFER_STEPS.md)。
旧サーバーの既存ログイン済みターミナルで最初のPython here-docを実行すると、NPZ byteを読んでpacket/manifest/SHA256SUMSをhome内へ新規作成する。
続くSCPを旧サーバー側で実行するだけでpacketとSHA256SUMSを新hostの `/home/AbeHiromu/projects/h4-handoff-evidence/20261008/receipt-final-review/incoming/` へコピーできる。
認証できなければ既存SFTPで手元を経由して同じ2 filesをcopyする。秘密情報は通常の認証promptだけに入力しchatへ送らない。
鍵・認証設定・共有環境を変更せず、旧fileは変更/削除/移動しない。packet/control/内部SSH情報はGit外。
利用者へcopy完了/pathを依頼し、その間に独立reviewと資料を完成した。未受領のままbytes一致を宣言しない。

## 独立reviewのtechnical blocker

reviewer `/root/independent_final_review` は実装authorとは別にSOURCE/REVIEW/固定19 artifacts/外部25証拠/live profileをread-onlyで検査した。
SOURCE32/blob/SHA、候補environment/compilerの一致、CPU14 distinct physical cores、容量・quota・累積budget案はPASS。
**P1：所有processがsample成功後、pidfd送信前に退出する競合でcleanup loopが中断する。**

- [OwnedIdentity.terminate](../../src/trottertracks/resource_applicability/h4_geometry/observer.py#L76) と `terminate_after_parent_exit()` は pidfd_send_signal のESRCH/ProcessLookupErrorをcatchしない。
- 2 owner fixtureで送信side_effect=[ESRCH,None]にすると、signals_attempted=1、second_owned_child_visited=false。
- `IndependentObserver.stop_children()`、observerのworker/driver停止、`OwnedPool.shutdown()`のwait/pipe/executor終了が例外で途中終了する可能性がある。
- 最小修正：両送信経路でESRCHだけをalready-exitedとして処理して後続cleanupを続け、他の所有/権限errorはfail-closed。限定mock回帰・新SOURCE固定・binding更新・独立再reviewが必要。

[純mock再現](../../artifacts/resource_applicability/track_a_h4_receipt_final_review/2026-10-08/cleanup_ESRCH_mock_reproduction_v1.json)を親側でも確認。実signal/実process/科学アクセス0。
指定SOURCEは今回保持し、source修正は行っていない。CPU等を承認するだけではP1は解消しない。

## 技術状態と未承認の区別

| 区分 | 判定 |
|---|---|
| 固定source/profile・既存57人工証拠 | PASS。旧compiler outputとの完全同一性は未検証 |
| CPU/observer資源/容量/inode/quota・74804案 | 技術案の計算は整合。利用許可・reservationではない |
| own-run STOP/cleanup | TECHNICAL FAIL：P1 ESRCH競合 |
| input/freeze/native stop/control receipt | NOT_EVALUABLE：転送待ち、bytes/manifest確認待ち |
| technical binding / reseal | 不成立。SOURCE修正と受領証拠が必要 |
| env/CPU/observer/+20/launch | UNAPPROVED。技術的不合格とは別の利用者判断 |

現在resource再観測は採取時刻のfresh条件 `PASS_AT_CAPTURE`、profile一致、quota KNOWN（user/group/project DISABLED）。
filesystem available約413.74GiB、free inodes 220497383、memory available約466.08GiB。
採取時刻の条件成立はlaunch時成立・CPU使用許可ではない。launch直前に5秒以内のmemory/FS/quota/inode/pressure/OOMと3秒CPU低負荷sampleを取り直す。

## 保持する次段承認案

H4 linear neutral singlet/STO-3G/DF12、8system＋ancilla1、T0.8、二次DF-prefix PF/canonical finite-RTE、6距離・218 templates/点・32 paired trajectories・1308 signals/74784 logical mapは不変。
新private環境/compilerの採用案、worker12 CPU `[2,4,5,6,8,9,10,11,12,13,14,15]`、driver16、observer18を保持。
observer AS256MiB/RSS64MiB、admission120.25GiB、driver/worker各AS/RSS8GiB/headroom16GiB/monitor5秒。
carry20 actual /165214360 bytes /5466.188392877579秒、現cap74784、残74764を保持する。cache節約20件の保証なし、最小累積cap74804案は未承認。
追加output5GiB/301000 inodes、charge8902109720B（約8.29GiB）、copy前の暫定6GiB案を保持。10GiB/72h上限不変。

allowed_cpus=[]、approved=false、runtime_authorization=false、sealed=false。元plan/auth/reviewはbyte-identicalに保存し、新binding評価だけを追加。
入力未受領・P1不合格があるためplan再sealは行わない。未実行absolute commandは従来資料の提案として保持し、実行しない。
残る手順は受領/停止proof確認→P1修正SOURCE・限定回帰・再binding→独立再review→環境/CPU/observer/+20契約変更を一括承認→利用者の明示launch。
その先の許可範囲もH4 map一度だけ、完了またはfail-closed STOP後に停止、自動retryなし。
