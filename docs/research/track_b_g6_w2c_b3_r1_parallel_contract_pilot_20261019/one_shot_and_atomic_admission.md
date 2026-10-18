# Global one-shotとatomic admission

既存のhost/admission SOURCEは不変。人工テストでは実hostの呼出し経路を使うが、ProductionAdmission factoryをtest内だけArtificialBoundaryBridgeへ差し替え、driverの実行先も人工bundleへ注入する。fake contract/authority/SQLite/outputは全てfresh `/tmp/b3-r1-artificial-*`。実host SOURCEには注入用の引数・環境flag・mock入口を追加しない。

1 global scopeに対して親がBoundary.enterを1回行う。SQLiteは既存のatomic UNIQUE scope/claim reservationと、RESERVED以降spentの実装をそのまま使用する。workerごとのclaimはない。worker起動は科学trialの許可ではなく、各trialに別の親gateがある。

一意trial IDはphase/pass/arm/layer/axis/index。親は期待順序・state byte・global call枠を確認してSTART_RESERVEDを保存してから返信する。枠は返却しない。reservation数はplannedと31,710,848の小さい方を超えない。異常終了・結果不採用・commit ACK不明でも追加samplingで置き換えない。開始許可件数、完了、失敗、不明、未開始、取消を別記する。

既存の正式予算はcorrectness256、baseline15,855,296、instrumented15,855,296、合計31,710,848。保存Nはordinary1,174,526、partial_return_tail894,664、closed_P3_tail880,312、full_return1,014,322（各layer/axis）。人工benchmarkの小Nは性能fixtureであり正式予算を書き換えない。実Nの計画offset/総数だけはsamplingなしで検証する。

ClaimLedgerはreserve/transition/rowの各呼出しで接続し、finallyでcloseする。親は開いたSQLite connection/transactionをforkへ渡さず、workerはledger操作を行わない。worker結果の回収はpipe、親のbyte/state管理も単一process。scopeを同時に2 hostが起動するraceでは1つだけがadmitted。reservation commit直後の例外・親死亡・worker死亡・output前失敗・STOP記録失敗後も再利用を拒否する。release/retry/resume APIは新設しない。

fake authorityの追加input pinで人工policy/settings/prototype hash/fixture hashを束縛する。これは**正式parallel契約を旧v2のまま採択したものではない**。正式v2のscheduleは逐次条件を要求し、実hostは旧 `D.execute` に固定されている。将来は新しいschema/entrypoint/SOURCE rolesと全pin、runtime/import location、契約digest、独立approval、fresh namespaceとstore identityの技術検証が必要。その後も実authorization/claimは別のユーザー承認を要する。今回のartifactは全てrun_ready=false。
