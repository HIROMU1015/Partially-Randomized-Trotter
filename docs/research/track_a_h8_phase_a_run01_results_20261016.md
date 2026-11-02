# Track A H8 Phase A run01：追加152候補のsignal完了・mandatory STOP

**H8 Phase Aは正常終了した。追加152候補のsignal・bias・経験的適格性が揃い、35 workerは全exit0・全回収。wall72.293秒、観測aggregate peak RSS6.846 GiBだった。中断・科学計算のretry・修正再実行は0。費用model固定、compile、費用再会計へ進まずSTOPした。**

実行SOURCE：`fc463d9e07825f34e496acac6b40db7aec1aed6d`。
branch：`track-a-h8-strengthened-deterministic-readiness-20261016`。
[実装・未承認時の契約](track_a_h8_phase_a_implementation_readiness_20261016.md)／[全保存signal・適格性表・終了監査](../../artifacts/resource_applicability/track_a_h8_phase_a_run01/2026-10-16/README.md)。

## 実行承認と固定条件

利用者の「計算を開始して　計算が中断されたら、修正を行って」を、公開済みH8 Phase Aだけの実行承認として新grantへ固定した。失敗時に旧grantを再利用することや、次Phaseへ自動移行することは認めていない。今回のrunは一度で完了した。

linear H8、1.00 Å、STO-3G、保存tol-only DF28、tol1e-8/cutoff0、signed generation order、weighted Hermitian projection。16 spin modes、Nα=Nβ=4、sector4900、T=.8、q1/2/4/8、δ=.8/.4/.2/.1。εはcomplex signal精度の.05/.01/.005/.001/.0005。経験的allowance factor1/10、scalar once、full-H reference不回転、symmetric-directionalとの接続条件を保持した。

新S2はp8〜20から既存p14を除いた96候補、新打切りS4はp15〜21の56候補で、B0・B0I双方を含む。旧32候補・既存PR・旧full p28は再計算していない。新SCF/DF/state、PR law/sampling、full-H reference exponential、native full-wrapper build/compile、費用fit/再会計は0。

## 完了した処理と並列化

|処理|実績|
|---|---:|
|保存snapshot load / full preparation復元|各1|
|新B0 signal / 対応B0I解析的overlay|76 / 76|
|prefix task / truncated-reference pair|13 / 13|
|新tail scalar / 保存p21 scalar再利用|12 / 1|
|primitive task / probe action|22 / 1,650|
|reference matvec実績 / 不可逆予約上限|314,478 / 48,000,000（Phase cap50,000,000）|
|worker / 最大並列数|35 / 16|
|worker CPU合計|499.659235秒|
|launcher wall|72.293307秒|
|観測aggregate peak RSS|6.845833 GiB|
|監視最大gap / 契約上限|3.817427秒 / 5秒|

primitive waveを16並列以内で完了・回収後、13 prefix taskを処理した。snapshot・保存Gaussian操作列・full-H参照を共有し、prefix内ではq・S2/S4間のtruncated referenceを共有、B0IはB0のsignalを再利用した。各role thread1、全35 native exit0、cleanup error0、次Phase呼出0。

worker CPUは第2wave receiptの累積値であり、第1wave値を再加算していない。上記wallは起動前gateの時間を含まない。逐次実行との対照を行っていないため、CPU/wall比を高速化率とは呼ばない。

## 保存適格性の件数

以下は**追加152候補だけ**の保存判定件数。旧32候補を足した最良費用やPRとの費用比ではない。factor1/10の適格性は今回全件同じだった。全候補でεとbiasが同じという意味ではない。

|signal ε|B0 S2（48中）|B0I S2（48中）|B0 S4（28中）|B0I S4（28中）|計|
|---:|---:|---:|---:|---:|---:|
|.05|31|45|28|28|132|
|.01|18|18|28|28|92|
|.005|18|18|21|21|78|
|.001|6|6|21|21|54|
|.0005|6|6|18|21|51|

少なくとも1つのε/factor条件に適格な追加候補は132/152。残20候補をゼロ費用として扱わず、費用model適用や追加compileの理由にもしない。1,520条件行のsigned real/imag bias・allowance・headroom・保存shotは圧縮CSVに、全recordは元JSONのbytesを保つgzipに収録した。shotは既存規則による解析的割当であり、新たな量子shot samplingではない。

primitive経路の最大差は9.74892e-15、B0のnative/spin対独立forward signal差は最大2.29164e-13、state差は最大4.03793e-13。全候補のsigned discard/PF/total分解closureを確認した。B0Iはfull-Fock tail scalarによる位相補正と経験的誤差予算の合成である。これらの小さい経路差は、未知の真の数値誤差・厳密biasの上界や、PR母平均・winner保証ではない。

## 終了・来歴・保全監査

source15、provider265、runtime34,273、input11を実行前後・公開監査でSHA照合した。full-H参照は元workerの完了lineage、保存p21 scalarは既存receiptへ結合した。quotaはread-only `Q_GETFMT`をuser/group/projectで照会し、各ESRCH（当該filesystemでquota未有効）を記録。割当18 physical core・idle、cgroup各祖先のCPU/memory limit、PSI/OOM、disk/inode、旧保護状態のfresh gate通過後に起動した。

旧671 grant/ledger等（従来667と準備中の無効draftを含む）、111他worktreeのHEAD/statusは不変。V1・V2・H5/H7と旧結果・rawを保全した。新grantの消費とworker予約はそのまま保持し、refund/resetしない。public bundleには有効grant本体、private path、分子配列、runtime、credentialを含めない。

原worker receiptのうち`prefix_19`だけは`result_sha256`欄が欠けていた。旧poolの/proc退出競合を待つ分岐では、wait4でexit0を回収した後にこの欄を付けず完了する経路がある。原receiptは修正していない。全35件でexit0・reap・結果fileの存在・task予算照合を確認し、公開時に別のSHA receiptを作成した。prefixごとの保存rowsと全152行の結合も照合した。34件の実行時SHAと、1件の公開時のみのSHAを区別し、欠測欄を遡及補完したとは扱わない。科学結果の欠測・再計算はない。

primary agent自身の読取専用の保存結果・ledger・metadata監査であり、独立レビューや実科学の外部再現ではない。科学runは完了STOP。現段階では新たな人工campaignや実H8再実行を重ねない。

## 次の判断とSTOP

次候補は**別承認のPhase B1**で、保存H8 anchorのSOURCE・compiler/profile/control/scalar整合性を確認して費用modelを固定する。H6係数は使わない。anchorが整合しなければSTOPし、自動で不足compileを追加しない。その後の最大4軸S4検証B2も別承認で、各軸2%以上のmodel誤差なら即STOP、再fitや追加compileなし。5精度の費用再会計Cも別承認である。

**今回確定したのは追加signal・bias・経験的適格性とPhase Aの実行完了であり、H8のPR費用優位の変化はまだ判定していない。V2現行fitはNO_GO終了を維持する。結果をcommit/pushし、remote SHA/tree照合後mandatory STOP。**
